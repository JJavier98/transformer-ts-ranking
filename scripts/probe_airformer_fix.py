"""Test candidate fixes for AirFormer's NaN on weather (GPU node only).

``diagnose_airformer.py`` showed that the stochastic latent block makes the
loss go to ``inf``/NaN within a few steps on weather, and that the model
trains normally without it.  The block computes ``exp(logvar)`` and divides by
``exp(logvar_prior)`` with an unbounded log-variance.

This script monkeypatches ``LatentLayer.forward`` with a log-variance clamp of
different widths (without touching the library), trains for a few hundred
steps on the benchmark cell with the library's own ``train_step`` and
reports, for each bound and seed:

* whether every loss stayed finite and the step where it first did not,
* the reconstruction and KL terms separately (to see which one explodes),
* the largest |log-variance| reached before the clamp,
* the final training loss.

A bound is a good fix when it keeps the run finite on every seed while being
loose enough not to change the model where it already works (log-variances on
ETT/exchange/illness stay inside it).  Output:
``artifacts/review/airformer_fix_probe.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "s-transformers-lib"))

from transformer_ts_ranking.benchmark.model_configs import (  # noqa: E402
    build_long_term_config,
    filter_config_for_model,
)
from transformer_ts_ranking.benchmark.window_dataset import LongTermWindowDataset  # noqa: E402
from transformer_ts_ranking.data.long_term import load_long_term_dataset  # noqa: E402
from s_transformers_lib.models import create_model  # noqa: E402
from s_transformers_lib.models.airformer import encoder as enc  # noqa: E402

_ORIGINAL_FORWARD = enc.LatentLayer.forward
STATS: dict = {"max_abs_logvar": 0.0}


def patched_forward(bound: float | None):
    """Return a ``LatentLayer.forward`` that clamps the log-variance to ±bound."""
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        STATS["max_abs_logvar"] = max(STATS["max_abs_logvar"],
                                      float(logvar.detach().abs().max()))
        if bound is not None:
            logvar = logvar.clamp(-bound, bound)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar
    return forward


def run(dataset, horizon: int, bound: float | None, seed: int, steps: int, device: str) -> dict:
    """Train one configuration and summarise the loss trajectory."""
    enc.LatentLayer.forward = patched_forward(bound)
    STATS["max_abs_logvar"] = 0.0
    cfg = filter_config_for_model("airformer", build_long_term_config(
        "airformer", len(dataset.feature_columns), horizon,
        dataset.seq_len, dataset.label_len, dataset.frequency))
    config = dict(cfg.to_dict()) if hasattr(cfg, "to_dict") else dict(cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_model("airformer", config=config).to(device)
    loader = DataLoader(LongTermWindowDataset(dataset, horizon, "train"), batch_size=16,
                        shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()
    recon, kl_terms = [], []
    model.train()
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        x, y = batch["x"].to(device), batch["y"].to(device)
        opt.zero_grad()
        pred, kl = model.forecast(x)
        r = mse(pred, y)
        loss = r + 0.01 * kl
        recon.append(float(r))
        kl_terms.append(float(kl))
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
    r, k = np.array(recon), np.array(kl_terms)
    bad = ~np.isfinite(r + 0.01 * k)
    return {
        "bound": bound, "seed": seed, "steps_run": len(r),
        "finite": bool(not bad.any()),
        "first_non_finite_step": int(np.argmax(bad)) if bad.any() else None,
        "recon_at_failure": float(r[np.argmax(bad)]) if bad.any() else None,
        "kl_at_failure": float(k[np.argmax(bad)]) if bad.any() else None,
        "recon_first5": [float(v) for v in r[:5]],
        "recon_last100_mean": float(r[-100:].mean()) if not bad.any() else None,
        "kl_max_finite": float(k[np.isfinite(k)].max()) if np.isfinite(k).any() else None,
        "max_abs_logvar_before_clamp": STATS["max_abs_logvar"],
    }


def main() -> None:
    """Run the grid of bounds × seeds and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="weather")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--bounds", default="none,10,6,4,2")
    args = parser.parse_args()
    dataset = load_long_term_dataset(
        REPO, args.dataset, REPO / "configs" / "benchmark" / "long_term_datasets.yaml")
    rows = []
    for b in args.bounds.split(","):
        bound = None if b == "none" else float(b)
        for seed in [int(s) for s in args.seeds.split(",")]:
            row = run(dataset, args.horizon, bound, seed, args.steps, "cuda")
            rows.append(row)
            print(json.dumps(row), flush=True)
    enc.LatentLayer.forward = _ORIGINAL_FORWARD
    suffix = "" if args.dataset == "weather" else f"_{args.dataset}"
    out = REPO / "artifacts" / "review" / f"airformer_fix_probe{suffix}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "horizon": args.horizon,
                               "gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
