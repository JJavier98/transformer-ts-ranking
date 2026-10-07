"""Find why AirFormer produces NaN on the weather dataset (GPU node only).

Every AirFormer run on weather (4 horizons × 3 seeds) finished with NaN
metrics and a NaN training loss from the very first epoch, while the same
model trains normally on ETTh1/ETTh2/ETTm1/ETTm2/exchange_rate/illness.

This script reproduces the benchmark's setup for one cell (same data loader,
same config, same optimiser, same seed), then:

1. checks that the input batches are finite;
2. runs the first training steps and records, with forward hooks, the first
   module whose output is non-finite, plus value statistics of the stochastic
   block (log-variance range, KL term);
3. repeats the experiment with candidate fixes (no stochastic block; bounded
   log-variance) and reports whether the loss stays finite over a short
   training run.

Output: ``artifacts/review/airformer_weather_diagnosis.json``.

Usage (inside an sbatch job):
    python scripts/diagnose_airformer.py [--dataset weather] [--horizon 96]
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


def build(dataset, horizon: int, overrides: dict, seed: int = 42):
    """Create an AirFormer exactly as the benchmark does, plus config overrides."""
    cfg = build_long_term_config(
        "airformer", len(dataset.feature_columns), horizon,
        dataset.seq_len, dataset.label_len, dataset.frequency)
    model_cfg = filter_config_for_model("airformer", cfg)
    torch.manual_seed(seed)
    config = dict(model_cfg.to_dict()) if hasattr(model_cfg, "to_dict") else dict(model_cfg)
    config.update(overrides)
    return create_model("airformer", config=config), config


def first_non_finite(model, batch) -> dict:
    """Run one training forward pass and report the first non-finite module output."""
    found: dict = {}
    hooks = []

    def make_hook(name):
        def hook(_m, _inp, out):
            tensors = out if isinstance(out, tuple) else (out,)
            for t in tensors:
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    found.setdefault("module", name)
                    found.setdefault("shape", list(t.shape))
                    found.setdefault("nan", int(torch.isnan(t).sum()))
                    found.setdefault("inf", int(torch.isinf(t).sum()))
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))
    model.train()
    with torch.no_grad():
        out, kl = model.forecast(batch["x"])
    for h in hooks:
        h.remove()
    found["output_finite"] = bool(torch.isfinite(out).all())
    found["kl_finite"] = bool(torch.isfinite(kl).all())
    found["kl_value"] = float(kl) if torch.isfinite(kl).all() else None
    return found


def logvar_stats(model, batch) -> dict:
    """Range of the log-variance produced by every latent layer on one batch."""
    stats: dict = {}
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "logvar_layer"):
            def hook(_m, _i, out, name=name):
                z, mu, logvar = out
                stats[name] = {
                    "logvar_min": float(logvar.min()), "logvar_max": float(logvar.max()),
                    "mu_absmax": float(mu.abs().max()),
                }
            hooks.append(module.register_forward_hook(hook))
    model.train()
    with torch.no_grad():
        model.forecast(batch["x"])
    for h in hooks:
        h.remove()
    return stats


def short_training(model, loader, steps: int, device: str) -> dict:
    """Train for a few steps with the library's train_step; report loss finiteness."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        losses.append(model.train_step(batch, opt, loss_fn, device=device))
    arr = np.array(losses)
    return {"steps": len(losses), "all_finite": bool(np.isfinite(arr).all()),
            "first_non_finite_step": int(np.argmax(~np.isfinite(arr))) if (~np.isfinite(arr)).any() else None,
            "first_losses": [float(x) for x in arr[:5]], "last_loss": float(arr[-1])}


def main() -> None:
    """Run the diagnosis and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="weather")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    device = "cuda"

    dataset = load_long_term_dataset(
        REPO, args.dataset, REPO / "configs" / "benchmark" / "long_term_datasets.yaml")
    train = LongTermWindowDataset(dataset, args.horizon, "train")
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    report: dict = {
        "gpu": torch.cuda.get_device_name(0), "dataset": args.dataset, "horizon": args.horizon,
        "n_channels": len(dataset.feature_columns),
        "input": {"x_finite": bool(torch.isfinite(batch["x"]).all()),
                  "y_finite": bool(torch.isfinite(batch["y"]).all()),
                  "x_absmax": float(batch["x"].abs().max()),
                  "scaled_values_finite": bool(np.isfinite(dataset.scaled_values).all()),
                  "scaled_absmax": float(np.abs(dataset.scaled_values).max())},
        "variants": {},
    }
    variants = {
        "benchmark_config": {},
        "no_stochastic": {"stochastic_flag": False},
        "no_spatial": {"spatial_flag": False},
        "no_stochastic_no_spatial": {"stochastic_flag": False, "spatial_flag": False},
    }
    for name, overrides in variants.items():
        model, config = build(dataset, args.horizon, overrides)
        model.to(device)
        entry = {"config": {k: config.get(k) for k in (
            "d_model", "n_heads", "e_layers", "dropout", "spatial_flag", "stochastic_flag",
            "mlp_expansion", "end_channels")}}
        entry["first_forward"] = first_non_finite(model, batch)
        if name == "benchmark_config" and model.stochastic_flag:
            entry["logvar"] = logvar_stats(model, batch)
        model, _ = build(dataset, args.horizon, overrides)
        model.to(device)
        entry["short_training"] = short_training(model, loader, args.steps, device)
        report["variants"][name] = entry
        print(name, json.dumps(entry["first_forward"]), json.dumps(entry["short_training"]),
              flush=True)

    out = REPO / "artifacts" / "review" / f"airformer_{args.dataset}_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
