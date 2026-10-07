"""Cost probe for ContiFormer and ContiFormer-fast.

ContiFormer could not complete a single long-term cell within the 4-day
wall-time.  The library now offers (a) an optimised ContiFormer (identical
arithmetic, less per-step overhead, bounded memory through gradient
checkpointing) and (b) ContiFormer-fast, the same continuous-time model with
one ODE trajectory per start point instead of one solve per pair of time
steps.

This probe measures, on the GPU it runs on, the peak memory and the wall time
of one training step (forward + backward + optimizer) and one inference batch
for a list of configurations, and extrapolates:

* the time of one training epoch on every core benchmark dataset, and
* the GPU-hours of the full core long-term grid (7 datasets × 4 horizons ×
  3 seeds) assuming ``EPOCHS_ASSUMED`` epochs per cell.

Random inputs are used: the cost of the ODE solver depends only on tensor
shapes.  The model configuration mirrors the one the benchmark builds for
ContiFormer (d_model=128, 4 layers, 4 heads, d_inner=256, rk4,
atol_ode=1e-2).  It does not produce benchmark results.

Output: ``--out`` (default ``artifacts/review/contiformer_probe.json``).

Usage:
    python scripts/probe_contiformer.py [--steps 3]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "s-transformers-lib"))

from s_transformers_lib.models import create_model  # noqa: E402

SEQ_LEN = 96
ENC_IN = 7
# Typical epochs per cell before early stopping (mean over the completed
# core cells of the other models is 19.5; median 15; the cap is 30).
EPOCHS_ASSUMED = 20
SEEDS = 3
# Training rows per core dataset (standard splits); windows per epoch are
# rows - seq_len - horizon + 1.
TRAIN_ROWS = {
    "ETTh1": 8640, "ETTh2": 8640, "ETTm1": 34560, "ETTm2": 34560,
    "exchange_rate": 5311, "weather": 36887, "illness": 676,
}
HORIZONS = {name: (24, 36, 48, 60) if name == "illness" else (96, 192, 336, 720)
            for name in TRAIN_ROWS}
# Validation + test windows are roughly 3/7 of the training windows and are
# evaluated every epoch / once; inference costs ~1/4 of a training step per
# sample, so they add about 15 % on top of training.
EVAL_OVERHEAD = 1.15

# (model, batch_size, gradient checkpointing, steps_per_interval, CUDA graphs)
CONFIGS = [
    ("contiformer_fast", 16, True, 1, False),
    ("contiformer_fast", 16, True, 1, True),
    ("contiformer_fast", 32, True, 1, True),
    ("contiformer_fast", 64, True, 1, True),
    ("contiformer_fast", 128, True, 1, True),
]


def _grid_gpu_hours(sec_per_sample: float) -> dict:
    """GPU-hours of the core long-term grid for a given training cost.

    Args:
        sec_per_sample: Training wall time per sample (seconds).

    Returns:
        Dict with per-dataset GPU-hours and the total.
    """
    per_ds = {}
    for name, rows in TRAIN_ROWS.items():
        windows = sum(max(rows - SEQ_LEN - h + 1, 0) for h in HORIZONS[name])
        per_ds[name] = round(
            windows * sec_per_sample * EPOCHS_ASSUMED * SEEDS * EVAL_OVERHEAD / 3600, 1
        )
    per_ds["total"] = round(sum(per_ds.values()), 1)
    return per_ds


def _measure(model_name: str, batch: int, ckpt: bool, spi: int | None,
             graphs: bool, steps: int, device: str) -> dict:
    """Time one configuration and return peak memory and per-step timings.

    Args:
        model_name: ``contiformer`` or ``contiformer_fast``.
        batch: Batch size.
        ckpt: Whether ODE gradient checkpointing is enabled.
        spi: ``steps_per_interval`` for the fast model (``None`` otherwise).
        graphs: Whether the fast model replays its ODE sweeps from CUDA graphs.
        steps: Number of timed training steps (after one warm-up step).
        device: Torch device string.

    Returns:
        Dict with the configuration, peak memory (MB), training-step and
        inference times, epoch-time and full-grid extrapolations, or the
        error on failure.
    """
    row = {"model": model_name, "seq_len": SEQ_LEN, "batch_size": batch,
           "checkpointing": ckpt, "steps_per_interval": spi,
           "cuda_graphs": graphs}
    cfg = dict(seq_len=SEQ_LEN, pred_len=96, enc_in=ENC_IN, d_model=128,
               d_inner=256, n_layers=4, n_head=4, use_gradient_checkpointing=ckpt)
    if spi is not None:
        cfg["steps_per_interval"] = spi
        cfg["use_cuda_graphs"] = graphs
    torch.manual_seed(0)
    model = create_model(model_name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(batch, SEQ_LEN, ENC_IN, device=device)
    y = torch.randn(batch, 96, ENC_IN, device=device)
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        times = []
        for i in range(steps + 1):  # step 0 is warm-up (and graph capture)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if i > 0:
                times.append(time.perf_counter() - t0)
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            infer_s = time.perf_counter() - t0
        step_s = sum(times) / len(times)
        per_sample = step_s / batch
        row.update(
            status="ok",
            peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            train_step_s=round(step_s, 3),
            train_s_per_sample=round(per_sample, 4),
            infer_batch_s=round(infer_s, 3),
            epoch_hours_h96={
                name: round((rows - SEQ_LEN - 96 + 1) * per_sample / 3600, 2)
                for name, rows in TRAIN_ROWS.items() if name != "illness"
            },
            core_grid_gpu_hours=_grid_gpu_hours(per_sample),
        )
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status="oom", error=str(exc).splitlines()[0])
    except RuntimeError as exc:  # e.g. a CUDA-graph capture failure
        row.update(status="error", error=str(exc).splitlines()[0][:300])
    finally:
        del model, opt, x, y
        torch.cuda.empty_cache()
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    """Run the probe grid and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", default="artifacts/review/contiformer_probe.json")
    args = parser.parse_args()

    rows = [_measure(m, b, c, s, g, args.steps, "cuda") for (m, b, c, s, g) in CONFIGS]
    report = {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model_config": "seq_len=96 d_model=128 d_inner=256 n_layers=4 n_head=4 rk4 atol_ode=1e-2",
        "epochs_assumed": EPOCHS_ASSUMED,
        "rows": rows,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
