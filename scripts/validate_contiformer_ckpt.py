"""Numerical check for the ContiFormer changes in the library.

Runs one forward + backward pass on a fixed random batch with fixed weights
(``zero_init_ode=False`` so the ODE is non-trivial) and saves the output and
the flattened parameter gradients, so that four variants can be compared:

  * ``old``    — the library before the changes (pinned commit, separate path)
  * ``nockpt`` — the new ContiFormer with ``use_gradient_checkpointing=False``
  * ``ckpt``   — the new ContiFormer with ``use_gradient_checkpointing=True``
  * ``fast``   — ContiFormer-fast (shared trajectories, 2 steps per
    interval, plain autograd) with the same weights; ``fast1`` uses 1 step
    per interval with per-step recomputation (eager) and ``fast1g`` the same
    replayed from CUDA graphs

``ckpt`` vs ``nockpt`` must match exactly (same arithmetic, recomputed).
``nockpt`` vs ``old`` must match to fp32 noise: the optimisations remove
overhead but keep the arithmetic (the solver's time loop runs on Python
floats instead of float32 tensors, which can shift the grid by ~1e-7).
``fast`` vs ``nockpt`` measures the discretisation difference of the shared
trajectory schedule: it should be small relative to the output scale.

Each variant runs in its own process (two versions of the same package cannot
be imported side by side).  Must run on a GPU node.

Usage:
    python validate_contiformer_ckpt.py run --lib PATH --mode ckpt --out f.pt
    python validate_contiformer_ckpt.py compare --dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SEQ_LEN, BATCH, ENC_IN = 32, 2, 7


def run(lib: str, mode: str, out: str) -> None:
    """Run one forward/backward pass and save output, gradients and memory.

    Args:
        lib: Path to the library root to import (old or new version).
        mode: ``old``, ``nockpt``, ``ckpt``, ``fast``, ``fast1`` or ``fast1g``.
        out: Path of the ``.pt`` file to write.
    """
    sys.path.insert(0, lib)
    from s_transformers_lib.models import create_model  # noqa: PLC0415

    torch.manual_seed(0)
    cfg = dict(seq_len=SEQ_LEN, pred_len=24, enc_in=ENC_IN, d_model=128,
               n_layers=2, n_head=4, dropout=0.0, zero_init_ode=False)
    if mode != "old":  # the old library has no such option
        cfg["use_gradient_checkpointing"] = mode == "ckpt"
    # ContiFormer-fast builds the same modules in the same order, so the same
    # seed gives it exactly the same initial weights as ContiFormer.
    if mode.startswith("fast"):
        cfg["steps_per_interval"] = 1 if mode.startswith("fast1") else 2
        cfg["use_gradient_checkpointing"] = mode.startswith("fast1")
        cfg["use_cuda_graphs"] = mode == "fast1g"
    model = create_model("contiformer_fast" if mode.startswith("fast") else "contiformer", cfg)
    model = model.cuda().train()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, SEQ_LEN, ENC_IN, generator=gen).cuda()

    torch.cuda.reset_peak_memory_stats()
    y = model(x)
    y.pow(2).mean().backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    torch.save({
        "y": y.detach().cpu(),
        "g": grads.cpu(),
        "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }, out)


def compare(directory: str) -> None:
    """Compare the three saved variants and print/write a JSON summary.

    Args:
        directory: Folder holding ``old.pt``, ``nockpt.pt``, ``ckpt.pt`` and
            ``fast.pt``, ``fast1.pt`` and ``fast1g.pt``.
    """
    d = Path(directory)
    old, nockpt, ckpt, fast, fast1, fast1g = (
        torch.load(d / f"{k}.pt")
        for k in ("old", "nockpt", "ckpt", "fast", "fast1", "fast1g")
    )

    def diff(a: dict, b: dict) -> dict:
        """Max absolute output difference and relative gradient difference."""
        return {
            "max_abs_output_diff": float((a["y"] - b["y"]).abs().max()),
            "rel_grad_diff": float((a["g"] - b["g"]).norm() / b["g"].norm()),
        }

    summary = {
        "shape": {"seq_len": SEQ_LEN, "batch": BATCH, "n_layers": 2, "d_model": 128,
                  "zero_init_ode": False},
        "ckpt_vs_nockpt": diff(ckpt, nockpt),
        "nockpt_vs_old": diff(nockpt, old),
        "fast_vs_nockpt": diff(fast, nockpt),
        "fast1_vs_nockpt": diff(fast1, nockpt),
        "fast1g_vs_fast1": diff(fast1g, fast1),
        "output_scale": float(nockpt["y"].abs().mean()),
        "peak_mb": {k: round(v["peak_mb"], 1) for k, v in
                    (("old", old), ("nockpt", nockpt), ("ckpt", ckpt), ("fast", fast),
                     ("fast1", fast1), ("fast1g", fast1g))},
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Dispatch the ``run`` / ``compare`` sub-commands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--lib", required=True)
    p_run.add_argument("--mode", required=True, choices=["old", "nockpt", "ckpt", "fast", "fast1", "fast1g"])
    p_run.add_argument("--out", required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--dir", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.lib, args.mode, args.out)
    else:
        compare(args.dir)


if __name__ == "__main__":
    main()
