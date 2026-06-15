"""Quick GPU sanity check for the benchmark environment.

Run on a compute node (via srun) to confirm the installed PyTorch build can
actually execute kernels on that node's GPU. Exercises:

  1. CUDA availability + device/arch metadata.
  2. A matmul (core CUDA kernels).
  3. A conv2d (cuDNN) — this is what breaks on V100 when cuDNN dropped Volta.

Exit code 0 means the node is usable for the benchmark; non-zero means the
torch build is incompatible with this GPU.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    """Run the GPU checks and return a process exit code."""
    print(f"torch           : {torch.__version__}")
    print(f"cuda build      : {torch.version.cuda}")
    print(f"cudnn           : {torch.backends.cudnn.version()}")
    print(f"compiled archs  : {torch.cuda.get_arch_list()}")
    print(f"cuda available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available on this node.")
        return 1

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"device          : {name}  (sm_{cap[0]}{cap[1]})")

    try:
        # Core CUDA kernels.
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        c = (a @ b).sum().item()
        print(f"matmul OK       : sum={c:.3f}")

        # cuDNN path — the actual Volta breakage point.
        x = torch.randn(8, 3, 32, 32, device=dev)
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)
        y = conv(x)
        torch.cuda.synchronize()
        print(f"conv2d/cuDNN OK : out={tuple(y.shape)}")
    except Exception as exc:  # noqa: BLE001 - we want to report any GPU failure
        print(f"FAIL: GPU kernel execution failed: {type(exc).__name__}: {exc}")
        return 2

    print("PASS: this node can run the benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
