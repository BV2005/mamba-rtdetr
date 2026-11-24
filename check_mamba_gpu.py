"""Checks that mamba_ssm's CUDA kernels run (forward + backward) on the GPU; used by train_full_coco.sh.
Exit code 0 if they do, 1 otherwise. A mamba-ssm installed without its CUDA build imports fine but fails here.

    python check_mamba_gpu.py [--quiet]
"""
import sys


def main():
    quiet = "--quiet" in sys.argv
    try:
        import torch
        from mamba_ssm import Mamba
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible to PyTorch")
        m = Mamba(d_model=256).cuda()
        x = torch.randn(2, 64, 256, device="cuda", requires_grad=True)
        m(x).sum().backward()
        torch.cuda.synchronize()
    except Exception as e:  # ImportError (not installed / no CUDA extension) or a CUDA error
        if not quiet:
            print(f"  mamba_ssm on the GPU: FAILED ({type(e).__name__}: {e})")
        return 1
    if not quiet:
        print(f"  mamba_ssm CUDA kernels work on {torch.cuda.get_device_name(0)} (forward + backward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
