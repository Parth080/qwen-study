"""Phase 0 check: does the machine see the GPU from both PyTorch (MPS) and MLX?"""
import platform, sys, time

print(f"python {sys.version.split()[0]}  arch {platform.machine()}  macOS {platform.mac_ver()[0]}")

import torch
print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}")
x = torch.randn(4096, 4096, device="mps", dtype=torch.bfloat16)
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(10):
    y = x @ x
torch.mps.synchronize(); dt = time.perf_counter() - t
print(f"  mps bf16 matmul 4096^2 x10: {dt*1000:.0f} ms  ({10*2*4096**3/dt/1e12:.1f} TFLOP/s)")

import mlx.core as mx
print(f"mlx {mx.__version__}  device={mx.default_device()}")
a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
mx.eval(a); t = time.perf_counter()
for _ in range(10):
    b = a @ a
mx.eval(b); dt = time.perf_counter() - t
print(f"  mlx bf16 matmul 4096^2 x10: {dt*1000:.0f} ms  ({10*2*4096**3/dt/1e12:.1f} TFLOP/s)")

import transformers, mlx_lm, peft, datasets, nnsight
for m in (transformers, mlx_lm, peft, datasets, nnsight):
    print(f"{m.__name__} {m.__version__}")
print("OK")
