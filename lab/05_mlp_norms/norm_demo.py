"""Make RMSNorm versus LayerNorm visible on a tiny batch."""
import torch
from qwen_lab.model.qwen3_torch import RMSNorm

x = torch.tensor([[1.0, 2.0, 5.0, 10.0], [-2.0, 0.0, 3.0, 7.0]])
rms = RMSNorm(4); layer = torch.nn.LayerNorm(4, elementwise_affine=False)
print("input mean: ", x.mean(-1).tolist(), "RMS:", x.square().mean(-1).sqrt().tolist())
print("RMSNorm mean:", rms(x).mean(-1).tolist(), "RMS:", rms(x).square().mean(-1).sqrt().tolist())
print("LayerNorm mean:", layer(x).mean(-1).tolist(), "std:", layer(x).std(-1, unbiased=False).tolist())
