"""Quantise a known 64-value vector and report reconstruction error."""
import torch
from qwen_lab.model.quantization import symmetric_dequantize, symmetric_quantize

x = torch.linspace(-2.0, 3.0, 64)
for bits in (8, 4):
    q, scale, shape = symmetric_quantize(x, bits=bits, group_size=64)
    restored = symmetric_dequantize(q, scale, shape)
    print(f"int{bits}: scale={scale.item():.6f}, mse={(x - restored).square().mean().item():.8f}, "
          f"max_error={(x - restored).abs().max().item():.6f}")
