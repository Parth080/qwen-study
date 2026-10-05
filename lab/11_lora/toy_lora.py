"""Verify that a zero-initialised LoRA B keeps the initial base output intact."""
import torch
from qwen_lab.train.lora import LoRALinear, lora_parameter_count

torch.manual_seed(0)
base = torch.nn.Linear(4, 3, bias=False)
adapter = LoRALinear(base, rank=2, alpha=4)
x = torch.randn(2, 4)
print("initial update max:", adapter.delta_weight().abs().max().item())
print("adapter parameters:", lora_parameter_count(base, 2), "vs full:", base.weight.numel())
before = adapter(x); adapter.merge(); after = adapter(x)
print("merge max output difference:", (before - after).abs().max().item())
