"""Capture and patch a toy activation before applying the same API to Qwen."""
import torch
from qwen_lab.analysis.hooks import capture, replace_output

model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU(), torch.nn.Linear(2, 1, bias=False))
x = torch.tensor([[1.0, -2.0]])
with capture(model[1]) as saved:
    baseline = model(x)
with replace_output(model[1], torch.zeros_like(saved["value"])):
    patched = model(x)
print("captured activation:", saved["value"].tolist())
print("baseline:", baseline.item(), "patched:", patched.item())
