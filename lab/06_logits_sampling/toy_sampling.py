"""A deterministic, inspectable playground for Lab 06 sampling filters."""
import torch
from qwen_lab.eval.sampling import min_p, top_k, top_p, temperature

logits = torch.tensor([4.0, 2.0, 1.0, 0.0, -2.0])
for name, result in {
    "original": logits,
    "temperature=0.7": temperature(logits, 0.7),
    "top_k=3": top_k(logits, 3),
    "top_p=0.8": top_p(logits, 0.8),
    "min_p=0.1": min_p(logits, 0.1),
}.items():
    print(name, "logits=", result.tolist(), "probs=", result.softmax(-1).tolist())
