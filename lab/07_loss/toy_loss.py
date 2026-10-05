"""Show the causal one-token shift and a per-token loss vector."""
import torch
from qwen_lab.eval.losses import causal_cross_entropy, per_token_causal_loss

logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0]]])
labels = torch.tensor([[0, 1, 2]])
print("labels:", labels.tolist(), "; logits positions 0 and 1 predict labels 1 and 2")
print("per-token shifted loss:", per_token_causal_loss(logits, labels).tolist())
print("mean:", causal_cross_entropy(logits, labels).item())
