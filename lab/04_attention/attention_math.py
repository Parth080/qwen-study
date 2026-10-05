"""Print a causal mask and one hand-checkable attention calculation."""
import math
import torch

q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
k = q.clone(); v = torch.tensor([[[10.0], [20.0], [30.0]]])
scores = q @ k.transpose(-2, -1) / math.sqrt(2)
mask = torch.full((3, 3), float("-inf")).triu(1)
weights = (scores + mask).softmax(-1)
print("causal mask:\n", mask)
print("attention weights:\n", weights)
print("outputs:\n", weights @ v)
