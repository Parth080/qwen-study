"""Show AdamW's moments and decoupled update for a single parameter."""
import torch
from qwen_lab.train.optimizers import AdamW

p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
opt = AdamW([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.01)
for step, grad in enumerate((torch.tensor([0.5, -0.25]), torch.tensor([-0.2, 0.4])), 1):
    p.grad = grad; opt.step(); state = opt.state[p]
    print(f"step {step}: parameter={p.tolist()} m={state['exp_avg'].tolist()} v={state['exp_avg_sq'].tolist()}")
