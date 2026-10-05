"""Reference checks for Lab 08's hand-written AdamW.

Expected public interface in qwen_lab.train.optimizers:

    AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer.step()
    optimizer.zero_grad()

The class should update torch Parameters in place using their .grad tensors and
must match torch.optim.AdamW's decoupled weight decay semantics.
"""
import pytest
import torch


@pytest.fixture
def optimizers():
    try:
        from qwen_lab.train.optimizers import AdamW
    except ImportError as error:
        pytest.fail(
            "write src/qwen_lab/train/optimizers.py with AdamW (see "
            "lab/08_backward_optimizers/README.md): "
            f"{error}"
        )
    return AdamW


def test_adamw_matches_torch_over_fixed_gradient_sequence(optimizers):
    AdamW = optimizers
    initial = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.5]], dtype=torch.float64)
    mine_parameter = torch.nn.Parameter(initial.clone())
    torch_parameter = torch.nn.Parameter(initial.clone())
    options = dict(lr=0.03, betas=(0.8, 0.93), eps=1e-9, weight_decay=0.07)
    mine = AdamW([mine_parameter], **options)
    reference = torch.optim.AdamW([torch_parameter], foreach=False, fused=False, **options)
    gradients = [
        torch.tensor([[0.2, -0.4, 0.1], [0.0, 0.3, -0.2]], dtype=torch.float64),
        torch.tensor([[-0.1, 0.6, 0.3], [0.4, -0.2, 0.2]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.8], [-0.5, 0.1, -0.7]], dtype=torch.float64),
    ]
    for gradient in gradients:
        mine_parameter.grad = gradient.clone()
        torch_parameter.grad = gradient.clone()
        mine.step()
        reference.step()
        torch.testing.assert_close(mine_parameter, torch_parameter, rtol=1e-12, atol=1e-12)


def test_adamw_skips_parameters_without_a_gradient(optimizers):
    AdamW = optimizers
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.1)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([1.0, -1.0]))


def test_zero_grad_clears_gradient_tensors(optimizers):
    AdamW = optimizers
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    parameter.grad = torch.tensor([0.2, -0.3])
    optimizer = AdamW([parameter], lr=0.1)
    optimizer.zero_grad()
    assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
