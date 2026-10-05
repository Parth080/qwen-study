"""Reference checks for Lab 03's from-scratch Qwen3 forward pass.

Expected public interface in qwen_lab.model.qwen3_torch:

    RMSNorm(hidden_size: int, eps: float = 1e-6)
    Qwen3ForCausalLM.from_pretrained(snapshot_path, dtype=torch.bfloat16)
    model(input_ids) -> logits  # [batch, sequence, vocab_size]

The full checkpoint check is intentionally substantial. Run its small RMSNorm
check while developing, then run this whole file once the loader is ready:
    pytest tests/test_03_forward.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest
import torch
from huggingface_hub import snapshot_download


REPO = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def implementation():
    try:
        from qwen_lab.model.qwen3_torch import Qwen3ForCausalLM, RMSNorm
    except ImportError as error:
        pytest.fail(
            "write src/qwen_lab/model/qwen3_torch.py with RMSNorm and "
            "Qwen3ForCausalLM (see lab/03_forward_from_scratch/README.md): "
            f"{error}"
        )
    return RMSNorm, Qwen3ForCausalLM


def test_rmsnorm_matches_reference_formula(implementation):
    RMSNorm, _ = implementation
    layer = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 0.5, 2.0, -1.0]))
    values = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.0, 2.0, -1.0, 3.0]])
    expected = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * layer.weight
    torch.testing.assert_close(layer(values), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_loaded_06b_logits_match_transformers_fp32(implementation):
    """Numerically tight check in fp32 on a short fixed sequence.

    Measured noise floor between two *correct* fp32 implementations on this
    machine is ~1e-4 (op order); a real bug produces differences > 0.1.
    """
    _, Qwen3ForCausalLM = implementation
    from transformers import AutoModelForCausalLM

    snapshot = snapshot_download(REPO)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = torch.tensor([[9707, 1879, 0, 220, 17, 15, 17, 19]], device=device)

    reference = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).to(device).eval()
    mine = Qwen3ForCausalLM.from_pretrained(snapshot, dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        expected = reference(input_ids=ids).logits
        actual = mine(ids)

    assert actual.shape == expected.shape == (1, ids.shape[1], 151936)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_loaded_06b_bf16_top1_agrees(implementation):
    """In bf16 two correct implementations differ by ~0.3 in raw logits purely
    from accumulation order, so bf16 is checked on the decision, not the values."""
    _, Qwen3ForCausalLM = implementation
    from transformers import AutoModelForCausalLM

    snapshot = snapshot_download(REPO)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = torch.tensor([[9707, 1879, 0, 220, 17, 15, 17, 19]], device=device)

    reference = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16).to(device).eval()
    mine = Qwen3ForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        expected = reference(input_ids=ids).logits.float()
        actual = mine(ids).float()

    assert torch.equal(actual.argmax(-1), expected.argmax(-1)), "top-1 token differs somewhere in the sequence"
    assert (actual - expected).abs().mean() < 0.1
