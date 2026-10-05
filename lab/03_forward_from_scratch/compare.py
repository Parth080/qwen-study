"""Compare the local readable Qwen3 forward pass with Hugging Face logits."""
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from qwen_lab.model.qwen3_torch import Qwen3ForCausalLM

snapshot = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32
ids = torch.tensor([[9707, 1879, 0, 220, 17, 15, 17, 19]], device=device)
reference = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype).to(device).eval()
mine = Qwen3ForCausalLM.from_pretrained(snapshot, dtype=dtype).to(device).eval()
with torch.no_grad():
    expected = reference(input_ids=ids).logits.float()
    actual = mine(ids).float()
print("shape:", tuple(actual.shape))
print("max abs error:", (actual - expected).abs().max().item())
print("reference next id:", expected[0, -1].argmax().item())
print("local next id:    ", actual[0, -1].argmax().item())
