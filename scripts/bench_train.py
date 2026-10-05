"""LoRA training tokens/sec with mlx-lm on a synthetic chat dataset. Fills §7 of PLAN.md.

usage: python scripts/bench_train.py Qwen/Qwen3-0.6B [--iters 30] [--batch 4] [--seq 1024]
"""
import argparse, json, random, re, subprocess, sys, tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("model", nargs="?", default="Qwen/Qwen3-0.6B")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--seq", type=int, default=1024)
p.add_argument("--extra", default="", help="extra args passed to mlx_lm.lora, e.g. '--num-layers 8'")
a = p.parse_args()

random.seed(0)
words = "attention gradient tokenizer residual softmax embedding logits rotary norm optimizer loss batch sequence kernel memory".split()
def sample():
    q = " ".join(random.choices(words, k=random.randint(20, 40)))
    ans = " ".join(random.choices(words, k=random.randint(150, 250)))
    return {"messages": [{"role": "user", "content": f"Explain: {q}"}, {"role": "assistant", "content": ans}]}

d = Path(tempfile.mkdtemp(prefix="bench_data_"))
for split, n in (("train", 400), ("valid", 40)):
    with open(d / f"{split}.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(sample()) + "\n")

cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", a.model, "--train", "--data", str(d),
       "--iters", str(a.iters), "--batch-size", str(a.batch), "--max-seq-length", str(a.seq),
       "--steps-per-report", "5", "--steps-per-eval", "1000", "--adapter-path", str(d / "adapters"),
       "--save-every", "1000", *a.extra.split()]
print(" ".join(cmd))
out = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in (out.stdout + out.stderr).splitlines() if "Tokens/sec" in l or "Trainable" in l or "Peak mem" in l]
print("\n".join(lines[-6:]) if lines else (out.stdout + out.stderr)[-2000:])
tps = [float(m) for m in re.findall(r"Tokens/sec ([\d.]+)", out.stdout + out.stderr)]
if tps:
    print(f"\n{a.model}: median train tok/s = {sorted(tps)[len(tps)//2]:.0f} (batch {a.batch}, seq {a.seq})")
