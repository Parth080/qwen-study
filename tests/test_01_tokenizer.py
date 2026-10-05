"""Lab 01 reference check.

Compares YOUR tokenizer (src/qwen_lab/tokenizer/bpe.py) against the Hugging Face one, token for token.
Run:  pytest tests/test_01_tokenizer.py
      pytest tests/test_01_tokenizer.py -k basic     # only the first tier while you build up
"""
import os, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

REPO = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def ref():
    return AutoTokenizer.from_pretrained(REPO)


@pytest.fixture(scope="module")
def mine():
    try:
        from qwen_lab.tokenizer.bpe import BPETokenizer
    except ImportError as e:
        pytest.fail(f"write src/qwen_lab/tokenizer/bpe.py with class BPETokenizer (see lab/01_tokenizer/README.md): {e}")
    path = snapshot_download(REPO, allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"])
    return BPETokenizer.from_pretrained(path)


def ref_ids(ref, text):
    return ref(text, add_special_tokens=False).input_ids


# ---- tier 1: plain text, no special tokens ---------------------------------------------------

BASIC = [
    "hello world",
    "Hello World",
    " hello",
    "the The  the   The",
    "I can't, won't, and shouldn't. They're here; we've left.",
    "def f(x):\n    return x * 2\n\n\nclass A:\n\tpass\n",
    "        eight leading spaces",
    "\t\t\ttabs\tinside\t\n",
    "1234567890123 and 3.14159 and 1,000,000 and +91 98765 43210",
    "नमस्ते दुनिया, यह एक परीक्षण वाक्य है।",
    "日本語のテキスト。中文文本。한국어 텍스트.",
    "https://huggingface.co/Qwen/Qwen3-8B?x=1&y=2  user.name+tag@example.com",
    "emoji 🙂🚀👩‍💻 and symbols ∑∫≈ → ← ©",
    "mixed: café naïve résumé Straße",
    "\n\n\n",
    "   ",
    "",
    "a",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "...!!!???---===***",
    "x" * 300,
    "The quick brown fox jumps over the lazy dog. " * 20,
]


@pytest.mark.parametrize("text", BASIC, ids=[repr(t)[:24] for t in BASIC])
def test_basic_encode_matches_reference(ref, mine, text):
    assert mine.encode(text) == ref_ids(ref, text)


@pytest.mark.parametrize("text", BASIC, ids=[repr(t)[:24] for t in BASIC])
def test_basic_roundtrip(mine, text):
    assert mine.decode(mine.encode(text)) == text


def test_basic_decode_matches_reference(ref, mine):
    for text in BASIC:
        ids = ref_ids(ref, text)
        assert mine.decode(ids) == ref.decode(ids)


# ---- tier 2: NFC normalization --------------------------------------------------------------

def test_nfc_normalization(ref, mine):
    decomposed = "cafe\u0301 re\u0301sume\u0301 n\u0303"        # e + combining acute, n + combining tilde
    assert mine.encode(decomposed) == ref_ids(ref, decomposed)


# ---- tier 3: random strings built from the vocab itself ---------------------------------------

def test_random_vocab_concatenations(ref, mine):
    rng = random.Random(0)
    vocab = ref.get_vocab()
    ids = [i for i in vocab.values() if i < 151643]
    for _ in range(300):
        pieces = [ref.decode([rng.choice(ids)]) for _ in range(rng.randint(1, 12))]
        text = "".join(pieces)
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")  # a partial-UTF-8 token decodes to U+FFFD; both sides then see the same string
        assert mine.encode(text) == ref_ids(ref, text), repr(text)


def test_random_unicode_and_whitespace(ref, mine):
    rng = random.Random(1)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,;:!?'\"()-_/\\\n\t") + list("नमस्तेदुनियाकआ") + list("日本語中文") + ["🙂", "é", "ß"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert mine.encode(text) == ref_ids(ref, text), repr(text)


# ---- tier 4: special (added) tokens ---------------------------------------------------------

SPECIAL = [
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>assistant\n<think>\n\n</think>\n\nParis<|im_end|>",
    "text<|endoftext|>more text<|endoftext|>",
    "not a special token: <|im_start|x> and <|im_startx|> and <im_start>",
    "<|im_start|><|im_end|><|im_start|>",
]


@pytest.mark.parametrize("text", SPECIAL, ids=[repr(t)[:24] for t in SPECIAL])
def test_special_tokens(ref, mine, text):
    assert mine.encode(text) == ref_ids(ref, text)


def test_chat_template_string_roundtrips(ref, mine):
    msgs = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nobvious\n</think>\n\n4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    for thinking in (True, False):
        text = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        assert mine.encode(text) == ref_ids(ref, text)
        assert mine.decode(mine.encode(text)) == text
