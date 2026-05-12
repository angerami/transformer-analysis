"""Tests for corpus loading configuration in compute_perplexity.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compute_perplexity import load_corpus_tokens


def _make_tokenizer(fixed_ids=None):
    """Stub tokenizer that returns fixed token IDs."""
    tok = MagicMock()
    ids = fixed_ids if fixed_ids is not None else list(range(100))
    tok.return_value = MagicMock(input_ids=torch.tensor([ids]))
    tok.side_effect = None
    return tok


def test_unknown_corpus_raises():
    tok = _make_tokenizer()
    with pytest.raises(ValueError, match="Unknown corpus"):
        load_corpus_tokens("notacorpus", tok)


def test_wikitext103_returns_tensor():
    # Return an object that supports both iteration and column access like a HF Dataset
    fake_texts = ["hello world", "foo bar baz", ""]
    fake_ds = MagicMock()
    fake_ds.__getitem__ = MagicMock(return_value=fake_texts)

    tok = MagicMock()
    tok.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))

    with patch("compute_perplexity.load_dataset", return_value=fake_ds):
        result = load_corpus_tokens("wikitext103", tok)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 1


def test_pile_subset_token_limit():
    # Simulate streaming dataset that yields large texts
    long_ids = list(range(500))

    class FakeStreamDS:
        def shuffle(self, **kw): return self
        def __iter__(self):
            for _ in range(20):
                yield {"text": "word " * 200}

    tok = MagicMock()
    tok.return_value = MagicMock(input_ids=torch.tensor([long_ids]))

    with patch("compute_perplexity.load_dataset", return_value=FakeStreamDS()):
        result = load_corpus_tokens("pile", tok, pile_tokens=1024)

    assert len(result) == 1024


def test_pile_subset_exact_length():
    target = 512
    chunk_ids = list(range(100))

    class FakeStreamDS:
        def shuffle(self, **kw): return self
        def __iter__(self):
            for _ in range(20):
                yield {"text": "x"}

    tok = MagicMock()
    tok.return_value = MagicMock(input_ids=torch.tensor([chunk_ids]))

    with patch("compute_perplexity.load_dataset", return_value=FakeStreamDS()):
        result = load_corpus_tokens("pile", tok, pile_tokens=target)

    assert len(result) == target
