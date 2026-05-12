"""Tests for the eval loop and collector pattern in compute_perplexity.py."""
import sys
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compute_perplexity import NLLCollector, eval_loop, to_long_format, run_inference


# ---------------------------------------------------------------------------
# NLLCollector
# ---------------------------------------------------------------------------

def test_nll_collector_basic():
    c = NLLCollector()
    # Simulate 100 tokens with NLL = 2.0
    out = types.SimpleNamespace(loss=torch.tensor(2.0))
    c.update(out, 100)
    assert abs(c.result() - 2.0) < 1e-6


def test_nll_collector_weighted_average():
    c = NLLCollector()
    out1 = types.SimpleNamespace(loss=torch.tensor(4.0))
    out2 = types.SimpleNamespace(loss=torch.tensor(2.0))
    c.update(out1, 100)   # 400 total
    c.update(out2, 100)   # 200 total
    assert abs(c.result() - 3.0) < 1e-6   # (400+200)/200 = 3.0


def test_nll_collector_empty():
    c = NLLCollector()
    assert math.isnan(c.result())


def test_perplexity_from_nll():
    nll = 3.0
    ppl = math.exp(nll)
    assert abs(ppl - math.e ** 3) < 1e-6


# ---------------------------------------------------------------------------
# eval_loop with a tiny stub model
# ---------------------------------------------------------------------------

class StubOutput:
    def __init__(self, loss):
        self.loss = torch.tensor(loss)
        self.hidden_states = None
        self.attentions = None


class StubModel(torch.nn.Module):
    """Returns a fixed loss value for any input."""
    def __init__(self, fixed_loss=2.5, max_pos=128):
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=max_pos)
        self.fixed_loss = fixed_loss

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=False, output_attentions=False):
        return StubOutput(self.fixed_loss)


def test_eval_loop_constant_loss():
    model = StubModel(fixed_loss=3.0, max_pos=64)
    tokens = torch.randint(0, 1000, (200,))
    results = eval_loop(model, tokens, device=torch.device("cpu"), stride=64)
    assert abs(results["nll"] - 3.0) < 1e-4
    assert abs(math.exp(results["nll"]) - math.exp(3.0)) < 1e-3


def test_eval_loop_respects_max_tokens():
    model = StubModel(fixed_loss=1.0, max_pos=64)
    tokens = torch.randint(0, 1000, (1000,))
    results = eval_loop(model, tokens, device=torch.device("cpu"),
                        stride=64, max_tokens=128)
    assert abs(results["nll"] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# run_inference flags
# ---------------------------------------------------------------------------

def test_run_inference_returns_loss():
    model = StubModel(fixed_loss=2.0, max_pos=32)
    ids = torch.randint(0, 100, (1, 10))
    mask = torch.ones_like(ids)
    out = run_inference(model, ids, mask)
    assert hasattr(out, "loss")
    assert abs(out.loss.item() - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# to_long_format
# ---------------------------------------------------------------------------

def test_to_long_format_schema():
    row = {"model": "gpt2", "revision": "main", "step": None,
           "perplexity": 30.5, "nll": math.log(30.5), "bpb": 4.5, "corpus": "wikitext103"}
    records = to_long_format(row)
    metrics = {r["metric"] for r in records}
    assert metrics == {"perplexity", "nll", "bpb"}
    for r in records:
        assert r["source"] == "eval_pass"
        assert r["corpus"] == "wikitext103"
        assert r["model"] == "gpt2"
        assert isinstance(r["value"], float)
