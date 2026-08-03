# Copyright (c) 2026 Huawei Technologies Co., Ltd.
import pytest
import torch

from vllm_ascend.ops.fused_moe.experts_selector import (
    _apply_block_prefix_topk_with_top1_escape,
)


_ENV_NAMES = (
    "VLLM_ASCEND_DFLASH_MOE_PREFIX_K",
    "VLLM_ASCEND_DFLASH_MOE_PREFIX_MIN_TOKENS",
    "VLLM_ASCEND_DFLASH_MOE_PREFIX_MAX_TOKENS",
)


@pytest.fixture(autouse=True)
def clear_prefix_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def apply(logits, top_k=2):
    return _apply_block_prefix_topk_with_top1_escape(
        logits,
        top_k=top_k,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
    )


def test_disabled_is_identity():
    logits = torch.randn(4, 8)
    assert apply(logits) is logits


def test_prefix_topk_and_suffix_top1_form_allowed_set(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", "2")
    logits = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 5.0, 4.0],
            [0.0, 0.0, 1.0, 2.0, 6.0],
        ]
    )
    masked = apply(logits)
    active = (masked[0] != torch.finfo(masked.dtype).min).nonzero().flatten()
    assert active.tolist() == [0, 1, 2, 3, 4]


def test_suffix_preserves_only_its_top1_escape(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", "1")
    logits = torch.tensor([[9.0, 8.0, 0.0, 0.0], [0.0, 1.0, 7.0, 6.0]])
    masked = apply(logits)
    active = (masked[0] != torch.finfo(masked.dtype).min).nonzero().flatten()
    assert active.tolist() == [0, 1, 2]


def test_full_prefix_is_identity_control(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", "4")
    logits = torch.randn(4, 8)
    assert torch.equal(apply(logits), logits)


def test_token_guards_exclude_other_forwards(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", "2")
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_MIN_TOKENS", "8")
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_MAX_TOKENS", "16")
    short = torch.randn(1, 8)
    long = torch.randn(17, 8)
    assert apply(short) is short
    assert apply(long) is long


def test_unsupported_routing_is_unchanged(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", "2")
    logits = torch.randn(4, 8)
    result = _apply_block_prefix_topk_with_top1_escape(
        logits,
        top_k=2,
        scoring_func="sigmoid",
        use_grouped_topk=False,
        custom_routing_function=None,
    )
    assert result is logits


@pytest.mark.parametrize("value", ["-1", "bad"])
def test_invalid_prefix_fails(monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_PREFIX_K", value)
    with pytest.raises(ValueError, match="PREFIX_K"):
        apply(torch.randn(4, 8))
