# Copyright (c) 2026 Huawei Technologies Co., Ltd.
import pytest
import torch

from vllm_ascend.ops.fused_moe.experts_selector import _apply_block_top_p_coreset


_ENV_NAMES = (
    "VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P",
    "VLLM_ASCEND_DFLASH_MOE_CORESET_MIN_TOKENS",
    "VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_TOKENS",
    "VLLM_ASCEND_DFLASH_MOE_CORESET_MIN_EXPERTS",
    "VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_EXPERTS",
)


@pytest.fixture(autouse=True)
def clear_coreset_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def apply(logits, top_k=2):
    return _apply_block_top_p_coreset(
        logits,
        top_k=top_k,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
    )


def test_disabled_is_identity():
    logits = torch.randn(4, 8)
    assert apply(logits) is logits


def test_empty_top_p_is_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "")
    logits = torch.randn(4, 8)
    assert apply(logits) is logits


def test_shared_coreset_masks_experts(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "0.60")
    logits = torch.tensor(
        [[6.0, 5.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0], [6.0, 5.0, 0.0, 0.0]]
    )
    masked = apply(logits, top_k=1)
    active = (masked[0] != torch.finfo(masked.dtype).min).nonzero().flatten()
    assert active.tolist() == [0, 1]
    assert torch.equal(masked[:, active], logits[:, active])


def test_minimum_experts_preserves_top_k(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "0.01")
    logits = torch.tensor([[9.0, 8.0, 7.0, 0.0], [9.0, 8.0, 7.0, 0.0]])
    masked = apply(logits, top_k=3)
    assert (masked[0] != torch.finfo(masked.dtype).min).sum().item() == 3


def test_token_guards_exclude_other_forwards(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "0.5")
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_MIN_TOKENS", "8")
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_TOKENS", "16")
    short = torch.randn(1, 8)
    long = torch.randn(17, 8)
    assert apply(short) is short
    assert apply(long) is long


def test_max_experts_caps_coreset(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "0.99")
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_EXPERTS", "3")
    logits = torch.zeros(4, 8)
    masked = apply(logits, top_k=2)
    assert (masked[0] != torch.finfo(masked.dtype).min).sum().item() == 3


def test_unsupported_routing_is_unchanged(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", "0.5")
    logits = torch.randn(4, 8)
    result = _apply_block_top_p_coreset(
        logits,
        top_k=2,
        scoring_func="sigmoid",
        use_grouped_topk=False,
        custom_routing_function=None,
    )
    assert result is logits


@pytest.mark.parametrize("value", ["0", "1.1", "bad"])
def test_invalid_top_p_fails(monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P", value)
    with pytest.raises(ValueError, match="TOP_P"):
        apply(torch.randn(4, 8))
