# Copyright (c) 2026 Huawei Technologies Co., Ltd.
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops.fused_moe.experts_selector as selector


def apply(monkeypatch, logits, context, top_k=2):
    monkeypatch.setattr(selector, "get_forward_context", lambda: context)
    return selector._apply_dflash_request_coresets(
        logits,
        top_k=top_k,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
    )


def make_context(mode, rows, prefix_k=1, escape_rank=1):
    return SimpleNamespace(
        dflash_mode=mode,
        dflash_verify_rows=rows,
        dflash_prefix_k=prefix_k,
        dflash_escape_rank=escape_rank,
        dflash_layer_baseline_union_sizes=[],
        dflash_layer_union_sizes=[],
    )


def test_baseline_records_union_without_masking(monkeypatch):
    logits = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 6.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 6.0],
        ]
    )
    rows = (("a", torch.tensor([0, 1])), ("b", torch.tensor([2, 3])))
    context = make_context("baseline", rows)
    result = apply(monkeypatch, logits, context)
    assert result is logits
    assert context.dflash_layer_baseline_union_sizes[0].item() == 6
    assert context.dflash_layer_union_sizes[0].item() == 6


def test_union_shares_request_coresets_only_on_exposed_rows(monkeypatch):
    logits = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 6.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.0, 8.0, 0.0],
            [7.0, 0.0, 0.0, 0.0, 6.0, 5.0],
        ]
    )
    rows = (("a", torch.tensor([0, 1])), ("b", torch.tensor([2, 3])))
    context = make_context("union", rows)
    masked = apply(monkeypatch, logits, context)
    floor = torch.finfo(logits.dtype).min
    assert torch.equal(masked[0], logits[0])
    assert torch.equal(masked[2], logits[2])
    assert masked[1, 3] != floor
    assert masked[3, 1] != floor
    assert context.dflash_layer_union_sizes[0].item() == 4


def test_version_b_keeps_request_coresets_private(monkeypatch):
    logits = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 6.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.0, 8.0, 0.0],
            [7.0, 0.0, 0.0, 0.0, 6.0, 5.0],
        ]
    )
    rows = (("a", torch.tensor([0, 1])), ("b", torch.tensor([2, 3])))
    context = make_context("version_b", rows)
    masked = apply(monkeypatch, logits, context)
    floor = torch.finfo(logits.dtype).min
    assert masked[1, 3] == floor
    assert masked[3, 1] == floor
    assert masked[1, 1] != floor
    assert masked[3, 0] != floor


def test_escape_rank_contributes_multiple_suffix_experts(monkeypatch):
    logits = torch.tensor(
        [[9.0, 8.0, 0.0, 0.0], [0.0, 1.0, 7.0, 6.0]]
    )
    rows = (("a", torch.tensor([0, 1])),)
    context = make_context("version_b", rows, escape_rank=2)
    masked = apply(monkeypatch, logits, context)
    floor = torch.finfo(logits.dtype).min
    assert masked[1, 2] != floor
    assert masked[1, 3] != floor


@pytest.mark.parametrize("mode", ["bad", "controller"])
def test_unknown_mode_fails(monkeypatch, mode):
    context = make_context(mode, (("a", torch.tensor([0, 1])),))
    with pytest.raises(ValueError, match="unsupported DFlash prefix mode"):
        apply(monkeypatch, torch.randn(2, 4), context)
