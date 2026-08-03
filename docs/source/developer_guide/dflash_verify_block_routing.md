# DFlash verify-block expert routing experiments

These experimental policies alter expert selection only for ordinary softmax
routing. They are disabled by default and leave grouped or custom routing
unchanged.

## Block Top-p coreset

Set `VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P` to a value in `(0, 1]`. Router
probabilities are summed across the verification block, experts are sorted by
that aggregate score, and the smallest prefix reaching Top-p is shared by all
positions.

For a 16-token verification block:

```bash
export VLLM_ASCEND_DFLASH_MOE_CORESET_TOP_P=0.5
export VLLM_ASCEND_DFLASH_MOE_CORESET_MIN_TOKENS=16
export VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_TOKENS=16
export VLLM_ASCEND_DFLASH_MOE_CORESET_MIN_EXPERTS=8
export VLLM_ASCEND_DFLASH_MOE_CORESET_MAX_EXPERTS=128
```

## Prefix Top-k with suffix Top-1 escape

The allowed set is the union of the original Top-k experts from the first
`K` positions and the original Top-1 expert from every remaining position.

```bash
export VLLM_ASCEND_DFLASH_MOE_PREFIX_K=6
export VLLM_ASCEND_DFLASH_MOE_PREFIX_MIN_TOKENS=16
export VLLM_ASCEND_DFLASH_MOE_PREFIX_MAX_TOKENS=16
```

Set only one policy at a time. An empty variable disables that policy. Setting
the prefix length to the full block length is an identity control: set-building
and masking operators run, but all logits remain available.

## Experimental infrastructure

Router-logit collection and the DFlash KV-zeroer compatibility guard are not
part of these routing policies. Keep them in separate commits and enable trace
patching only for trace collection runs. Accuracy runs should start from a
clean vLLM/vLLM-Ascend checkout plus the routing commits and the independently
reviewed DFlash compatibility fix.
