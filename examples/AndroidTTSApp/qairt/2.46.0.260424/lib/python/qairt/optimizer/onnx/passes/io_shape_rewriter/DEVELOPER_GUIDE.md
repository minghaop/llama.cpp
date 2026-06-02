# IO Shape Rewriter — Developer Guide

Covers `change_seq_and_context_length` and the axis denotation inference system it
depends on. Read this before modifying either subsystem or adding support for a new
model or op type.

---

## Concepts

**AR (Attention Range / Sequence Length)** — tokens processed per forward pass.
AR=1 is decode mode; AR=N is prefill mode.

**CL (Context Length)** — total KV cache window size.

Both values are baked into every tensor shape at export time. This pass rewrites them
without retraining.

**Axis denotation** — a label attached to one axis of a tensor that records its
semantic role. Labels used here:

| Label | Meaning | Rewritten to |
|-------|---------|--------------|
| `SEQ_LENGTH` | AR tokens | `new_AR` |
| `CONTEXT_LENGTH` | Full KV window | `new_CL` |
| `PAST_SEQ_LENGTH` | Past-only cache slice | `new_CL - new_AR` |
| `SLIDING_CONTEXT_LENGTH` | Sliding window variant | `new_sliding_CL` |
| `BATCH` | Batch dimension | unchanged |
| `UNKNOWN` | Cannot be determined | unchanged |

---

## Graph I/O Shape Contract

For a concat-based model (AR=73, CL=3073):

```
INPUTS
  input_ids           [1, 73]           SEQ_LENGTH at axis 1
  attention_mask      [1, 1, 73, 3073]  SEQ_LENGTH at axis 2, CONTEXT_LENGTH at axis 3
  past_key_N_in       [1, 8, 64, 3000]  PAST_SEQ_LENGTH at axis 3  (CL - AR = 3000)
  past_value_N_in     [1, 8, 3000, 64]  PAST_SEQ_LENGTH at axis 2

OUTPUTS
  logits              [1, 73, vocab]    SEQ_LENGTH at axis 1
  past_key_N_out      [1, 8, 64, 73]   SEQ_LENGTH at axis 3  ← new keys only, NOT full cache
  past_value_N_out    [1, 8, 73, 64]   SEQ_LENGTH at axis 2
```

`past_key_N_out` carries only the **new AR tokens**. The runtime appends them to its
rolling buffer to form the next `past_key_N_in`. The full CL-sized buffer lives outside
the graph.

Inside the graph, `past_key_N_in` is used for attention (concatenated with current keys
to form a `[B, H, D, CL]` tensor) but is **not** connected to `past_key_N_out`.

---

## Pipeline: 8 Steps

Entry point: `change_seq_and_context_length` in `api.py` → `IOShapeRewriter.apply`.

| Step | Pass | What it does |
|------|------|--------------|
| 1 | `ComputeSeqAndContextLength` | Reads current AR and CL from `attention_mask` shape |
| 2 | `AxisDenotationInference` | Labels every tensor axis (seed + propagate) |
| 3 | `IOShapeRewriter.validate_config` | Enforces `1 ≤ new_AR ≤ new_CL - 1` |
| 4 | `UpdateGraphIO` | Rewrites graph input/output shapes using denotation map |
| 5 | `UpdateReshapeNodes`, `UpdateExpandNodes`, `UpdateConstantTensors` | Updates shape constants inside data-transformation nodes; regenerates sequential index arrays (e.g. `[0…AR-1]`) |
| 6 | `DeadCodeRemoval` / `DeadWeightRemoval` | Prunes unreachable nodes and weights |
| 7 | `ShapeInference` | Re-propagates shapes through all intermediate tensors |

Only tensors that received a denotation in Step 2 are touched in Steps 4–6. Everything
else is left unchanged and corrected by Step 8.

---

## Axis Denotation Inference (Step 2)

### Seeding

`AxisDenotationInitializer` walks `graph.inputs` only and assigns denotations by
name pattern and shape. Key rules:

- `attention_mask` → `[BATCH, UNKNOWN, SEQ_LENGTH, CONTEXT_LENGTH]`
- `past_key_N_in` → `PAST_SEQ_LENGTH` or `CONTEXT_LENGTH` at the cache axis,
  depending on whether `cache_dim == CL - AR` or `cache_dim == CL`
- `cache_index` (scatter pattern) → seeds the downstream `Add` output with `SEQ_LENGTH`

The cache axis denotation is determined by shape measurement, not assumption:

```python
if cache_dim == context_length - seq_length:  → PAST_SEQ_LENGTH
elif cache_dim == context_length:             → CONTEXT_LENGTH
elif cache_dim == sliding_context_length:     → SLIDING_CONTEXT_LENGTH
else: raise ValueError(...)
```

Graph **outputs** are never seeded directly — they receive denotations only if
propagation reaches them.

### Propagation

`AxisDenotationInference.apply` iterates nodes in topological order and applies the
first matching `Infer*AxisDenotation` pass. Each pass implements one rule:

- **Identity** (`Sigmoid`, `Cast`, `Slice`, `ScatterElements`, …): output = input[0]
- **Transpose**: permutes denotations by `perm` attribute
- **Concat** (critical): on the concat axis, `PAST_SEQ_LENGTH + SEQ_LENGTH → CONTEXT_LENGTH`
- **Reshape**: matches input/output dims by shape arithmetic
- **Reduce**: drops the reduced axis from denotations
- **MatMul / Gemm**: `[…, M, K] @ […, K, N] → […, M, N]`

If no pass matches a node **and** at least one of its inputs has denotations, a
`WARNING` is logged:

```
No axis denotation propagation rule for op 'X' (node 'Y').
Denotations will not propagate past this node; downstream tensors may not be resized correctly.
```

This is the signal that a new pass is needed (see below).

---

## KV Cache Patterns

### Concat-based (standard)

`past_key_N_in` has shape `[B, H, D, CL-AR]` → seeded `PAST_SEQ_LENGTH`.
Inside the graph it is concatenated with current keys to form `[B, H, D, CL]` for
attention. The graph output `past_key_N_out` carries only the new `AR` keys and
receives `SEQ_LENGTH` via propagation through the head-reassembly path.

### Scatter-based (in-place)

`past_key_N_in` has shape `[B, H, D, CL]` → seeded `CONTEXT_LENGTH`.
New tokens are written in-place via `ScatterElements` at position `cache_index`.
The output inherits `CONTEXT_LENGTH` via identity propagation and is resized to
`new_CL`.

---

## Adding a Propagation Pass for a Missing Op

### 1. Check what is registered

```python
from qairt.optimizer.onnx.passes.axis_denotation_infer.propagation_passes import (
    get_registered_op_types,
)
for name, ops in sorted(get_registered_op_types().items()):
    print(f"{name}: {sorted(ops)}")
```

### 2. Shape-preserving op? One-line fix

Add the op name to `InferIdentityAxisDenotation.op_types` in `propagation_passes.py`.

### 3. Non-trivial output shape? Write a new pass

```python
# propagation_passes.py

@register_pass
class InferMyOpAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for MyOp(data, weight) -> output.

    Output shape = input[0] shape with last axis replaced by weight.shape[-1].
    """

    op_types = {"MyOp"}

    def infer_output_denotations(self, node: ir.Node) -> list[AxisDenotation] | None:
        input_denotations = self._get_input_denotations(node, 0)
        if not input_denotations:
            return None
        # Last axis is determined by weight — mark UNKNOWN
        return input_denotations[:-1] + [AxisDenotation.UNKNOWN]
```

`@register_pass` automatically adds the class to the registry returned by
`get_registered_op_types()`.

For ops with **multiple outputs** (like `Split`, `TopK`), override `rewrite` instead
and call `set_axis_denotations(output, denotations)` per output.

### 4. Add to the propagation pipeline

In `axis_denotation_infer.py`, import the class and add an instance to
`propagation_passes`:

```python
from qairt.optimizer.onnx.passes.axis_denotation_infer.propagation_passes import (
    ...
    InferMyOpAxisDenotation,
)

propagation_passes = [
    ...
    InferMyOpAxisDenotation(),
]
```

### 5. Does the op carry a shape constant that must be rewritten?

Ops that **transform tensor shape** (e.g. `Reshape`, `Expand`) embed the target
shape as a constant input. An infer pass propagates denotations through the op, but
it does **not** update that constant — a separate node update pass is required for
that.

If your op falls into this category, write an `Update<Op>Nodes` pass anywhere in
the codebase and decorate it with `@register_node_update_pass`:

```python
from qairt.optimizer.onnx.passes.io_shape_rewriter import register_node_update_pass
from qairt.optimizer.onnx.passes.io_shape_rewriter.config import IOShapeRewriterConfig
from qairt.optimizer.onnx.passes.base import BasePredicatePass

@register_node_update_pass
class UpdateMyOpNodes(BasePredicatePass):
    def __init__(self, config: IOShapeRewriterConfig):
        super().__init__()
        self.config = config

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "MyOp":
            return False
        return bool(get_axis_denotations(node.outputs[0])) and get_constant_np(node.inputs[1]) is not None

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        # update node.inputs[1] based on denotations ...
```

The pass is registered at import time. As long as the module containing it is
imported before `IOShapeRewriter.apply` runs, it will be included automatically —
no manual list editing required.

### 6. Verify

- Warning for `MyOp` is gone.
- Downstream tensors now have denotations.
- Shape inference (Step 8) succeeds.

---

## Supporting a Non-Standard Model

If the initializer raises `ValueError` because no built-in rule matches your model's
input names, provide a custom seed rule:

```python
from qairt.optimizer.onnx.passes.io_shape_rewriter import (
    change_seq_and_context_length,
    AxisDenotationConfig,
    AxisDenotationSeedRule,
)
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

config = AxisDenotationConfig(
    custom_seed_rules=[
        AxisDenotationSeedRule(
            name_pattern=r"kv_cache_layer_\d+",
            denotations=[
                AxisDenotation.BATCH,
                AxisDenotation.UNKNOWN,          # heads
                AxisDenotation.PAST_SEQ_LENGTH,
                AxisDenotation.UNKNOWN,          # head_dim
            ],
        )
    ]
)

change_seq_and_context_length(
    ctx, new_seq_length=32, new_context_length=4096,
    axis_denotation_config=config,
)
```

Custom rules are checked **before** built-in rules and can override them.

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: Could not determine axis denotation for input X` | KV cache dim doesn't match `CL`, `CL-AR`, or `sliding_CL` | Check `attention_mask` shape; AR and CL are read from it first |
| `ValueError: Rank mismatch for graph input X` | Custom seed rule has wrong number of denotations | Match denotation list length to tensor rank |
| `ValueError: Invalid sequence length N for context length M` | `new_AR >= new_CL` | Ensure `1 ≤ new_AR ≤ new_CL - 1` |
| Warning: `No axis denotation propagation rule for op X` | Op has no registered pass | Add to `InferIdentityAxisDenotation.op_types` or write a new pass |
| Shape inference fails after rewriting | A `Reshape` constant was not updated | Check whether the Reshape output has denotations; if not, propagation stopped upstream |
| Downstream tensor has wrong shape | Denotation chain broken at an unhandled op | Find the warning log; add a pass for that op |
| ONNX checker reports `Incompatible dimensions` at a data-transformation op (`Reshape`, `Expand`, etc.) after a successful rewrite | Denotations propagated correctly but the op's shape constant was never updated | An infer pass is not enough for ops that embed shape as a constant input — check whether a corresponding `@register_node_update_pass` exists for that op; if not, write one (see step 5 above) |
