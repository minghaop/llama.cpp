# Validate Encoding — JSON Rule Configuration Format

This document describes the format of the JSON rule configuration file accepted by the
`validate_encoding` component. It is intended for customers and QA engineers who want to
define their own validation rules without modifying any source code.

> **Built-in rules run by default.**  When `--rule_config` is not supplied, the full
> set of built-in rules is active automatically.  When `--rule_config` **is** supplied,
> the built-in rules are **replaced entirely** by the rules listed in that file — only
> the explicitly listed rules will run.  Use `configs/example_rules.json` as a starting
> point and add or remove entries to match your model's requirements.

---

## How to Use a Rule Configuration File

Pass the path to your JSON file via the `--rule_config` argument:

```bash
qairt-accuracy-debugger validate_encoding \
    --encoding_path       /path/to/encoding.json \
    --rule_config         /path/to/my_rules.json \
    --working_directory   /path/to/output
```

When `--rule_config` is supplied, the rules defined in the file **replace** all
built-in rules.  Any built-in rule that is not listed in the file will not run.
If `--rule_config` is omitted, all built-in rules are active.

---

## Top-Level Structure

```json
{
  "description":       "<string>",
  "version":           "<string>",
  "validation_rules":  [ <rule>, ... ],
  "context_aware_rules": [ <rule>, ... ]
}
```

| Field                  | Type            | Required | Description |
|------------------------|-----------------|----------|-------------|
| `description`          | string          | No       | Free-text description of this rule set. Not used by the tool; for human reference only. |
| `version`              | string          | No       | Version label for your rule file (e.g. `"1.0"`). Not used by the tool; for human reference only. |
| `validation_rules`     | array of rules  | No       | Rules that are applied to each tensor **independently**, using only that tensor's own encoding attributes (bitwidth, scale, symmetry, etc.). |
| `context_aware_rules`  | array of rules  | No       | Rules that are applied to activation tensors and require knowledge of the **surrounding graph structure** (e.g. which op produced this tensor, what its predecessor's encoding is). Requires a DLC file via `--dlc_file_path`; without one these rules are skipped entirely. |

Both arrays are optional. An empty array `[]` means no rules of that type will be applied.

---

## Rule Object

Every entry in `validation_rules` or `context_aware_rules` is a **rule object**:

```json
{
  "name":        "<string>",
  "type":        "builtin" | "parametric" | "custom",
  "enabled":     true | false,
  "description": "<string>",
  ...
}
```

| Field         | Type    | Required | Description |
|---------------|---------|----------|-------------|
| `name`        | string  | Yes      | Unique identifier for this rule entry. For `builtin` and `parametric` rules it must match one of the supported names listed below. For `custom` rules it is a free-text label used in violation reports. |
| `type`        | string  | Yes      | Determines how the rule is created. Must be one of `"builtin"`, `"parametric"`, or `"custom"`. See the sections below for details. |
| `enabled`     | boolean | No       | When `false` the rule is skipped entirely. Defaults to `true`. Use this to suppress a rule without removing it from the file. |
| `description` | string  | No       | Human-readable explanation of what the rule checks. Appears in violation reports when the rule is violated. |

---

## Rule Types

### 1. `builtin`

Activates a pre-implemented rule that is shipped with the tool. No additional fields are needed.

```json
{
  "name":    "conv_weights_symmetric",
  "type":    "builtin",
  "enabled": true
}
```

#### Available `validation_rules` built-ins

| `name`                        | What it checks |
|-------------------------------|----------------|
| `conv_weights_symmetric`      | Convolution weight tensors must be **symmetrically** quantized (`is_symm == "true"`). Identified by tensor names containing both `"conv"` and `"weight"` or `"kernel"`. |
| `rmsnorm_weights_16bit`       | RMSNorm weight tensors must use **16-bit** quantization. Identified by tensor names containing `"rms"`, `"norm"`, and `"weight"`. |
| `rmsnorm_weights_asymmetric`  | RMSNorm weight tensors must be **asymmetrically** quantized (`is_symm == "false"`). |
| `layernorm_weights_asymmetric`| LayerNorm weight tensors must be **asymmetrically** quantized. Identified by tensor names containing `"layernorm"` or `"layer_norm"` and `"weight"` or `"gamma"`. |
| `batchnorm_weights_asymmetric`| BatchNorm weight tensors must be **asymmetrically** quantized. Identified by tensor names containing `"batchnorm"`, `"batch_norm"`, or `"bn"` and `"weight"` or `"gamma"`. |
| `layernorm_weights_16bit`     | LayerNorm weight tensors must use **16-bit** quantization. |
| `all_weights_8bit`            | All tensors whose name contains `"weight"` or `"kernel"` must use **8-bit** quantization. |
| `linear_weights_per_channel`  | Linear/Dense/FC weight tensors must use **per-channel** quantization (number of channels > 1). Identified by tensor names containing `"linear"`, `"dense"`, or `"fc"` and `"weight"` or `"kernel"`. |

#### Available `context_aware_rules` built-ins

| `name`                                   | What it checks |
|------------------------------------------|----------------|
| `reshape_encoding_matches_predecessor`   | The quantization encoding (scale and offset) of a Reshape output tensor must match the encoding of its input tensor. |
| `transpose_encoding_matches_predecessor` | The quantization encoding of a Transpose output tensor must match the encoding of its input tensor. |
| `concat_inputs_same_range`               | All input tensors feeding into a Concat operation must share the same quantization range (scale and offset). |

---

### 2. `parametric`

A pre-implemented rule whose behaviour is controlled by one or more numeric parameters.
Supply the parameters in a `"parameters"` object.

```json
{
  "name":        "large_quantization_range_warning",
  "type":        "parametric",
  "enabled":     true,
  "description": "Warn when any tensor's quantization range exceeds the threshold",
  "parameters": {
    "threshold": 500.0
  }
}
```

#### Available parametric rules

| `name`                            | Parameter    | Type  | Default  | Description |
|-----------------------------------|--------------|-------|----------|-------------|
| `large_quantization_range_warning`| `threshold`  | float | `1000.0` | A violation is reported for any tensor whose quantization range (`max − min`) exceeds this value. Applies to both weight and activation tensors that have `min`/`max` values. |

---

### 3. `custom`

Lets you define a completely new rule in JSON without writing any Python code.
A custom rule is described by two sub-objects: `"conditions"` (which tensors the rule
applies to) and `"checks"` (what must be true about those tensors).

```json
{
  "name":        "my_rule_name",
  "type":        "custom",
  "enabled":     true,
  "description": "Human-readable description shown in violation reports",
  "conditions":  { ... },
  "checks":      { ... }
}
```

A tensor is validated by this rule only when **all** specified conditions match.
If no conditions are specified, the rule applies to every tensor.

#### `conditions` fields

| Field                | Type            | Description |
|----------------------|-----------------|-------------|
| `tensor_name_pattern`| string (regex)  | A **case-insensitive** Python regular expression. The rule applies only to tensors whose name matches this pattern. Example: `"(attention\|attn).*weight"` matches any tensor with `attention` or `attn` followed by `weight` anywhere in its name. |
| `tensor_type`        | string          | Restricts the rule to a specific category of tensor. Accepted values: `"weight"` (name contains `"weight"` or `"kernel"`), `"bias"` (name contains `"bias"`), `"activation"` (name does not contain `"weight"`, `"bias"`, or `"kernel"`). |
| `layer_types`        | array of strings| A list of layer-type keywords. The rule applies only to tensors whose name contains **at least one** of these keywords (case-insensitive). Example: `["conv", "linear"]`. |

All specified conditions must be satisfied simultaneously (logical AND).

#### `checks` fields

| Field          | Type             | Description |
|----------------|------------------|-------------|
| `bitwidth`     | integer          | The tensor's quantization bitwidth must equal this value. Common values: `4`, `8`, `16`. |
| `is_symm`      | boolean          | `true` requires symmetric quantization; `false` requires asymmetric quantization. |
| `min_channels` | integer          | The number of quantization channels must be **greater than or equal to** this value. Use `2` to require per-channel quantization (more than one channel). |
| `max_channels` | integer          | The number of quantization channels must be **less than or equal to** this value. Use `1` to require per-tensor quantization. |
| `scale_range`  | object           | Each scale value in the tensor must fall within the specified range. Contains `"min"` (float, inclusive lower bound) and/or `"max"` (float, inclusive upper bound). |
| `offset_range` | object           | Each offset value in the tensor must fall within the specified range. Contains `"min"` (integer, inclusive lower bound) and/or `"max"` (integer, inclusive upper bound). |

All specified checks must pass simultaneously (logical AND). If a check field is omitted,
that attribute is not validated.

---

## Complete Example

The following file demonstrates all three rule types working together:

```json
{
  "description": "Example rule set for a transformer model",
  "version": "1.0",

  "validation_rules": [

    {
      "name":    "conv_weights_symmetric",
      "type":    "builtin",
      "enabled": true,
      "description": "Convolution weights must be symmetrically quantized"
    },

    {
      "name":    "rmsnorm_weights_16bit",
      "type":    "builtin",
      "enabled": true,
      "description": "RMSNorm weights must use 16-bit quantization"
    },

    {
      "name":    "rmsnorm_weights_asymmetric",
      "type":    "builtin",
      "enabled": false,
      "description": "Disabled: our model uses symmetric RMSNorm weights"
    },

    {
      "name":        "large_quantization_range_warning",
      "type":        "parametric",
      "enabled":     true,
      "description": "Warn when quantization range exceeds 500",
      "parameters": {
        "threshold": 500.0
      }
    },

    {
      "name":        "attention_weights_4bit_symmetric",
      "type":        "custom",
      "enabled":     true,
      "description": "Attention weights must use 4-bit symmetric quantization",
      "conditions": {
        "tensor_name_pattern": "(attention|attn).*weight",
        "tensor_type": "weight"
      },
      "checks": {
        "bitwidth": 4,
        "is_symm":  true
      }
    },

    {
      "name":        "linear_weights_per_channel_8bit",
      "type":        "custom",
      "enabled":     true,
      "description": "Linear weights must use 8-bit per-channel quantization",
      "conditions": {
        "layer_types": ["linear", "dense", "fc"],
        "tensor_type": "weight"
      },
      "checks": {
        "bitwidth":     8,
        "min_channels": 2
      }
    },

    {
      "name":        "conv_scale_range",
      "type":        "custom",
      "enabled":     true,
      "description": "Convolution weight scales must be in [0.001, 0.1]",
      "conditions": {
        "layer_types": ["conv"],
        "tensor_type": "weight"
      },
      "checks": {
        "scale_range": { "min": 0.001, "max": 0.1 }
      }
    }
  ],

  "context_aware_rules": [

    {
      "name":    "reshape_encoding_matches_predecessor",
      "type":    "builtin",
      "enabled": true,
      "description": "Reshape output encoding must match its input encoding"
    },

    {
      "name":    "concat_inputs_same_range",
      "type":    "builtin",
      "enabled": true,
      "description": "All inputs to a Concat op must share the same quantization range"
    }
  ]
}
```

---

## Violation Report

When a rule is violated, the following information is recorded in the output reports
(`validation_report.json` and `validation_report.csv`):

| Field              | Description |
|--------------------|-------------|
| `tensor_name`      | Full name of the tensor that violated the rule. |
| `rule_description` | The `description` field of the violated rule, plus details about the specific mismatch (e.g. actual vs. expected bitwidth). |
| `dtype`            | Data type of the tensor (`int`, `sfxp`, `ufxp`, `float`). |
| `bitwidth`         | Quantization bitwidth of the tensor. |
| `is_symm`          | Whether the tensor uses symmetric quantization (`"true"` or `"false"`). |
| `channels`         | Number of quantization channels (1 = per-tensor, >1 = per-channel). |

---

## Tips

- **Start from the example template:** Copy `configs/example_rules.json`, keep only the
  rules relevant to your model, and pass it via `--rule_config`.
- **Omit `--rule_config` to use all built-in rules:** The full built-in rule set runs
  automatically when no config file is provided.
- **Suppress a rule without deleting it:** Set `"enabled": false`. This makes it easy to
  re-enable the rule later and keeps a record of why it was turned off.
- **Combine conditions:** All fields inside `"conditions"` are applied together (AND logic).
  If you need OR logic across different tensor groups, define two separate custom rules.
- **Regex patterns are case-insensitive:** `"tensor_name_pattern": "CONV.*weight"` and
  `"tensor_name_pattern": "conv.*weight"` behave identically.
- **Context-aware rules need a DLC file:** Pass `--dlc_file_path` when using
  `reshape_encoding_matches_predecessor`, `transpose_encoding_matches_predecessor`, or
  `concat_inputs_same_range`. Without a DLC file these rules are skipped entirely.
- **Order does not affect results:** Rules are applied independently to each tensor; the
  order of rules in the JSON array does not change which violations are detected.
