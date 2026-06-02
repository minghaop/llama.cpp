# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
Validation rules for encoding validation.

This module contains predefined validation rules that can be used with the
ValidateEncodings class. Each rule is a function that takes (tensor_name, tensor_encoding)
and returns (is_valid, rule_description).
"""

from typing import Tuple

from qti.aisw.accuracy_debugger.encodings.encodings import TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import TensorType


# ==============================================================================
# Tensor Identification Helper Functions
# ==============================================================================


def is_conv_weight(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify convolution weight tensors.

    Matches when the tensor is a param tensor (V0/V1/DLC) or when the name
    contains both a conv keyword and a weight keyword (V2 / name-only fallback).

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a convolution weight.
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
        or "kernel" in name
    )
    is_conv = "conv" in name
    return is_conv and is_weight


def is_rmsnorm_weight(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify RMSNorm weight tensors.

    Matches when the tensor is a param tensor (V0/V1/DLC) or when the name
    contains a weight keyword (V2 / name-only fallback), and the name contains
    both "rms" and "norm".

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as an RMSNorm weight.
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
    )
    is_rmsnorm = "rms" in name and "norm" in name
    return is_rmsnorm and is_weight


def is_layernorm_weight(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify LayerNorm weight tensors.

    Matches when the tensor is a param tensor (V0/V1/DLC) or when the name
    contains a weight/gamma keyword (V2 / name-only fallback), and the name
    contains a layernorm keyword.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a LayerNorm weight.
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
        or "gamma" in name
    )
    is_layernorm = "layernorm" in name or "layer_norm" in name
    return is_layernorm and is_weight


def is_batchnorm_weight(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify BatchNorm weight tensors.

    Matches when the tensor is a param tensor (V0/V1/DLC) or when the name
    contains a weight/gamma keyword (V2 / name-only fallback), and the name
    contains a batchnorm keyword.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a BatchNorm weight.
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
        or "gamma" in name
    )
    is_batchnorm = "batchnorm" in name or "batch_norm" in name or "bn" in name
    return is_batchnorm and is_weight


def is_linear_weight(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify Linear/Dense layer weight tensors.

    Matches when the tensor is a param tensor (V0/V1/DLC) or when the name
    contains a weight/kernel keyword (V2 / name-only fallback), and the name
    contains a linear-layer keyword.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a Linear weight.
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
        or "kernel" in name
    )
    is_linear = "linear" in name or "dense" in name or "fc" in name
    return is_linear and is_weight


def is_reshape_activation(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify Reshape operation activation tensors.

    Matches when the tensor is an activation tensor (V0/V1/DLC) or when the
    name contains "reshape" (V2 / name-only fallback).

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a Reshape activation.
    """
    name = tensor_name.lower()
    is_activation = (
        tensor.tensor_type == TensorType.ActivationEncodings
        or tensor.tensor_type == TensorType.Encodings
    )
    return is_activation and "reshape" in name


def is_transpose_activation(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify Transpose operation activation tensors.

    Matches when the tensor is an activation tensor (V0/V1/DLC) or when the
    name contains a transpose keyword (V2 / name-only fallback).

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a Transpose activation.
    """
    name = tensor_name.lower()
    is_activation = (
        tensor.tensor_type == TensorType.ActivationEncodings
        or tensor.tensor_type == TensorType.Encodings
    )
    return is_activation and ("transpose" in name or "permute" in name)


def is_concat_activation(tensor_name: str, tensor: TensorEncoding) -> bool:
    """Identify Concat operation activation tensors.

    Matches when the tensor is an activation tensor (V0/V1/DLC) or when the
    name contains a concat keyword (V2 / name-only fallback).

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        bool: True if tensor is identified as a Concat activation.
    """
    name = tensor_name.lower()
    is_activation = (
        tensor.tensor_type == TensorType.ActivationEncodings
        or tensor.tensor_type == TensorType.Encodings
    )
    return is_activation and ("concat" in name or "cat" in name)


# ==============================================================================
# Validation Rules
# ==============================================================================


def rule_conv_weights_symmetric(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: Convolution weights must be symmetric-quantized.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_conv_weight(tensor_name, tensor):
        return True, None
    is_valid = bool(tensor.is_symm and tensor.is_symm.lower() == "true")
    return is_valid, "Convolution weight tensors must be symmetric quantized"


def rule_rmsnorm_weights_16bit(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: RMSNorm weights must be quantized with a 16-bit bitwidth.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_rmsnorm_weight(tensor_name, tensor):
        return True, None
    is_valid = tensor.bitwidth == 16
    return is_valid, "RMSNorm weight tensors must be quantized with 16-bit bitwidth"


def rule_rmsnorm_weights_asymmetric(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: RMSNorm weights must be asymmetrically quantized.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_rmsnorm_weight(tensor_name, tensor):
        return True, None
    is_valid = bool(tensor.is_symm and tensor.is_symm.lower() == "false")
    return is_valid, "RMSNorm weight tensors must be asymmetrically quantized"


def rule_layernorm_weights_asymmetric(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: LayerNorm weights must be asymmetrically quantized.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_layernorm_weight(tensor_name, tensor):
        return True, None
    is_valid = bool(tensor.is_symm and tensor.is_symm.lower() == "false")
    return is_valid, "LayerNorm weight tensors must be asymmetrically quantized"


def rule_batchnorm_weights_asymmetric(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: BatchNorm weights must be asymmetrically quantized.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_batchnorm_weight(tensor_name, tensor):
        return True, None
    is_valid = bool(tensor.is_symm and tensor.is_symm.lower() == "false")
    return is_valid, "BatchNorm weight tensors must be asymmetrically quantized"


def rule_all_weights_8bit(tensor_name: str, tensor: TensorEncoding) -> Tuple[bool, str | None]:
    """Validation rule: All weight tensors must use 8-bit quantization.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    name = tensor_name.lower()
    is_weight = (
        tensor.tensor_type == TensorType.ParamEncodings
        or "weight" in name
        or "kernel" in name
    )
    if not is_weight:
        return True, None

    is_valid = tensor.bitwidth == 8
    return is_valid, "All weight tensors must use 8-bit quantization"


def rule_linear_weights_per_channel(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: Linear layer weights must use per-channel quantization.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_linear_weight(tensor_name, tensor):
        return True, None
    is_valid = tensor.channels > 1
    return is_valid, "Linear layer weights must use per-channel quantization"


def rule_layernorm_weights_16bit(
    tensor_name: str, tensor: TensorEncoding
) -> Tuple[bool, str | None]:
    """Validation rule: LayerNorm weights must be quantized with a 16-bit bitwidth.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    if not is_layernorm_weight(tensor_name, tensor):
        return True, None
    is_valid = tensor.bitwidth == 16
    return is_valid, "LayerNorm weight tensors must be quantized with 16-bit bitwidth"


def rule_large_quantization_range_warning(
    tensor_name: str, tensor: TensorEncoding, threshold: float = 1000.0
) -> Tuple[bool, str | None]:
    """Validation rule: Warning for very large activation and weight quantization ranges.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.
        threshold: Maximum allowed range (max - min). Default: 1000.0

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
            - is_valid: True if rule passes or doesn't apply
            - rule_description: Description of the rule if it applies, None otherwise
    """
    # Skip float tensors
    if not tensor.min or not tensor.max:
        return True, None

    # Calculate quantization range for each channel
    ranges = [max_val - min_val for min_val, max_val in zip(tensor.min, tensor.max)]
    max_range = max(ranges) if ranges else 0

    is_valid = max_range <= threshold
    if not is_valid:
        return (
            is_valid,
            f"WARNING: Large quantization range detected ({max_range:.2f} > {threshold}). "
            f"This may indicate quantization issues.",
        )
    return True, None


# ==============================================================================
# Rule Collections
# ==============================================================================

# These lists are kept for programmatic use by callers who want to register
# a curated subset of rules in Python code.  They are NOT applied automatically
# by ValidateEncodings — no rules run unless the caller explicitly loads a
# JSON config file or registers rules via add_validation_rule().

# A representative set covering the most common quantization constraints.
COMMON_RULES = [
    rule_conv_weights_symmetric,
    rule_rmsnorm_weights_16bit,
    rule_rmsnorm_weights_asymmetric,
    rule_layernorm_weights_asymmetric,
    rule_batchnorm_weights_asymmetric,
    rule_layernorm_weights_16bit,
    lambda name, tensor: rule_large_quantization_range_warning(name, tensor, threshold=1000.0),
]
