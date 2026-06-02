# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum

from qti.aisw.accuracy_debugger.encodings.encodings import (
    TensorEncoding,
)
from qti.aisw.accuracy_debugger.encodings.encodings_utils import (
    TensorDtype,
)


VALID_BITWIDTHS = [4, 8, 16, 32]


class ComparisonStatus(Enum):
    """Overall comparison status for encodings comparison"""

    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNMAPPED = "UNMAPPED"


class FieldStatus(Enum):
    """comparison status for various field present in the encoding like
    dtype, is_symm, scale, offset, channels
    """

    NOT_COMPARED = "NOT_COMPARED"
    SAME = "SAME"
    DATA_TYPE_NOT_SAME = "Data type not same: {} vs {}"
    IS_SYMMETRIC_NOT_SAME = "is_symmetric not same: {} vs {}"
    OFFSET_NOT_SAME = "Offset not same: {} vs {} at channel number: {}"
    OFFSET_NOT_PRESENT = "Offset not present"
    SCALE_NOT_SAME = "Scale not in bound of {}: {} vs {} at channel number: {}"
    SCALE_NOT_PRESENT = "Scale not present"
    BITWIDTH_NOT_SIMILAR = "Bitwidth not similar: {} vs {}"
    INVALID_BITWIDTH = "Bitwidth is invalid"
    NUM_CHANNELS_NOT_SAME = "Number of channels are different: {} vs {}"
    TWO_FLOAT_ARE_NOT_SAME = "Two float encodings are not same"
    ENCODING_TYPE_NOT_SAME = "Encoding Types: {} vs {}. Can not be compared"
    BLOCK_SIZE_NOT_SAME = "Block size not same: {} vs {}"
    COMPRESSED_BW_NOT_SAME = "Compressed bitwidth not same: {} vs {}"
    NUM_BLOCKS_NOT_SAME = "Number of blocks are different: {} vs {}"
    PER_BLOCK_INT_SCALE_NOT_SAME = "Per block integer scales are not same"


def initialize_comparison_dict() -> dict:
    """Creates comparison dict for: dtype, is_symm, bitwidth, channels, scale, offset
    Args:
        initializer(FieldStatus | None): value with which each key in the dictionary will be initialized

    Returns:
        (dict): comparison dictionary initialized with NOT_COMPARED
    """
    keys = [
        "encoding_type",
        "dtype",
        "is_symm",
        "bitwidth",
        "channels",
        "scale",
        "offset",
        "block_size",
        "compressed_bw",
        "per_block_int_scale",
    ]
    compare_dict = {key: FieldStatus.NOT_COMPARED.value for key in keys}

    return compare_dict


def get_comparison_structure(tensor_names: set) -> dict:
    """Creates comaprison structure for the given tensor names

    Args:
        tensor_names (set): set of tensor names

    Return:
        (dict): dictionary of comparison structure
    """
    comparison_struct = {}
    for tensor_name in tensor_names:
        comparison_struct[tensor_name] = {}
        comparison_struct[tensor_name]["compare_info"] = {}
        comparison_struct[tensor_name]["Status"] = {}
        comparison_struct[tensor_name]["Mapping"] = []

    return comparison_struct


def updated_comparison_status(
    status1: ComparisonStatus, status2: ComparisonStatus
) -> ComparisonStatus:
    """Compare two error level and return the higher error level.

    Args:
        status1 (ComparisonStatus): comparison status level 1
        status2 (ComparisonStatus): comparison status level 2

    Returns:
        (ComparisonStatus): updated error level
    """
    if ComparisonStatus.ERROR == status1 or ComparisonStatus.ERROR == status2:
        return ComparisonStatus.ERROR
    elif ComparisonStatus.WARNING == status1 or ComparisonStatus.WARNING == status2:
        return ComparisonStatus.WARNING

    return ComparisonStatus.SUCCESS


def compare_scale_offset(
    tensor_encoding1: TensorEncoding,
    tensor_encoding2: TensorEncoding,
    scale_threshold: float = 1e-3,
) -> tuple[dict, ComparisonStatus]:
    """Compares scale and offset of two encodings given the bitwidthds

    Args:
        tensor_encoding1 (TensorEncoding): TensorEncoding object for tensor1
        tensor_encoding2 (TensorEncoding): TensorEncoding object for tensor2
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (dict, str): Returns the following:
        2. dict of comparision info
        3. ComparisonStatus of the encodings comparision
    """
    compare_info = {
        "channels": FieldStatus.NOT_COMPARED.value,
        "scale": FieldStatus.NOT_COMPARED.value,
        "offset": FieldStatus.NOT_COMPARED.value,
    }

    # Compare the number of channels
    if tensor_encoding1.channels != tensor_encoding2.channels:
        compare_info["channels"] = FieldStatus.NUM_CHANNELS_NOT_SAME.value.format(
            tensor_encoding1.channels, tensor_encoding2.channels
        )
        return compare_info, ComparisonStatus.ERROR
    else:
        compare_info["channels"] = FieldStatus.SAME.value

    comparison_status = ComparisonStatus.SUCCESS
    # Compare scale and offsets
    # Since two encodings with different bitwidths can be algebrically converted into one another
    # we need to compare the scale and offset accordingly by scaling them
    multiplier = pow(2, tensor_encoding1.bitwidth - tensor_encoding2.bitwidth)
    for index, scale in enumerate(zip(tensor_encoding1.scale, tensor_encoding2.scale)):
        s1, s2 = scale
        threshold = scale_threshold * min(s1, s2)
        if abs(s1 * multiplier - s2) > threshold:
            compare_info["scale"] = FieldStatus.SCALE_NOT_SAME.value.format(
                threshold, s1, s2, index
            )
            comparison_status = ComparisonStatus.ERROR
            break
    if compare_info["scale"] == FieldStatus.NOT_COMPARED.value:
        compare_info["scale"] = FieldStatus.SAME.value

    for index, offset in enumerate(zip(tensor_encoding1.offset, tensor_encoding2.offset)):
        o1, o2 = offset
        if o1 != o2 * multiplier:
            compare_info["offset"] = FieldStatus.OFFSET_NOT_SAME.value.format(o1, o2, index)
            comparison_status = ComparisonStatus.ERROR

            break
    if compare_info["offset"] == FieldStatus.NOT_COMPARED.value:
        compare_info["offset"] = FieldStatus.SAME.value

    return compare_info, comparison_status


def compare_dtype(dtype1: TensorDtype, dtype2: TensorDtype) -> tuple[dict, ComparisonStatus]:
    """Compares dtype of two encodings

    Args:
        dtype1 (str): dtype of encoding1
        dtype2 (str): dtype of encoding2

    Returns:
        (dict, ComparisonStatus): Returns the following:
            1. dictionary of comparison info with "dtype" being key
            2. ComparisonStatus of the encodings comparision
    """
    compare_info = {"dtype": FieldStatus.SAME.value}
    # Case1: both dtypes are not same
    if dtype1 != dtype2:
        compare_info["dtype"] = FieldStatus.DATA_TYPE_NOT_SAME.value.format(
            dtype1.value, dtype2.value
        )
    # Case2: both dtypes are float
    elif dtype1 == TensorDtype.FLOAT and dtype2 == TensorDtype.FLOAT:
        compare_info["dtype"] = FieldStatus.TWO_FLOAT_ARE_NOT_SAME.value

    comparison_status = (
        ComparisonStatus.SUCCESS
        if compare_info["dtype"] == FieldStatus.SAME.value
        else ComparisonStatus.ERROR
    )

    return compare_info, comparison_status
