# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" Input Config for Emitter """
import abc
import warnings
from typing import Dict, List, Tuple, Union
import numpy as np
from qti.aisw.converters.common import ir_graph as ir_graph_lib
from qti.aisw.emitter import ENCODING_VERSION


def gate_min_max(
        min_val: Union[float, np.ndarray], max_val: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Gates min and max encoding values to retain zero in the range representation.
    Rules : min at maximum can be zero, max at minimum can be zero and
    if max and min are equal, adds epsilon to maintain range.
    :param min_val: min encoding value
    :param max_val: max encoding value
    :return: gated min and max values
    """

    epsilon = 1e-5
    # For per channel quantization
    if isinstance(min_val, np.ndarray):
        gated_min = np.clip(min_val, None, 0.0)
        gated_max = np.clip(max_val, 0.0, None)
        gated_max = np.clip(gated_max, gated_min + epsilon, None)
    else:
        gated_min = min(min_val, 0.0)
        gated_max = max(max_val, 0.0)
        gated_max = max(gated_max, gated_min + epsilon)

    return gated_min, gated_max


def calculate_delta_offset(
        min_val: Union[float, np.ndarray],
        max_val: Union[float, np.ndarray],
        bitwidth: int,
        use_symmetric_encodings: bool,
        use_strict_symmetric: bool,
) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
    """
    Calculates delta and offset given min and max.

    Quantization policy:
    - Asymmetric quantization is applied if all channels have strictly non-negative ranges (i.e., np.all(min_val >= 0)).
    - Symmetric quantization is applied only if `use_symmetric_encodings=True` and
     at least one channel has a negative range (i.e., np.any(min_val < 0)).

    :param min_val: min encoding value
    :param max_val: max encoding value
    :param bitwidth: bitwidth used for quantization
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :return: delta and offset values computed
    """
    num_steps = 2**bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    min_val, max_val = gate_min_max(min_val, max_val)

    # Check if both delta and offset are scalars
    if np.isscalar(min_val) and np.isscalar(max_val):
        # Use only max val to compute delta in the case of signed symmetric
        if use_symmetric_encodings and min_val < 0:
            num_positive_steps = np.floor(num_steps / 2)
            delta = max_val / num_positive_steps
            offset = -num_positive_steps
            if not use_strict_symmetric:
                offset -= 1
        else:
            delta = (max_val - min_val) / num_steps
            offset = round(min_val / delta)
        return delta, offset

    # np.array case
    min_val = np.asarray(min_val, dtype=np.float32)
    max_val = np.asarray(max_val, dtype=np.float32)

    delta = np.empty_like(min_val, dtype=np.float32)
    offset = np.empty_like(min_val, dtype=np.int32)

    num_positive_steps = np.floor(num_steps / 2)

    apply_symmetric = use_symmetric_encodings and not np.all(min_val >= 0)
    if apply_symmetric:
        delta[:] = max_val / num_positive_steps
        offset[:] = -num_positive_steps
        if not use_strict_symmetric:
            offset[:] -= 1
    else:
        delta[:] = (max_val - min_val) / num_steps
        offset[:] = np.round(min_val / delta).astype(np.int32)

    return delta, offset


def compute_min_max_given_delta_offset(
        delta: Union[float, np.ndarray],
        offset: Union[int, np.ndarray],
        bitwidth: int,
        use_symmetric_encodings: bool,
        use_strict_symmetric: bool,
) -> Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
    """
    Compute min and max given delta and offset.

    :param delta: Delta to compute with
    :param offset: Offset to compute with
    :param bitwidth: Bitwidth for finding number of steps
    :param use_symmetric_encodings: True if symmetric, False otherwise
    :param use_strict_symmetric: True if using strict symmetric, False otherwise
    :return: Tuple of computed min and max values
    """
    num_steps = 2**bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    # Check if both delta and offset are scalars
    is_scalar = np.isscalar(delta) and np.isscalar(offset)

    delta = np.asarray(delta, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)

    min_val = delta * offset
    max_val = (num_steps + offset) * delta

    # If inputs were scalars, return scalars
    if is_scalar:
        return float(min_val), float(max_val)

    return min_val, max_val


class TensorEncoding:
    dtype: str
    bitwidth: int
    enc_type: str
    is_symmetric: str
    scale: Union[float, List[float]]
    offset: Union[int, List[int], float, List[float]]
    min: Union[float, List[float]]
    max: Union[float, List[float]]

    def __init__(self, dtype, bitwidth, enc_type, is_symmetric, scale, offset, min, max):
        self.dtype = dtype
        self.bitwidth = bitwidth
        self.enc_type = enc_type
        self.is_symmetric = is_symmetric
        self.scale = scale
        self.offset = offset
        self.min = min
        self.max = max

    def _fill_partial_encoding(self):
        """
        Partial encoding case, only either of scale/offset or min/max are provided.
        """
        if self.bitwidth != 0 and (self.scale == 0 or (self.min == 0 and self.max == 0)):
            # only min/max provided
            if self.scale == 0:
                self.scale, self.offset = calculate_delta_offset(self.min, self.max,
                                                                 self.bitwidth, self.is_symmetric,
                                                                 False)
            # only scale/offset provided
            elif self.min == 0 and self.max == 0:
                self.min, self.max = compute_min_max_given_delta_offset(self.scale, self.offset,
                                                                        self.bitwidth, self.is_symmetric,
                                                                        False)

    @abc.abstractmethod
    def to_encoding_dict_0_6_1(self):
        pass

    @abc.abstractmethod
    def to_encoding_dict_1_0_0(self):
        pass

    @staticmethod
    def validate_symmetric_for_pcq(is_symmetric):
        assert is_symmetric, 'Per Channel Quantization requires is_symmetric to be set to True'
        return is_symmetric

    @staticmethod
    def validate_symmetric_for_lpbq(is_symmetric):
        assert is_symmetric, 'LPBQ requires is_symmetric to be set to True'
        return is_symmetric

    def to_encoding_dict(self):
        if ENCODING_VERSION == "0.6.1":
            if isinstance(self, (LPBQEncoding, PerBlockEncoding)):
                warnings.warn("BQ/LPBQ encodings are not supported directly with ENCODING_VERSION 0.6.1, "
                              "please use ENCODING_VERSION 1.0.0 for more reliability!")
            return self.to_encoding_dict_0_6_1()
        elif ENCODING_VERSION == "1.0.0":
            return self.to_encoding_dict_1_0_0()
        else:
            raise ValueError(f"Invalid ENCODING_VERSION, {ENCODING_VERSION} is set, only ['0.6.1', '1.0.0'] are allowed!")


class PerTensorEncoding(TensorEncoding):
    def __init__(self, enc_info: ir_graph_lib.IrQuantizationInfo, dtype: str):
        super().__init__(dtype=dtype, bitwidth=enc_info.encInfo.bw,
                         enc_type='PER_TENSOR', is_symmetric=str(enc_info.encInfo.is_symmetric),
                         scale=enc_info.encInfo.scale, offset=enc_info.encInfo.offset,
                         min=enc_info.encInfo.min, max=enc_info.encInfo.max)
        if self.dtype == 'int':
            self._fill_partial_encoding()

    def to_encoding_dict_0_6_1(self):
        if self.dtype == "int":
            return {
                'bitwidth': self.bitwidth,
                'scale': self.scale,
                'offset': self.offset,
                'min': self.min,
                'max': self.max,
                'is_symmetric': self.is_symmetric,
                'dtype': self.dtype.lower(),
            }
        else:
            return {
                'bitwidth': self.bitwidth,
                'dtype': 'float'
            }

    def to_encoding_dict_1_0_0(self):
        if self.dtype == "int":
            return {
                'dtype': self.dtype.upper(),
                'scale': [self.scale],
                'offset': [self.offset],
                'bw': self.bitwidth,
                'is_sym': self.is_symmetric,
                'enc_type': self.enc_type
            }
        else:
            return {
                'bw': self.bitwidth,
                'dtype': 'FLOAT'
            }


class PerChannelEncoding(TensorEncoding):
    def __init__(self, enc_info: ir_graph_lib.IrQuantizationInfo):
        scale, offset, min_, max_ = [], [], [], []
        for enc in enc_info.axisEncInfo.encInfos:
            scale.append(enc.scale)
            offset.append(enc.offset - 2 ** (enc.bw - 1))
            min_.append(enc.min)
            max_.append(enc.max)
        super().__init__(dtype="int",
                         bitwidth=enc_info.axisEncInfo.encInfos[0].bw,
                         enc_type='PER_CHANNEL',
                         is_symmetric=self.validate_symmetric_for_pcq(enc_info.axisEncInfo.encInfos[0].is_symmetric),
                         scale=scale, offset=offset, min=min_, max=max_)
        self._fill_partial_encoding()

    def to_encoding_dict_0_6_1(self):
        return [{
            'bitwidth': self.bitwidth,
            'scale': self.scale[idx],
            'offset': self.offset[idx],
            'min': self.min[idx],
            'max': self.max[idx],
            'is_symmetric': self.is_symmetric,
            'dtype': self.dtype.lower(),
        } for idx in range(len(self.scale))]

    def to_encoding_dict_1_0_0(self):
        return {
            'dtype': self.dtype.upper(),
            'scale': self.scale,
            'offset': self.offset,
            'bw': self.bitwidth,
            'is_sym': self.is_symmetric,
            'enc_type': self.enc_type
        }


class PerBlockEncoding(TensorEncoding):
    def __init__(self, enc_info: ir_graph_lib.IrQuantizationInfo):
        scale, offset, min_, max_ = [], [], [], []
        for enc in enc_info.bqEncInfo.encInfos:
            scale.append(enc.scale)
            offset.append(enc.offset - 2 ** (enc.bw - 1))
            min_.append(enc.min)
            max_.append(enc.max)
        super().__init__(dtype="int",
                         bitwidth=enc_info.bqEncInfo.encInfos[0].bw,
                         enc_type='PER_BLOCK',
                         is_symmetric=str(enc_info.bqEncInfo.encInfos[0].is_symmetric),
                         scale=scale, offset=offset, min=min_, max=max_)
        self.block_size = enc_info.bqEncInfo.blockSize
        self._fill_partial_encoding()

    def to_encoding_dict_0_6_1(self):
        return [{
            'bitwidth': self.bitwidth,
            'scale': self.scale[idx],
            'offset': self.offset[idx],
            'min': self.min[idx],
            'max': self.max[idx],
            'is_symmetric': self.is_symmetric,
            'dtype': self.dtype.lower(),
        } for idx in range(len(self.scale))]

    def to_encoding_dict_1_0_0(self):
        return {
            'dtype': self.dtype.upper(),
            'scale': self.scale,
            'offset': self.offset,
            'bw': self.bitwidth,
            'is_sym': self.is_symmetric,
            'block_size': self.block_size,
            'enc_type': self.enc_type
        }


class LPBQEncoding(TensorEncoding):
    def __init__(self, enc_info: ir_graph_lib.IrQuantizationInfo):
        super().__init__(dtype="int",
                         bitwidth=enc_info.lpbqEncInfo.encInfos[0].bw,
                         enc_type="LPBQ",
                         is_symmetric=self.validate_symmetric_for_lpbq(enc_info.lpbqEncInfo.encInfos[0].is_symmetric),
                         scale=[], offset=[], min=[], max=[])
        self.compressed_bw = enc_info.lpbqEncInfo.blockScaleBitwidth
        self.block_size = enc_info.lpbqEncInfo.blockSize
        self.per_block_int_scale = enc_info.lpbqEncInfo.blocksScale8 \
            if len(enc_info.lpbqEncInfo.blocksScale8) != 0 else enc_info.lpbqEncInfo.blocksScale16
        self.block_factor = len(self.per_block_int_scale)//len(enc_info.lpbqEncInfo.encInfos)
        self._extract_encoding(enc_info)

    def _extract_encoding(self, enc_info):
        for channel_enc in enc_info.lpbqEncInfo.encInfos:
            self.scale.append(channel_enc.scale)
            self.offset.append(channel_enc.offset - 2 ** (channel_enc.bw - 1))

        for i, block_scale in enumerate(self.per_block_int_scale):
            channel_enc_idx = i // self.block_factor
            channel_enc = enc_info.lpbqEncInfo.encInfos[channel_enc_idx]
            min_ = - (2 ** (enc_info.lpbqEncInfo.blockScaleBitwidth - 1) - 1) * channel_enc.scale * block_scale - channel_enc.scale * block_scale
            self.min.append(min_)
            max_ = (2 ** (enc_info.lpbqEncInfo.blockScaleBitwidth - 1) - 1) * channel_enc.scale * block_scale
            self.max.append(max_)

    def to_encoding_dict_0_6_1(self):
        return [{
            'bitwidth': self.bitwidth,
            'scale': self.scale[block_idx // self.block_factor],
            'offset': self.offset[block_idx // self.block_factor],
            'min': self.min[block_idx],
            'max': self.max[block_idx],
            'is_symmetric': self.is_symmetric,
            'dtype': self.dtype.lower(),
        } for block_idx in range(len(self.per_block_int_scale))]

    def to_encoding_dict_1_0_0(self):
        return {
            'dtype': self.dtype.upper(),
            'bw': self.bitwidth,
            'is_sym': self.is_symmetric,
            'compressed_bw': self.compressed_bw,
            'block_size': self.block_size,
            'scale': self.scale,
            'offset': self.offset,
            'enc_type': 'LPBQ',
            'per_block_int_scale': self.per_block_int_scale
        }
