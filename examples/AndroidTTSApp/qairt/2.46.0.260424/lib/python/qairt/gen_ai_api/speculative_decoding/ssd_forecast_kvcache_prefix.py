# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import re
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import torch

from qairt.gen_ai_api.builders.gen_ai_utils import _load_encoding, quantize
from qairt.utils import loggers

_logger = loggers.get_logger(__name__)


def process_ssd_param(
    onnxfile: Path,
    encodingfile: str | os.PathLike | None,
    output_dir: Path,
    ssd_param_filename: str,
    transposed_key_cache: bool = True,
) -> Path:
    """
    Process an SSD checkpoint to generate a quantized KV cache prefix file.

    Args:
        onnxfile: Path to the original ONNX model (used only to infer the number of layers).
        encodingfile: Path to a JSON file containing quantization encodings, or ``None``.
        output_dir: Directory where the generated ``kv-cache.primary.qnn-htp`` file will be written.
        ssd_param_filename: Path to the SSD checkpoint containing ``forecast_prefix``.
        transposed_key_cache: Whether to transpose the key cache to match runtime layout.

    Returns:
        Path to ``output_dir`` (convenient for chaining).
    """
    ssd_param = _load_ssd_checkpoint(Path(ssd_param_filename))
    prefix = _extract_prefix(ssd_param)
    prefix = _maybe_transpose(prefix, transposed_key_cache)

    _logger.info("SSD: forecast prefix length = %d", prefix[0][0].shape[2])
    onnxmodel = onnx.load(onnxfile, load_external_data=False)

    num_layers = len([i for i in onnxmodel.graph.output if re.search(r"_value_\d+_out$", i.name)])
    if num_layers == 0:
        raise RuntimeError("Unable to determine number of layers from ONNX model outputs.")

    # Quantize kvcache prefix
    encodings = _file_to_dict(encodingfile)
    cache_filename = os.path.join(output_dir, "kv-cache.primary.qnn-htp")
    _write_kv_cache(prefix, encodings, cache_filename, num_layers=num_layers)
    _logger.info("Saved forecast KV-cache prefix as %s", cache_filename)
    return output_dir


def _load_ssd_checkpoint(ssd_param_path: Path) -> dict:
    """Load the SSD checkpoint and return the raw dictionary."""
    if not ssd_param_path.is_file():
        raise FileNotFoundError(f"SSD checkpoint not found: {ssd_param_path}")
    return torch.load(str(ssd_param_path), map_location="cpu")


def _extract_prefix(ssd_param: dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Extract the forecast prefix tensor and split it into per-layer KV pairs.

    Returns:
        Tuple of (key_tensor, value_tensor) for each layer.
    """
    if "forecast_prefix" not in ssd_param:
        raise KeyError("Missing 'forecast_prefix' in SSD checkpoint.")
    prefix = ssd_param["forecast_prefix"].to(torch.float32)
    n_layer, _, _, _, len_prefix, _ = prefix.shape
    # Keep only the first token dimension (index 0) for each layer
    return tuple((prefix[layer_idx][0], prefix[layer_idx][1]) for layer_idx in range(n_layer))


def _maybe_transpose(
    prefix: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    transposed_key_cache: bool,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    If ``transposed_key_cache`` is True, transpose each key tensor from
    (batch, seq, head, dim) to (batch, seq, dim, head) to match the
    expected layout of the runtime.
    """
    if not transposed_key_cache:
        return prefix
    return tuple((kv[0].permute(0, 1, 3, 2), kv[1]) for kv in prefix)


def _file_to_dict(encodingfile: str | os.PathLike | None, no_merge: bool = False) -> dict:
    """
    Load a JSON encoding file and optionally merge activation/parameter encodings.
    """
    if encodingfile is not None:
        with open(encodingfile) as json_file:
            quant_encoding_dict = json.load(json_file)
        if no_merge:
            return quant_encoding_dict
        return _load_encoding(quant_encoding_dict)
    return {}


def _write_kv_cache(
    prefix: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    encodings: dict,
    cache_path: str,
    num_layers: int,
) -> None:
    """
    Quantize the KV-cache prefix using the supplied encodings and write the
    binary cache file.

    Args:
        prefix: Tuple of (key, value) tensors for each layer.
        encodings: Dictionary mapping tensor names to their quantization encodings.
        cache_path: Destination file path.
        num_layers: Number of layers expected in the model (used for validation).
    """

    def _is_kv_in(layer: int, name: str) -> bool:
        return re.search(f"_(key|value)_{layer}_in$", name) is not None

    layerwise_kv_in_names = [
        sorted([name for name in encodings.keys() if _is_kv_in(layer, name)]) for layer in range(num_layers)
    ]

    # Validate that each layer has exactly two entries (key and value)
    for idx, names in enumerate(layerwise_kv_in_names):
        if len(names) != 2:
            raise ValueError(f"Layer {idx} does not have both key and value encodings: {names}")

    key_value_encodings = [[encodings[name] for name in kv_names] for kv_names in layerwise_kv_in_names]

    key_q = [quantize(cache[0], encoding[0]) for cache, encoding in zip(prefix, key_value_encodings)]
    value_q = [quantize(cache[1], encoding[1]) for cache, encoding in zip(prefix, key_value_encodings)]

    key_cache = np.concatenate(key_q)
    value_cache = np.concatenate(value_q)

    CACHE_FILE_SPEC = "IIBxHHH"  # struct format: num_tensors, magic, dtype, n_head, n_kv_dim, n_tok
    CACHE_FILE_MAGIC = 0xC0DE  # Magic number used by Qualla cache files

    dtype_index = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        None,
        np.float16,
        np.float32,
        np.float64,
        np.bool_,
    ].index(key_cache.dtype)
    assert struct.calcsize(CACHE_FILE_SPEC) == 16, "Cache file header size must be 16 bytes"

    with open(cache_path, "wb") as handle:
        n_layer, n_head, n_tok, n_kv_dim = value_cache.shape
        num_tensors = n_layer * 2
        handle.write(
            struct.pack(
                CACHE_FILE_SPEC,
                num_tensors,
                CACHE_FILE_MAGIC,
                dtype_index,
                n_head,
                n_kv_dim,
                n_tok,
            )
        )
        key_cache.tofile(handle)
        value_cache.tofile(handle)
