# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Helper function to record LoRA slice metadata.
"""

import os
from collections import defaultdict
from typing import Any

import numpy as np

from qairt.optimizer.onnx.utils.utils import (
    get_lora_weight_shapes,
    infer_slice_output_shape,
)
from qairt.optimizer.utils.logger import logger


def check_weights_identical(
    name: str, mha_named_safetensors: dict[str, dict], sha_named_safetensors: dict[str, dict]
):
    for encset in mha_named_safetensors:
        if name not in sha_named_safetensors[encset]:
            return False
        if name not in mha_named_safetensors[encset]:
            return False
        if not np.equal(mha_named_safetensors[encset][name], sha_named_safetensors[encset][name]).all():
            return False
    return True


def record_lora_split_metadata_from_tracing(
    merged_tracing_info: list[dict[str, Any]],
    mha_named_safetensors: dict[str, dict],
    sha_named_safetensors: dict[str, dict],
    metadata_graph_id: str,
    transforms_metadata: str | None = None,
    split_no: int | None = None,
    ar_n: int | None = None,
) -> dict[str, Any | None]:
    """
    Extracts LoRA slice transformation metadata directly from tracing information
    and initial safetensor data. It records individual 'Slice' operations
    for each head, avoiding runtime graph lookups for shapes and value metadata.
    Also records 'Identity' transforms for LoRA index tensors.

    Args:
        merged_tracing_info: A list of merged tracing information entries.
        mha_named_safetensors: A dictionary containing all the *MHA* LoRA safetensors information.
                               Used to retrieve initial shapes for tracing source tensors and to
                               identify LoRA index tensors.
        sha_named_safetensors: A dictionary containing all the *SHA* LoRA safetensors information.
                               Expected to be nested (e.g., {'encset_name': {'tensor_name': data}}).
        metadata_graph_id: The group ID to use for recording transforms.
        transforms_metadata: Optional path to an existing metadata JSON file. If provided,
                             transforms will be loaded from this file into the manager.

    Returns:
        A dictionary representing the generated metadata, or None if no transforms were recorded
        or if TransformManager is not active.
    """

    from qti.aisw.converters.common.tensor_transforms.operator import Operator
    from qti.aisw.converters.common.tensor_transforms.transform_manager import (
        TransformManager,
    )

    logger.debug("--- Starting record_lora_split_metadata_from_tracing (Metadata-Driven Slice Approach) ---")

    active_transform_manager = TransformManager()
    if transforms_metadata:
        if os.path.exists(transforms_metadata):
            logger.info(f"Loading existing metadata from: {transforms_metadata}")
            try:
                active_transform_manager.load_metadata(transforms_metadata)
                logger.debug("Successfully loaded metadata.")
            except Exception as e:
                logger.error(f"Failed to load metadata from {transforms_metadata}: {e}")
        else:
            logger.warning(
                f"Input metadata path '{transforms_metadata}' does not exist. Starting with an empty TransformManager."
            )
    else:
        logger.debug("No input metadata path provided. Starting with an empty TransformManager.")

    # Create the transform graph for this specific metadata_graph_id if it doesn't already exist
    if not active_transform_manager.has_graph(metadata_graph_id):
        active_transform_manager.create_transform_graph(
            metadata_graph_id, order_num=1, split_num=split_no, ar_n=ar_n
        )

    # Collect SHA tensor names and their dtypes
    sha_lora_tensor_names = set()
    sha_tensor_dtypes = {}
    for _, tensors_dict in sha_named_safetensors.items():
        for tname, tensor in tensors_dict.items():
            sha_lora_tensor_names.add(tname)
            dt = getattr(tensor, "dtype", None)
            if dt is not None:
                sha_tensor_dtypes[tname] = getattr(dt, "name", None) or str(dt) or None

    logger.debug(
        f"Flattened SHA named safetensors. Total individual tensor names: {len(sha_lora_tensor_names)}"
    )

    # Get shapes of original MHA LoRA weights for looking up src_name shapes
    mha_lora_weight_shapes = get_lora_weight_shapes(mha_named_safetensors)
    logger.debug(f"MHA LoRA weights identified: {len(mha_lora_weight_shapes)}.")

    # 1. Pre-process merged_tracing_info for efficient lookup by dst_name
    tracings_by_dst_name = defaultdict(list)
    for i, trace_entry in enumerate(merged_tracing_info):
        dst_name = trace_entry.get("dst_name")
        if (
            trace_entry.get("tracing_type") == "M2sTracingInfo"
            and dst_name
            and all(k in trace_entry for k in ["src_name", "axes", "starts", "ends"])
        ):
            tracings_by_dst_name[dst_name].append((i, trace_entry))

    lora_slice_transforms_recorded = 0
    sha_weights_with_tracing = set()

    # 2. Iterate over the SHA named_safetensors
    logger.debug(f"Iterating over flattened SHA named safetensors. Total: {len(sha_lora_tensor_names)}.")
    for dest_tensor_name in sha_lora_tensor_names:
        # Check if this SHA LoRA weight (destination tensor) has any tracing entries
        if dest_tensor_name not in tracings_by_dst_name:
            if dest_tensor_name in mha_lora_weight_shapes and check_weights_identical(
                dest_tensor_name, mha_named_safetensors, sha_named_safetensors
            ):
                # same tensor shown in mha safetensors and sha safentensors, and also has same value
                # which means it is lora related tensors but not touched by mha2sha
                tensor_shape = mha_lora_weight_shapes[dest_tensor_name]

                active_transform_manager.add_transform(
                    src_tensors=[dest_tensor_name],
                    dest_tensors=[dest_tensor_name],
                    tensor_dtypes={dest_tensor_name: sha_tensor_dtypes[dest_tensor_name]}
                    if dest_tensor_name in sha_tensor_dtypes
                    else None,
                    operator=Operator(
                        op_type="Identity", input_shapes=[tensor_shape], output_shapes=[tensor_shape]
                    ),
                    graph_id=metadata_graph_id,
                )
            else:
                logger.error(
                    f"  Skipping '{dest_tensor_name}': No M2sTracingInfo found for this SHA LoRA weight."
                )
                continue
        else:
            # Process all tracing entries associated with this specific SHA LoRA weight
            if len(tracings_by_dst_name[dest_tensor_name]) > 1:
                logger.warning(
                    f"  Found multiple ({len(tracings_by_dst_name[dest_tensor_name])}) tracing entries for destination '{dest_tensor_name}'. Expected one. Will attempt to use the first valid one."
                )

            transform_added_for_this_dest = False
            for original_idx, trace_entry in tracings_by_dst_name[dest_tensor_name]:
                if transform_added_for_this_dest:
                    logger.debug(
                        f"  Skipping additional tracing entry for '{dest_tensor_name}' (entry {original_idx}) as a transform has already been added."
                    )
                    continue

                src_tensor_name = trace_entry["src_name"]

                # Get the original MHA shape using src_name
                src_shape_for_metadata = mha_lora_weight_shapes.get(src_tensor_name)
                if src_shape_for_metadata is None:
                    continue

                # Validate slice axis and indices from tracing info
                axes = trace_entry["axes"]
                starts = trace_entry["starts"]
                ends = trace_entry["ends"]

                if (
                    not isinstance(axes, list)
                    or not axes
                    or not isinstance(starts, list)
                    or not starts
                    or not isinstance(ends, list)
                    or not ends
                ):
                    continue

                # Infer dest_shape using the helper function
                dest_shape = infer_slice_output_shape(
                    src_shape_for_metadata, axes, starts, ends, src_tensor_name, original_idx
                )

                if dest_shape is None:
                    continue

                # Prepare attributes for a 'Slice' operator
                slice_attributes: dict[str, Any] = {
                    "axes": axes,
                    "starts": starts,
                    "ends": ends,
                }

                lora_slice_operator = Operator(
                    op_type="Slice",
                    attributes=slice_attributes,
                    input_shapes=[src_shape_for_metadata] if src_shape_for_metadata else [],
                    output_shapes=[dest_shape] if dest_shape else [],
                )

                active_transform_manager.add_transform(
                    src_tensors=[src_tensor_name],
                    dest_tensors=[dest_tensor_name],
                    operator=lora_slice_operator,
                    graph_id=metadata_graph_id,
                )
                lora_slice_transforms_recorded += 1
                sha_weights_with_tracing.add(dest_tensor_name)
                transform_added_for_this_dest = True

            if not transform_added_for_this_dest:
                logger.error(
                    f"  Skipping '{dest_tensor_name}': No valid M2sTracingInfo found for this SHA LoRA weight."
                )
                continue

    if lora_slice_transforms_recorded == 0:
        logger.info(
            "No LoRA weight slice transforms identified and recorded from tracing info for SHA tensors."
        )
    else:
        logger.info(
            f"Successfully identified and recorded {lora_slice_transforms_recorded} LoRA weight slice transforms for SHA tensors."
        )

    logger.debug("--- Exiting record_lora_split_metadata_from_tracing ---")
    return active_transform_manager.get_metadata()
