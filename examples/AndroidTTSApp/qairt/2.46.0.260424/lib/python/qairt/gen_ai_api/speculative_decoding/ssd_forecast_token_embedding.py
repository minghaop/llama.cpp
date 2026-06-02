# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.numpy_helper import from_array, to_array

log = logging.getLogger(__name__)


def add_forecast_token_embeddings(
    onnxfile: Path,
    ssd_param_filename: Path,
    output_dir: Path,
) -> Path:
    """
    Append forecast token embeddings to an ONNX model.

    Args:
        onnxfile (Path): Path to the original ONNX model.
        ssd_param_filename (Path): Path to the SSD checkpoint containing ``forecast_embedding``.
        output_dir (Path): Destination path where the modified model will be written.

    Returns:
        Path: The path to the written model (``output_dir``) or the original model
        if no embedding table is present.
    """
    onnxmodel = onnx.load(onnxfile, load_external_data=True)

    onnx_input_names = [i.name for i in onnxmodel.graph.input]
    if "input_ids" in onnx_input_names:
        embed_forecast_token_embeddings(
            onnxmodel,
            ssd_param_filename,
            base_dir=onnxfile.parent,
        )
        onnx.save(onnxmodel, output_dir, save_as_external_data=True)
        return output_dir
    else:
        if not any(name in onnx_input_names for name in ("inputs_embeds", "input_embeds")):
            raise ValueError(
                f"input_ids and inputs_embeds/input_embeds are not found in the ONNX inputs: {onnx_input_names}"
            )
        log.info("Skipping embed_forecast_token_embeddings as embeddings table is not present in the model")
        return onnxfile


def concat_ssd_param_embeddings_into_embeddings_table(
    embedding_table: np.ndarray, ssd_param_path: Path
) -> np.ndarray:
    """
    Append forecast token embeddings from an SSD checkpoint to the existing
    embedding table.

    Args:
        embedding_table (np.ndarray): Original token embedding matrix of shape (vocab_size, hidden_dim).
        ssd_param_path (Path): Path to the SSD checkpoint containing ``forecast_embedding``.

    Returns:
        np.ndarray: New embedding matrix with forecast tokens appended.
    """
    ssd_param = torch.load(str(ssd_param_path), map_location="cpu")
    forecast_token_embeddings = ssd_param["forecast_embedding"].to(torch.float32)

    if embedding_table.shape[1] != forecast_token_embeddings.shape[1]:
        raise ValueError(
            f"Mismatching token embedding size: model dim={embedding_table.shape[1]}, "
            f"forecast dim={forecast_token_embeddings.shape[1]}"
        )

    log.info("SSD: the number of forecast tokens = %d", len(forecast_token_embeddings))
    new_embedding_table = np.concatenate((embedding_table, forecast_token_embeddings), axis=0)
    return new_embedding_table


def get_embedding_table_and_proto_and_node(
    onnxmodel: onnx.ModelProto, base_dir: Path
) -> tuple[np.ndarray, onnx.TensorProto, onnx.NodeProto]:
    """
    Retrieve the embedding table tensor, its initializer proto, and the Gather node
    that consumes it.

    Args:
        onnxmodel (onnx.ModelProto): The loaded ONNX model.
        base_dir (Path): Directory containing any external tensor data.

    Returns:
        tuple: (embedding_table, embedding_table_proto, embedding_gather_node)
    """
    embedding_table_node = _find_embedding_table_node(onnxmodel)
    embedding_table_name = embedding_table_node.input[0]

    embedding_table_proto = next(
        (i for i in onnxmodel.graph.initializer if i.name == embedding_table_name),
        None,
    )
    if embedding_table_proto is None:
        raise RuntimeError(f"Initializer '{embedding_table_name}' not found in the model.")

    embedding_table = to_array(embedding_table_proto, base_dir=str(base_dir))
    return embedding_table, embedding_table_proto, embedding_table_node


def embed_forecast_token_embeddings(onnxmodel: onnx.ModelProto, ssd_param_path: Path, base_dir: Path) -> None:
    """
    Modify the ONNX model in‑place by replacing its embedding initializer with a
    version that includes forecast token embeddings.

    Args:
        onnxmodel (onnx.ModelProto): The model to modify.
        ssd_param_path (Path): Path to the SSD checkpoint.
        base_dir (Path): Directory containing external data for the model.

    Returns:
        None
    """
    embedding_table, embedding_table_proto, _ = get_embedding_table_and_proto_and_node(onnxmodel, base_dir)

    new_embedding_table = concat_ssd_param_embeddings_into_embeddings_table(embedding_table, ssd_param_path)
    onnxmodel.graph.initializer.remove(embedding_table_proto)
    onnxmodel.graph.initializer.append(from_array(new_embedding_table, embedding_table_proto.name))


def _find_embedding_table_node(onnxmodel: onnx.ModelProto) -> onnx.NodeProto:
    """
    Return the first Gather node that provides the embedding table.

    Args:
        onnxmodel (onnx.ModelProto): The ONNX model to search.

    Raises:
        RuntimeError: If no Gather node is found in the graph.

    Returns:
        onnx.NodeProto: The first matching Gather node.
    """
    gathers = [node for node in onnxmodel.graph.node if node.op_type == "Gather"]
    if not gathers:
        raise RuntimeError("No Gather node found in the ONNX graph.")
    return gathers[0]
