# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import qairt
from qairt.modules.dlc_module.dlc_utils import GraphInfo
from qti.aisw.tools.core.modules.net_runner.utils import NetRunnerInputData

MultiGraphInputData = Union[List[NetRunnerInputData], str, Path]


def get_graph_info(dlc_path: Union[Path, str]) -> list[GraphInfo]:
    """
    Get information about the graphs and their inputs and outputs from the given DLC.

    Args:
        dlc_path (Union[Path, str]): Path to the DLC. Must be provided.

    Returns:
        list[GraphInfo]: Graph info corresponding to the DLC.

    Raises:
        FileNotFoundError: If given DLC path does not exist.
    """
    if not os.path.exists(dlc_path):
        raise FileNotFoundError(f"Given DLC {dlc_path} does not exist.")
    loaded_model = qairt.load(dlc_path)
    graphs_info = loaded_model.module.info.graphs
    return graphs_info


def is_model_multi_graph(
    dlc_path: Optional[Path | str] = None, graphs_info: Optional[list[GraphInfo]] = None
) -> bool:
    """
    Check if given model has multiple graphs. Either DLC path or graphs_info has to be provided.

    Args:
        dlc_path (Union[Path, str]): Path to the DLC. Must be provided if graphs_info is not provided.
        graphs_info (list[GraphInfo]): Graphs info for a model. Must be provided if DLC is not provided.

    Returns:
        bool: Boolean to indicate whether model has multiple graphs or not.

    Raises:
        ValueError: If both DLC path and graphs_info are not provided.
    """
    if not graphs_info and not dlc_path:
        raise ValueError(
            f"Either dlc_path or graphs_info required to determine if model has multiple graphs."
        )
    dlc_path = dlc_path or ""
    graphs_info = graphs_info if graphs_info else get_graph_info(dlc_path=dlc_path)
    return len(graphs_info) > 1


def get_all_graphs_input_dtype_pair(graph_info: list[GraphInfo]) -> List[Dict[str, List[Tuple[str, str]]]]:
    """
    Map all graphs to their inputs and the corresponding datatypes.
    Sample output:
    [
        {
            "graph1": [
                ("input1", "QNN_DATATYPE_INT_64"),
                ("input2", "QNN_DATATYPE_FLOAT_32"),
                ("input3", "QNN_DATATYPE_BOOL_8"),
            ]
        },
        {
            "graph2": [
                ("input1", "QNN_DATATYPE_INT_64"),
                ("input2", "QNN_DATATYPE_INT_64"),
                ("input3", "QNN_DATATYPE_INT_64")
            ]
        }
    ]

    Args:
        graph_info (list[GraphInfo]): Graphs info for a model.

    Returns:
        List[Dict[str, List[Tuple[str, str]]]]: Mapping of graph name with inputs and datatypes.
    """
    graph_inp_name_dtype_list = []
    for info in graph_info:
        input_dtype_pair_list = []
        for tensor_info in info.inputs:
            dtype_list = tensor_info.data_type.split(".")
            datatype = dtype_list[0] if len(dtype_list) == 1 else dtype_list[1]
            input_dtype_pair_list.append((tensor_info.name, datatype))
        graph_inp_name_dtype_list.append({info.name: input_dtype_pair_list})
    return graph_inp_name_dtype_list


def get_input_to_graph_map(
    input_data: MultiGraphInputData, graphs_info: list[GraphInfo], use_native_input: bool
) -> Dict[str, Union[Dict, List, np.ndarray]]:
    """
    Given all inputs for a multi-graph model, map them to their corresponding graphs.

    Args:
        input_data (MultiGraphInputData): All inputs to a multi-graph model. Could be a list or a yaml file.
        graphs_info (list[GraphInfo]): Graphs info for all the graphs in the model.
        use_native_input (bool): Boolean to indicate if inputs are native (non-float32).

    Returns:
        Dict[str:Union[Dict, List, np.ndarray]]: Mapping of graph name to its inputs.

    Examples: Example to get input to graph map
        >>> graphs_info = get_graph_info(model.dlc)
        >>> input_map = get_input_to_graph_map(input_data=input.yaml,
                                               graphs_info=graphs_info,
                                               use_native_input=True)
        Sample output:
        {
            "graph1": {"input1": <ndarray>, "input2": <ndarray>},
            "graph2": {"input1": <ndarray>, "input2": <ndarray>},
        }
    """
    input_to_graph_map = {}

    if isinstance(input_data, (str, Path)):
        inp_datatype_map = get_all_graphs_input_dtype_pair(graph_info=graphs_info)
        from qti.aisw.core.model_level_api.utils.yaml_utils import extract_and_load_input_tensors

        input_to_graph_map = extract_and_load_input_tensors(
            yaml_path=input_data,
            native_inputs=use_native_input,
            native_input_tensor_names=None,
            graph_input_name_dtype_pairs=inp_datatype_map,
        )
    elif isinstance(input_data, List):
        for _input in input_data:
            if isinstance(_input, (Path, str)):
                # need to figure out how to handle this case
                raise ValueError("Input of type txt file not supported.")
            inp_dims = []
            if isinstance(_input, List) and isinstance(_input[0], Dict):
                inp_dims = [i.shape for i in _input[0].values()]
            elif isinstance(_input, Dict):
                inp_dims = [i.shape for i in _input.values()]
            elif isinstance(_input, List) and isinstance(_input[0], np.ndarray):
                inp_dims = [_input[0].shape]
            elif isinstance(_input, np.ndarray):
                inp_dims = [_input.shape]
            else:
                raise ValueError("Unsupported input format.")
            for info in graphs_info:
                name = info.name
                graph_inp_dims = [tensor_info.dimensions for tensor_info in info.inputs]
                if all(list(dim) in graph_inp_dims for dim in inp_dims):
                    input_to_graph_map[name] = _input
    return input_to_graph_map
