# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic.json_schema import SkipJsonSchema

from qairt.modules.dlc_module.dlc_utils import GraphInfo
from qairt.modules.multi_graph_execution.multi_graph_utils import MultiGraphInputData, get_input_to_graph_map
from qti.aisw.tools.core.modules.api import (
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.utils.configure_backend import get_supported_backends
from qti.aisw.tools.core.modules.net_runner import (
    NetRunner,
    NetRunnerRunArgConfig,
    NetRunnerRunOutputConfig,
)
from qti.aisw.tools.core.modules.net_runner.utils import NamedTensorMapping
from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger


class MultiGraphExecutionInputConfig(NetRunnerRunArgConfig):
    """
    Defines arguments for a multi graph execution.

    Graphs info comprises of information of the graph including its name, inputs and outputs.
    Information about the input(s) tensor name, dimensions, datatype is also included. Same goes for
    the output tensors.

    Input data can either be a str or Path, representing the input yaml file or a list of inputs, each
    corresponding to the graphs.
    """

    graphs_info: list[GraphInfo]
    input_data: MultiGraphInputData
    graph_name: Optional[List[str]] = None
    optional_output_tensor_names: Optional[Dict[str, Optional[List[str]]]] = None


class MultiGraphExecutionOutputConfig(NetRunnerRunOutputConfig):
    output_data: Dict[str, List[NamedTensorMapping]]


class MultiGraphExecutionModuleSchema(ModuleSchema):
    _BACKENDS = get_supported_backends()
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)

    name: Literal["MultiGraphExecutionModule"] = "MultiGraphExecutionModule"
    path: Path = Path(__file__)
    arguments: MultiGraphExecutionInputConfig
    outputs: SkipJsonSchema[Optional[MultiGraphExecutionOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class MultiGraphExecution(Module):
    """Executes a model with multiple graphs."""

    _SCHEMA = MultiGraphExecutionModuleSchema

    def __init__(self, logger: Optional[logging.Logger] = None, net_runner: Optional[NetRunner] = None):
        """Initializes a MultiGraphExecution module instance

        Args:
            logger (Optional[logging.Logger]): A logger instance to be used by the MultiGraphExecution module.
            net_runner (Optional[NetRunner]): Netrunner object to call the run function
        """
        if logger:
            self._logger = QAIRTLogger.get_logger("MultiGraphExecutionLogger", parent_logger=logger)
        else:
            self._logger = QAIRTLogger.get_logger(
                "MultiGraphExecutionLogger",
                level="INFO",
                formatter_val="extended",
                handler_list=["dev_console"],
            )
        self._net_runner = net_runner or NetRunner()

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def execute(self, config: MultiGraphExecutionInputConfig) -> MultiGraphExecutionOutputConfig:
        """
        Runs inferences on a backend, model, and target based on the provided identifier.
        Inference is run on all the graphs present in the model, by mapping the correct input
        to its corresponding graph.

        Args:
            config (MultiGraphExecutionInputConfig): Arguments of the inference, including inputs,
            graphs_info and an identifier for a model, backend, and target.

        Returns:
            Output data as a list of list of tensor name -> np array mappings, encompassing outputs
            of all the graphs.
        """
        graphs_info = config.graphs_info
        net_runner_run_arg_config = NetRunnerRunArgConfig(
            identifier=config.identifier,
            input_data=[],
            backend_config_file=config.backend_config_file,
            backend_config_dict=config.backend_config_dict,
            context_config=config.context_config,
            inference_config=config.inference_config,
            output_dir=config.output_dir,
            input_tensor_names=config.input_tensor_names,
        )
        output_data_dict = {}
        graph_inp_map = None
        graph_inp_map = get_input_to_graph_map(
            config.input_data, graphs_info, net_runner_run_arg_config.inference_config.use_native_input_data
        )

        filtered_graphs_info = graphs_info
        if config.graph_name:
            filtered_graphs_info = [info for info in graphs_info if info.name in config.graph_name]

        for info in filtered_graphs_info:
            net_runner_run_arg_config.input_data = graph_inp_map[info.name]
            net_runner_run_arg_config.graph_name = info.name

            # Extract graph-specific output tensor names
            if config.optional_output_tensor_names is not None:
                # Get ONLY the output tensor names for THIS graph
                graph_specific_output_tensors = config.optional_output_tensor_names.get(info.name)
                net_runner_run_arg_config.optional_output_tensor_names = graph_specific_output_tensors
            else:
                net_runner_run_arg_config.optional_output_tensor_names = None

            try:
                inference_output_config = self._net_runner.run(net_runner_run_arg_config)
                output_data_dict[info.name] = inference_output_config.output_data
            except Exception as e:
                self._logger.error(f"Inference failed for graph {info.name}: {e}")
                output_data_dict[info.name] = []

        return MultiGraphExecutionOutputConfig(output_data=output_data_dict)
