# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import platform
import shutil
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy
from typing_extensions import Unpack

from qairt.api.bases.model_base import ModelBase
from qairt.api.configs import (
    BackendType,
    Device,
    DevicePlatformType,
    ExecutionInputData,
    ExecutionResult,
    RemoteDeviceIdentifier,
    Target,
)
from qairt.api.executor.config_util import get_config_api_options_dict
from qairt.api.executor.execution_config import ExecutionConfig, supported_initialize_execute_platforms
from qairt.api.profiler.profiler import profile
from qairt.modules.cache_module.cache_module import CacheModule
from qairt.modules.common.graph_info_models import TensorInfo
from qairt.modules.dlc_module import DlcModule
from qairt.modules.dlc_module.dlc_utils import GraphInfo
from qairt.modules.lora.lora_config import (
    UseCaseOutputConfig,
    load_use_case_config,
    serialize_lora_adapter_weight_config,
)
from qairt.modules.qti_import import qti_module_api
from qairt.utils.asset_utils import AssetMapping, AssetType, check_asset_type
from qairt.utils.exceptions import UnknownAssetError
from qairt.utils.loggers import get_logger
from qti.aisw.core.model_level_api import Target as mlapi_target
from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
    InferenceIdentifier,
    NetRunner,
    NetRunnerLoadArgConfig,
    NetRunnerRunArgConfig,
    NetRunnerUnloadArgConfig,
)

__ALL__ = ["Model"]

_model_logger = get_logger("qairt.model")


class Model(ModelBase):
    """
    Representative entity that is executable on a QAIRT Backend.

    It has the following additional properties:

    - It can be saved to an asset that remains executable on a QAIRT Backend.
    - It can be loaded from a DLC. If load is successful, the Model object can be executed similar
      to the asset.
    - It can be queried to return properties that identify the asset from which it was loaded.
    """

    def __init__(self, module: DlcModule, *, name: str = ""):
        """
        Initializes a new Model object.

        Args:
            module (Optional[DlcModule]): The module object to associate with this model.
            name (str): The name of the model. If not provided, the name of the module is used.

        """
        super().__init__(name=name)

        # A module is compositionally related to its graphs, which may be in memory or serialized on disk.
        self.module: DlcModule = module

        # Assets that are generated as part of model execution, or as part of operations performed
        # on this model object that may be needed for execution. Users are not expected to interact with
        # this structure.
        self.assets = AssetMapping[str]()
        # The lora use case configurations
        self.lora_use_cases: Optional[Union[List[UseCaseOutputConfig], str, os.PathLike, None]] = None

    @property
    def module(self) -> DlcModule:
        """
        Returns the dlc module associated with this model.
        """
        return self._module

    @module.setter
    def module(self, module: DlcModule):
        """
        Sets the dlc module associated with this model.
        """
        if not isinstance(module, DlcModule):
            raise TypeError("Module must be of type DlcModule")

        if self._module is None:
            self._module: DlcModule = module
        else:
            raise AttributeError("Module cannot be reset after initialization.")

    @module.deleter
    def module(self):
        if self._module is None:
            return
        raise RuntimeError("Cannot delete module after initialization.")

    @property
    def graphs_info(self) -> list[GraphInfo]:
        """
        Returns a list of graph information details of the model.
        """
        return self._module.info.graphs

    def _normalize_optional_output_tensor_names(
        self,
        optional_output_tensor_names: Optional[Union[List[str], Dict[str, Optional[List[str]]]]],
    ) -> Optional[Union[List[str], Dict[str, Optional[List[str]]]]]:
        """
        Validates and normalizes optional_output_tensor_names according to model graph count.

        - Single-graph models: returns None or a copied List[str]
        - Multi-graph models: returns a sanitized Dict[graph_name, Optional[List[str]]] (with lists copied),
          warning and filtering out unknown graph names.

        Raises:
            TypeError on invalid shapes/types (e.g., wrong data types or non-str elements).
        """
        # Fast path: None means "use all outputs" — no validation needed
        if optional_output_tensor_names is None:
            return None

        # Fast path for single-graph style usage: list[str]
        if isinstance(optional_output_tensor_names, list):
            if not all(isinstance(n, str) for n in optional_output_tensor_names):
                raise TypeError("optional_output_tensor_names must be a list of strings")
            return list(optional_output_tensor_names)  # defensive copy

        # The only remaining valid option is multi-graph style usage: dict[str, Optional[list[str]]]
        if not isinstance(optional_output_tensor_names, dict):
            raise TypeError(
                "optional_output_tensor_names must be one of: None, List[str], or Dict[str, Optional[List[str]]]"
            )

        graphs_info = self.graphs_info
        num_graphs = len(graphs_info)

        if num_graphs <= 0:
            raise ValueError("Model must contain at least one graph")

        if num_graphs == 1:
            raise TypeError(
                "For single-graph models, optional_output_tensor_names must be a List[str] or None, "
                "containing only the output tensor names"
            )

        valid_graph_names = {g.name for g in graphs_info}
        # Warn and drop unknown graph keys
        unknown = [k for k in optional_output_tensor_names if k not in valid_graph_names]
        if unknown:
            _model_logger.warning(
                "Ignoring unknown graph name(s) in optional_output_tensor_names: %s",
                ", ".join(sorted(unknown)),
            )

        # Build a sanitized copy to avoid returning caller's dict/lists
        sanitized: Dict[str, Optional[List[str]]] = {}
        for gname, names in optional_output_tensor_names.items():
            if gname not in valid_graph_names:
                continue
            if names is None:
                sanitized[gname] = None
                continue
            if not isinstance(names, list):
                raise TypeError(
                    f"optional_output_tensor_names['{gname}'] must be None or List[str], got {type(names)}"
                )
            if not all(isinstance(n, str) for n in names):
                raise TypeError(f"optional_output_tensor_names['{gname}'] must be a list of strings")
            sanitized[gname] = list(names)  # defensive copy

        return sanitized

    def initialize(
        self,
        backend: Union[str | BackendType] = "CPU",
        device: Optional[Device] = None,
        **extra_args,
    ) -> None:
        """
        Initializes the QAIRT model and loads required backend artifacts needed for executing on device.

        This function is optional, and should be used if you intend to call execute multiple times
        with the same model, backend, and device. In addition to enabling a single initialization,
        this method controls the lifetime of backend library artifacts.

        Args:
            backend (Optional[BackendType], optional): The intended QAIRT Backend for execution.
                Defaults to "CPU" if no backend is specified.
            device (Optional[Device], optional): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution.
                See :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.
        """
        _model_logger.info("Initializing model for execution")

        if hasattr(self, "_inference_handle"):
            self.destroy()

        (
            inference_config,
            inference_identifier,
            net_runner,
            backend_custom_config_dict,
            context_custom_config_dict,
        ) = self._create_execution_context(backend, device, extra_args)
        if inference_identifier.target.type not in supported_initialize_execute_platforms:
            _model_logger.error(
                f"Model initialization is not supported for target type: {inference_identifier.target.type}."
                f"Detected OS: {platform.system()}, Architecture: {platform.machine()}"
            )

            raise RuntimeError("Unsupported platform. Exiting.")

        net_runner_load_arg_config = NetRunnerLoadArgConfig(
            identifier=inference_identifier,
            inference_config=inference_config,
            backend_config_dict=backend_custom_config_dict,
            context_config=qti_module_api.QNNContextConfig(**context_custom_config_dict),
        )

        try:
            load_config = net_runner.load(net_runner_load_arg_config)
        except Exception as e:
            _model_logger.error(f"Failed to initialize the model for execution.")
            raise e

        # set inference handle
        setattr(
            self,
            "_inference_handle",
            (
                inference_config,
                load_config.handle,
                net_runner,
                backend_custom_config_dict,
                context_custom_config_dict,
            ),
        )

    def _execute(
        self,
        inputs: ExecutionInputData,
        *,
        backend: str | BackendType = "CPU",
        device: Optional[Device] = None,
        **extra_args,
    ) -> ExecutionResult:
        """
        Performs inference on a QAIRT backend.

        This method is triggered via the __call__ method, and must be implemented by
        any subclasses. The behavior of this method is not guaranteed if it is called directly.

        Args:
            inputs (ExecutionInputData): Input data to be used for execution.

            backend (Optional[BackendType]): The intended QAIRT Backend for execution. Defaults to
                                             "CPU" if no backend is specified.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution. See submodule
                            :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.

        Returns:
            ExecutionResult: The result contains the inference output data in memory, and
            any additional output generated from profiling.

        Raises:
            ValidationError: if provided extra args are not valid
            ExecutionError: if an error occurs during model execution
        """
        optional_output_tensor_names = extra_args.pop("optional_output_tensor_names", None)

        if not hasattr(self, "_inference_handle"):
            (
                inference_config,
                inference_identifier,
                net_runner_module,
                backend_custom_config_dict,
                context_custom_config_dict,
            ) = self._create_execution_context(backend, device, extra_args)

        else:
            (
                inference_config,
                inference_identifier,
                net_runner_module,
                backend_custom_config_dict,
                context_custom_config_dict,
            ) = getattr(self, "_inference_handle")

        graphs_info = self.graphs_info

        if len(graphs_info) > 1:
            raise RuntimeError(
                "Multi-graph execution is not supported for Model (DLC-only execution)"
                "Please use CompiledModel or generate context binaries for multi-graph runs"
            )

        optional_output_tensor_names = self._normalize_optional_output_tensor_names(
            optional_output_tensor_names
        )

        net_runner_run_arg_config = NetRunnerRunArgConfig(
            identifier=inference_identifier,
            input_data=inputs,
            inference_config=inference_config,
            input_tensor_names=self._input_tensor_names,
            backend_config_dict=backend_custom_config_dict,
            context_config=qti_module_api.QNNContextConfig(**context_custom_config_dict),
            optional_output_tensor_names=optional_output_tensor_names,
        )

        try:
            inference_output_config = net_runner_module.run(net_runner_run_arg_config)
        except Exception as e:
            _model_logger.error(f"Failed to execute the Model: {self.name}.")
            raise e

        if isinstance(inputs, (str, PathLike)):
            # if input is a file, then output data can be a list of dictionaries
            # as per this API. Support for this is added primarily for legacy use cases
            # involving input list files.
            output_data = inference_output_config.output_data
        else:
            # Output data is a dictionary
            output_data = inference_output_config.output_data[0]

        return ExecutionResult(data=output_data, profiling_data=inference_output_config.profiling_data)

    @profile("inference")
    def __call__(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str] = "CPU",  # TODO AISW-129336: Set default to None after refactoring
        device: Optional[Device] = None,
        optional_output_tensor_names: Optional[Union[List[str], Dict[str, Optional[List[str]]]]] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Public method to execute the model. Handles the execution flow internally by calling self._execute.

        Args:
            inputs: Input data to be used for execution. Input data could be of the following types.

             - np.ndarray | Sequence[np.ndarray]: A single numpy array or sequence of numpy arrays.

             - str | PathLike: A path to an input list text file containing paths to raw data.
                               This type is intended for data that will be executed on remote devices,
                               where the data must be pushed to the execution environment.
                               The format of this file is defined further in documentation.
                               See `QAIRT SDK` for details.

             - Dict[str, np.ndarray]: A dictionary of tensor names to
               numpy arrays.

            backend (Optional[str]): The intended QAIRT Backend for execution. Defaults to
                                "CPU" if no backend is specified.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.

            optional_output_tensor_names (Optional[Union[List[str], Dict[str, Optional[List[str]]]]]):
                Controls which optional output tensors to return from the model. This parameter does not affect
                mandatory outputs, which are always included.

                For single-graph models:
                - If None (default): Returns all optional outputs (same as CLI behavior when optionalOutputTensorNames is omitted)
                - If empty list ([]): Returns no optional outputs (only mandatory outputs)
                - If list of strings: Returns only the specified optional outputs plus all mandatory outputs

                For multi-graph models:
                - Should be a dictionary mapping graph names to their respective output selections
                - Each value follows the same rules as single-graph models.

            **kwargs: Keyword arguments for execution.

                     kwargs may contain
                          - Extra keyword arguments to pass for execution.
                            See :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.
                          - Arguments to pre or post execute hooks.

        Examples:
            .. code-block:: python

                model = qairt.load("model.dlc")

                # Execute the model with a single input
                result = model(inputs=np.ndarray(shape=(1, 3, 224, 224)))

                # Execute the model on device
                from qairt import Device, RemoteDeviceIdentifier, DevicePlatformType

                device = Device(RemoteDeviceIdentifier(serial_id="abcd123"), type=DevicePlatformType.ANDROID)
                result = model(inputs=np.ndarray(shape=(1, 3, 224, 224)), backend="HTP", device=device)

                # Execute the model with kwargs
                result = model(inputs=np.ndarray(shape=(1, 3, 224, 224)), synchronous=False, use_mmap=True)

                # Execute the single graph model with output tensor names
                result = model(inputs=inputs, optional_output_tensor_names=["tensor1", "tensor2"])

                # Execute single graph model, get only mandatory outputs
                result = model(inputs=inputs, optional_output_tensor_names=[])

                # Execute Multi-graph model with different selections per graph
                result = model(inputs=inputs, optional_output_tensor_names={
                    "graph1": ["tensorA", "tensorB"],
                    "graph2": None,  # All optional outputs for graph2
                    "graph3": []      # No optional outputs for graph3
                })

        Returns:
            ExecutionResult: The result after applying pre-execute hooks, execution, and post-execute hooks.
            The result contains the inference output data in memory, and any additional output generated from
            profiling.

        Raises:
            ExecutionError: if a runtime error occurs.
        """
        return super().__call__(
            inputs,
            backend=backend,
            device=device,
            optional_output_tensor_names=optional_output_tensor_names,
            **kwargs,
        )

    @classmethod
    def load(
        cls, path: str | os.PathLike, *, name: str = "", load_weight_data: bool = True, **load_args
    ) -> "Model":
        """
        Loads a DLC from the specified path.

        Args:
            path (str): The path to a DLC.
            name (str): The name of the model
            load_weight_data (bool): Whether to load the weights of the DLC on initialization.
            load_args (Optional[Dict[str, Any]]): Additional arguments for loading a DLC.
                                                  See DlcModule.load for details.

        Returns:
            The loaded model object.

        """
        if not check_asset_type(AssetType.DLC, path):
            raise UnknownAssetError(f"File: {path} is not a valid DLC")

        try:
            _model_logger.debug(f"Loading DLC from {path}")
            dlc_module = DlcModule.load(path, enable_lazy_weight_loading=load_weight_data)
        except Exception as e:
            _model_logger.error(f"Error loading DLC from path: {path}")
            raise e

        _model_logger.info(f"Loaded DLC from {path}")
        model = cls(module=dlc_module, name=name)

        source_dir = Path(path).parent

        # Load lora_use_cases.yaml
        config_path = source_dir / "lora_use_cases.yaml"
        if config_path.exists():
            try:
                model.lora_use_cases = load_use_case_config(config_path)
                _model_logger.debug(f"Loaded LoRA use cases from {config_path}")
            except Exception as e:
                _model_logger.error(f"Unable to load LoRA use cases from {config_path}: {e}")
                raise RuntimeError(f"Failed to load LoRA use cases: {e}")

        return model

    @property
    def quantized(self) -> bool:
        """
        Returns True if the model is quantized False otherwise.
        """

        name = self.module.graph_names()[0]
        graph_info = self.module.graphs_info
        graph = graph_info[name]
        is_quantized = graph.outputs[0].is_quantized

        return is_quantized

    def save(self, path: str | os.PathLike = "", **kwargs) -> str:
        """
        Saves a model as a DLC (.dlc).

        Args:
            path (str): The path where the model will be saved.

        Examples:

             .. code-block:: python

                model.save("model.dlc")


        Returns:
            The path where the model, along with any associated assets, were saved.
        """
        saved_path = self.module.save(path)
        base_dir = Path(saved_path).parent

        # Save lora_use_cases.yaml
        if self.lora_use_cases:
            if isinstance(self.lora_use_cases, list):
                temp_yaml_path = base_dir / "lora_use_cases.yaml"
                try:
                    serialize_lora_adapter_weight_config(
                        self.lora_use_cases, str(temp_yaml_path), str(base_dir)
                    )
                    _model_logger.debug(f"Serialized LoRA use cases to {temp_yaml_path}")
                except Exception as e:
                    _model_logger.warning(f"Failed to serialize LoRA use cases: {e}")
            elif isinstance(self.lora_use_cases, (str, os.PathLike)):
                yaml_path = Path(self.lora_use_cases)
                if yaml_path.exists():
                    shutil.copy2(yaml_path, base_dir / "lora_use_cases.yaml")
                    _model_logger.debug(f"Copied LoRA config from {yaml_path}")
                else:
                    _model_logger.warning(f"Provided LoRA config path does not exist: {yaml_path}")

        # Copy .safetensors and .encodings files
        source_dir = getattr(self.module, "working_directory", None)
        if source_dir and Path(source_dir).exists():
            for file in Path(source_dir).iterdir():
                if file.suffix in [".safetensors", ".encodings"]:
                    shutil.copy2(file, base_dir / file.name)

        return str(saved_path)

    def save_with_assets(self, dir_path: str | os.PathLike = ".", **kwargs) -> str:
        """
        Saves all assets associated with this model. This method can be used to save the model and additional
        assets such as profiling logs and binaries generated during execution.

        Args:
            dir_path (DirectoryPath): The path directory where the model, along with any associated assets,
                                      should be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path where the assets were saved.
        """

        if not os.path.isdir(dir_path):
            raise IOError(f"Path {dir_path} does not exist")

        dir_path = Path(dir_path).resolve()
        dlc_path = dir_path / self.module.path

        self.save(dlc_path)

        for _, asset in self.assets.items():
            asset.save(dir_path)

        return str(dir_path)

    def _create_execution_context(
        self,
        backend: Union[str, BackendType],
        device: Optional[Device] = None,
        extra_args: Optional[Dict] = None,
    ) -> Tuple[InferenceConfig, InferenceIdentifier, NetRunner, Dict, Dict]:
        """
        Creates an execution context by validating extra arguments, determining the device, and creating an inference identifier.

        Args:
            backend: The intended QAIRT Backend for execution
            device: The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra key-value to validate and use for creating the inference configuration.


        Returns:
            A tuple containing:
                - inference_config: The inference configuration.
                - inference_identifier: The inference identifier.
                - net_runner: The NetRunner instance.

        """

        if extra_args is None:
            extra_args = {}

        # Validating extra args
        # If backend is present in extra_args, the backend arg in ExecutionConfig needs to be given preference.
        if "backend" in extra_args:
            backend = extra_args.pop("backend", "")
        execution_config = ExecutionConfig(backend=backend, **extra_args)
        extra_args_copy = extra_args.copy()
        for arg in [
            "log_level",
            "context_custom_configs",
            "device_custom_configs",
            "graph_custom_configs",
            "memory_custom_config",
            "context_execute_custom_config",
            "runtime_custom_config",
        ]:
            extra_args_copy.pop(arg, None)
        inference_config = InferenceConfig(log_level=execution_config.log_level, **extra_args_copy)

        if device is None:
            localhost = mlapi_target.create_host_target()
            target_device = Target(type=localhost.target_platform_type.value)
            _model_logger.info("No device provided. Defaulting to localhost.")
        else:
            if device.type == DevicePlatformType.WOS:
                raise NotImplementedError(
                    "Remote execution on WoS (Windows on Snapdragon) is not supported."
                    "To run inference on a WoS device, execute the Python API directly "
                    "on the WoS device."
                )
            soc_model = None
            # Set chipset to enable automatic skel/stub selection for HTP targets
            if chipset := device.get_chipset():
                if isinstance(device.info.identifier, RemoteDeviceIdentifier):
                    # TODO: Change soc_model to chipset
                    soc_model = device.info.identifier.soc_model
                    device.info.identifier.soc_model = chipset

            target_device = Target(
                type=device.info.platform_type.value,
                identifier=device.info.identifier,
                soc_model=soc_model,
            )
            if hasattr(device.info, "credentials"):
                target_device.credentials = device.info.credentials

        if isinstance(self.module, DlcModule) and self.module.path:
            model = qti_module_api.ModelConfig(path=Path(self.module.path))
        elif isinstance(self.module, CacheModule) and self.module.path:
            if inference_config.set_output_tensors:
                inference_config.set_output_tensors = []
                _model_logger.warning(
                    "The 'set_output_tensors' option cannot be used when the graph is retrieved from "
                    "the context binary. It has been changed to an empty list."
                )
            model = qti_module_api.ModelConfig(path=Path(self.module.path))

        else:
            raise AttributeError("Module is missing the required path attribute.")

        inference_identifier = InferenceIdentifier(
            model=model,
            target=target_device,
            backend=backend,
        )

        qnn_api_config_options = get_config_api_options_dict(execution_config)
        backend_custom_config_dict = qnn_api_config_options["backend_extensions"]["config_dict"]
        _model_logger.debug(f"Backend custom options set to: {backend_custom_config_dict}")
        context_custom_config_dict = qnn_api_config_options["context_configs"]
        _model_logger.debug(f"Context custom options set to: {context_custom_config_dict}")

        net_runner = NetRunner(logger=get_logger("qairt.execute"))
        return (
            inference_config,
            inference_identifier,
            net_runner,
            backend_custom_config_dict,
            context_custom_config_dict,
        )

    def __del__(self) -> None:
        """Garbage collection on the model, primarily used to delete any
        persistent inference handles and any unsaved assets"""

        # destroy inferencer handle
        self.destroy()

    def destroy(self) -> None:
        """
        This method is called when the model object is being garbage collected.
        It unloads the model from memory and releases any associated resources.

        :return: None
        """
        if not hasattr(self, "_inference_handle"):
            return
        _, unload_handle, netrunner, bk, ctx = getattr(self, "_inference_handle")
        config = NetRunnerUnloadArgConfig(handle=unload_handle)
        netrunner.unload(config)
        delattr(self, "_inference_handle")

    @property
    def _input_tensor_names(self) -> List[str]:
        """
        Returns a flat list of all input tensor names across all graphs.
        """
        return [tensor_info.name for _, tensor_list in self.input_tensors for tensor_info in tensor_list]

    @property
    def input_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the input tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and input tensors information.
        """
        return [(graph.name, graph.inputs) for graph in self.module.info.graphs]

    @property
    def output_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the output tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and output tensors information.
        """
        return [(graph.name, graph.outputs) for graph in self.module.info.graphs]
