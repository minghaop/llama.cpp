# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from typing_extensions import Unpack

from qairt.api.compiler import CompileConfig
from qairt.api.configs import (
    BackendType,
    Device,
    ExecutionInputData,
    ExecutionResult,
)
from qairt.api.executor.execution_config import ExecutionConfig
from qairt.api.model import Model, qti_module_api
from qairt.modules.cache_module import CacheModule
from qairt.modules.dlc_module import DlcModule
from qairt.modules.dlc_module.dlc_utils import GraphInfo
from qairt.modules.lora.lora_config import (
    UseCaseOutputConfig,
    load_use_case_config,
    serialize_lora_adapter_weight_config,
)
from qairt.modules.multi_graph_execution import MultiGraphExecution, MultiGraphExecutionInputConfig
from qairt.utils.asset_utils import AssetType, check_asset_type, get_asset_type
from qairt.utils.exceptions import UnknownAssetError
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
    NetRunnerRunArgConfig,
)

__ALL__ = ["CompiledModel"]

_cmodel_logger = get_logger("qairt.execute")


class CompiledModel(Model):
    """
    Representative entity that has been prepared for execution on a QAIRT Backend,
    and is executable only that backend.This means that graph initialization and composition
    has been performed, and the model contains a reference to a serialized cache or a container
    composed of serialized caches.

    It has the following additional properties:
        - It can be saved to an asset that remains executable only on the QAIRT Backend for which it has been
          compiled.
        - It can be loaded from a DLC or Binary asset. If load is successful, the CompiledModel object
          can be executed similar to the asset.
        - It can be queried to return properties that identify the asset from which it was loaded. No properties
          may be changed after the object has been created.
    """

    def __init__(
        self,
        module: DlcModule | CacheModule,
        backend: Optional[str | BackendType] = None,
        *,
        name: str = "",
        config: Optional["CompileConfig"] = None,
    ):
        """
        Initializes a new CompiledModel object.

        Args:
            module (DlcModule | CacheModule): A representation of this model as a serialized module.
            backend (str | BackendType): The backend this model is associated with. If no backend is specified,
                then it is assumed that a backend will be passed to execute.
                Note that if the module is of type: DlcModule, then the model may be
                executed on a different backend from which it was compiled.
            name (str): The name of the model.
            config (Optional[CompileConfig]): Compilation configuration options used to create the module.
        """

        super().__init__(module=module, name=name)  # type: ignore[arg-type]
        self._config = config
        self._backend = backend
        self.lora_use_case_binary_map: Dict[str, Path] = {}

    @property  # type: ignore[override]
    def module(self) -> DlcModule | CacheModule:
        """
        Returns the module associated with this model.
        """
        return self._module

    @module.setter
    def module(self, module: DlcModule | CacheModule):
        if self._module is not None:
            raise AttributeError("Cannot set module after initialization.")
        self._module = module

    @module.deleter
    def module(self):
        if self._module is None:
            return
        raise RuntimeError("Cannot delete module after initialization.")

    @property
    def backend(self) -> Optional[str | BackendType]:
        """
        Returns the backend associated with this model.
        """
        return self._backend

    @property
    def config(self) -> Optional[CompileConfig]:
        """
        Returns the configuration used to compile this model.
        """
        return self._config

    def _verify_and_set_backend(
        self, backend: Optional[Union[str, BackendType]] = None
    ) -> Union[str, BackendType]:
        """
        Verifies and sets the backend for the compiled model if backend is None.

        Args:
            backend: The backend to use for execution. Can be a string or a BackendType.

        Returns:
            The backend to use for execution.

        Raises:
            AttributeError: If the backend does not match the one used for compilation.
        """
        if backend is None:
            if not self.backend:
                raise AttributeError("Backend must be specified when executing a compiled model.")
            backend = self.backend
        elif self.backend and backend != self.backend:
            raise AttributeError(
                f"Compiled Model cannot be executed on a different backend than it was compiled."
                f" Expected: {self.backend}. Got: {backend}"
            )
        return backend

    def _configure_lora_binary_updates(self, inference_config) -> None:
        """
        Generate and set up the binary_updates.yaml file for LoRA adapters.
        Args:
            inference_config (InferenceConfig): The inference configuration object that will be
                                               updated with the path to the binary_updates.yaml file.

        YAML Structure Generated:
            version: 1
            graphs:
              - <graph_name>:
                  - <adapter_binary_path_1>
                  - <adapter_binary_path_2>
                  - ...
        """
        # Collect adapter binaries, excluding the base model (already loaded)
        adapter_binaries = [
            str(adapter_path)
            for adapter_name, adapter_path in self.lora_use_case_binary_map.items()
            if adapter_name != "base"
        ]
        graph_name = self.graphs_info[0].name

        # Create YAML structure with version and graph-to-adapters mapping
        yaml_data = {"version": 1, "graphs": [{graph_name: adapter_binaries}]}

        if hasattr(self.module, "working_directory") and self.module.working_directory:
            temp_dir = self.module.working_directory
        else:
            temp_dir = self.module.path.parent

        # Create a temporary file with delete=False to avoid name collisions
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="binary_updates_", dir=temp_dir, delete=False
        ) as temp_file:
            yaml.dump(yaml_data, temp_file)
            binary_updates_path = temp_file.name

        # Configure inference_config object to use the binary updates file
        inference_config.binary_updates = binary_updates_path

    def initialize(
        self,
        backend: Optional[str | BackendType] = None,
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
            device (Optional[Device], optional): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution.
                See :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.
        """
        backend = self._verify_and_set_backend(backend)
        super().initialize(backend, device, **extra_args)

    def _execute(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str | BackendType] = None,
        device: Optional[Device] = None,
        **extra_args,
    ) -> ExecutionResult:
        """
         Performs inference on a QAIRT backend.

        This method is triggered via the __call__ method, and must be implemented by
        any subclasses. The behavior of this method is not guaranteed if it is called directly.

        Args:
            inputs: Input data to be used for execution. See `qairt.configs.common.ExecutionInputData` for types.

            backend (Optional[BackendType]): The intended QAIRT Backend for execution. If no backend is specified,
                                             then self._backend is used.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.

            extra_args: Extra keyword arguments to pass for execution. See submodule
                             :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.

        Returns:
            ExecutionResult: The result after applying pre-execute hooks, execution, and post-execute hooks.
            The result contains the inference output data in memory, and any additional output generated from
            profiling.

        Raises:
            ValidationError: if provided extra args are not valid
            ExecutionError: if an error occurs during compiled model execution


        """
        if not self.module.executable:
            raise RuntimeError("Could not execute model")

        optional_output_tensor_names = extra_args.pop("optional_output_tensor_names", None)

        optional_output_tensor_names = self._normalize_optional_output_tensor_names(
            optional_output_tensor_names
        )

        graph_names = extra_args.pop("graph_names", None)

        if not hasattr(self, "_inference_handle"):
            backend = self._verify_and_set_backend(backend)
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
            multi_graph_input_config = MultiGraphExecutionInputConfig(
                graphs_info=graphs_info,
                input_data=inputs,
                graph_name=graph_names,
                identifier=inference_identifier,
                inference_config=inference_config,
                backend_config_dict=backend_custom_config_dict,
                context_config=qti_module_api.QNNContextConfig(**context_custom_config_dict),
                optional_output_tensor_names=optional_output_tensor_names,
            )
            multi_graph_module = MultiGraphExecution(net_runner=net_runner_module)
            try:
                inference_output_config = multi_graph_module.execute(multi_graph_input_config)
            except Exception as e:
                _cmodel_logger.error(
                    f"Failed to execute the Compiled Model for multi-graph execution: {self.name} with error {e}"
                )
        else:
            # For single-graph models, extract the graph name to fix native_input_tensor_names bug
            single_graph_name = graphs_info[0].name if graphs_info else None

            # Configure LoRA adapter binary updates if use cases are defined
            if self.lora_use_cases:
                self._configure_lora_binary_updates(inference_config)

            net_runner_run_arg_config = NetRunnerRunArgConfig(
                identifier=inference_identifier,
                input_data=inputs,
                inference_config=inference_config,
                input_tensor_names=self._input_tensor_names,
                backend_config_dict=backend_custom_config_dict,
                context_config=qti_module_api.QNNContextConfig(**context_custom_config_dict),
                optional_output_tensor_names=optional_output_tensor_names,
                graph_name=single_graph_name,  # Pass graph name for single-graph models
            )
            try:
                inference_output_config = net_runner_module.run(net_runner_run_arg_config)
            except Exception as e:
                _cmodel_logger.error(f"Failed to execute the Compiled Model: {self.name}.")
                raise e

        profiling_data = None
        if inference_output_config.profiling_data:
            profiling_data = inference_output_config.profiling_data
            for _, asset in self.assets.items():
                if asset.type == AssetType.SCHEMATIC_BIN:
                    if profiling_data.backend_profiling_artifacts is None:
                        profiling_data.backend_profiling_artifacts = []
                    profiling_data.backend_profiling_artifacts.append(asset.path)

        if isinstance(inputs, (str, PathLike)):
            # For multi graph execution, output data would be a nested list of dictionaries.
            if isinstance(inference_output_config.output_data, dict) and not self.lora_use_cases:
                output_data = {
                    name: output[0] for name, output in inference_output_config.output_data.items()
                }
            else:
                # if input is a file, then output data can be a list of dictionaries
                # as per this API. Support for this is added primarily for legacy use cases
                # involving input list files.
                output_data = inference_output_config.output_data
        elif isinstance(inputs, (np.ndarray, dict)) and isinstance(inference_output_config.output_data, list):
            # Output data is a dictionary, for single graph execution output
            output_data = inference_output_config.output_data[0]
        elif isinstance(inputs, list) and isinstance(inputs[0], (np.ndarray, dict)):
            # Output data is a List of dictionaries, for multi graph execution
            output_data = {name: output[0] for name, output in inference_output_config.output_data.items()}

        return ExecutionResult(data=output_data, profiling_data=profiling_data)

    def __call__(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str] = None,
        device: Optional[Device] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Public method to execute the model. Handles the execution flow internally by calling self._execute.

        Args:
            inputs: Input data to be used for execution.

            backend (Optional[str]): The intended QAIRT Backend for execution. Defaults to self.backend
                                     if no backend is specified.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.

            **kwargs: Keyword arguments for execution.

                     kwargs may contain:
                          - Extra keyword arguments to pass for execution. See submodule
                            :class:`qairt.api.executor.execution_config.ExecutionConfig` for details.
                          - Arguments to pre or post execute hooks.

        Examples:
            .. code-block:: python

                compiled_model = qairt.load("model.bin")

                # Execute the model with a single input
                result = compiled_model(inputs=np.ndarray(shape=(1, 3, 224, 224)))

                # Execute the model on device
                from qairt import Device, RemoteDeviceIdentifier, DevicePlatformType

                device = Device(RemoteDeviceIdentifier(serial_id="abcd123"), type=DevicePlatformType.ANDROID)
                result = compiled_model(inputs=np.ndarray(shape=(1, 3, 224, 224)), device=device)

                # Execute the model with kwargs
                result = compiled_model(inputs=np.ndarray(shape=(1, 3, 224, 224)), synchronous=False, use_mmap=True)

        Returns:
            ExecutionResult: The result after applying pre-execute hooks, execution, and post-execute hooks.
            The result contains the inference output data in memory, and any additional output generated from
            profiling.
        Raises:
            AttributeError: if no backend can be identified
            ExecutionError: if an error occurs during model execution
        """
        backend = self._verify_and_set_backend(backend)

        return super().__call__(
            inputs,
            backend=backend,
            device=device,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike,
        *,
        name: str = "",
        compile_config: Optional[CompileConfig] = None,
        backend: Optional[BackendType] = None,
        lora_config_or_path: Optional[Union[str, PathLike, List[UseCaseOutputConfig]]] = None,
        **load_args,
    ) -> "CompiledModel":
        """
        Loads a model from a specified context binary (.bin) or DLC (.dlc).

        Args:
            path (str): The path to a binary or DLC.
            name (str): The name of the model
            compile_config (Optional[CompileConfig]): The specifications used to compile this model.
            backend (Optional[BackendType]): The backend this model is associated with.
            lora_config_or_path (Optional[Union[str, PathLike, List[UseCaseOutputConfig]]]):
                Path to lora_use_cases.yaml file or a list of UseCaseOutputConfig objects.
            load_args (Optional[Dict[str, Any]]): Additional arguments for loading a DLC. See DLCModule.load
            for details.

        Returns:
            CompiledModel: The loaded model object.
        """
        if not (check_asset_type(AssetType.DLC, path) or check_asset_type(AssetType.CTX_BIN, path)):
            raise UnknownAssetError(f"{path}: is not a valid compiled asset")

        asset_type = get_asset_type(path)

        if asset_type == AssetType.DLC:
            dlc_module = DlcModule.load(path, **load_args)

            if not dlc_module.caches:
                raise RuntimeError(f"DLC: {path} is not compiled.")

            model = cls(name=name, module=dlc_module, backend=backend, config=compile_config)

        else:
            cache_module = CacheModule.load(path=path)
            source_dir = Path(path).parent
            lora_use_case_binary_map: Dict[str, Path] = {}

            # Base binary
            lora_use_case_binary_map["base"] = Path(path)

            # Identify backend on load if none is provided
            if not backend and cache_module.info.backend:
                backend = cache_module.info.backend

            model = cls(name=name, module=cache_module, backend=backend, config=compile_config)

            if lora_config_or_path:
                if isinstance(lora_config_or_path, list):
                    # User provided in-memory list of UseCaseOutputConfig objects
                    model.lora_use_cases = lora_config_or_path
                else:
                    # User provided a path to lora_use_cases.yaml
                    config_path = Path(lora_config_or_path)
                    if config_path.exists():
                        model.lora_use_cases = load_use_case_config(config_path)
                    else:
                        raise FileNotFoundError(f"Provided lora config path does not exist: {config_path}")
            else:
                # Auto-load lora_use_cases.yaml from the same directory if it exists
                config_path = source_dir / "lora_use_cases.yaml"
                if config_path.exists():
                    model.lora_use_cases = load_use_case_config(config_path)

            bin_files = {file.stem: file for file in source_dir.glob("*.bin")}
            # Add default_adapter if present
            default_adapter_file = next(
                (file for file in bin_files.values() if "default_adapter" in file.name), None
            )
            if default_adapter_file:
                lora_use_case_binary_map["default_adapter"] = default_adapter_file

            if isinstance(model.lora_use_cases, list):
                matched_files = set()
                for use_case in model.lora_use_cases:
                    if not isinstance(use_case, UseCaseOutputConfig):
                        _cmodel_logger.warning(f"Unexpected use case type: {type(use_case)}")
                        continue
                    uc_name = use_case.name
                    for stem, file in bin_files.items():
                        if file in matched_files:
                            continue
                        # Match if stem ends with the exact use_case name, preceded by a delimiter
                        if stem.endswith(uc_name) and (
                            stem == uc_name or stem[-len(uc_name) - 1] in ["_", "-"]
                        ):
                            lora_use_case_binary_map[uc_name] = file
                            matched_files.add(file)
                            break
                    else:
                        _cmodel_logger.warning(f"No matching binary found for use case: {uc_name}")
            elif model.lora_use_cases is not None:
                _cmodel_logger.warning(f"model.lora_use_cases is not iterable: {type(model.lora_use_cases)}")

            model.lora_use_case_binary_map = lora_use_case_binary_map

        return model

    @property
    def quantized(self) -> bool:
        if isinstance(self.module, DlcModule):
            return super().quantized
        else:
            # TODO: Find a better way to do this. This approach uses
            # QNN_BACKEND_ID definitions for HTP, HTP_MCP and HTP_QEMU
            # This is not ideal as it doesn't account for float dtype on
            # on these runtimes, or quant dtypes on CPU.
            module: CacheModule = self.module
            return module.info.backend.id in [6, 11, 13]

    def save(self, path: str | os.PathLike = "", asset_type: Optional[AssetType] = None, **kwargs) -> str:
        """
        Saves a model

        Args:
            path (str): The path where the model should be saved.
            asset_type (Optional[AssetType]): The type of asset to write to disk. The option can be used
                                              to save a DLC loaded as a compiled model as a directory
                                              of context binaries by setting the asset_type to AssetType.CTX_BIN.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for saving the model.

        Examples:

             .. code-block:: python

                compiled_model.save("model.bin")

        Returns:
            str: The path where the model was saved.
        """
        if asset_type is None:
            if isinstance(self.module, DlcModule):
                asset_type = AssetType.DLC
            elif str(path).endswith("_backend") or Path(path).stem.endswith("_backend"):
                asset_type = AssetType.BACKEND_BIN
            else:
                asset_type = AssetType.CTX_BIN

        if asset_type == AssetType.CTX_BIN:
            if isinstance(self.module, DlcModule):
                if self.module.caches:
                    if not os.path.isdir(path):
                        path = str(Path(path).parent)
                        os.makedirs(path, exist_ok=True)
                    cache_modules = self.module.extract_caches(str(path))
                    for cache_module in cache_modules:
                        file_path = os.path.join(
                            path, cache_module.name + AssetType.get_extension(AssetType.CTX_BIN)
                        )
                        cache_module.save(file_path)
                else:
                    raise ValueError("Cannot extract binaries from DLC")
            else:
                path_obj = Path(path)
                if path_obj.suffix == AssetType.get_extension(AssetType.CTX_BIN):
                    dir_path = str(path_obj.parent)
                    filename = path_obj.stem
                elif os.path.isdir(path):
                    dir_path = str(path)
                    filename = self.module.name
                else:
                    # User provided a path without extension or non-existent path
                    # Treat the last component as the desired filename
                    dir_path = str(path_obj.parent) if path_obj.parent != path_obj else "."
                    filename = path_obj.stem if path_obj.stem else self.module.name

                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, filename + AssetType.get_extension(AssetType.CTX_BIN))

                # Store original module name before save (as save might change it)
                original_module_name = self.module.name
                self.module.save(file_path)

                # Copy all lora use case binaries (exclude backend binaries, schematic binaries, and the main model)
                source_dir = self.module.working_directory
                if source_dir is not None:
                    for file in source_dir.iterdir():
                        if (
                            file.is_file()
                            and file.suffix == ".bin"
                            and file.stem != original_module_name
                            and not file.stem.endswith(("_backend", "_schematic"))
                        ):
                            shutil.copy2(file, Path(dir_path) / file.name)

                    if self.lora_use_cases:
                        if isinstance(self.lora_use_cases, list):
                            temp_yaml_path = source_dir / "lora_use_cases.yaml"
                            serialize_lora_adapter_weight_config(
                                self.lora_use_cases, str(temp_yaml_path), str(source_dir)
                            )
                            if temp_yaml_path.exists():
                                shutil.copy2(temp_yaml_path, Path(dir_path) / "lora_use_cases.yaml")
                            else:
                                _cmodel_logger.warning("Failed to serialize lora_use_cases.yaml")
                        elif isinstance(self.lora_use_cases, (str, os.PathLike)):
                            yaml_path = Path(self.lora_use_cases)

                            if yaml_path.exists():
                                shutil.copy2(yaml_path, Path(dir_path) / "lora_use_cases.yaml")
                            else:
                                _cmodel_logger.warning(
                                    f"Provided lora_use_cases path does not exist: {yaml_path}"
                                )
        elif asset_type == AssetType.BACKEND_BIN:
            backend_binary = None
            for asset in self.assets.values():
                if asset.type == asset_type:
                    backend_binary = asset.path
                    break
            if not backend_binary:
                _cmodel_logger.warning("Trying to save a non-existent backend binary")
            else:
                shutil.copyfile(backend_binary, path)
        else:
            path = super().save(path)
        return str(path)

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

        Raises:
            IOError: If the provided path does not exist or is not a directory.
        """
        if isinstance(self.module, DlcModule):
            return super().save_with_assets(dir_path, **kwargs)
        else:
            if not os.path.isdir(dir_path):
                raise IOError(f"Path {dir_path} does not exist")

            self.save(dir_path, **kwargs)

            for _, asset in self.assets.items():
                asset.save(dir_path)

        return str(dir_path)
