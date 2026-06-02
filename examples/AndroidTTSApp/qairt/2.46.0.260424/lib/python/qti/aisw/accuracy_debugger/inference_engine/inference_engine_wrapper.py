# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import math
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import psutil
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngine,
    InferenceEngineInputConfig,
    InferenceEngineOutputConfig,
)
from qti.aisw.accuracy_debugger.utils.constants import GRAPH_FINALIZE_FAILURE_MSG
from qti.aisw.accuracy_debugger.utils.exceptions import GenerateBinaryFailure
from qti.aisw.dlc_utils import modeltools
from qti.aisw.tools.core.modules.api import BackendType
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


def get_adb_device_available_memory(device_id: Optional[str] = None, logger: Any = None) -> int:
    """Gets the available memory (MemAvailable) from an ADB-connected Android device.

    Runs 'adb devices' to discover connected devices, then reads /proc/meminfo on the
    target device to retrieve the MemAvailable value in MB.

    Args:
        device_id (str, optional): Serial ID of the target ADB device. If None, the first
            available device from 'adb devices' is used. If multiple devices are connected
            and no device_id is specified, the first device in the list is used.
        logger (optional): Logger instance for debug/warning messages.

    Returns:
        int: MemAvailable value in MB from the target device's /proc/meminfo.

    Raises:
        RuntimeError: If 'adb' is not found in PATH, if no ADB devices are connected,
            if the specified device_id is not found among connected devices, or if
            MemAvailable cannot be parsed from /proc/meminfo.

    Example:
        Sample output of ``adb -s <device_id> shell cat /proc/meminfo``::

            MemTotal:       15523268 kB
            MemFree:        10910708 kB
            MemAvailable:   12291184 kB

        In this example, ``MemAvailable: 12291184 kB`` → the function returns ``12003`` MB
        (12291184 kB / 1024 = 12003 MB).
    """

    def _log(level: str, msg: str) -> None:
        if logger:
            getattr(logger, level)(msg)

    # Step 1: Get list of connected ADB devices
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("'adb' command not found. Ensure ADB is installed and available in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("'adb devices' command timed out after 10 seconds.")

    # Parse device list — skip the header line "List of devices attached"
    devices = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if line and "\t" in line:
            serial, status = line.split("\t", 1)
            if status.strip() == "device":
                devices.append(serial.strip())

    if not devices:
        raise RuntimeError(
            "No ADB devices found. Ensure a device is connected and authorized via ADB."
        )

    # Step 2: Determine the target device
    if device_id:
        if device_id not in devices:
            raise RuntimeError(
                f"Specified device_id '{device_id}' was not found among connected ADB devices: "
                f"{devices}. Ensure the device is connected and authorized."
            )
        target_device = device_id
        _log("debug", f"Using specified ADB device: {target_device}")
    else:
        target_device = devices[0]
        if len(devices) > 1:
            _log(
                "debug",
                f"Multiple ADB devices found: {devices}. Using first device: {target_device}",
            )
        else:
            _log("debug", f"Using ADB device: {target_device}")

    # Step 3: Read /proc/meminfo from the target device
    try:
        result = subprocess.run(
            ["adb", "-s", target_device, "shell", "cat", "/proc/meminfo"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        meminfo = result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timed out reading /proc/meminfo from ADB device '{target_device}'."
        )

    # Step 4: Parse MemAvailable from /proc/meminfo
    mem_available_kb = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                mem_available_kb = int(parts[1])
            break

    if mem_available_kb is None:
        raise RuntimeError(
            f"Could not parse 'MemAvailable' from /proc/meminfo on ADB device '{target_device}'."
        )

    mem_available_mb = mem_available_kb // 1024
    _log("debug", f"ADB device '{target_device}' MemAvailable: {mem_available_mb} MB")

    return mem_available_mb


class InferenceEngineWrapper:
    """User interface class for model inference in Wrapper mode.
    Contains methods to convert, quantize, generate_binary and execute the model,
    based on the backend and platform provided in the InferenceEngineInputConfig.
    If not able to dump all intermediate outputs at once, context-binary and net-run will be
    executed iteratively to dump outputs in chunks.
    """

    def __init__(self, logger: Any = None) -> None:
        """Initialize InferenceEngineWrapper
        Args:
            logger (Any): Desired python logger
        """
        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("Inference")
            self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")

    def run(self, config: InferenceEngineInputConfig) -> InferenceEngineOutputConfig:
        """Execute Inference Engine in Wrapper mode with LoRA phase-based execution
        Args:
            config: InferenceEngineInputConfig object containing args for inference

        Returns:
            InferenceEngineOutputConfig: Compilation artifacts and inference results
        """
        try:
            self.inference_engine = InferenceEngine(logger=self.logger)

            # Phase 1: DLC Generation (LoRA Model Creator → Converter → Quantizer)
            if config.input_model.suffix == ".bin":
                # User only wants to validate config; do NOT run inference
                if not config.net_run_input_data:
                    self.logger.info(
                        "No net_run_input_data provided for .bin model.Skipping inference"
                    )
                    return InferenceEngineOutputConfig()
                # Build inference config for .bin
                inference_config = self._get_inference_config(config, config.input_model)
                inference_output = self._execute_inference_wrapper(inference_config)
                return inference_output

            if config._source_model:
                # When source model is supplied generate quantized dlc using inference engine.
                # Generate dlc: Includes Conversion, Optimization, Serialization and Quantization
                # Also includes LoRA Model Creator if lora_model_creator_args provided
                generate_dlc_config = self._get_dlc_generation_config(config)
                generated_dlc = self.inference_engine.run(generate_dlc_config)
                dlc_path = (
                    generated_dlc.quantizer_dlc
                    if generated_dlc.quantizer_dlc
                    else generated_dlc.converter_dlc
                )
            else:
                # Input model is a dlc file
                dlc_path = config.input_model
                generated_dlc = InferenceEngineOutputConfig()

            # Phase 1.5: LoRA Importer (separate call with just importer args)
            # Run importer if:
            # 1. User explicitly provided lora_importer_args, OR
            # 2. LoRA Model Creator was run and produced output (auto-populate importer args)
            if config.lora_importer_args or (
                hasattr(generated_dlc, "lora_model_creator_output")
                and generated_dlc.lora_model_creator_output
                and hasattr(generated_dlc.lora_model_creator_output, "importer_config")
            ):
                self.logger.info("=== Phase 1.5: LoRA Importer ===")

                # Create importer config with just importer arguments
                importer_config = self._create_lora_importer_config(config, generated_dlc, dlc_path)
                lora_importer_result = self.inference_engine.run(importer_config)

                # Store LoRA importer output in generated_dlc
                generated_dlc.lora_importer_output = lora_importer_result.lora_importer_output

            # Early return if no inference needed (after LoRA Importer completes)
            if not config.net_run_input_data:
                return generated_dlc

            # Get the size of the model's dlc file in Mega bytes
            dlc_size = os.path.getsize(dlc_path) / 1024**2

            # Estimate memory required to dump all intermediate tensors outputs
            self.intermediate_tensors_size = self.estimate_intermediate_outputs_size(
                dlc_path, config.backend
            )

            # If user has passed set_output_tensors argument then filter intermediate outputs to
            # include only those specific outputs to dump
            set_output_tensors = None
            if config.context_bin_gen_arguments:
                set_output_tensors = config.context_bin_gen_arguments.set_output_tensors
            elif config.net_run_arguments:
                set_output_tensors = config.net_run_arguments.set_output_tensors

            if set_output_tensors:
                self.intermediate_tensors_size = {
                    output: self.intermediate_tensors_size.get(output, 0)
                    for output in set_output_tensors
                }

            # Calculate total memory required to dump all intermediate outputs
            total_outputs_size = sum(self.intermediate_tensors_size.values())
            self.logger.debug(
                f"Estimated required memory: {int(dlc_size)}MB (dlc size) + {int(total_outputs_size)}MB (outputs size)"
            )

            # Fetch memory limit on the backend to be used
            device_memory_limit = self.get_memory_limit(
                config.backend, config.platform, config.remote_host_details
            )
            self.logger.debug(f"Target device memory limit = {device_memory_limit}MB")

            # Create inference config which includes offline-prepare (if eligible) and net-run inference
            inference_config = self._get_inference_config(config, dlc_path, generated_dlc)

            if device_memory_limit < (dlc_size + total_outputs_size):
                # Create batches of intermediate outputs which can fit into memory and execute inference iteratively
                intermediate_outputs_batches = self.divide_output_tensors(
                    intermediate_tensors_size=self.intermediate_tensors_size,
                    available_memory=device_memory_limit - dlc_size,
                )
                inference_output = None
                output_data = {}
                for idx, batch_outputs in enumerate(intermediate_outputs_batches):
                    self.logger.debug(
                        f"Dumping {idx + 1}/{len(intermediate_outputs_batches)} batch of intermediate outputs..."
                    )
                    inference_output = self._execute_inference_wrapper(
                        inference_config, batch_outputs
                    )
                    # Club output data dictionaries of each iteration
                    output_data = output_data | inference_output.output_data[0]
                inference_output.output_data = [output_data]
            else:
                # Otherwise, execute inference to dump all or specified intermediate outputs at once
                self.logger.debug(
                    "Attempting to dump all or specified intermediate outputs at once..."
                )
                inference_output = self._execute_inference_wrapper(
                    inference_config, set_output_tensors
                )

            self.logger.debug("Inference engine wrapper completed successfully!")

            # Append converter and quantizer dlc's to final inference output (if they exist)
            if hasattr(generated_dlc, 'converter_dlc') and generated_dlc.converter_dlc:
                inference_output.converter_dlc = generated_dlc.converter_dlc
            if hasattr(generated_dlc, 'quantizer_dlc') and generated_dlc.quantizer_dlc:
                inference_output.quantizer_dlc = generated_dlc.quantizer_dlc

            # Append LoRA outputs to final inference output for cleanup (if they exist)
            if hasattr(generated_dlc, 'lora_model_creator_output') and generated_dlc.lora_model_creator_output:
                inference_output.lora_model_creator_output = generated_dlc.lora_model_creator_output
            if hasattr(generated_dlc, 'lora_importer_output') and generated_dlc.lora_importer_output:
                inference_output.lora_importer_output = generated_dlc.lora_importer_output

            return inference_output
        except Exception as e:
            self.logger.error(f"Inference engine wrapper failed: {e}")
            raise Exception("Inference engine failed.")

    def _execute_inference_wrapper(
        self, inference_config: InferenceEngineInputConfig, outputs: List = None
    ) -> InferenceEngineOutputConfig:
        """Executes inference and gathers intermediate outputs as per given inference config.
        If inference run fails due to memory limitations, then this function will be recursively
        called twice to dump outputs in two half batches

        Args:
            inference_config (InferenceEngineInputConfig): InferenceEngineInputConfig object containing args for inference
            outputs (List, optional): Dumps only specified intermediate outputs. Defaults to None

        Returns:
            InferenceEngineOutputConfig: Compilation artifacts and inference results
        """
        inference_config = self._update_inference_config(
            config=inference_config, set_output_tensors=outputs
        )

        try:
            inference_output = self.inference_engine.run(inference_config)
            return inference_output
        except GenerateBinaryFailure as context_binary_error:
            if str(context_binary_error) == GRAPH_FINALIZE_FAILURE_MSG:
                self.logger.debug(
                    "Inference run failed due to memory constraints, so further splitting outputs list into two batches"
                )

                # Handle enable_intermediate_outputs/debug scenario where outputs list won't be passed
                if outputs is None:
                    outputs = [key for key in self.intermediate_tensors_size]

                if len(outputs) <= 1:
                    self.logger.warning(
                        f"Couldn't dump following intermediate output: {outputs}. Reason: {context_binary_error}"
                    )
                    return self._get_empty_inference_output()

                # Further split outputs into two chunks
                first_half = self._execute_inference_wrapper(
                    inference_config, outputs[: len(outputs) // 2]
                )
                second_half = self._execute_inference_wrapper(
                    inference_config, outputs[len(outputs) // 2 :]
                )

                # Club inference outputs data dictionaries
                clubbed_outputs = first_half.output_data[0] | second_half.output_data[0]
                inference_output = first_half
                inference_output.output_data = [clubbed_outputs]
                return inference_output
            else:
                return self._handle_other_errors(outputs, context_binary_error)
        except Exception as err:
            return self._handle_other_errors(outputs, err)

    def _get_dlc_generation_config(
        self, config: InferenceEngineInputConfig
    ) -> InferenceEngineInputConfig:
        """Creates config for generating dlc which includes following: Conversion, Optimization,
        Serialization and Quantization. Also includes LoRA Model Creator if provided.

        Args:
            config (InferenceEngineInputConfig): InferenceEngineInputConfig object containing args for inference

        Returns:
            InferenceEngineInputConfig: InferenceEngineInputConfig object containing args for inference
        """
        return InferenceEngineInputConfig(
            input_model=config.input_model,
            converter_arguments=config.converter_arguments,
            quantizer_arguments=config.quantizer_arguments,
            backend=config.backend,
            working_directory=config.working_directory,
            soc_model=config.soc_model,
            # Only pass LoRA Model Creator args for Phase 1 (DLC Generation)
            lora_model_creator_args=config.lora_model_creator_args,
            # Pass LoRA alpha tensor to modify quantization input list
            lora_alpha_tensor=config.lora_alpha_tensor,
        )

    def _create_lora_importer_config(
        self,
        config: InferenceEngineInputConfig,
        generated_dlc: InferenceEngineOutputConfig,
        dlc_path: Path,
    ) -> InferenceEngineInputConfig:
        """Creates config for LoRA importer with just importer arguments

        Args:
            config: Original InferenceEngineInputConfig
            generated_dlc: Output from DLC generation phase
            dlc_path: Path to the DLC file

        Returns:
            InferenceEngineInputConfig: Config with only LoRA importer arguments
        """
        # Auto-populate LoRA importer args from Phase 1 output if model creator was run
        if generated_dlc.lora_model_creator_output and config.lora_importer_args is None:
            # Create importer args from model creator output (fully auto-populated)
            from qti.aisw.accuracy_debugger.lora import LoRAImporterInputConfig

            # Create output directory if it doesn't exist
            lora_importer_output_dir = config.working_directory / "lora_importer_output"
            lora_importer_output_dir.mkdir(parents=True, exist_ok=True)

            lora_importer_args = LoRAImporterInputConfig(
                lora_config=generated_dlc.lora_model_creator_output.importer_config,
                input_dlc=dlc_path,
                input_network=generated_dlc.lora_model_creator_output.base_model_path,
                input_list=config.quantizer_arguments.input_list
                if config.quantizer_arguments
                else None,
                output_dir=lora_importer_output_dir,
            )
        elif generated_dlc.lora_model_creator_output and config.lora_importer_args is not None:
            # Model creator was run AND user provided importer args
            # Auto-populate required fields from model creator output, but preserve user's optional fields
            from qti.aisw.accuracy_debugger.lora import LoRAImporterInputConfig

            # Start with user's provided args
            user_args = config.lora_importer_args.model_dump(exclude_unset=True)

            # Create output directory if it doesn't exist
            lora_importer_output_dir = config.working_directory / "lora_importer_output"
            lora_importer_output_dir.mkdir(parents=True, exist_ok=True)

            # Auto-populate required fields from model creator output
            auto_populated_args = {
                "lora_config": generated_dlc.lora_model_creator_output.importer_config,
                "input_dlc": dlc_path,
                "input_network": generated_dlc.lora_model_creator_output.base_model_path,
                "input_list": config.quantizer_arguments.input_list
                if config.quantizer_arguments
                else None,
                "output_dir": lora_importer_output_dir,
            }

            # Merge: auto-populated required fields + user's optional fields
            merged_args = {**auto_populated_args, **user_args}
            lora_importer_args = LoRAImporterInputConfig(**merged_args)
        else:
            # Use provided importer args as-is (standalone scenario)
            lora_importer_args = config.lora_importer_args

        # Create config with just importer arguments
        return InferenceEngineInputConfig(
            input_model=dlc_path,  # Not used by importer but required for validation
            working_directory=config.working_directory,
            # Only LoRA importer arguments
            lora_importer_args=lora_importer_args,
            # Pass LoRA alpha tensor to modify importer input list
            lora_alpha_tensor=config.lora_alpha_tensor,
        )

    def _get_inference_config(
        self,
        config: InferenceEngineInputConfig,
        input_model: Path,
        generated_dlc_output: InferenceEngineOutputConfig = None,
    ) -> InferenceEngineInputConfig:
        """Creates inference config which includes offline-prepare (optional) and net-run inference

        Args:
            config (InferenceEngineInputConfig): InferenceEngineInputConfig object containing args for inference
            input_model (Path): The path to the DLC/BIN file
            generated_dlc_output (InferenceEngineOutputConfig, optional): Output from DLC generation phase

        Returns:
            InferenceEngineInputConfig: InferenceEngineInputConfig object containing args for inference
        """
        # Create a copy of context_bin_gen_arguments to avoid modifying the original
        context_bin_args = config.context_bin_gen_arguments

        # If we have LoRA importer output with lora_output_files, add it to context_bin_args
        if (
            generated_dlc_output
            and generated_dlc_output.lora_importer_output
            and generated_dlc_output.lora_importer_output.lora_output_files
        ):
            # Create context_bin_args if it doesn't exist
            if context_bin_args is None:
                from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig

                context_bin_args = GenerateConfig()

            context_bin_args.adapter_weight_config_file = (
                generated_dlc_output.lora_importer_output.lora_output_files
            )
            self.logger.debug(
                f"Setting adapter_weight_config_file: {generated_dlc_output.lora_importer_output.lora_output_files}"
            )

        # Phase 2: Execution - Create config for Context Binary + Net Runner
        # Note: At this stage, LoRA processing is complete, so we pass the DLC and necessary
        # LoRA artifacts for context binary generation and binary_updates creation
        return InferenceEngineInputConfig(
            input_model=input_model,
            backend=config.backend,
            platform=config.platform,
            context_bin_gen_arguments=context_bin_args,
            context_bin_backend_extension=config.context_bin_backend_extension,
            offline_prepare=config.offline_prepare,
            net_run_arguments=config.net_run_arguments,
            net_run_input_data=config.net_run_input_data,
            net_run_backend_extension=config.net_run_backend_extension,
            dump_output=config.dump_output,
            remote_host_details=config.remote_host_details,
            working_directory=config.working_directory,
            # Pass use_case_names for binary_updates creation in qairt_inference_engine.py
            use_case_names=config.use_case_names,
            # Pass LoRA alpha tensor to modify net run input list
            lora_alpha_tensor=config.lora_alpha_tensor,
        )

    def _update_inference_config(
        self, config: InferenceEngineInputConfig, set_output_tensors: List = None
    ) -> InferenceEngineInputConfig:
        """Updates inference config with appropriate values for enable_intermediate_outputs, debug
        and set_output_tensors flags based on offline/online prepare

        Args:
            config (InferenceEngineInputConfig): InferenceEngineInputConfig object containing args for inference
            set_output_tensors (List, optional): Dumps only specified intermediate outputs. Defaults to None

        Returns:
            InferenceEngineInputConfig: InferenceEngineInputConfig object containing args for inference
        """
        if set_output_tensors:
            if config.offline_prepare:
                config.context_bin_gen_arguments.set_output_tensors = set_output_tensors
                config.context_bin_gen_arguments.enable_intermediate_outputs = None
            else:
                config.net_run_arguments.set_output_tensors = set_output_tensors
                config.net_run_arguments.debug = None
        else:
            if config.offline_prepare:
                config.context_bin_gen_arguments.enable_intermediate_outputs = True
                config.context_bin_gen_arguments.set_output_tensors = None
            else:
                if config.input_model.suffix == ".bin":
                    self.logger.warning(
                        ".bin models do not support intermediate output dumping. "
                        "Debug mode has been disabled."
                    )
                    config.net_run_arguments.debug = False
                else:
                    config.net_run_arguments.debug = True
                config.net_run_arguments.set_output_tensors = None
        return config

    def _get_empty_inference_output(self):
        """Creates a dummy empty inference output config and returns it

        Returns:
            InferenceEngineOutputConfig: Empty Inference output config
        """
        inference_output = InferenceEngineOutputConfig()
        inference_output.output_data = [{}]
        return inference_output

    def _handle_other_errors(self, outputs, error):
        """Handling function to handle errors other than memory errors

        Args:
            outputs (List): Specific outputs dumping list
            error (Exception): Raised Exception

        Returns:
            InferenceEngineOutputConfig: Empty Inference output config
        """
        # Raise error when inference fails while dumping all intermediate outputs at once
        # Note: All intermediate outputs will be dumped when outputs is None
        if outputs is None:
            raise error
        # Raise warning when inference fails while dumping specific outputs and continue
        else:
            self.logger.warning(
                f"Couldn't dump following intermediate outputs: {outputs}. Reason: {error}"
            )
            return self._get_empty_inference_output()

    def estimate_intermediate_outputs_size(self, dlc_path: Path, backend: BackendType) -> dict:
        """Estimates memory requirment to execute the model and dump intermediate outputs

        Args:
            dlc_path (Path): The path to the DLC file
            backend (BackendType): Backed type

        Returns:
            dict: A dictionary where keys are tensor names and values are their corresponding sizes in megabytes
        """
        # Get the IR graph from the DLC file
        model_reader = modeltools.IrDlcReader()
        model_reader.open(str(dlc_path))
        ir_graph = model_reader.get_ir_graph()

        # Iterate over all tensors present in the IR graph and calculate size of each tensor
        intermediate_tensors_size = {}
        for name, tensor in ir_graph.get_tensor_map().items():
            if tensor.tensor_type() in ["NATIVE", "APP_WRITE", "APP_READ"]:
                # Get Bitwidth for the tensor's data type
                # Example format for tensor.data_type().name is QNN_DATATYPE_FLOAT_32
                bitwidth = tensor.data_type().name.rsplit("_", 1)[1]

                # If bitwidth is missing/invalid then make an assumption that tensor is 32 bit
                # Example for missing bitwidth is QNN_DATATYPE_UNDEFINED
                bitwidth = int(bitwidth) if bitwidth.isnumeric() else 32

                # Estimate memory size of the tensor using bitwidth and dimension information
                tensor_size_in_bits = math.prod(tensor.dims()) * bitwidth

                # Sanitize tensor names except for AIC backend (as AIC doesn't use sanitization)
                tensor_name = tensor.name()
                if backend != BackendType.AIC:
                    tensor_name = Helper.transform_node_names(tensor_name)

                # Convert memory size into megabytes and store
                intermediate_tensors_size[tensor_name] = tensor_size_in_bits / (8 * 1024**2)

        return intermediate_tensors_size

    def get_memory_limit(
        self,
        backend: BackendType,
        platform: DevicePlatformType,
        remote_host_details: Optional[Any] = None,
    ) -> int:
        """Returns memory limit for the given backend.

        For HTP/HTP_MCP backends, dynamically queries the MemAvailable value from the
        attached ADB device. If a device_id is provided via remote_host_details then,
        that specific device is queried; otherwise the first
        available ADB device is used. Falls back to 3584 MB if ADB query fails.

        Args:
            backend (BackendType): Backend type
            platform (DevicePlatformType): Target platform
            remote_host_details (optional): RemoteHostDetails object containing device
                identifier. If provided and has a serial_id, that device is queried.

        Returns:
            int: Memory limit for the given backend in MB
        """
        if backend in [BackendType.HTP, BackendType.HTP_MCP]:
            # Extract device_id from remote_host_details if provided
            device_id = None
            if remote_host_details and remote_host_details.identifier:
                device_id = remote_host_details.identifier.serial_id

            # Dynamically query MemAvailable from the ADB device
            try:
                device_memory_limit = get_adb_device_available_memory(
                    device_id=device_id, logger=self.logger
                )
                self.logger.debug(
                    f"Retrieved MemAvailable from ADB device"
                    + (f" '{device_id}'" if device_id else "")
                    + f": {device_memory_limit} MB"
                )
            except RuntimeError as e:
                self.logger.warning(
                    f"Could not retrieve device memory via ADB: {e}. "
                    f"Falling back to default HTP memory limit of 3584 MB."
                )
                device_memory_limit = 3584
        elif backend == BackendType.AIC:
            # Each NSP will have 3.7GB memory limit but leave 700MB as buffer
            device_memory_limit = 3000
        elif (
            backend in [BackendType.GPU, BackendType.CPU] and platform == DevicePlatformType.ANDROID
        ):
            # Mobile device RAM usually starts from 8GB but leave 1GB as buffer
            device_memory_limit = 7000
        elif backend == BackendType.CPU and platform == DevicePlatformType.X86_64_LINUX:
            # Fetch available RAM on the linux host and convert into megabites
            device_memory_limit = psutil.virtual_memory().available / 1024**2
        else:
            device_memory_limit = 1024  # Default limit set as 1GB

        return device_memory_limit

    def divide_output_tensors(
        self, intermediate_tensors_size: dict[str, float], available_memory: float
    ) -> List[List[str]]:
        """Divides a dictionary of tensors into batches based on memory constraints.

        This method implements a First Fit Decreasing (FFD) bin-packing algorithm that groups
        tensors into batches where each batch's total size fits within the available memory
        constraint. The FFD algorithm sorts tensors by size in descending order before packing,
        which typically produces near-optimal results for bin-packing problems.

        The algorithm works as follows:
        1. Validates that available memory is positive
        2. Filters out tensors that exceed available memory (with warnings)
        3. Sorts remaining tensors by size in descending order
        4. Iteratively places each tensor in the first batch with sufficient space
        5. Creates a new batch if no existing batch can accommodate the tensor

        Args:
            intermediate_tensors_size (dict[str, float]): A dictionary mapping tensor names
                (str) to their memory sizes (float). Sizes should be in consistent units
                (should be in MBs). Empty dictionaries are allowed and will return an
                empty list.
            available_memory (float): The maximum memory capacity allowed for each batch,
                in the same units as tensor sizes. Must be greater than 0.

        Returns:
            List[List[str]]: A list of batches, where each batch is a list of tensor names.
                Each batch's combined tensor sizes do not exceed available_memory.
                Returns an empty list if intermediate_tensors_size is empty.

        Raises:
            ValueError: If available_memory is less than or equal to 0.
            RuntimeError: If no batches could be created because all tensors exceed the
                available memory limit (after filtering).

        Example:
            >>> tensors = {"tensor1": 100, "tensor2": 150, "tensor3": 80, "tensor4": 50}
            >>> batches = self.divide_output_tensors(tensors, available_memory=200)
            >>> # Result: [["tensor2", "tensor4"], ["tensor1", "tensor3"]]
            >>> # Batch 1: 150 + 50 = 200 (fits exactly)
            >>> # Batch 2: 100 + 80 = 180 (fits within 200)

        Note:
            - Tensors larger than available_memory are automatically filtered out with warnings
            - The algorithm does not guarantee optimal packing but provides good approximations
        """
        # ============================================================================
        # Step 1: Validate Input Parameters
        # ============================================================================

        # Ensure available memory is positive to prevent invalid memory constraints
        if available_memory <= 0:
            raise ValueError(f"Available memory must be positive, got {available_memory}.")

        # Handle edge case: empty input dictionary
        if not intermediate_tensors_size:
            self.logger.info("No tensors provided for batching. Returning empty list.")
            return []

        # ============================================================================
        # Step 2: Filter Tensors Based on Memory Constraints
        # ============================================================================

        # Create a filtered dictionary containing only tensors that can fit in available memory
        # Tensors exceeding the limit are logged and excluded from processing
        filtered_tensor_size = {}
        for tensor_name, tensor_size in intermediate_tensors_size.items():
            # Validate tensor size is non-negative
            if tensor_size < 0:
                self.logger.warning(
                    f"Tensor '{tensor_name}' has negative size {tensor_size}. Skipping."
                )
                continue

            # Check if tensor fits within available memory
            if tensor_size <= available_memory:
                filtered_tensor_size[tensor_name] = tensor_size
            else:
                # Log warning for tensors that are too large to fit
                self.logger.warning(
                    f"Tensor '{tensor_name}' with size {tensor_size} exceeds available memory "
                    f"{available_memory}, so it will be skipped."
                )

        # ============================================================================
        # Step 3: Sort Tensors Using First Fit Decreasing (FFD) Strategy
        # ============================================================================

        # Sort tensors by size in descending order for better bin-packing efficiency
        sorted_tensors = sorted(
            filtered_tensor_size.items(),
            key=lambda x: x[1],  # Sort by tensor size (second element of tuple)
            reverse=True,  # Largest first
        )

        # ============================================================================
        # Step 4: Initialize Data Structures for Batch Management
        # ============================================================================

        # batches: List of lists, where each inner list contains tensor names in that batch
        batches = []

        # batch_sizes: Parallel list tracking the current total size of each batch
        batch_sizes = []

        # ============================================================================
        # Step 5: Perform First Fit Decreasing Bin Packing
        # ============================================================================

        # Iterate through sorted tensors and place each in the first available batch
        for tensor_name, tensor_size in sorted_tensors:
            placed = False

            # Try to fit the tensor in an existing batch (First Fit strategy)
            for i, current_size in enumerate(batch_sizes):
                # Check if adding this tensor would exceed the batch's memory limit
                if current_size + tensor_size <= available_memory:
                    # Tensor fits in this batch - add it
                    batches[i].append(tensor_name)
                    batch_sizes[i] += tensor_size
                    placed = True
                    break  # Move to next tensor once placed

            # If tensor doesn't fit in any existing batch, create a new batch
            if not placed:
                batches.append([tensor_name])
                batch_sizes.append(tensor_size)

        # ============================================================================
        # Step 6: Validate Results and Return
        # ============================================================================

        # Ensure at least one batch was created (all tensors weren't filtered out)
        if not batches:
            raise RuntimeError(
                f"Could not create any output batches. All tensors exceed the available "
                f"memory limit of {available_memory}."
            )

        return batches
