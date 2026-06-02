# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import ConfigDict, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from qairt.api.configs.common import BackendType
from qairt.gen_ai_api.configs.pipeline_config import GeniePipelineConfig
from qairt.modules.genie_execution.genie_cli_mixin import GenieCLIMixin
from qairt.utils import loggers
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DeviceInfo,
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface

genie_app_runner_logger = loggers.get_logger(name="GenieAppRunner")


class GenieAppRunExecutionConfig(AISWBaseModel):
    """
    Defines execution configuration for Genie App pipeline execution.
    Supports text prompt, image input, and pipeline configuration.
    """

    text: Optional[str] = None
    system_prompt: Optional[str] = None
    image: Optional[str | os.PathLike] = None
    pipeline_config: GeniePipelineConfig
    qairt_sdk_root: Optional[str | os.PathLike] = None
    tokenizer_path: Optional[str | os.PathLike] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.text or not self.image:
            raise AttributeError("Either text or image is required for Genie App execution")

        if self.qairt_sdk_root and not os.path.isdir(self.qairt_sdk_root):
            raise NotADirectoryError(
                f"Provided path to QAIRT SDK does not point to an existing directory: {self.qairt_sdk_root}"
            )

        return self


class GenieAppRunOutputConfig(AISWBaseModel):
    """
    Output configuration for Genie App-specific outputs
    May be extended to include more output modalities in the future.
    """

    return_code: int
    stdout: str
    stderr: str
    profile_record: Optional[Dict[str, Any]] = None


class GenieAppRunnerModuleSchema(ModuleSchema):
    _BACKENDS = ["HTP"]
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)

    name: Literal["GenieAppRunnerModule"] = "GenieAppRunnerModule"
    path: Path = Path(__file__)
    arguments: GenieAppRunExecutionConfig
    outputs: SkipJsonSchema[Optional[GenieAppRunOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class GenieAppRunner(Module, GenieCLIMixin):
    """
    Genie App Runner that connects multimodal models like text and vision as Genie nodes
    through a Genie Node Pipeline execution workflow
    """

    _SCHEMA = GenieAppRunnerModuleSchema
    _LOGGER = genie_app_runner_logger

    _PIPELINE_SCRIPT_FILENAME = "LMMScript.cfg"
    _PROFILE_FILENAME = "profile.json"

    def __init__(
        self,
        pipeline_config: GeniePipelineConfig,
        backend: BackendType,
        device: Union[DeviceInfo, DeviceInterface],
        qairt_sdk_root: Optional[Union[str, os.PathLike]] = None,
        tokenizer_path: Optional[Union[str, os.PathLike]] = None,
        backend_extensions_config_path: Optional[Union[str, os.PathLike]] = None,
        logger: Any = None,
        clean_up: bool = False,
    ):
        """
        Initialize a GenieAppRunner module instance

        Args:
            pipeline_config (GeniePipelineConfig): Pipeline configuration containing all node configs
            backend (BackendType): Desired backend type for execution
            device (DeviceInfo | DeviceInterface): Device to execute on
            qairt_sdk_root: Path to QAIRT SDK
            tokenizer_path: Path to tokenizer file (can be overridden in run config)
            backend_extensions_config_path: Path to backend extensions config file
            logger: Logger instance
            clean_up (bool): Whether to clean up on-device workspace on destruction
        """
        super().__init__(logger)

        # Initialize the Genie CLI mixin with common functionality
        self._init_genie_cli_mixin(backend, device, qairt_sdk_root, clean_up)

        # GenieApp-specific initialization
        self._pipeline_config = pipeline_config
        self._tokenizer_path = tokenizer_path
        self._backend_extensions_config_path = backend_extensions_config_path

        # Process pipeline config artifacts
        self._process_pipeline_config_artifacts(self._tokenizer_path, self._backend_extensions_config_path)

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    @property
    def clean_up_on_exit(self) -> bool:
        return self._clean_up

    @clean_up_on_exit.setter
    def clean_up_on_exit(self, clean_up: bool) -> None:
        self._clean_up = clean_up

    def load(self, pipeline_config: Optional[GeniePipelineConfig] = None):
        """Load pipeline node configs and QAIRT SDK artifacts to the on-device workspace.

        Args:
            pipeline_config: Optional pipeline configuration to override the current one
        """
        self._ensure_target_root_exists()

        if not self._sdk_artifacts_loaded:
            self._push_sdk_artifacts("genie-app")

        if pipeline_config:
            self._pipeline_config = pipeline_config
            self._process_pipeline_config_artifacts(
                self._tokenizer_path, self._backend_extensions_config_path
            )

        # Load all artifacts using mixin functionality
        self._push_config_artifacts()

    def run(self, run_config: GenieAppRunExecutionConfig) -> GenieAppRunOutputConfig:
        """Execute the Genie App pipeline with the given configuration.

        Args:
            run_config: Configuration containing text prompt, image path, and pipeline configuration

        Returns:
            GenieAppRunOutputConfig: The execution result with output, return code, and stderr
        """
        # Determine which tokenizer path to use (run config takes precedence)
        tokenizer_path = run_config.tokenizer_path if run_config.tokenizer_path else self._tokenizer_path

        # Check if pipeline config or tokenizer path has changed
        pipeline_config_changed = (
            run_config.pipeline_config and run_config.pipeline_config != self._pipeline_config
        )
        tokenizer_path_changed = tokenizer_path != self._tokenizer_path

        if pipeline_config_changed:
            assert (
                run_config.pipeline_config is not None
            )  # mypy - already validated in GenieAppRunExecutionConfig
            self._pipeline_config = run_config.pipeline_config
            self._process_pipeline_config_artifacts(tokenizer_path, self._backend_extensions_config_path)
            # Only invalidate config artifacts when pipeline config changes
            self._config_artifacts_loaded = False
        elif tokenizer_path_changed:
            # If only tokenizer path is different, reprocess artifacts with new tokenizer
            self._process_pipeline_config_artifacts(tokenizer_path, self._backend_extensions_config_path)
            # Only invalidate config artifacts when tokenizer changes
            self._config_artifacts_loaded = False

        if run_config.qairt_sdk_root and run_config.qairt_sdk_root != self._qairt_sdk_root:
            self._qairt_sdk_root = run_config.qairt_sdk_root
            # Only invalidate SDK artifacts when SDK root changes
            self._sdk_artifacts_loaded = False

        self.load()

        # Generate pipeline script
        pipeline_script = self._generate_pipeline_script(run_config)

        # Push pipeline script to device
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, self._PIPELINE_SCRIPT_FILENAME)
            with open(script_path, "w") as f:
                f.write(pipeline_script)

            target_script_path = self._target_sep.join([self._target_root, self._PIPELINE_SCRIPT_FILENAME])
            self._check_device_return(
                self._device.push(script_path, target_script_path),
                "Failed to push pipeline script to device\t",
            )

        # Push image file to device
        if run_config.image:
            target_image_path = self._target_sep.join([self._target_root, "preprocessed_image.raw"])
            self._check_device_return(
                self._device.push(run_config.image, target_image_path),
                f"Failed to push image file {run_config.image} to device\t",
            )

        env = self._get_device_environment()
        env.shell = True
        cmd = self._prepare_genie_app_command(run_config)

        genie_app_runner_logger.debug(f"Executing Genie App command: {cmd} from cwd {env.cwd}")
        device_return = self._check_device_return(
            self._device.execute([cmd], device_env_context=env), f"Failed to execute command: {cmd}\t"
        )

        profile_record = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                target_file_path = self._target_sep.join([self._target_root, self._PROFILE_FILENAME])
                tmp_file_path = os.path.join(temp_dir, self._PROFILE_FILENAME)
                self._check_device_return(
                    self._device.pull(target_file_path, tmp_file_path),
                    f"Failed to pull profiling json: {self._PROFILE_FILENAME}\t",
                )
                self._check_device_return(
                    self._device.remove(target_file_path),
                    f"Failed to remove profiling json: {target_file_path}",
                )

                with open(tmp_file_path, "r") as f:
                    profile_record = json.load(f)

        except RuntimeError as e:
            genie_app_runner_logger.warning(f"Failed to get profiling data: {str(e)}")

        return GenieAppRunOutputConfig(
            return_code=device_return.returncode,
            stderr=device_return.stderr,
            stdout=device_return.stdout,
            profile_record=profile_record,
        )

    def unload(self):
        """Remove artifacts from the device."""
        self._cleanup_device_artifacts()

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def _generate_pipeline_script(self, run_config: GenieAppRunExecutionConfig) -> str:
        """Generate the Genie App pipeline script using the run configuration.

        Args:
            run_config: Configuration containing runtime inputs like prompts and image paths

        Returns:
            str: Generated pipeline script with placeholders replaced by runtime values
        """

        # Use the pipeline config from the runner instance
        if not self._pipeline_config:
            raise ValueError("Pipeline configuration is required but not set in runner")

        pipeline_config = self._pipeline_config

        # Update pipeline inputs with runtime input values if they contain placeholders
        updated_inputs = []
        for node_name, input_value in pipeline_config.inputs:
            # Check if input value is a placeholder that needs runtime replacement
            if isinstance(input_value, str):
                # Replace runtime placeholders with actual values
                if input_value == "{{system_prompt}}":
                    updated_inputs.append((node_name, run_config.system_prompt or ""))
                elif input_value == "{{text_prompt}}":
                    updated_inputs.append((node_name, run_config.text or ""))
                elif input_value == "{{image_filename}}":
                    updated_inputs.append((node_name, "preprocessed_image.raw"))
                elif input_value.startswith("{{") and input_value.endswith("}}"):
                    # Handle other runtime placeholders - keep as empty string if not found
                    placeholder_name = input_value[2:-2]  # Remove {{ and }}
                    runtime_value = (
                        getattr(run_config, placeholder_name, "")
                        if hasattr(run_config, placeholder_name)
                        else ""
                    )
                    updated_inputs.append((node_name, runtime_value))
                else:
                    # Use the value as-is (already populated)
                    updated_inputs.append((node_name, input_value))
            else:
                # Non-string values, use as-is
                updated_inputs.append((node_name, str(input_value)))

        # Create pipeline config with updated inputs if any changes were made
        if updated_inputs != pipeline_config.inputs:
            pipeline_config = GeniePipelineConfig(
                version=pipeline_config.version,
                nodes=pipeline_config.nodes,
                inputs=updated_inputs,
                connections=pipeline_config.connections,
            )

        return pipeline_config.generate_pipeline_script()

    def _prepare_genie_app_command(self, run_config: GenieAppRunExecutionConfig) -> str:
        """Prepare the command to execute the Genie App pipeline.

        Args:
            run_config: Configuration containing execution parameters

        Returns:
            str: Command string to execute the genie-app binary with appropriate arguments
        """
        cmd = []
        assert self._device.device_info is not None

        if self._device.device_info.platform_type in self._SUPPORTED_PLATFORMS:
            # Use genie-app for pipeline execution
            cmd.append(self._target_sep.join([self._target_qairt_dir(), "genie-app"]))
        else:
            raise RuntimeError(
                f"Requested platform {self._device.device_info.platform_type} is currently not supported."
            )

        cmd.append("-s")
        cmd.append(self._PIPELINE_SCRIPT_FILENAME)

        return " ".join(cmd)

    # Methods specific to GenieAppRunner (not covered by mixin)

    def _process_pipeline_config_artifacts(
        self,
        tokenizer_path: Optional[str | os.PathLike] = None,
        backend_extensions_config_path: Optional[str | os.PathLike] = None,
    ):
        """Process artifacts from pipeline configuration including tokenizer and backend extensions.

        Args:
            tokenizer_path: Optional path to tokenizer file
            backend_extensions_config_path: Optional path to backend extensions config file
        """
        self._config_artifacts.clear()
        for node in self._pipeline_config.nodes.values():
            # Add node config files directly to root
            if os.path.exists(node.config["path"]):
                target_path = self._target_sep.join(
                    [self._target_root, os.path.basename(node.config["path"])]
                )
                self._config_artifacts[target_path] = node.config["path"]

            # Add model files directly to root
            if node.model_paths:
                for model_path in node.model_paths:
                    if os.path.exists(model_path):
                        self._add_config_artifact(model_path)

            # Add LoRA adapter files directly to root
            if node.lora_adapter_paths:
                for lora_path in node.lora_adapter_paths:
                    if os.path.exists(lora_path):
                        self._add_config_artifact(lora_path)

        # Add tokenizer file if provided
        if tokenizer_path and os.path.exists(tokenizer_path):
            target_path = self._target_sep.join([self._target_root, os.path.basename(tokenizer_path)])
            self._config_artifacts[target_path] = tokenizer_path

        # Add backend extensions config file if provided
        if backend_extensions_config_path and os.path.exists(backend_extensions_config_path):
            target_path = self._target_sep.join(
                [self._target_root, os.path.basename(backend_extensions_config_path)]
            )
            self._config_artifacts[target_path] = backend_extensions_config_path

        self._config_artifacts_loaded = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    def __del__(self):
        if self._clean_up:
            self.unload()
