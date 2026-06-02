# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import ConfigDict, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from qairt.api.configs.common import BackendType
from qairt.modules.genie_execution.genie_cli_mixin import GenieCLIMixin
from qairt.modules.genie_execution.genie_config import (
    DialogType,
    EagletDialog,
    EagletDraftDialogEngine,
    EagletTargetDialogEngine,
    EngineModelType,
    GenieConfig,
    Sampler,
    SSDDialog,
)
from qairt.modules.lora.lora_config import UseCaseRunConfig
from qairt.utils.loggers import QAIRTLogger, get_logger
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    Target,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.utils.configure_backend import create_backend
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceEnvironmentContext,
    DeviceInfo,
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCode,
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)

from ...api.configs.device import log_device_output
from .genie_t2t_runner_helper import GenieT2TRunnerHelper

genie_t2t_runner_logger = get_logger(name="GenieT2TRunner")


class E2TQuantizedType(str, Enum):
    """
    Defines supported quantization datatypes for the GenieT2TRun embedding input workflow
    """

    INT8 = "int8"
    INT16 = "int16"
    UINT8 = "uint8"
    UINT16 = "uint16"


class EmbeddingQuantization(AISWBaseModel):
    """
    Quantization parameters for a quantized embedding input or quantized embedding table
    """

    datatype: E2TQuantizedType
    scale: float
    offset: int

    def __str__(self):
        return ",".join([str(self.datatype.value), str(self.scale), str(self.offset)])


class EmbeddingInputConfig(AISWBaseModel):
    """
    Embedding input and embedding table arguments consist of the path to the raw data and optional quantization
    parameters
    """

    path: str | os.PathLike
    quantization: Optional[EmbeddingQuantization] = None


class EmbeddingConfig(AISWBaseModel):
    """
    Defines embedding input for e2t use cases
    """

    input: EmbeddingInputConfig
    """
    Embedding of prompt and/or multimodal inputs
    """
    embedding_table: EmbeddingInputConfig
    """
    Embedding table for converting token ids to embedding vectors
    """

    @model_validator(mode="after")
    def check_quantization(self) -> Self:
        if bool(self.input.quantization) ^ bool(self.embedding_table.quantization):
            raise AttributeError(
                "Quantization parameters must be provided for both the input and embedding table, or neither."
            )
        if self.input.quantization and self.embedding_table.quantization:
            signed = [E2TQuantizedType.INT8, E2TQuantizedType.INT16]
            if bool(self.input.quantization.datatype in signed) ^ bool(
                self.embedding_table.quantization.datatype in signed
            ):
                raise AttributeError("Input and embedding table quantization types' signedness must match")

        return self


class GenieT2TRunExecutionConfig(AISWBaseModel):
    """
    Defines supported genie-t2t-run options

    If a new genie config is passed it will take the place of the previously loaded config

    User specifies one of 'prompt', 'prompt_file', 'token_file', or 'embedding_config' as input to the model

    If Lora adapter bins are given in the genie config, the user may specify the name of the adapter they would
    like applied to the model, as well as the lora alpha value, using the 'lora_config'

    The user may specify a set of genie config sampler parameters via the 'sampler'. This will avoid redeploying
    the assets pointed to by a genie config if only the sampler parameters are changed.

    Passing 'qairt_sdk_root' will update the location that the instance pulls qairt libraries from for execution.
    """

    prompt: Optional[str] = None
    prompt_file: Optional[str | os.PathLike] = None
    token_file: Optional[str | os.PathLike] = None
    embedding_config: Optional[EmbeddingConfig] = None
    config: Optional[GenieConfig] = None
    lora_config: Optional[UseCaseRunConfig] = None
    sampler: Optional[Sampler] = None
    max_num_tokens: Optional[int] = None
    qairt_sdk_root: Optional[str | os.PathLike] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        inputs = {
            "prompt": self.prompt,
            "prompt_file": self.prompt_file,
            "token_file": self.token_file,
            "embedding_config": self.embedding_config,
        }
        provided_inputs = [x for x in inputs.keys() if inputs[x]]

        if not provided_inputs:
            raise AttributeError(f"No input provided for execution. Please provide one of: {inputs.keys()}")

        if len(provided_inputs) > 1:
            raise AttributeError(
                f"Too many inputs provided: {provided_inputs}. Please provide one of: {inputs.keys()}"
            )

        if self.lora_config and self.config:
            # with eaglet there can be two models: draft and target
            engines = (
                [self.config.dialog.engine]
                if not isinstance(self.config.dialog, EagletDialog)
                else self.config.dialog.engine
            )
            for engine in engines:
                if not (engine.model.binary and engine.model.binary.lora):
                    raise AttributeError(
                        "Lora execution argument provided, however, the provided GenieConfig has no lora"
                        "definition"
                    )
                available_use_cases = [x.name for x in engine.model.binary.lora.adapters]
                if self.lora_config.use_case_name not in available_use_cases:
                    raise AttributeError(
                        f'Requested lora use case, "{self.lora_config.use_case_name}", not present in'
                        "provided GenieConfig"
                    )
        if self.qairt_sdk_root and not os.path.isdir(self.qairt_sdk_root):
            raise NotADirectoryError(
                f"Provided path to QAIRT SDK does not point to an existing directory: {self.qairt_sdk_root}"
            )

        return self


class GenieT2TRunOutputConfig(AISWBaseModel):
    """
    Defines output from execution of the GenieT2TRunner
    """

    return_code: int
    stdout: str
    stderr: str
    profile_record: Optional[Dict[str, Any]] = None


class GenieT2TRunnerModuleSchema(ModuleSchema):
    _BACKENDS = ["HTP"]
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)

    name: Literal["GenieT2TRunnerModule"] = "GenieT2TRunnerModule"
    path: Path = Path(__file__)
    arguments: GenieT2TRunExecutionConfig
    outputs: SkipJsonSchema[Optional[GenieT2TRunOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class GenieT2TRunner(Module, GenieCLIMixin):
    _SCHEMA = GenieT2TRunnerModuleSchema
    _LOGGER = genie_t2t_runner_logger

    _GENIE_CONFIG_FILENAME = "genie_config.json"
    _SSD_KVCACHE_PREFIX_FILENAME = "kv-cache.primary.qnn-htp"
    _PROFILE_FILENAME = "profile.json"

    def __init__(
        self,
        config: GenieConfig,
        backend: BackendType,
        device: Union[DeviceInfo, DeviceInterface],
        qairt_sdk_root: Optional[Union[str, os.PathLike]] = None,
        logger: Any = None,
        clean_up: bool = True,
    ):
        """
        Initializes a GenieT2TRunner module instance

        Args:
            config (GenieConfig): Genie config representing the model to run
            backend (BackendType): Desired backend type for execution
            device (DeviceInfo | DeviceInterface): Device to execute the model on
            qairt_sdk_root(Optional[Union[str, PathLike]]): User may specify the path of a QAIRT SDK to
                pull execution libraries from, otherwise, libraries are derived from the currently installed SDK
            logger (any): A logger instance to be used by the NetRunner module
            clean_up (bool): Indicates whether the on device workspace should be deleted when the instance is destroyed.
        """
        super().__init__(logger)

        # Initialize the Genie CLI mixin with common functionality
        self._init_genie_cli_mixin(backend, device, qairt_sdk_root, clean_up)

        # T2T-specific initialization
        self._config_loaded = False
        self._config, self._config_artifacts = GenieT2TRunnerHelper.prepare_config(config)
        if qairt_sdk_root:
            if os.path.isdir(qairt_sdk_root):
                self._qairt_sdk_root = qairt_sdk_root
            else:
                raise NotADirectoryError(f"Provided QAIRT SDK root is not a directory: {qairt_sdk_root}")
        else:
            self._qairt_sdk_root = str(os.environ.get("QAIRT_SDK_ROOT"))
        self._sdk_artifacts_loaded = False

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

    def load(self, config: Optional[GenieConfig] = None):
        """
        This function will load model and QAIRT sdk artifacts to the on device workspace for execution. Subsequent
        calls to 'run' will only need to push inputs or new model artifacts if the genie config is updated
        Args:
            config: (Optional[GenieConfig]) The user may optionally pass a GenieConfig. This will replace the
            GenieConfig currently associated with the instance. Assets associated with the previous config will be
            unloaded from the device, and the assets associated with the new config will be loaded.
        """
        self._ensure_target_root_exists()

        if not self._sdk_artifacts_loaded:
            # Use mixin to push SDK artifacts with T2T-style subdirectories
            self._push_sdk_artifacts("genie-t2t-run", use_subdirectories=True)

        # prepare_config will populate a class variable that is a dictionary, and it's opinionated about where things are.
        if config:
            self._config, self._config_artifacts = GenieT2TRunnerHelper.prepare_config(config)
            self._config_loaded = False
            # Only invalidate config artifacts when config changes
            self._config_artifacts_loaded = False

        # this is just pushing the genie config
        if not self._config_loaded:
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(os.path.join(temp_dir, self._GENIE_CONFIG_FILENAME), "w") as f:
                    f.write(json.dumps(self._config.export(), indent=2))

                self._device.remove(self._target_sep.join([self._target_root, self._GENIE_CONFIG_FILENAME]))
                self._check_device_return(
                    self._device.push(
                        os.path.join(temp_dir, self._GENIE_CONFIG_FILENAME),
                        self._target_sep.join([self._target_root, self._GENIE_CONFIG_FILENAME]),
                    ),
                    "Failed to push genie config to device\t",
                )
                self._config_loaded = True

        # Push T2T-specific config artifacts (different structure than mixin default)
        self._push_t2t_config_artifacts()

    def run(self, run_config: GenieT2TRunExecutionConfig) -> GenieT2TRunOutputConfig:
        """
        Runs the execution configuration specified by the 'run_config' on the device associated with the instance.
        Any required artifacts not already present on target will be pushed to the working directory so the user is
        not required to call 'load' before 'run'.

        If a GenieConfig is passed in the GenieT2TRunExecutionConfig, it will replace the GenieConfig previously
        associated with the instance. Assets from the previous config will be unloaded from the target as appropriate,
        and the artifacts associated with the new GenieConfig will be loaded.
        Args:
            run_config (GenieT2TRunExecutionConfig): Defines inputs and allows configuration updates after construction
        Returns:
            GenieT2TRunOutputConfig: Output from on device execution
        """
        if run_config.config:
            self._config, self._config_artifacts = GenieT2TRunnerHelper.prepare_config(run_config.config)
            self._config_loaded = False
            # Only invalidate config artifacts when config changes
            self._config_artifacts_loaded = False

        if run_config.sampler:
            self._config.dialog.sampler = run_config.sampler
            self._config_loaded = False

        if run_config.max_num_tokens:
            self._config.dialog.max_num_tokens = run_config.max_num_tokens
            self._config_loaded = False

        if run_config.qairt_sdk_root and run_config.qairt_sdk_root != self._qairt_sdk_root:
            self._qairt_sdk_root = run_config.qairt_sdk_root
            # Only invalidate SDK artifacts when SDK root changes
            self._sdk_artifacts_loaded = False

        self.load()
        env = self._get_device_environment(use_subdirectories=True)
        env.shell = True
        cmd = self._prepare_command(run_config)
        genie_t2t_runner_logger.debug(f"Executing command: {cmd} from cwd {env.cwd}")
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
            genie_t2t_runner_logger.warning(f"Failed to get profiling data: {str(e)}")
        return GenieT2TRunOutputConfig(
            return_code=device_return.returncode,
            stderr=device_return.stderr,
            stdout=device_return.stdout,
            profile_record=profile_record,
        )

    def unload(self):
        """
        Removes artifacts from the device that the instance pushed
        """
        self._cleanup_device_artifacts()
        # Reset T2T-specific state
        self._config_loaded = False

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def _prepare_command(self, run_config: GenieT2TRunExecutionConfig) -> str:
        cmd = []
        assert self._device.device_info is not None
        if self._device.device_info.platform_type in self._SUPPORTED_PLATFORMS:
            cmd.append(self._target_sep.join([self._target_bin_dir(), "genie-t2t-run"]))
        else:
            raise RuntimeError(
                f"Requested platform {self._device.device_info.platform_type} is currently not supported."
            )
        cmd.append("-c")
        cmd.append(self._GENIE_CONFIG_FILENAME)
        cmd.append("--profile")
        cmd.append(self._PROFILE_FILENAME)
        if run_config.prompt:
            # genie-t2t-run -c genie_config.json -p "prompt"
            cmd.append("-p")
            # Escape the prompt properly for shell execution
            escaped_prompt = run_config.prompt.replace("\\", "\\\\").replace('"', '\\"')
            cmd.append(f'"{escaped_prompt}"')
        elif run_config.prompt_file:
            # genie-t2t-run -c genie_config.json --prompt_file /path/to/prompt_file.txt
            dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.prompt_file)]
            )
            self._device.remove(dst_path)
            self._check_device_return(
                self._device.push(run_config.prompt_file, dst_path),
                f"Failed to push prompt file: {run_config.prompt_file}\t",
            )
            cmd.append("--prompt_file")
            cmd.append(dst_path)
        elif run_config.token_file:
            # genie-t2t-run -c genie_config.json --token_file /path/to/token_file.raw
            dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.token_file)]
            )
            self._check_device_return(
                self._device.push(run_config.token_file, dst_path),
                f"Failed to push token file: {run_config.token_file}\t",
            )
            cmd.append("--token_file")
            cmd.append(dst_path)
        elif run_config.embedding_config:
            # float32 embedding input and embedding table
            # genie-t2t-run -c genie_config.json -e /path/to/embedding_input.raw -t /path/to/embedding_input.raw

            # quantized embedding input and embedding table
            # genie-t2t-run -c genie_config.json -e /path/to/embedding_input.raw,DATATYPE,SCALE,OFFSET  \
            # -t /path/to/embedding_input.raw,DATATYPE,SCALE,OFFSET
            input_dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.embedding_config.input.path)]
            )
            self._check_device_return(
                self._device.push(run_config.embedding_config.input.path, input_dst_path),
                (f"Failed to push embedding input file: {run_config.embedding_config.input.path}\t"),
            )

            embedding_table_dst_path = self._target_sep.join(
                [
                    self._target_root,
                    "input",
                    os.path.basename(run_config.embedding_config.embedding_table.path),
                ]
            )
            self._check_device_return(
                self._device.push(run_config.embedding_config.embedding_table.path, embedding_table_dst_path),
                (
                    "Failed to push embedding table file: "
                    f"{run_config.embedding_config.embedding_table.path}\t"
                ),
            )

            input_arg = input_dst_path
            if run_config.embedding_config.input.quantization:
                input_arg += "," + str(run_config.embedding_config.input.quantization)

            embedding_table_arg = embedding_table_dst_path
            if run_config.embedding_config.embedding_table.quantization:
                embedding_table_arg += "," + str(run_config.embedding_config.embedding_table.quantization)

            cmd.extend(["-e", str(input_arg), "-t", str(embedding_table_arg)])
        else:
            raise ValueError("Please provide a prompt, prompt file, or embedding input and embedding table")

        if run_config.lora_config:
            # genie-t2t-run -c genie_config.json <one of above inputs> \
            # -l use_case_name,alpha_name_1,alpha_value_1,alpha_name_2,alpha_value_2,...

            engine = (
                self._config.dialog.engine
                if not isinstance(self._config.dialog, EagletDialog)
                else next(
                    engine
                    for engine in self._config.dialog.engine
                    if isinstance(engine, EagletTargetDialogEngine)
                )
            )

            if not (engine.model.binary and engine.model.binary.lora):
                raise AttributeError(
                    "Lora execution argument provided, however, the provided GenieConfig has no lora definition"
                )
            uc_run_config_name = run_config.lora_config.use_case_name
            available_use_cases = [x.name for x in engine.model.binary.lora.adapters]
            if not (
                engine.model.binary and engine.model.binary.lora and uc_run_config_name in available_use_cases
            ):
                raise ValueError(f"Requested lora use case not present in genie config: {uc_run_config_name}")

            adapter_idx = available_use_cases.index(uc_run_config_name)

            # Iterate through all adapters in use case
            for i in range(len(run_config.lora_config.adapters)):
                genie_lora_adapters = engine.model.binary.lora.adapters
                alpha_name = genie_lora_adapters[adapter_idx].alphas[i]
                alpha_value = run_config.lora_config.adapters[i].alpha
                if i == 0:
                    cmd.append("-l")
                    cmd.append(f"{uc_run_config_name},{alpha_name},{alpha_value}")
                else:
                    cmd[-1] += f",{alpha_name},{alpha_value}"
            if isinstance(run_config.config, EagletDialog):
                cmd.extend(["--engine_role", "target"])
        # check if logger is set. If set, retrieve log level as string
        if self._logger is not None:
            log_level_str = QAIRTLogger.get_default_logging_level(
                self._logger.name, use_qairt_levels=True
            ).lower()
            cmd.append("--log")
            cmd.append(log_level_str)

        return " ".join(cmd)

    def _push_t2t_config_artifacts(self):
        """Push T2T-specific config artifacts with their special structure."""
        if not self._config_artifacts_loaded:
            for dst, src in self._config_artifacts.items():
                self._check_device_return(
                    self._device.push(src, self._target_sep.join([self._target_root, str(dst)])),
                    f"Failed to push model artifact {src} to device\t",
                )
            self._config_artifacts_loaded = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    def __del__(self):
        if self._clean_up:
            self.unload()
