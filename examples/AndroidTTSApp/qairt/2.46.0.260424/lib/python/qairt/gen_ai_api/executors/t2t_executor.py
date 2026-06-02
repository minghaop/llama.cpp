# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import asyncio
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from typing_extensions import Self

from qairt.api.compiled_model import CompiledModel
from qairt.api.configs.common import BackendType
from qairt.api.configs.device import Device, DevicePlatformType, log_device_output
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.executors.gen_ai_executor import (
    GenAIExecutor,
    GenerationMetrics,
    TextGenerationResult,
    parse_genie_profile_record,
    process_prompt,
)
from qairt.gen_ai_api.executors.genie_dialog_factory import GenieDialogFactory
from qairt.modules.genie_execution.genie_config import (
    GenieConfig,
)
from qairt.modules.genie_execution.genie_t2t_run_module import GenieT2TRunExecutionConfig, GenieT2TRunner
from qairt.modules.genie_execution.native_t2t_module import GenieNativeT2TRunner
from qairt.modules.lora.lora_config import UseCaseRunConfig
from qairt.utils import loggers


class T2TExecutor(GenAIExecutor):
    """
    The T2TExecutor handles text-to-text generation on target via Genie. It supports two modes of execution:

    - Native: This is the default mode of execution on platforms with native python support if no device is
      specified. Execution is performed via native python bindings.
    - Device: This is the mode of execution when a device is specified. Execution is performed via subprocess.
      See :class:`qairt.api.configs.device.Device` for supported device types.
    """

    _logger = loggers.get_logger(name=__name__)

    def __init__(
        self,
        models: List[CompiledModel],
        genai_config: GenAIConfig,
        backend: BackendType,
        device: Optional[Device] = None,
        *,
        backend_extensions_config: Optional[Dict] = None,
        qairt_sdk_root: Optional[str | os.PathLike] = None,
        clean_up: bool = True,
        draft_models: Optional[List[CompiledModel]] = None,
        draft_model_backend_extensions_config: Optional[Dict] = None,
    ):
        """
        The executor has a compiled container and device to run it on. The GenAIExecutor will use the
        device to maintain an execution environment and run the compiled container

        Args:
            models (List[CompiledModel]): The compiled models to run
            genai_config (GenAIConfig): GenaiConfig to base GenieConfig on for genie-t2t-run execution
            backend (BackendType): Backend to use for genie-t2t-run execution.
            device (Device): Device to be used for execution. If set to none, native execution will be presumed.
            backend_extensions_config (Dict): Backend extensions configuration for genie-t2t-run execution
            qairt_sdk_root (str | os.PathLike): Path to QAIRT SDK with execution libraries (if different from installed QAIRT)
            clean_up (bool): Will delete artifacts pushed to device if True

        """
        self._clean_up = clean_up
        self._runner: GenieT2TRunner | None = None
        self._native_runner: GenieNativeT2TRunner | None = None

        self._environment_prepared = False
        self._work_dir: tempfile.TemporaryDirectory | None = None

        self._models: List[CompiledModel] = models
        self._draft_models: Optional[List[CompiledModel]] = draft_models
        if not self._models:
            raise ValueError("No models provided for execution")
        self._gen_ai_config = genai_config
        self._device: Optional[Device] = device
        self._backend: BackendType = backend
        self._backend_extensions_config: Optional[Dict] = backend_extensions_config
        self._draft_model_backend_extensions_config: Optional[Dict] = draft_model_backend_extensions_config
        self._qairt_sdk_root: str | os.PathLike = qairt_sdk_root or str(os.environ.get("QAIRT_SDK_ROOT", ""))

        if self._backend_extensions_config and self._backend == BackendType.CPU:
            self._logger.warning(
                f"Ignoring provided backend extensions as requested backend, {self._backend}, does not support extensions"
            )

    def prepare_environment(self) -> Self:
        """
        Prepares artifacts for execution on target
        """
        if not self._environment_prepared:
            self._work_dir = tempfile.TemporaryDirectory()

            backend_extensions_path = ""
            if self._backend_extensions_config:
                backend_extensions_path = os.path.join(self._work_dir.name, "backend_extensions.json")
                with open(backend_extensions_path, "w") as f:
                    f.write(json.dumps(self._backend_extensions_config, indent=2))

            draft_model_backend_extensions_path = ""
            if self._draft_model_backend_extensions_config:
                draft_model_backend_extensions_path = os.path.join(
                    self._work_dir.name, "draft_model_backend_extensions.json"
                )
                with open(draft_model_backend_extensions_path, "w") as f:
                    f.write(json.dumps(self._draft_model_backend_extensions_config, indent=2))
            genie_config = GenieConfig(
                dialog=GenieDialogFactory.create_dialog(
                    self._backend,
                    self._gen_ai_config,
                    self._models,
                    backend_extensions_path,
                    self._is_native_execution(),
                    self._draft_models,
                    draft_model_backend_extensions_path,
                )
            )

            if self._is_native_execution():
                self._native_runner = GenieNativeT2TRunner(genie_config=genie_config)

            else:
                assert isinstance(self._device, Device)  # for mypy

                self._runner = GenieT2TRunner(
                    genie_config,
                    self._backend,
                    self._device.info,
                    self._qairt_sdk_root,
                    clean_up=self._clean_up,
                )
                self._runner.load()
            self._environment_prepared = True
        return self

    def clean_environment(self) -> Self:
        """
        Removes artifacts from target environment
        """
        self._logger.info("Cleaning environment")
        if self._runner:
            self._runner.unload()
            self._runner = None

        if self._work_dir:
            self._work_dir.cleanup()

        if self._native_runner:
            del self._native_runner
            self._native_runner = None

        self._environment_prepared = False
        return self

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]], Path],
        *,
        lora_config: Optional[UseCaseRunConfig] = None,
    ) -> TextGenerationResult:
        """
        Generates a response from a given prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]], Path]): The prompt to be used for generation.
                Can be one of:

                - str: A raw text string prompt
                - Path: Path to a JSON file containing chat messages
                - List[Dict[str, str]]: A list of dicts with "role" and "content" keys. Each dict should contain:

                  - "role": The role of the message sender (e.g., "system", "user", "assistant")
                  - "content": The actual message content

                  Example::

                      [
                          {"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": "What is the capital of France?"}
                      ]

            lora_config (UseCaseRunConfig): Configuration used to control how LoRA adapters
                are applied during model inference.

        Returns:
            TextGenerationResult: The result of the generation containing the output text and associated
            generation metrics.
        """
        if not self._environment_prepared:
            self.prepare_environment()

        formatted_prompt = process_prompt(prompt, self._gen_ai_config)

        if self._is_native_execution() and self._native_runner:
            result = self._native_runner.query(formatted_prompt, lora_config=lora_config)
            self._native_runner.reset_dialog()
            return result
        else:
            return self._generate_non_native(formatted_prompt, lora_config)

    def stream_generate(
        self, prompt: Union[str, List[Dict[str, str]], Path], q: asyncio.Queue
    ) -> asyncio.Task:
        """
        Starts streaming generation and returns the task that will produce the final result.

        Args:
            prompt (Union[str, List[Dict[str, str]], Path]): The prompt to be used for generation.
                Can be one of:

                - str: A raw text string prompt
                - Path: Path to a JSON file containing chat messages
                - List[Dict[str, str]]: A list of dicts with "role" and "content" keys. Each dict should contain:

                  - "role": The role of the message sender (e.g., "system", "user", "assistant")
                  - "content": The actual message content

            q (asyncio.Queue): An asyncio queue used to stream output chunks back to the caller.

        Returns:
            asyncio.Task: A task that will eventually return TextGenerationResult.
        """
        if not self._environment_prepared:
            self.prepare_environment()

        formatted_prompt = process_prompt(prompt, self._gen_ai_config)

        if self._is_native_execution() and self._native_runner:
            try:
                return asyncio.create_task(self._native_runner.stream_query(formatted_prompt, q))
            except Exception as e:
                self._logger.warning(f"Streaming failed to start: {e}")
                raise
        else:
            raise NotImplementedError("Non-native execution is not supported.")

    def _generate_non_native(
        self, prompt: str, lora_config: Optional[UseCaseRunConfig] = None
    ) -> TextGenerationResult:
        """
        Generates a response from a given prompt using the GenieT2TRunner. A device must be specified
        for this method to be called.

        Args:
            prompt (str): The prompt to be used for generation.
            lora_config (UseCaseRunConfig): Configuration used to control how LoRA adapters are applied during model inference.

        Returns:
            TextGenerationResult: The result of the generation containing the output text and associated
            generation metrics.
        """
        out = TextGenerationResult()

        if not self._runner:
            raise RuntimeError("Environment preparation failed")
        else:
            t2t_result = self._runner.run(GenieT2TRunExecutionConfig(prompt=prompt, lora_config=lora_config))

            out.output = t2t_result.stdout

            if t2t_result.return_code != 0:
                out.error = t2t_result.stderr
            else:
                out.output += "\n" + t2t_result.stderr
            self._logger.debug(f"stdout: {t2t_result.stdout}\nstderr:{t2t_result.stderr}")

            # log device output
            log_device_output(self._device, self._logger, log_filter=r"^(?=.*(Genie)).+$")

            begin_idx = out.output.find("[BEGIN]:")
            end_idx = out.output.find("[END]", begin_idx)
            if begin_idx > -1 and end_idx > begin_idx:
                out.generated_text = out.output[begin_idx + len("[BEGIN]:") : end_idx].strip()

            if t2t_result.profile_record:
                out.metrics = parse_genie_profile_record(t2t_result.profile_record)

        return out

    def _is_native_execution(self) -> bool:
        if not self._device:
            return True
        elif self._device.identifier is None:
            if (
                platform.system() == "Linux"
                and platform.machine() == "x86_64"
                and self._device.type == DevicePlatformType.X86_64_LINUX
            ):
                return True

            if platform.system() == "Windows":
                if (
                    "AMD64" in platform.processor() or "Intel64" in platform.processor()
                ) and self._device.type == DevicePlatformType.X86_64_WINDOWS_MSVC:
                    return True
                elif (
                    "ARM64" in platform.processor()
                    or "AARCH64" in platform.processor()
                    or "ARMv8" in platform.processor()
                ) and self._device.type == DevicePlatformType.WOS:
                    return True

        return False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clean_up:
            self.clean_environment()
        return False

    def __del__(self):
        if getattr(self, "_clean_up", False):
            self.clean_environment()
