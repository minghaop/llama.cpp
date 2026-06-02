# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from typing_extensions import Self

from qairt.api.configs.common import BackendType
from qairt.api.configs.device import Device
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.configs.pipeline_config import GeniePipelineConfig
from qairt.gen_ai_api.executors.gen_ai_executor import (
    GenAIExecutor,
    TextGenerationResult,
    parse_genie_profile_record,
    process_prompt,
)
from qairt.modules.genie_execution.genie_app_run_module import GenieAppRunExecutionConfig, GenieAppRunner
from qairt.utils import loggers


class ImageT2TExecutor(GenAIExecutor):
    """
    The ImageT2TExecutor handles Large Multimodal Model execution on target via Genie pipeline using Genie App.
    It supports both text and vision inputs and coordinates their processing
    """

    _logger = loggers.get_logger(name=__name__)

    def __init__(
        self,
        genai_config: GenAIConfig,
        pipeline_config: GeniePipelineConfig,
        backend: BackendType,
        device: Optional[Device] = None,
        backend_extensions_config: Optional[Dict] = None,
        qairt_sdk_root: Optional[str | os.PathLike] = None,
        clean_up: bool = True,
    ):
        """
        Initialize the ImageT2TExecutor

        Args:
            genai_config (GenAIConfig): GenAI configuration for model parameters
            pipeline_config (GeniePipelineConfig): Pipeline configuration for multimodal execution
            backend (BackendType): Backend to use for execution
            device (Device): Device to be used for execution
            backend_extensions_config (Dict): Backend extensions configuration
            qairt_sdk_root (str | os.PathLike): Path to QAIRT SDK
            clean_up (bool): Whether to delete artifacts after execution
        """
        self._clean_up = clean_up
        self._runner: GenieAppRunner | None = None

        self._environment_prepared = False
        self._work_dir: tempfile.TemporaryDirectory | None = None

        self._gen_ai_config: GenAIConfig = genai_config
        self._pipeline_config: GeniePipelineConfig = pipeline_config
        self._device: Optional[Device] = device
        self._backend: BackendType = backend
        self._backend_extensions_config: Optional[Dict] = backend_extensions_config
        self._qairt_sdk_root: str | os.PathLike = qairt_sdk_root or str(os.environ.get("QNN_SDK_ROOT", ""))

        if self._backend_extensions_config and self._backend == BackendType.CPU:
            self._logger.warning(
                f"Ignoring provided backend extensions as requested backend, {self._backend}, does not support extensions"
            )

    def prepare_environment(self) -> Self:
        """Prepares artifacts for Genie App execution on target.

        Returns:
            Self: The executor instance
        """
        if not self._environment_prepared:
            self._work_dir = tempfile.TemporaryDirectory()

            backend_extensions_path = ""
            if self._backend_extensions_config:
                backend_extensions_path = os.path.join(self._work_dir.name, "htp_backend_ext_config.json")
                with open(backend_extensions_path, "w") as f:
                    f.write(json.dumps(self._backend_extensions_config, indent=2))

            assert isinstance(self._device, Device)  # for mypy
            # The pipeline configuration contains all the node configurations
            # with their respective JSON config files specified in node_config_filename
            self._runner = GenieAppRunner(
                pipeline_config=self._pipeline_config,
                backend=self._backend,
                device=self._device.info,
                qairt_sdk_root=self._qairt_sdk_root,
                tokenizer_path=self._gen_ai_config.tokenizer_path,
                backend_extensions_config_path=backend_extensions_path,
                clean_up=self._clean_up,
            )
            self._runner.load()

        self._environment_prepared = True
        return self

    def clean_environment(self) -> Self:
        """Removes artifacts from target environment.

        Returns:
            Self: The executor instance
        """
        self._logger.info("Cleaning ImageText2Text environment")
        if self._runner:
            self._runner.unload()
            self._runner = None

        if self._work_dir:
            self._work_dir.cleanup()

        self._environment_prepared = False
        return self

    def generate(
        self,
        text: Union[str, List[Dict[str, str]], Path],
        image: Optional[str] = None,
    ) -> TextGenerationResult:
        """Generates a response from text and optionally image inputs.

        Args:
            text: The text to be used for generation. Can be one of:
                - str: A raw text string
                - Path: Path to a JSON file containing chat messages
                - List[Dict[str, str]]: A list of dicts with "role" and "content" keys. Each dict should contain:
                  - "role": The role of the message sender (e.g., "system", "user", "assistant")
                  - "content": The actual message content
            image: Optional path to the input image

        Returns:
            TextGenerationResult: The result containing generated text and metadata
        """
        if not self._environment_prepared:
            self.prepare_environment()

        # Process the text using the shared utility function
        formatted_text = process_prompt(text, self._gen_ai_config)

        # TODO: Vision Processor will be built in builder and stored in container
        # which will be used here in executor eventually. For now, .raw image is expected.
        preprocessed_image_path = image if image is not None else None

        # Create the execution config
        run_config = GenieAppRunExecutionConfig(
            text=formatted_text,
            image=preprocessed_image_path,
            pipeline_config=self._pipeline_config,
            tokenizer_path=self._gen_ai_config.tokenizer_path,
        )

        # Execute the Genie App LMM pipeline
        if self._runner:
            run_result = self._runner.run(run_config)

            result = TextGenerationResult()
            result.output = run_result.stdout

            if run_result.return_code != 0:
                result.error = run_result.stderr
            else:
                result.output += "\n" + run_result.stderr

            # Parse generated text from output
            begin_idx = result.output.find("[BEGIN]:")
            end_idx = result.output.find("[END]", begin_idx)
            if begin_idx > -1 and end_idx > begin_idx:
                result.generated_text = result.output[begin_idx + len("[BEGIN]:") : end_idx].strip()

            if run_result.profile_record:
                result.metrics = parse_genie_profile_record(run_result.profile_record)

            return result

        raise RuntimeError("Genie App Runner not initialized")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clean_up:
            self.clean_environment()
        return False

    def __del__(self):
        if self._clean_up:
            self.clean_environment()
