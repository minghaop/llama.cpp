# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
from enum import Enum
from typing import Literal, Optional, Union, cast, overload

# This import needs to be at the top of other imports so that
# required modules in transformers are modified during runtime to enable GGUF workflow.
try:
    from qti.aisw.converters.gguf_builder import gguf_builder

except ImportError as e:
    print("WARNING: Unable to import GGUF Builder")
from typing import Literal, Optional, Union, cast, overload

from qairt.api.configs.common import BackendType
from qairt.api.converter.converter_config import CalibrationConfig
from qairt.gen_ai_api.builders.baichuan.builder import BaichuanBuilderHTP
from qairt.gen_ai_api.builders.gen_ai_builder_cpu import GenAIBuilderCPU
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP
from qairt.gen_ai_api.builders.gen_ai_utils import load_pretrained_config
from qairt.gen_ai_api.builders.indus.builder import IndusBuilderHTP
from qairt.gen_ai_api.builders.jais.builder import JaisBuilderHTP
from qairt.gen_ai_api.builders.llama.builder import LlamaBuilderHTP
from qairt.gen_ai_api.builders.mistral.builder import MistralBuilderHTP
from qairt.gen_ai_api.builders.phi.builder import PhiBuilderHTP
from qairt.gen_ai_api.builders.plamo.builder import PlamoBuilderHTP
from qairt.gen_ai_api.builders.qwen.builder import Qwen3MoeBuilderHTP, QwenBuilderHTP

logger = logging.getLogger(__name__)


_SUPPORTED_BACKENDS = [BackendType.CPU, BackendType.HTP]


# TODO: This should be co-located with HTP Builder
class SupportedLLMs(Enum):
    """Enumeration of preconfigured builder architectures for the HTP backend"""

    LLAMA = "LlamaForCausalLM"
    BAICHUAN = "BaiChuanForCausalLM"
    PHI = "Phi3ForCausalLM"
    QWEN = "Qwen2ForCausalLM"
    QWEN3_MOE = "Qwen3MoeForCausalLM"
    MISTRAL = "MistralForCausalLM"
    JAIS = "JAISLMHeadModel"
    PLAMO = "PlamoForCausalLM"
    INDUS = "GPT2LMHeadModel"


class GenAIBuilderFactory:
    """
    Factory class to create :class:`qairt.gen_ai_api.gen_ai_builder.GenAIBuilder` instances
    """

    @overload
    @classmethod
    def create(
        cls,
        pretrained_model_path: str | os.PathLike,
        backend_type: Literal[BackendType.CPU] | Literal["CPU"],
        *,
        cache_root: Optional[str | os.PathLike] = None,
    ) -> GenAIBuilderCPU: ...

    @overload
    @classmethod
    def create(
        cls,
        pretrained_model_path: str | os.PathLike,
        backend_type: Literal[BackendType.HTP] | Literal["HTP"] = BackendType.HTP,
        *,
        cache_root: Optional[str | os.PathLike] = None,
        tokenizer_path: Optional[str | os.PathLike] = None,
        config_path: Optional[str | os.PathLike] = None,
    ) -> GenAIBuilderHTP: ...

    @overload
    @classmethod
    def create(
        cls,
        pretrained_model_path: str | os.PathLike,
        backend_type: str | BackendType,
        *,
        cache_root: Optional[str | os.PathLike] = None,
        tokenizer_path: Optional[str | os.PathLike] = None,
        config_path: Optional[str | os.PathLike] = None,
    ) -> Union[GenAIBuilderHTP, GenAIBuilderCPU]: ...

    @classmethod
    def create(
        cls,
        pretrained_model_path: str | os.PathLike,
        backend_type: str | BackendType = BackendType.HTP,
        *,
        cache_root: Optional[str | os.PathLike] = None,
        tokenizer_path: Optional[str | os.PathLike] = None,
        config_path: Optional[str | os.PathLike] = None,
    ) -> Union[GenAIBuilderCPU, GenAIBuilderHTP]:
        """
        Creates a GenAIBuilder instance based on the provided pretrained model path and backend type.

        This function makes the following assumptions:

         - Directory Contents: The following directory and naming structure is expected

            - For the HTP Backend:

               <pretrained_model_path>.dir
                - <model>.onnx
                - <model>.encodings (a file containing quantization overrides)
                - <model>.data (optional)
                - config.json (a Hugging Face transformers configuration for the model)
                - tokenizer.json (the corresponding configuration for the tokenizer from Hugging Face)

             If the pretrained model path is a file, the directory containing it will be used to locate
             additional required artifacts. If the pretrained model path is a directory, then the directory
             is assumed to contain a single onnx model and a single set of corresponding encodings. A warning
             will be returned if more than one model or encodings are found.

            - For the CPU Backend:

               <pretrained_model_path>.dir where the directory contains the transformers configuration,
               model weights, tokenizer and other artifacts as if the model where downloaded directly from
               Hugging Face.

         - Model Architecture:

           The builder will attempt to identify the model and provide a pre-configured instance based on the
           architecture. If HTP is requested and the model architecture is not recognized,
           then a default GenAIBuilderHTP instance will be returned, with a warning.
           See :class:`qairt.gen_ai_api.gen_ai_builder_factory.SupportedLLMs` for a list of pre-configured
           architectures.

        Args:
            pretrained_model_path (str): The path to the pretrained ONNX model.  The pretrained model path may
             be a directory containing the model or a file path to the model itself.
            backend_type (BackendType): The type of backend to use. Defaults to BackendType.HTP.
            cache_root (Path, optional): The root directory for caching, if desired.
            tokenizer_path (str | PathLike, optional): Explicit path to the tokenizer. If provided,
             this takes precedence over convention-based discovery. Can be a file path or directory
             (will look for tokenizer.json inside). Defaults to None.
            config_path (str | PathLike, optional): Explicit path to the model config. If provided,
             this takes precedence over convention-based discovery. Can be a file path or directory
             (will look for config.json inside). Defaults to None.

        Returns:
            GenAIBuilder: The created GenAIBuilder instance.

        Raises:
            ValueError: If the pretrained model path does not exist or if required files are missing.
        """

        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"Pretrained model path '{pretrained_model_path}' does not exist")

        if backend_type not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Backend type '{backend_type}' is not supported")

        if backend_type == BackendType.CPU:
            return GenAIBuilderCPU.from_pretrained(pretrained_model_path)

        apply_gguf_config = False
        if os.path.splitext(pretrained_model_path)[1] == ".gguf":
            gguf_artifacts_paths = gguf_builder.GGUFBuilder(pretrained_model_path).build_from_gguf()
            pretrained_model_path = gguf_artifacts_paths[0]
            apply_gguf_config = True

        # if pretrained_model_path is a file, get the path to the directory containing it
        pretrained_model_path_dir = pretrained_model_path
        if os.path.isfile(pretrained_model_path):
            pretrained_model_path_dir = os.path.dirname(pretrained_model_path)

        # Load config - use explicit path if provided, otherwise use convention
        config = load_pretrained_config(config_path if config_path else pretrained_model_path_dir)
        builder = None
        if hasattr(config, "architectures") and config.architectures is not None:
            if SupportedLLMs.LLAMA.value in config.architectures:
                builder = LlamaBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.QWEN.value in config.architectures:
                builder = QwenBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.QWEN3_MOE.value in config.architectures:
                builder = Qwen3MoeBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.PHI.value in config.architectures:
                builder = PhiBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.MISTRAL.value in config.architectures:
                builder = MistralBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.BAICHUAN.value in config.architectures:
                builder = BaichuanBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.JAIS.value in config.architectures:
                builder = JaisBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.PLAMO.value in config.architectures:
                builder = PlamoBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
            elif SupportedLLMs.INDUS.value in config.architectures:
                builder = IndusBuilderHTP.from_pretrained(
                    os.fspath(pretrained_model_path),
                    cache_root,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path,
                )
        if not builder:
            logger.warning(
                "Architecture is unknown or unsupported; Returning default. "
                "This builder may work but will probably require additional configuration."
            )
            builder = GenAIBuilderHTP.from_pretrained(
                os.fspath(pretrained_model_path),
                cache_root,
                tokenizer_path=tokenizer_path,
                config_path=config_path,
            )
        assert builder is not None

        if apply_gguf_config:
            calib_config = cast(CalibrationConfig, builder._calibration_config)
            calib_config.keep_weights_quantized = True
            setattr(calib_config, "float_bitwidth", 16)

        return builder
