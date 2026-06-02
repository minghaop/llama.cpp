# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Module for preparing a model for inference using the Genie Composer module.
"""

import pathlib
from enum import Enum
from os import PathLike
from typing import Optional

from qairt.api.compiled_model import CompiledModel
from qairt.api.configs.common import BackendType
from qairt.modules.cache_module import CacheModule
from qairt.utils.loggers import get_logger
from qti.aisw.genai.qnn_genai_transformer_composer_backend import (
    GGMLFileType,
    run_composer,
)

logger = get_logger(__name__)


class QuantizationLevel(Enum):
    """An enum for the quantization levels supported by the Genie Composer module."""

    Z4 = "Z4"
    Z4_FP16 = "Z4_FP16"
    Z4_BF16 = "Z4_BF16"
    Q4 = "Q4"
    Z8 = "Z8"
    FP32 = "FP32"


class GeniePreparationModule:
    """A class for preparing a model for inference using the Genie Composer module."""

    QUANTIZATION_LEVEL_TO_GGML_FILE_TYPE = {
        QuantizationLevel.Z4: GGMLFileType.MostlyZ4,
        QuantizationLevel.Z4_FP16: GGMLFileType.Z4_FP16,
        QuantizationLevel.Z4_BF16: GGMLFileType.Z4_BF16,
        QuantizationLevel.Q4: GGMLFileType.MostlyQ4_0_32,
        QuantizationLevel.Z8: GGMLFileType.MostlyZ8,
        QuantizationLevel.FP32: GGMLFileType.AllF32,
    }

    @classmethod
    def prepare(
        cls,
        framework_model_path: str | PathLike,
        *,
        quantization_level: Optional[QuantizationLevel] = None,
        export_tokenizer_json: bool = False,
        outfile: str | PathLike = pathlib.Path.cwd(),
        config_file: Optional[str | PathLike] = None,
        lora: Optional[str | PathLike] = None,
        lm_head_precision: Optional[QuantizationLevel] = None,
    ) -> CompiledModel:
        """
        Prepare a model for inference using the Genie Composer module.

        Args:
            framework_model_path (str | PathLike): Path to the framework model to be prepared.
            quantization_level (Optional[QuantizationLevel], optional): Quantization level to use for the model. Defaults to None.
            export_tokenizer_json (bool, optional): Whether to export the tokenizer as a JSON file. Defaults to False.
            outfile (str | PathLike, optional): Path to write the generated file. Defaults to the current working directory.
            config_file (Optional[str | PathLike], optional): Path to the configuration file.
                If none is provided, it will look for config.json in the framework_model_path
            lora (Optional[str | PathLike], optional): Path to the LoRA (Low-Rank Adaptation) file. Defaults to None.
            lm_head_precision (Optional[QuantizationLevel], optional): Quantization level to use for the language model head. Defaults to quantization_level

        Returns:
            CompiledModel: A CompiledModel object containing the prepared model.
        """
        outfile = pathlib.Path(outfile)
        if outfile.is_dir():
            outfile = outfile / "outfile.bin"

        run_composer(
            model=framework_model_path,
            quantize=quantization_level.value if quantization_level else None,
            export_tokenizer_json=export_tokenizer_json,
            outfile=outfile,
            config_file=config_file,
            lora=lora,
            lm_head_precision=lm_head_precision.value if lm_head_precision else None,
        )

        return CompiledModel(CacheModule.load(path=outfile), BackendType.CPU)
