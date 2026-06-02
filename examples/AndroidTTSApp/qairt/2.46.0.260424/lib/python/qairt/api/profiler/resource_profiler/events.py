# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Profile event types and event name registries.

All profiling event names and types are defined here so that every profiled
event across every pipeline is visible in one place.
"""

from __future__ import annotations

from enum import Enum


class ProfileEventType(str, Enum):
    """Classifies the context in which a profiling event was recorded.

    The ``str`` mixin means values work as plain strings in logs and JSON
    without needing ``.value``.
    """

    PIPELINE_STAGE = "pipeline_stage"  # captured by StageProfilerObserver
    CLASS_METHOD = "class_method"  # captured by @resource_profile on a class
    FUNCTION = "function"  # captured by @resource_profile(scope="function")
    CUSTOM = "custom"  # user-defined


class QuantizationEvents:
    """Standard event name constants for the quantization pipeline."""

    LOAD_MODEL = "quant.load_model"
    CALIBRATION = "quant.calibration"
    QUANTIZE = "quant.quantize"
    EXPORT = "quant.export"
    VALIDATE = "quant.validate"


class GenAIBuildEvents:
    """Standard event name constants for the GenAI build pipeline."""

    BUILD_LORA_GRAPH = "genaibuilder.build_lora_graph"
    AR_CL_CONVERSION = "genaibuilder.ar_cl_conversion"
    TRANSFORM = "genaibuilder.transform"
    CONVERT = "genaibuilder.convert"
    COMPILE = "genaibuilder.compile"
    SPECULATIVE_SETUP = "genaibuilder.speculative_config_setup"
    PREPARE_EMBEDDING_LUT = "genaibuilder.prepare_embedding_lut"


# New pipelines add a new class here.
