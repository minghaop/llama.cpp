# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum


class COMPONENT(Enum):
    """Enum for components of accuracy debugger"""

    FRAMEWORK_RUNNER = "framework_runner"
    INFERENCE_ENGINE = "inference_engine"
    VERIFICATION = "verification"
    COMPARE_ENCODINGS = "compare_encodings"
    TENSOR_VISUALIZER = "tensor_visualizer"
    SNOOPING = "snooping"
    SNOOPING_ORT_QNN = "snooping_ort_qnn"
    VALIDATE_ENCODING = "validate_encoding"
