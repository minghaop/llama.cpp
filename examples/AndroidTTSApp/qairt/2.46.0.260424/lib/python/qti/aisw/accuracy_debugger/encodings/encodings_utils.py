# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum


class EncodingType(Enum):
    UNDEFINED = "UNDEFINED"  # for float
    PER_TENSOR = "PER_TENSOR"
    PER_CHANNEL = "PER_CHANNEL"
    LPBQ = "LPBQ"


class UnsupportedEncodingsVersionError(Exception):
    """Exception class for Unsupported encodings version"""

    pass


class TensorType(Enum):
    """Enum class for Tensor Types"""

    ActivationEncodings = "activation_encodings"
    ParamEncodings = "param_encodings"
    Encodings = "encodings"


class TensorDtype(Enum):
    """Enum class for Tensor dtypes"""

    FLOAT = "float"
    SFXP = "int"
    UFXP = "uint"


class EncodingVersion(Enum):
    """Enum class for supported encodings version"""

    V0 = "0.6.0"
    V1 = "1.0.0"
    V2 = "2.0.0"


def get_encodings_version(encodings: dict) -> EncodingVersion:
    """Given encodings dictionary return encodings version
    Args:
        encodings (dict): encodings dictionary
    Returns:
        EncodingVersion: encoding version of encodings
    Raises:
        UnsupportedEncodingsVersionError: If version details could not be fetched for the
            given encodings
    """
    if (
        "activation_encodings" in encodings
        and "param_encodings" in encodings
        and isinstance(encodings["activation_encodings"], dict)
        and isinstance(encodings["param_encodings"], dict)
    ):
        return EncodingVersion.V0
    if "version" in encodings:
        return EncodingVersion(encodings["version"])

    raise UnsupportedEncodingsVersionError(
        "Could not fetch version details for the provided encodings"
    )
