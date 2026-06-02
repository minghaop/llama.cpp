# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import abc

import onnx
from onnxscript.values import Opset


def qcom_block_op_domain() -> str:
    """Returns the QTI BlockOps domain."""
    return "qti_aisw"


class QnnOnnxBlockOp(metaclass=abc.ABCMeta):
    """Abstract base for materializing a single Block Op in ONNX.

    Subclasses must set the following instance attributes BEFORE calling
    the base constructor:

        - self.name: str
        - self.min_opset: int   # minimum ONNX opset supported
        - self.max_opset: int   # maximum ONNX opset supported

    The base constructor validates the requested ONNX opset version and initializes:
        - self.opset: Opset("", onnx_opset_version)
        - self.aisw_opset: Opset(qcom_block_op_domain(), aisw_opset_version)
    """

    # These are initialized by derived classes before super().__init__
    name: str
    min_opset: int
    max_opset: int

    # Set by base constructor
    opset: Opset
    aisw_opset: Opset

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1) -> None:
        """Initialize the base BlockOp.

        Args:
            onnx_opset_version: ONNX default domain opset to materialize against.
            aisw_opset_version: AISW (qti_aisw) domain opset version. Defaults to 1.

        Raises:
            ValueError: If the requested ONNX opset is outside the supported range.
        """
        if onnx_opset_version < self.min_opset or onnx_opset_version > self.max_opset:
            raise ValueError(
                f"BlockOp {self.name} only supports ONNX opset versions between "
                f"{self.min_opset} and {self.max_opset}. "
                f"Requested: {onnx_opset_version}."
            )
        # Default (empty) domain is the standard ONNX operator set
        self.opset = Opset("", onnx_opset_version)
        # QTI BlockOp domain
        self.aisw_opset = Opset(qcom_block_op_domain(), aisw_opset_version)

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        """Return an ONNX FunctionProto for this Block Op.

        All BlockOps MUST implement this, regardless of their ability to
        return an onnxscript function.

        Returns:
            onnx.FunctionProto: The function proto representing this Block Op.

        Raises:
            NotImplementedError: If the derived class does not implement it.
        """
        raise NotImplementedError
