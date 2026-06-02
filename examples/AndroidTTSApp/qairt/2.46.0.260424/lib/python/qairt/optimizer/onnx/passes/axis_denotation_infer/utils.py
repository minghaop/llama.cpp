# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Utility functions for axis denotation inference and propagation"""

import onnx_ir as ir

from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation, VariableExtraInfo


def get_axis_denotations(value: ir.Value) -> list[AxisDenotation]:
    """Helper to get the axis denotations for a tensor"""
    try:
        return value.meta["extra_info"].axis_denotations
    except KeyError:
        return []


def set_axis_denotations(value: ir.Value, denotations: list[AxisDenotation]):
    """Helper to set the axis denotations for a tensor"""
    if "extra_info" not in value.meta:
        value.meta["extra_info"] = VariableExtraInfo()
    value.meta["extra_info"].axis_denotations = denotations
