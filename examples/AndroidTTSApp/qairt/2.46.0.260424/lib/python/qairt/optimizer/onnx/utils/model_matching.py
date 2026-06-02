# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Functions to match two models
"""

import copy
from typing import Dict, List

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.utils.utils import get_constant_np, is_constant


def match_tensor(
    src_v: ir.Value, dst_v: ir.Value, check_extra_info: bool = True, check_constants: bool = True
):
    """
    Match ir.Value, with or without extra info
    """
    # pylint: disable=too-many-return-statements
    # check tensor type
    if src_v.type != dst_v.type:
        return False

    # check tensor shape
    if src_v.shape != dst_v.shape:
        return False

    # check tensor value
    if check_extra_info:
        if src_v.meta["extra_info"] != dst_v.meta["extra_info"]:
            return False

    src_v_is_cst = is_constant(src_v)
    dst_v_is_cst = is_constant(dst_v)

    if not src_v_is_cst and not dst_v_is_cst:
        return True
    if src_v_is_cst and not dst_v_is_cst:
        return False
    if not src_v_is_cst and dst_v_is_cst:
        return False

    # src_v_is_cst and dst_v_is_cst
    if check_constants:
        src_v_cst = get_constant_np(src_v)
        dst_v_cst = get_constant_np(dst_v)
        if isinstance(src_v_cst, np.ndarray) and isinstance(dst_v_cst, np.ndarray):
            if not (src_v_cst == dst_v_cst).all():
                return False
    return True


def match_node(src_n: ir.Node | None, dst_n: ir.Node | None, check_constants: bool = True):
    """
    Match ir.Node
    """
    # pylint: disable=too-many-return-statements
    if src_n is None and dst_n is None:
        return True
    if src_n is None or dst_n is None:
        return False
    assert src_n is not None  # satisfy mypy
    assert dst_n is not None
    if src_n.domain != dst_n.domain:
        return False
    if src_n.op_type != dst_n.op_type:
        return False
    # compare attribute (directly call __eq__ on ir.Attr is not stable)
    src_attribute_proto = {k: ir.serde.serialize_attribute(v) for k, v in src_n.attributes.items()}
    dst_attribute_proto = {k: ir.serde.serialize_attribute(v) for k, v in dst_n.attributes.items()}
    if src_n.op_type == "Constant":
        if check_constants:
            src_np_data = get_constant_np(src_n.outputs[0])
            dst_np_data = get_constant_np(dst_n.outputs[0])
            if not (src_np_data == dst_np_data).all():
                return False

        del src_attribute_proto[
            "value"
        ]  # compare tensor is not stable (tensor can be expressed by raw or raw_data or external data)
        del dst_attribute_proto["value"]

    if src_attribute_proto != dst_attribute_proto:
        return False
    if len(src_n.inputs) != len(dst_n.inputs):
        return False
    if len(src_n.outputs) != len(dst_n.outputs):
        return False
    return True


def match_graph(
    src_graph: ir.Graph,
    dst_graph: ir.Graph,
    check_extra_info: bool = True,
    match_cst_as_initiliazer: bool = True,
    check_constants: bool = False,
):
    """
    Match src graph and dst graph
    Args:
        src_graph: src graph
        dst_graph: dst graph
        check_extra_info: whether to check extra info consistency
        match_cst_as_intitializer: whether to treat constant as initializer
    Returns:
        match_status: (bool) matched or not
        match_info: (dict) match information, contains tensor mapping between two graph

    Note:
        since onnxscript.script cannot express initializer,
        so we treat constant as initializer in this function if match_cst_as_initiliazer=True
    """
    # pylint: disable=too-many-return-statements, too-many-branches

    src2dst_tensor_map: Dict[str, ir.Value] = {}  # key: name of tensors in src_graph
    # value: value in dst_graph
    dst2src_tensor_map: Dict[str, ir.Value] = {}  # key: name of tensors in dst_graph
    # value: value in src_graph

    def add_tensor_mapping(src_v: ir.Value, dst_v: ir.Value):
        assert src_v.name is not None
        assert dst_v.name is not None
        src2dst_tensor_map[src_v.name] = dst_v
        dst2src_tensor_map[dst_v.name] = src_v

    # check op set
    if src_graph.opset_imports != src_graph.opset_imports:
        return False, src2dst_tensor_map

    match_info = {
        "src2dst_tensor_map": src2dst_tensor_map,
        "dst2src_tensor_map": dst2src_tensor_map,
    }

    # map inputs
    if len(src_graph.inputs) != len(dst_graph.inputs):
        return False, match_info
    for src_v, dst_v in zip(src_graph.inputs, dst_graph.inputs):
        if not match_tensor(src_v, dst_v, check_extra_info):
            return False, match_info
        add_tensor_mapping(src_v, dst_v)

    # map outputs
    if len(src_graph.outputs) != len(dst_graph.outputs):
        return False, match_info
    for src_v, dst_v in zip(src_graph.outputs, dst_graph.outputs):
        if not match_tensor(src_v, dst_v, check_extra_info):
            return False, match_info
        add_tensor_mapping(src_v, dst_v)

    # start from output, map all tensors in the graph
    check_src_tensors: List[ir.Value] = []
    check_src_tensors += src_graph.outputs[:]

    while len(check_src_tensors) > 0:
        src_v = check_src_tensors.pop()
        if src_v.name not in src2dst_tensor_map:
            return False, match_info
        dst_v = src2dst_tensor_map[src_v.name]

        src_v_producer = src_v.producer()
        dst_v_producer = dst_v.producer()

        if src_v_producer is None and dst_v_producer is None:
            # constant/input
            continue
        if not match_node(src_v_producer, dst_v_producer, check_constants=check_constants):
            if match_cst_as_initiliazer:
                if (
                    src_v_producer is None
                    and dst_v_producer is not None
                    and dst_v_producer.op_type == "Constant"
                ):
                    continue
                if (
                    dst_v_producer is None
                    and src_v_producer is not None
                    and src_v_producer.op_type == "Constant"
                ):
                    continue
            return False, match_info

        assert src_v_producer is not None  # satisfy mypy
        assert dst_v_producer is not None

        for src_prev_v, dst_prev_v in zip(src_v_producer.inputs, dst_v_producer.inputs):
            if src_prev_v is None and dst_prev_v is None:
                continue
            if src_prev_v is None or dst_prev_v is None:
                return False, match_info  # only one of them is None
            if src_prev_v.name in src2dst_tensor_map:
                if check_constants:
                    if is_constant(src_prev_v) and is_constant(dst_prev_v):
                        if (get_constant_np(src_prev_v) == get_constant_np(dst_prev_v)).all():
                            continue
                if src2dst_tensor_map[src_prev_v.name] is not dst_prev_v:
                    return False, match_info
            else:
                if not match_tensor(
                    src_prev_v, dst_prev_v, check_extra_info=check_extra_info, check_constants=check_constants
                ):
                    return False, match_info

                add_tensor_mapping(src_prev_v, dst_prev_v)
                assert src_prev_v is not None  # satisfy mypy
                check_src_tensors.append(src_prev_v)

    return True, match_info


def match_model(
    src_model: GraphContext,
    dst_model: GraphContext,
    check_extra_info: bool = True,
    match_cst_as_initiliazer: bool = True,
    check_constants: bool = False,
):
    """
    Match src model and dst model
    Args:
        src_model: src model
        dst_model: dst model
        check_extra_info: whether to check extra info consistency
        match_cst_as_intitializer: whether to treat constant as initializer
    Returns:
        match_status: (bool) matched or not
        match_info: (dict) match information, contains tensor mapping between two models

    Note:
        since onnxscript.script cannot express initializer,
        so we treat constant as initializer in this function if match_cst_as_initiliazer=True
    """
    return match_graph(
        src_model.graph_ir, dst_model.graph_ir, check_extra_info, match_cst_as_initiliazer, check_constants
    )


def match_model_with_functions(
    src_model: GraphContext,
    dst_model: GraphContext,
    check_extra_info: bool = True,
    match_cst_as_initiliazer: bool = True,
    check_constants: bool = False,
    match_function_list: list[str] | None = None,
):
    """
    Match src model and dst model
    Args:
        src_model: src model
        dst_model: dst model
        check_extra_info: whether to check extra info consistency
        match_cst_as_intitializer: whether to treat constant as initializer
    Returns:
        match_status: (bool) matched or not
        match_info: (dict) match information, contains tensor mapping between two models

    Note:
        since onnxscript.script cannot express initializer,
        so we treat constant as initializer in this function if match_cst_as_initiliazer=True
    """
    if match_function_list is None:
        match_function_list_set = set(x.name for x in src_model.model_ir.functions.values())
        match_function_list_set.update(set(x.name for x in dst_model.model_ir.functions.values()))
        match_function_list = list(match_function_list_set)

    success = True
    match_info: dict[str, dict[str, ir.Value]] = {}
    for func_name in match_function_list:
        src_func = None
        dst_func = None
        for func in src_model.model_ir.functions.values():
            if func.name == func_name:
                src_func = func
                break
        for func in dst_model.model_ir.functions.values():
            if func.name == func_name:
                dst_func = func
                break
        if src_func is None or dst_func is None:
            return False, match_info

        curr_success, curr_match_info = match_graph(
            src_func.graph, dst_func.graph, check_extra_info, match_cst_as_initiliazer, check_constants
        )
        if not curr_success:
            return False, match_info
        match_info[func_name] = curr_match_info

    curr_success, curr_match_info = match_graph(
        src_model.graph_ir, dst_model.graph_ir, check_extra_info, match_cst_as_initiliazer, check_constants
    )
    success = curr_success and success
    match_info["main_graph"] = curr_match_info
    return success, match_info
