# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module handles MoE transformation
"""

import copy
import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.block_ops.elementwise_mux import add_element_wise_mux_op
from qairt.optimizer.onnx.graph import (
    GraphContext,
    SubGraphDesc,
    get_subgraph_full_func_name,
    inline_function,
    parse_subgraph_full_func_name,
)
from qairt.optimizer.onnx.passes.adaptations.adapt_moe.extract_moe_components import (
    MOE_AGGREGATION_FUNC_NAME,
    MOE_EXPERT_FACTOR_FUNC_NAME,
    MOE_EXPERT_FUNC_NAME,
    MOE_EXPERT_OP_PRED_FUNC_NAME,
    MOE_FUNC_DOMAIN_NAME,
    MOE_ROUTER_FUNC_NAME,
)
from qairt.optimizer.onnx.passes.base import BasePass, BasePredicatePass, MatchInfoProtocol, PassConfig
from qairt.optimizer.onnx.passes.cleaning import DeadCodeRemovalRewriter, DeadWeightRemovalRewriter
from qairt.optimizer.onnx.utils.binop_binary_tree_utils import build_balanced_binary_tree
from qairt.optimizer.onnx.utils.encodings import EncType, TensorEncodingInfo
from qairt.optimizer.onnx.utils.ir_extra_info import (
    VariableExtraInfo,
    get_expert_encset_name,
    is_expert_encset_name,
)
from qairt.optimizer.onnx.utils.model_matching import match_graph
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    copy_value,
    get_constant_np,
    get_constant_tensor_proto,
    get_unique_name,
    get_value_numeric_shape,
    hash_values,
    is_constant,
    iter_all_values,
    join_name,
    make_constant_node,
    make_initializer_with_namehint,
    safe_insert_node_after,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger

MOE_DYN_EXPERT_FUNC_NAME = "MoE_DynExpert"
MOE_DYN_FACTOR_FUNC_NAME = "MoE_DynFactor"
MOE_DYN_OP_PRED_FUNC_NAME = "MoE_DynOpPred"

import os

FAKE_PER_TENSOR_ENCODINGS_FOR_MUX_OUT = os.getenv(
    "ONNX_G2G_FAKE_PER_TENSOR_ENCODINGS_FOR_MUX_OUT", "True"
).lower() in ("true", "1")


def copy_value_as_constant(origin_v: ir.Value, graph_copy_to: ir.Graph):
    # origin_v should be static (initializer/constant/identity of constant)
    # create Constant nodes for the expert constant, as share graph can't have initializers
    cst_proto = get_constant_tensor_proto(origin_v)
    assert cst_proto is not None
    cst_node = make_constant_node(graph_copy_to, origin_v.name, cst_proto)
    safe_insert_node_after(graph_copy_to, None, cst_node)

    # also copy the encodings from dst_v to cst_node.outputs[0]
    cst_node.outputs[0].meta["extra_info"] = origin_v.meta["extra_info"].copy()
    return cst_node.outputs[0]


class AdaptMoEComponents(BasePredicatePass):
    @dataclass
    class Config(PassConfig):
        overridden_subselection: int | None = None
        remove_op_predicate: bool = False

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        pass

    @classmethod
    def _get_template_expert_id(cls):
        return 1

    def _get_expert_slot_num(self, router_subgraph: ir.Graph):
        # for both AR1 and ARN graph, the expert slot is determined by the first top-k
        activate_top_k_indices = router_subgraph.outputs[1]
        first_top_k = activate_top_k_indices.producer()
        assert first_top_k is not None  # check for mypy
        assert first_top_k.op_type == "TopK"
        slot_num = int(get_constant_np(first_top_k.inputs[1]).item())
        assert isinstance(self.config, self.Config)
        if self.config.overridden_subselection is not None:
            assert self.config.overridden_subselection < slot_num
            return self.config.overridden_subselection
        return slot_num

    @classmethod
    def _simplify_expert_encodings(cls, extra_info: VariableExtraInfo, expert_num: int):
        for encset_name in list(extra_info.named_encodings.keys()):
            if is_expert_encset_name(encset_name):
                continue
            base_enc = extra_info.named_encodings[encset_name]
            all_same = True
            for expert_id in range(0, expert_num):
                expert_encset_name = get_expert_encset_name(encset_name, expert_id)
                if expert_encset_name not in extra_info.named_encodings:
                    all_same = False
                    break
                expert_enc = extra_info.named_encodings[expert_encset_name]

                if expert_enc != base_enc:
                    all_same = False
                    break

            if all_same:
                for expert_id in range(expert_num):
                    del extra_info.named_encodings[get_expert_encset_name(encset_name, expert_id)]

    @classmethod
    def _create_dynamic_weights_functions(
        cls, model_ir: ir.Model, caller_map: dict[int, ir.Node], dyn_op_type: str
    ):
        assert len(caller_map) >= 2

        expert_id_list = list(caller_map.keys())
        expert_id_list.sort()
        caller_list = [caller_map[x] for x in expert_id_list]

        # simplify all the constants connection in the graph (remove the reuse of constants)
        # for example: for expert_id == 0, in the op-pred subgraph, the constant "0" are reused for "Greator" and "Equal"
        for expert_id in expert_id_list:
            func = model_ir.functions[caller_map[expert_id].op_identifier()]
            for v in iter_all_values(func.graph):
                if is_constant(v):
                    if len(v.uses()) == 1 and (v.producer().op_type == "Constant"):
                        # Constant op and used by only one user
                        continue
                    else:
                        # used by multiple users or static tensor but not constant (like identity of initializer or constant)
                        # then we just make a new constant node to make the graph simpler to analyse
                        for use in list(v.uses()):
                            new_value = copy_value_as_constant(v, func.graph)
                            use.node.replace_input_with(use.idx, new_value)

            DeadCodeRemovalRewriter().apply(
                GraphContext(ir.Model(func.graph, ir_version=model_ir.ir_version), _skip_shape_infer=True)
            )

        # use the second graph (expert_id = 1) as template
        template_expert_id = cls._get_template_expert_id()
        template_graph = model_ir.functions[
            caller_map[expert_id_list[template_expert_id]].op_identifier()
        ].graph
        share_graph_desc = SubGraphDesc(
            template_graph,
            list(template_graph.inputs),
            list(template_graph.outputs),
            list(template_graph),
            allow_external_user=True,
        )
        share_graph = share_graph_desc.copy_as_graph("dyn_expert")

        all_possible_cst_v: dict[str, dict[int, ir.Value]] = {}  # key: value name in share graph
        # value: all possible ir.Values from expert-0 to expert-n, these values should belongs to share_graph
        for v in iter_all_values(share_graph):
            if is_constant(v):
                all_possible_cst_v[v.name] = {template_expert_id: v}
            else:
                # copy encodings as encset name "$MoE${base_encset_name}${template_expert_id}",
                # to align encset name with other experts
                v_extra_info: VariableExtraInfo = v.meta["extra_info"]
                for encset_name in list(v_extra_info.named_encodings.keys()):
                    if is_expert_encset_name(encset_name):
                        continue
                    v_extra_info.named_encodings[get_expert_encset_name(encset_name, template_expert_id)] = (
                        copy.deepcopy(v_extra_info.named_encodings[encset_name])
                    )

        for expert_id in expert_id_list[:template_expert_id] + expert_id_list[template_expert_id + 1 :]:
            caller = caller_map[expert_id]
            func = model_ir.functions[caller.op_identifier()]
            success, match_info = match_graph(
                share_graph, func.graph, check_constants=False, check_extra_info=False
            )
            if not success:
                return None

            for src_v_name, dst_v in match_info["src2dst_tensor_map"].items():
                src_v = match_info["dst2src_tensor_map"][dst_v.name]
                src_v_extra_info: VariableExtraInfo = src_v.meta["extra_info"]
                dst_v_extra_info: VariableExtraInfo = dst_v.meta["extra_info"]
                if src_v_name in all_possible_cst_v:
                    # create Constant nodes for the expert constant, as share graph can't have initializers
                    new_value = copy_value_as_constant(dst_v, share_graph)
                    all_possible_cst_v[src_v_name][expert_id] = new_value
                else:
                    # for activation (non-constant), we only collect the encodings of them
                    for encset_name, dst_enc in dst_v_extra_info.named_encodings.items():
                        if is_expert_encset_name(encset_name):
                            continue
                        src_v_extra_info.named_encodings[get_expert_encset_name(encset_name, expert_id)] = (
                            copy.deepcopy(dst_enc)
                        )

        # remove expert specific encodings if all the experts have same encodings
        for v in iter_all_values(share_graph):
            if not is_constant(v):
                cls._simplify_expert_encodings(v.meta["extra_info"], len(expert_id_list))

        # add another input for dynamic_expert_id
        dynamic_expert_id = ir.Value(
            None,
            name=get_unique_name(share_graph, "dynamic_expert_id"),
            shape=ir.Shape(
                [
                    1,
                ]
            ),
            type=ir.TensorType(ir.DataType.INT64),
        )
        dynamic_expert_id.meta["extra_info"] = VariableExtraInfo()
        share_graph.inputs.append(dynamic_expert_id)

        # change constant var into ElementWiseMux
        share_graph.opset_imports["qti_aisw"] = 1
        for name, v_map in all_possible_cst_v.items():
            expert_id_list = list(v_map.keys())
            expert_id_list.sort()
            all_possible_values = [v_map[x] for x in expert_id_list]

            # for small tensors, compare their values (small tensors should be already loaded even originally come from external data)
            v_np_map = {i: get_constant_np(x, load_external_data=False) for i, x in v_map.items()}
            small_cst = all(x is not None for x in v_np_map.values())
            if small_cst and all(
                (x == v_np_map[0]).all() and x.shape == v_np_map[0].shape for x in v_np_map.values()
            ):
                # same constant value, so we don't have to make it dynamic
                continue
            else:
                elemux = ir.Node(
                    domain="qti_aisw",
                    op_type="ElementWiseMux",
                    inputs=[
                        dynamic_expert_id,
                        *all_possible_values,
                    ],
                    name=get_unique_name(share_graph, f"{all_possible_values[0].name}/mux"),
                )
                elemux.outputs[0].name = get_unique_name(
                    share_graph, f"{v_map[template_expert_id].name}/mux_out"
                )
                elemux.outputs[0].shape = copy.deepcopy(v_map[template_expert_id].shape)
                dtype = v_map[template_expert_id].dtype
                if dtype is not None:
                    elemux.outputs[0].dtype = copy.deepcopy(dtype)

                elemux.outputs[0].meta["extra_info"] = v_map[template_expert_id].meta["extra_info"].copy()

                if FAKE_PER_TENSOR_ENCODINGS_FOR_MUX_OUT:
                    for encset_name, origin_enc in (
                        elemux.outputs[0].meta["extra_info"].named_encodings.items()
                    ):
                        assert isinstance(origin_enc, TensorEncodingInfo)
                        if is_expert_encset_name(encset_name):
                            continue
                        # backend requires the output encodings of Mux to be PerTensor or MultiQuant
                        # since other tools dosen't support MultiQuant for now, so we replace encodings
                        # with fake per tensor encodings (offset = 0, scale = 1)
                        fake_enc = TensorEncodingInfo(
                            EncType.PER_TENSOR,
                            bw=8,  # maybe 16 ?
                            dtype="INT",
                            is_sym=True,
                            offset=np.array(0),
                            scale=np.array(1),
                            max=None,
                            min=None,
                            enc_kind=origin_enc.enc_kind,
                        )
                        elemux.outputs[0].meta["extra_info"].named_encodings[encset_name] = fake_enc

                for encset_name in v_map[template_expert_id].meta["extra_info"].named_encodings.keys():
                    if is_expert_encset_name(encset_name):
                        continue
                    for expert_id in range(len(all_possible_values)):
                        expert_i_named_encodings = (
                            all_possible_values[expert_id].meta["extra_info"].named_encodings
                        )
                        assert encset_name in expert_i_named_encodings, (
                            f"tensor {v_map[template_expert_id].name} of expert {template_expert_id} has encodings, "
                            f"but its corresponding tensor {all_possible_values[expert_id].name} of expert {expert_id} has not"
                        )

                        elemux.outputs[0].meta["extra_info"].named_encodings[
                            get_expert_encset_name(encset_name, expert_id)
                        ] = copy.deepcopy(expert_i_named_encodings[encset_name])

                v_producer = v_map[template_expert_id].producer()
                assert v_producer is not None  # check for mypy
                share_graph.insert_after(v_producer, elemux)
                safe_replace_all_uses_with(
                    share_graph, v_map[template_expert_id], elemux.outputs[0], [elemux]
                )

        # make share_graph as function, and clean it
        shape_graph_ctx = GraphContext(
            ir.Model(share_graph, ir_version=model_ir.ir_version), _skip_shape_infer=True
        )
        RemoveNullElementWiseMux().apply(shape_graph_ctx)
        DeadCodeRemovalRewriter().apply(shape_graph_ctx)

        full_func_name = get_subgraph_full_func_name(dyn_op_type, hash_values(share_graph.outputs)[-16:])
        share_graph.name = full_func_name + ".graph"
        dyn_func = ir.Function(
            MOE_FUNC_DOMAIN_NAME,
            full_func_name,
            caller_list[0].overload,
            graph=share_graph,
            attributes=[],
        )
        model_ir.functions[dyn_func.identifier()] = dyn_func
        return dyn_func

    def _make_dynamic_expert_id_list(
        self,
        model_ir: ir.Model,
        router_caller: ir.Node,
    ) -> list[ir.Value]:
        router_subgraph = model_ir.functions[router_caller.op_identifier()].graph
        slot_num = self._get_expert_slot_num(router_subgraph)

        main_graph = router_caller.graph
        assert main_graph is not None

        # router output is in shape [1, k]
        # we need to firstly reshape to [k]
        topk_indices_v = router_caller.outputs[1]
        reshape_shape = [
            list(
                itertools.accumulate(
                    get_value_numeric_shape(topk_indices_v), func=lambda a, b: a * b, initial=1
                )
            )[-1]
        ]
        reshape_op = ir.Node(
            "",
            "Reshape",
            [
                topk_indices_v,
                make_initializer_with_namehint(
                    main_graph, join_name(topk_indices_v.name, "reshape/shape"), np.array(reshape_shape)
                ),
            ],
            name=get_unique_name(main_graph, join_name(topk_indices_v.name, "reshape")),
        )
        reshape_op_output = reshape_op.outputs[0]
        reshape_op_output.name = get_unique_name(main_graph, join_name(topk_indices_v.name, "reshape_out"))
        reshape_op_output.shape = ir.Shape(reshape_shape)
        if topk_indices_v.dtype is not None:
            reshape_op_output.dtype = topk_indices_v.dtype
        reshape_op_output.meta["extra_info"] = topk_indices_v.meta["extra_info"].copy()
        main_graph.insert_after(router_caller, reshape_op)

        # create dynamic expert id
        dynamic_expert_id_list = []
        for slot_id in range(slot_num):
            dynamic_expert_id_gather = ir.Node(
                "",
                "Gather",
                [
                    reshape_op_output,
                    make_initializer_with_namehint(
                        main_graph,
                        "expert_id_gather",
                        np.array([slot_id]),
                    ),
                ],
                outputs=[
                    ir.Value(
                        None,
                        name=get_unique_name(main_graph, "expert_id"),
                        shape=ir.Shape(
                            [
                                1,
                            ]
                        ),
                        type=ir.TensorType(ir.DataType.INT64),
                    )
                ],
            )
            dynamic_expert_id = dynamic_expert_id_gather.outputs[0]
            dynamic_expert_id.meta["extra_info"] = VariableExtraInfo()

            main_graph.insert_after(reshape_op, dynamic_expert_id_gather)
            dynamic_expert_id_list.append(dynamic_expert_id_gather.outputs[0])
        return dynamic_expert_id_list

    def match(self, graph, node):
        # match aggregation subgraph
        if (
            node.domain != MOE_FUNC_DOMAIN_NAME
            or parse_subgraph_full_func_name(node.op_type)[0] != MOE_AGGREGATION_FUNC_NAME
        ):
            return False
        return self.MatchInfo()

    def _get_subcomponent_callers(self, aggregation_caller: ir.Node):
        expert_caller_map: dict[int, ir.Node] = {}
        op_pred_caller_map: dict[int, ir.Node] = {}
        factor_caller_map: dict[int, ir.Node] = {}
        router_caller = None
        expert_num = aggregation_caller.attributes["expert_num"].as_int()
        op_predicate = aggregation_caller.attributes["op_predicate"].as_int()
        for i in range(0, expert_num):
            if op_predicate:
                op_pred_out = aggregation_caller.inputs[i + expert_num]
                assert op_pred_out is not None
                op_pred_caller = op_pred_out.producer()
                assert op_pred_caller is not None
                assert op_pred_caller.domain == MOE_FUNC_DOMAIN_NAME
                assert (
                    parse_subgraph_full_func_name(op_pred_caller.op_type)[0] == MOE_EXPERT_OP_PRED_FUNC_NAME
                )
                op_pred_caller_map[op_pred_caller.attributes["expert_id"].as_int()] = op_pred_caller

            expert_out = aggregation_caller.inputs[i]
            assert expert_out is not None

            expert_caller = expert_out.producer()
            assert expert_caller is not None
            assert expert_caller.domain == MOE_FUNC_DOMAIN_NAME
            assert parse_subgraph_full_func_name(expert_caller.op_type)[0] == MOE_EXPERT_FUNC_NAME
            expert_caller_map[expert_caller.attributes["expert_id"].as_int()] = expert_caller

            expert_factor = expert_caller.inputs[1]
            assert expert_factor is not None
            factor_caller = expert_factor.producer()
            assert factor_caller is not None
            assert factor_caller.domain == MOE_FUNC_DOMAIN_NAME
            assert parse_subgraph_full_func_name(factor_caller.op_type)[0] == MOE_EXPERT_FACTOR_FUNC_NAME
            factor_caller_map[factor_caller.attributes["expert_id"].as_int()] = factor_caller

            factors_v = factor_caller.inputs[0]
            assert factors_v is not None
            if router_caller is None:
                router_caller = factors_v.producer()
                assert router_caller is not None
                assert parse_subgraph_full_func_name(router_caller.op_type)[0] == MOE_ROUTER_FUNC_NAME
            else:
                assert router_caller is factors_v.producer()
        return {
            "expert_caller_map": expert_caller_map,
            "factor_caller_map": factor_caller_map,
            "op_pred_caller_map": op_pred_caller_map,
            "router_caller": router_caller,
        }

    def _infer_aggregation_block_output_encodings(
        cls, model_ir: ir.Model, origin_aggregation_caller: ir.Node
    ):
        if origin_aggregation_caller.outputs[0].meta["extra_info"].defined_encodings():
            return

        if origin_aggregation_caller.attributes["op_predicate"].as_int():
            # output encodings is not set, we need to infer it from the nodes of block

            # for ARN with op-predicate, the output encodings of the origin aggregation should be same as the
            # - last "Where" output encodings
            # - last "Add" output encodings
            subgraph = model_ir.functions[origin_aggregation_caller.op_identifier()].graph
            last_where_out = subgraph.outputs[0]
            last_where = last_where_out.producer()
            assert last_where is not None and last_where.op_type == "Where"
            last_add_out = last_where.inputs[1]
            assert last_add_out is not None
            origin_aggregation_caller.outputs[0].meta["extra_info"].merge(
                last_add_out.meta["extra_info"], encodings_only=True
            )
            origin_aggregation_caller.outputs[0].meta["extra_info"].merge(
                last_where_out.meta["extra_info"], encodings_only=True
            )

    def create_dyn_expert_factors(
        self,
        template_factor_expert: ir.Node,
        dyn_factor_func: ir.Function,
        router_caller: ir.Node,
        slot_num: int,
        dyn_expert_id_list: list[ir.Value],
    ) -> list[ir.Value]:
        main_graph = router_caller.graph
        assert main_graph is not None

        factor_shape = get_value_numeric_shape(dyn_factor_func.outputs[0])
        factor_numel = list(itertools.accumulate(factor_shape, lambda a, b: a * b, initial=1))[-1]
        if factor_numel > 1:
            expert_factor_list = []
            for slot_i, dyn_expert_i in enumerate(dyn_expert_id_list):
                inputs = list(template_factor_expert.inputs) + [dyn_expert_i]
                dyn_expert_caller = ir.Node(
                    dyn_factor_func.domain,
                    dyn_factor_func.name,
                    inputs=inputs,
                    outputs=[copy_value(main_graph, x, x.name) for x in dyn_factor_func.outputs],
                    overload=dyn_factor_func.overload,
                )
                main_graph.insert_after(template_factor_expert, dyn_expert_caller)
                expert_factor_list.append(dyn_expert_caller.outputs[0])
            return expert_factor_list

        else:
            # for AR1, factor can be gotten directly from topk results,
            # and don't have to compute the ScatterElements.
            # assum topk results are connected to ScatterElements
            normalized_topk_factors = router_caller.outputs[2]
            shape = get_value_numeric_shape(normalized_topk_factors)

            gather_axis: None | int = None
            for axis in range(len(shape)):
                if shape[axis] > 1:
                    gather_axis = axis
                    break
            assert gather_axis is not None

            expert_factor_list = []
            for slot_i in range(slot_num):
                gather_op = ir.Node(
                    "",
                    "Gather",
                    [
                        normalized_topk_factors,
                        make_initializer_with_namehint(main_graph, "indices", [slot_i]),
                    ],
                    name=get_unique_name(main_graph, f"slot_{slot_i}_factor_gather"),
                    attributes=[ir.AttrInt64("axis", gather_axis)],
                )
                assert isinstance(gather_op.name, str)  # check for mypy
                gather_op.outputs[0].name = get_unique_name(main_graph, gather_op.name + "/outputs")
                gather_op.outputs[0].meta["extra_info"] = VariableExtraInfo()
                main_graph.insert_after(router_caller, gather_op)
                expert_factor_list.append(gather_op.outputs[0])
            return expert_factor_list

    def rewrite(self, graph, node, match_info=None):
        assert isinstance(self.config, self.Config)
        model_ir = match_info.extra_info.model_ir
        aggregation_caller = node
        self._infer_aggregation_block_output_encodings(model_ir, aggregation_caller)

        callers = self._get_subcomponent_callers(aggregation_caller)
        expert_caller_map: dict[int, ir.Node] = callers["expert_caller_map"]
        factor_caller_map: dict[int, ir.Node] = callers["factor_caller_map"]
        router_caller: ir.Node = callers["router_caller"]

        dyn_expert_func = self._create_dynamic_weights_functions(
            model_ir, expert_caller_map, dyn_op_type=MOE_DYN_EXPERT_FUNC_NAME
        )
        dyn_factor_func = self._create_dynamic_weights_functions(
            model_ir, factor_caller_map, dyn_op_type=MOE_DYN_FACTOR_FUNC_NAME
        )
        dyn_expert_id_list = self._make_dynamic_expert_id_list(model_ir, router_caller)

        # create dynamic expert slot (by creating the call node of dynamic expert function)
        main_graph = router_caller.graph
        assert main_graph is not None

        template_expert_id = self._get_template_expert_id()
        template_expert = expert_caller_map[template_expert_id]

        dyn_expert_caller_map: dict[int, ir.Node] = {}

        # add ElementWiseMux function to model
        add_element_wise_mux_op(model_ir, len(expert_caller_map), simplified_version=True)
        dyn_factor_list = self.create_dyn_expert_factors(
            template_factor_expert=factor_caller_map[template_expert_id],
            dyn_factor_func=dyn_factor_func,
            router_caller=router_caller,
            slot_num=len(dyn_expert_id_list),
            dyn_expert_id_list=dyn_expert_id_list,
        )
        for slot_i, dyn_expert_i in enumerate(dyn_expert_id_list):
            dyn_factor = dyn_factor_list[slot_i]
            inputs = list(template_expert.inputs[:-1]) + [dyn_factor, dyn_expert_i]
            dyn_expert_caller = ir.Node(
                dyn_expert_func.domain,
                dyn_expert_func.name,
                inputs=inputs,
                outputs=[copy_value(main_graph, x, x.name) for x in template_expert.outputs],
                overload=dyn_expert_func.overload,
            )
            dyn_expert_caller_map[slot_i] = dyn_expert_caller
        main_graph.insert_after(template_expert, list(dyn_expert_caller_map.values()))

        dyn_op_pred_caller_map: dict[int, ir.Node] = {}
        if callers["op_pred_caller_map"]:
            # for ARN only
            op_pred_caller_map: dict[int, ir.Node] = callers["op_pred_caller_map"]
            dyn_op_pred_func = self._create_dynamic_weights_functions(
                model_ir, op_pred_caller_map, dyn_op_type=MOE_DYN_OP_PRED_FUNC_NAME
            )
            assert dyn_op_pred_func is not None
            template_op_pred = op_pred_caller_map[template_expert_id]

            for slot_i, dyn_expert_i in enumerate(dyn_expert_id_list):
                dyn_op_pred_caller = ir.Node(
                    dyn_op_pred_func.domain,
                    dyn_op_pred_func.name,
                    inputs=[*template_op_pred.inputs, dyn_expert_i],
                    outputs=[copy_value(main_graph, x, x.name) for x in template_op_pred.outputs],
                    overload=dyn_op_pred_func.overload,
                )
                dyn_op_pred_caller_map[slot_i] = dyn_op_pred_caller
            main_graph.insert_after(dyn_expert_caller, list(dyn_op_pred_caller_map.values()))

        op_predicate = False
        if not self.config.remove_op_predicate and aggregation_caller.attributes["op_predicate"].as_int():
            op_predicate = True

        self._create_new_aggregation_block(
            match_info.extra_info.model_ir,
            aggregation_caller,
            dyn_expert_caller_map,
            dyn_op_pred_caller_map,
            op_predicate=op_predicate,
        )

        logger.info(
            "applied {} on {} (total_expert_num={}, dynamic_expert_num={})".format(
                self.get_curr_pass_name(),
                aggregation_caller.outputs[0],
                len(expert_caller_map),
                len(dyn_expert_id_list),
            )
        )
        return True

    @classmethod
    def _create_new_aggregation_block(
        cls,
        model_ir: ir.Model,
        origin_aggregation_caller: ir.Node,
        dyn_expert_caller_map: dict[int, ir.Node],
        dyn_op_pred_caller_map: dict[int, ir.Node],
        op_predicate: bool,
    ):
        # add new aggregation block directly into main graph
        main_graph = origin_aggregation_caller.graph
        assert main_graph is not None  # check for mypy

        slot_ids = list(dyn_expert_caller_map.keys())
        slot_ids.sort()

        dyn_expert_caller_list = [dyn_expert_caller_map[i] for i in slot_ids]

        expected_extra_info = origin_aggregation_caller.outputs[0].meta["extra_info"].copy()

        values_to_aggregate: list[ir.Value] = []
        nodes_to_insert: list[ir.Node] = []

        if op_predicate:
            zero_shape = get_value_numeric_shape(origin_aggregation_caller.inputs[1])
            zero_cst_node = make_constant_node(main_graph, "zero", np.zeros(zero_shape, dtype=np.float32))
            nodes_to_insert.append(zero_cst_node)

            for i in slot_ids:
                expert_cond = dyn_op_pred_caller_map[i].outputs[0]
                expert_weighted_out = dyn_expert_caller_map[i].outputs[0]

                where_node = ir.Node(
                    "",
                    "Where",
                    [expert_cond, expert_weighted_out, zero_cst_node.outputs[0]],
                    name=get_unique_name(main_graph, join_name(expert_cond.name, "/op_pred_where")),
                )
                where_node.outputs[0].name = get_unique_name(
                    main_graph, join_name(expert_cond.name, "/op_pred_where/out")
                )

                # copy the encodings
                where_node.outputs[0].meta["extra_info"] = expert_weighted_out.meta["extra_info"].copy()
                where_node.outputs[0].shape = copy.deepcopy(expert_weighted_out.shape)
                if expert_weighted_out.dtype:
                    where_node.outputs[0].dtype = expert_weighted_out.dtype
                values_to_aggregate.append(where_node.outputs[0])
                nodes_to_insert.append(where_node)
        else:
            values_to_aggregate = [n.outputs[0] for n in dyn_expert_caller_list]
            nodes_to_insert = []

        # create aggregation add nodes
        tree = build_balanced_binary_tree(
            main_graph,
            values_to_aggregate,
            "Add",
            "",
            extra_info=expected_extra_info,
        )
        assert tree is not None
        main_graph.insert_before(origin_aggregation_caller, [*nodes_to_insert, *tree.new_nodes])
        safe_replace_all_uses_with(main_graph, origin_aggregation_caller.outputs[0], tree.output)
        return True


class InlineInternalFunctions(BasePass):
    def apply(self, ctx):
        function_names_to_inline = set(
            [
                MOE_DYN_EXPERT_FUNC_NAME,
                MOE_EXPERT_FUNC_NAME,
                MOE_DYN_OP_PRED_FUNC_NAME,
                MOE_EXPERT_OP_PRED_FUNC_NAME,
                MOE_DYN_FACTOR_FUNC_NAME,
                MOE_EXPERT_FACTOR_FUNC_NAME,
                MOE_ROUTER_FUNC_NAME,
                MOE_AGGREGATION_FUNC_NAME,
            ]
        )

        callers_to_inline: dict[ir.OperatorIdentifier, list[ir.Node]] = {}
        for n in list(ctx.graph_ir):
            if (
                n.domain == MOE_FUNC_DOMAIN_NAME
                and parse_subgraph_full_func_name(n.op_type)[0] in function_names_to_inline
            ):
                if n.op_identifier() not in callers_to_inline:
                    callers_to_inline[n.op_identifier()] = []
                callers_to_inline[n.op_identifier()].append(n)

        count = 1
        for caller_list in callers_to_inline.values():
            inline_function(ctx.model_ir, caller_list)
            count += len(caller_list)

        DeadCodeRemovalRewriter().apply(ctx)
        DeadWeightRemovalRewriter().apply(ctx)
        return count


class RemoveNullElementWiseMux(BasePredicatePass):
    def match(self, graph, node):
        if not (node.op_type == "ElementWiseMux" and node.domain == "qti_aisw"):
            return False

        check_static_shape_of_node_io(node)

        candidate_v_np_list = [get_constant_np(x, load_external_data=False) for x in node.inputs[1:]]

        if any(x is None for x in candidate_v_np_list):
            return False

        # candidate inputs of ElementWiseMux are all constants
        if all(x.size == 1 and (x == i).all() for i, x in enumerate(candidate_v_np_list)):
            # constants are [0,1,2,3...],
            # so we just need to reshape the expert_id if requried
            return True

        return False

    def rewrite(self, graph, node, match_info=None):
        # constants are [0,1,2,3...],
        # so we just need to reshape the expert_id if requried
        expert_id_shape = get_value_numeric_shape(node.inputs[0])
        out_shape = get_value_numeric_shape(node.outputs[0])
        if expert_id_shape == out_shape:
            safe_replace_all_uses_with(graph, node.outputs[0], node.inputs[0])
        else:
            reshape_cst_node = make_constant_node(graph, "expert_id_reshape/shape", out_shape)
            reshape_node = ir.Node(
                "",
                "Reshape",
                [node.inputs[0], reshape_cst_node.outputs[0]],
                name=get_unique_name(graph, "expert_id_reshape"),
            )
            reshape_node.outputs[0].name = get_unique_name(graph, "expert_id_reshape/output")
            reshape_node.outputs[0].meta["extra_info"] = node.inputs[0].meta["extra_info"].copy()
            graph.insert_before(node, [reshape_cst_node, reshape_node])
            safe_replace_all_uses_with(graph, node.outputs[0], reshape_node.outputs[0])
        return True
