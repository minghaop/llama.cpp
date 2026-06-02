# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides some basic rewriters for mha2sha ir modification
"""

import copy

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    BaseGroupSliceAttrs,
    GroupSliceAttrs,
    create_concat_node,
    get_gslice_attrs,
    get_slice_out_name,
    simplify_group_slice_attrs,
)
from qairt.optimizer.onnx.utils.utils import (
    clone_node_attribute,
    convert_attr_to_py,
    get_value_numeric_shape,
    safe_insert_node_after,
    safe_replace_all_uses_with,
)


class M2sBasePass(BasePredicatePass):
    """Base pass for MHA2SHA ir modification"""

    def create_simplified_gslice_node(
        self,  # pylint: disable=R0912,R0914
        graph: ir.Graph,
        value: ir.Value,
        gslice_attrs: GroupSliceAttrs,
        gslice_node_namehint=None,
        allow_reuse_other_gslice_node=True,
    ) -> tuple[ir.Node | None, list[ir.Value | None]]:
        """
        Simplify the gslice_attrs, and then create gslice ir.Node based on the simplified gslice_attrs

        Args:
            graph: ir.Graph instance
            value: value to apply groupslice
            gslice_attrs: group slice attributes
            gslice_node_namehint: namehint for the generated groupslice node
            allow_reuse_other_gslice_node: whether to use the already exsisted groupslice node
                                           rather than create a new one, if possible

        Returns:
            gslice_node: the created group slice node, None if reuse already exsited gslice
            outputs: the sliced values, None if the sliced tensor is empty (start=end)
        """
        # slices in gslice_attrs can have repeatation
        # this function will automatically handle this situation

        if allow_reuse_other_gslice_node:
            # check if there is already a gslice node with the same attrs
            gslice_out_cache = {}  # key (start, end)
            for user, _ in value.uses():
                if (
                    user.op_type == "GroupSlice"
                    and convert_attr_to_py(user.attributes["axis"], "as_int") == gslice_attrs.axis
                ):
                    for o_v, o_start_i, o_end_i in zip(
                        user.outputs,
                        convert_attr_to_py(user.attributes["starts"], "as_ints"),
                        convert_attr_to_py(user.attributes["ends"], "as_ints"),
                        strict=True,
                    ):
                        gslice_out_cache[(o_start_i, o_end_i)] = o_v

            outputs: list[ir.Value | None] = []
            all_use_cache = True
            for o_i in range(gslice_attrs.num_outputs()):
                if (gslice_attrs.starts[o_i], gslice_attrs.ends[o_i]) in gslice_out_cache:
                    outputs.append(gslice_out_cache[(gslice_attrs.starts[o_i], gslice_attrs.ends[o_i])])
                elif gslice_attrs.starts[o_i] == gslice_attrs.ends[o_i]:
                    outputs.append(None)
                else:
                    all_use_cache = False
                    break
            if all_use_cache:
                return None, outputs

        simplified_gslice_attrs, origin2simplified_idx_map = simplify_group_slice_attrs(gslice_attrs)

        # create gslice node
        gslice_node = self._create_groupslice_node(
            graph, value, simplified_gslice_attrs, gslice_node_namehint
        )

        outputs = []
        for i in range(gslice_attrs.num_outputs()):
            if gslice_attrs.starts[i] == gslice_attrs.ends[i]:
                outputs.append(None)
            else:
                outputs.append(gslice_node.outputs[origin2simplified_idx_map[i]])

        safe_insert_node_after(graph, value.producer(), gslice_node)

        return gslice_node, outputs

    def gslice_then_concat(
        self,
        graph: ir.Graph,
        value: ir.Value,
        gslice_attrs: GroupSliceAttrs,
        gslice_node_namehint=None,
        concat_node_namehint=None,
    ) -> tuple[ir.Value | None, ir.Node]:
        """
        Groupslice the given value, and then concat the results of the groupslice.
        Theoritically, if the group slice is complete,the output of the concat node can generally
        replace the original given value in the graph

        Args:
            graph: ir.Graph instance
            value: value to apply groupslice
            gslice_attrs: group slice attributes
            gslice_node_namehint: namehint for the generated groupslice node
            concat_node_namehint: namehint for the generated concat node

        Returns:
            recorvered_value: the output of the concat node, None if the group slice is not complete
            gslice_node: the created group slice node
        """
        # check for mypy
        assert value.name is not None
        assert value.shape is not None

        if gslice_node_namehint is None:
            gslice_node_namehint = value.name + "/gslice_node"
        if concat_node_namehint is None:
            concat_node_namehint = value.name + "/gconcat_node"
        gslice_node = self._create_groupslice_node(graph, value, gslice_attrs, gslice_node_namehint)

        filtered_gslice_outputs = [
            x for x in gslice_node.outputs if get_value_numeric_shape(x)[gslice_attrs.axis] > 0
        ]
        nodes_to_insert = [gslice_node]
        if len(filtered_gslice_outputs) > 1 and gslice_attrs.complete(value.shape[gslice_attrs.axis]):
            concat_node = create_concat_node(
                graph, filtered_gslice_outputs, gslice_attrs.axis, value.name + "/recover222", value.name
            )
            safe_replace_all_uses_with(graph, value, concat_node.outputs[0], except_users=(gslice_node,))

            # Provided by BasePredicatePass/BasePredicatePass through multiple inheritance
            self.mark_value_as_copy(graph, value, concat_node.outputs[0])  # type: ignore[attr-defined]

            nodes_to_insert.append(concat_node)
            recorvered_value = concat_node.outputs[0]
        elif len(filtered_gslice_outputs) == 1:
            recorvered_value = filtered_gslice_outputs[0]
        else:
            recorvered_value = None

        safe_insert_node_after(graph, value.producer(), nodes_to_insert)

        return recorvered_value, gslice_node

    def create_mini_inputs(
        self, graph: ir.Graph, input_v: ir.Value | None, input_gslice_attrs: BaseGroupSliceAttrs
    ) -> list[ir.Value | None]:
        """
        Create mini inputs of the speicifid value by applying group slice on it

        Args:
            graph: ir.Graph instance
            input_v: value to apply groupslice
            input_gslice_attrs: group slice attributes

        Returns:
            mini_inputs: the slice results of the speicifid value
        """
        # check if gslice is copy
        mini_inputs: list[ir.Value | None] = []
        if input_v is None:
            return [None] * input_gslice_attrs.num_outputs()
        if input_gslice_attrs.is_each_output_full(input_v):
            # FullGroupSliceAttrs must go here
            mini_inputs = [input_v] * input_gslice_attrs.num_outputs()
        else:
            assert isinstance(input_gslice_attrs, GroupSliceAttrs)  # check for mypy, definitely true
            _, mini_inputs = self.create_simplified_gslice_node(graph, input_v, input_gslice_attrs)
        return mini_inputs

    def get_mini_op_name(self, graph: ir.Graph, namehint, head_slice_id, batch_slice_id) -> str:
        """
        Get meaningful unique name of the mini op

        Args:
            graph: ir.Graph instance
            namehint: namehint of the mini op
            head_slice_id: head id
            batch_slice_id: batch id
        Returns:
            unique_name: the unique meaningful op name
        """
        suffixes = []
        if head_slice_id >= 0:
            suffixes.append(f"head_{head_slice_id}")
        if batch_slice_id >= 0:
            suffixes.append(f"batch_{batch_slice_id}")
        namehint = namehint + "/" + "/".join(suffixes)
        unique_name = graph.meta["extra_info"].get_unique_name(namehint)
        return unique_name

    def get_mini_tensor_name(self, graph: ir.Graph, namehint, head_slice_id, batch_slice_id) -> str:
        """
        Get meaningful unique name of the mini tensor

        Args:
            graph: ir.Graph instance
            namehint: namehint of the mini tensor
            head_slice_id: head id
            batch_slice_id: batch id
        Returns:
            unique_name: the unique meaningful tensor name
        """
        return self.get_mini_op_name(graph, namehint, head_slice_id, batch_slice_id)

    def create_mini_pattern(
        self,  # pylint: disable=R0913,R0917
        graph: ir.Graph,
        origin_op: ir.Node,
        mini_inputs: list[ir.Value | None],
        head_slice_id,
        batch_slice_id,
        slice_i,  # pylint: disable=W0613
        custom_kwargs: dict | None = None,
    ) -> list[ir.Value | None]:
        """
        Create a "mini pattern" for a single head/batch slice.

        The overall MHA-to-SHA rewrite transforms a subgraph like::

            Subgraph(in_a, in_b, ...) --> out
            {
                out = OriginalOp(in_a, in_b, ...)
            }

        into::

            Subgraph(in_a, in_b, ...) --> out
            {
                a0, a1, a2, ... = GroupSlice(in_a)   # omitted if input is shared across slices
                b0, b1, b2, ... = GroupSlice(in_b)

                out0 = OriginalOp(a0, b0, ...)        # <-- each of these is a "mini pattern"
                out1 = OriginalOp(a1, b1, ...)
                out2 = OriginalOp(a2, b2, ...)

                out = Concat(out0, out1, out2, ...)   # or reuse OriginalOp if only one slice
            }

        This function creates one mini op node (e.g. ``OriginalOp(a0, b0, ...)``
        for a given head/batch slice) and inserts it into the graph.

        Args:
            graph: ir.Graph instance
            origin_op: original op
            mini_inputs: the mini inputs of the op of current slice
            head_slice_id: head id
            batch_slice_id: batch id
            slice_i: current slice id
            custom_kwargs: custom defined kwargs, can be useful when this function is overriden in the specific pass
        Returns:
            the outputs of the generated mini op
        """
        # create mini op
        # sometimes, the origin_op.domain is not "",
        # for example, it maybe "com.microsoft" for QuantizeLinear node
        mini_op = ir.Node(
            domain=origin_op.domain,
            op_type=origin_op.op_type,
            inputs=mini_inputs,
            num_outputs=len(origin_op.outputs),
            name=self.get_mini_op_name(graph, origin_op.name, head_slice_id, batch_slice_id),
            attributes=[clone_node_attribute(attr) for attr in origin_op.attributes.values()],
        )

        graph.insert_before(origin_op, mini_op)
        return list(mini_op.outputs)

    def rewrite_based_on_gslice_attrs(
        self,  # pylint: disable=R0914
        graph: ir.Graph,
        op: ir.Node,
        output_gslice_nodes: list[ir.Node],
        inputs_gslice_attrs: list[BaseGroupSliceAttrs],
        mini_pattern_custom_kwargs: dict | None = None,
    ):
        """
        General rewrite method for group slice reordering, it has these steps:
        - create mini inputs of the op for each slice
            - apply groupslice (ir required) on each of the inputs of the original op
            - the behavior is controlled by inputs_gslice_attrs
        - create mini op for each slice
            - the inputs of each mini op are the mini inputs created for current slice
            - the behavior is controlled by create_mini_pattern
                - by default, create_mini_pattern create the mini ops with the same attributes
                  and the same topologicall with the original op
                - this behavior can be changed by overidding create_mini_pattern
        - recover the original op outputs if possible
            - if possible, concat the outputs of the mini ops for each slice, and then replace
              the original op outputs by the concat op outputs

        Args:
            graph: ir.Graph instance
            op: original op
            output_gslice_nodes: current groupslice nodes for the original op
            inputs_gslice_attrs: required group slice attrs for each inputs of the original op
        Returns:
            the recovered outputs
        """
        # check for mypy
        assert op.name is not None

        assert len(op.outputs) == len(output_gslice_nodes)
        slice_num = len(output_gslice_nodes[0].outputs)
        for x in inputs_gslice_attrs:
            assert slice_num == x.num_outputs()

        gsliced_outputs = [gslice_n.outputs for gslice_n in output_gslice_nodes]
        outputs_gslice_attrs = [get_gslice_attrs(gslice_n) for gslice_n in output_gslice_nodes]
        mini_inputs = []  # List of List
        mini_outputs: list[list[ir.Value]] = [[] for i in range(len(op.outputs))]  # List of list
        for in_i, _ in enumerate(op.inputs):
            split_in_i = self.create_mini_inputs(graph, op.inputs[in_i], inputs_gslice_attrs[in_i])
            mini_inputs.append(split_in_i)

        for slice_i in range(slice_num):
            # create mini op for each slice
            mini_op_inputs = []
            for in_i in range(len(op.inputs)):
                mini_op_inputs.append(mini_inputs[in_i][slice_i])
            mini_op_outputs = self.create_mini_pattern(
                graph,
                op,
                mini_op_inputs,
                inputs_gslice_attrs[0].head_slice_ids[slice_i],
                inputs_gslice_attrs[0].batch_slice_ids[slice_i],
                slice_i,
                mini_pattern_custom_kwargs,
            )
            origin_mini_outputs = []
            for out_i in range(len(op.outputs)):
                origin_mini_outputs.append(gsliced_outputs[out_i][slice_i])

            for out_i in range(len(op.outputs)):
                origin_out_v_slice_i = origin_mini_outputs[out_i]
                current_out_v_slice_i = mini_op_outputs[out_i]
                if origin_out_v_slice_i is None and current_out_v_slice_i is None:
                    continue
                if origin_out_v_slice_i is None or current_out_v_slice_i is None:
                    assert False, "should not happen"
                current_out_v_slice_i.name = graph.meta["extra_info"].get_unique_name(
                    origin_out_v_slice_i.name
                )
                safe_replace_all_uses_with(graph, origin_out_v_slice_i, current_out_v_slice_i)

                current_out_v_slice_i_extra_info_to_overwrite = None
                if (
                    "extra_info" in current_out_v_slice_i.meta
                    and current_out_v_slice_i.meta["extra_info"].defined_encodings()
                    and not (
                        "extra_info" in origin_out_v_slice_i.meta
                        and origin_out_v_slice_i.meta["extra_info"].defined_encodings()
                    )
                ):
                    # current_out_v_slice_i is a variable already exist in the graph
                    # and it may contains Encodings, but origin_out_v_slice_i does not,
                    # in this case, executing mark_value_as_copy will set current_out_v_slice_i's encodings to None
                    # so we need to backup it and recover it later
                    current_out_v_slice_i_extra_info_to_overwrite = current_out_v_slice_i.meta[
                        "extra_info"
                    ].copy(ignore_safetensors=True)

                self.mark_value_as_copy(graph, origin_out_v_slice_i, current_out_v_slice_i)  # type: ignore[attr-defined]

                if current_out_v_slice_i_extra_info_to_overwrite is not None:
                    current_out_v_slice_i.meta["extra_info"].merge(
                        current_out_v_slice_i_extra_info_to_overwrite, encodings_only=True
                    )

                mini_outputs[out_i].append(current_out_v_slice_i)

        # recover outputs by mini outputs
        recovered_outputs = []

        for out_i, out_v in enumerate(op.outputs):
            out_i_shape = out_v.shape
            out_i_name = out_v.name
            # check for mypy
            assert out_i_shape is not None
            assert out_i_name is not None
            assert all(x.shape is not None for x in mini_outputs[out_i])

            gslice_axis = outputs_gslice_attrs[out_i].axis
            filtered_gslice_outputs = [
                x for x in mini_outputs[out_i] if x.shape is not None and x.shape[gslice_axis] > 0
            ]

            if outputs_gslice_attrs[out_i].complete(out_i_shape[gslice_axis]):
                # we can only replace the original op output by (the concat of) group slice outputs,
                # when the group slice is complete.
                if len(filtered_gslice_outputs) == 1:
                    recovered_outputs.append(filtered_gslice_outputs[0])  # no need to concat

                    # replace all usage of op outputs by recovered outputs
                    safe_replace_all_uses_with(graph, op.outputs[out_i], recovered_outputs[out_i])
                else:
                    # TODO, using simplified outputs_gslice_attrs
                    assert not (
                        any(x.shape is None or x.shape[gslice_axis] == 0 for x in mini_outputs[out_i])
                    )
                    recover_concat = create_concat_node(
                        graph,
                        filtered_gslice_outputs,
                        gslice_axis,
                        op.name + "/recover111",
                        out_i_name + "/recover",
                    )
                    graph.insert_before(op, recover_concat)
                    recovered_outputs.append(recover_concat.outputs[0])

                    self.mark_value_as_copy(graph, op.outputs[out_i], recover_concat.outputs[0])  # type: ignore[attr-defined]

                    # replace all usage of op outputs by recovered outputs
                    safe_replace_all_uses_with(graph, op.outputs[out_i], recovered_outputs[out_i])

        # the output glisces can be removed safely
        graph.remove(output_gslice_nodes, safe=True)

        return recovered_outputs

    def try_replace_out_with_concat(
        self,
        graph: ir.Graph,
        origin_op: ir.Node,
        origin_op_gslice_attrs: GroupSliceAttrs,
        new_sliced_outputs: list[ir.Value],
    ):
        """
        Try to replace the output of a GroupSlice node with a Concat node.

        Args:
            graph: ir.Graph instance
            origin_op (ir.Node): The original GroupSlice node.
            origin_op_gslice_attrs (GroupSliceAttrs): The attributes of the original GroupSlice node.
            new_sliced_outputs (list[ir.Value]): The new sliced outputs.
        """
        # check for mypy
        assert origin_op.outputs[0].shape is not None
        assert origin_op.name is not None
        assert origin_op.outputs[0].name is not None

        out_shape = get_value_numeric_shape(origin_op.outputs[0])
        simplified_gslice_attrs, origin2simplified_idx_map = simplify_group_slice_attrs(
            origin_op_gslice_attrs
        )
        if simplified_gslice_attrs.complete(out_shape[simplified_gslice_attrs.axis]):
            simplified2origin_idx_map = {}
            for i in range(origin_op_gslice_attrs.num_outputs()):
                simplified_idx = origin2simplified_idx_map[i]
                if simplified_idx in simplified2origin_idx_map:
                    continue
                simplified2origin_idx_map[simplified_idx] = i
            complete_outputs = [
                new_sliced_outputs[simplified2origin_idx_map[i]]
                for i in range(simplified_gslice_attrs.num_outputs())
            ]
            output_concat_node = create_concat_node(
                graph,
                complete_outputs,
                simplified_gslice_attrs.axis,
                origin_op.name + "/gslice_concat",
                origin_op.outputs[0].name + "/gslice_concat_out",
            )

            self.mark_value_as_copy(graph, origin_op.outputs[0], output_concat_node.outputs[0])  # type: ignore[attr-defined]
            safe_replace_all_uses_with(graph, origin_op.outputs[0], output_concat_node.outputs[0])
            graph.insert_before(origin_op, output_concat_node)

    def _create_groupslice_node(
        self,
        graph: ir.Graph,
        data: ir.Value,
        gslice_attrs: GroupSliceAttrs,
        node_namehint: str | None = None,
    ) -> ir.Node:
        """
        Create GroupSlice node by the given group slice attrs

        Args:
            graph: ir.Graph instance
            data: input of the generated GroupSlice node
            gslice_attrs: group slice attributes to generate the GroupSlice node
            node_namehint: name hint of the generated GroupSlice node
        Returns:
            the generated group slice node
        """
        if node_namehint is None:
            assert data.name is not None  # check for mypy
            node_namehint = data.name + "/gslice_node"

        data_shape = get_value_numeric_shape(data)

        if len(gslice_attrs.starts) != len(gslice_attrs.ends):
            raise ValueError("MHAGroupSlice starts number should be equal to ends number")
        if gslice_attrs.axis < 0:
            raise ValueError(f"MHAGroupSlice axis should be positive. gslice axis {gslice_attrs.axis}")
        if gslice_attrs.axis >= len(data_shape):
            raise ValueError(f"MHAGroupSlice axis should be less than data shape, got {gslice_attrs.axis}")
        for start, end in zip(gslice_attrs.starts, gslice_attrs.ends):
            if start < 0:
                raise ValueError("MHAGroupSlice starts should be positive")
            if end < 0:
                raise ValueError(f"MHAGroupSlice ends should be positive, got {end}")
            if end > data_shape[gslice_attrs.axis]:
                raise ValueError(f"MHAGroupSlice ends should be less than data shape, got {end}")
        if gslice_attrs.num_outputs() == 0:
            raise ValueError("MHAGroupSlice num_outputs should be greater than 0")

        unqiue_name = graph.meta["extra_info"].get_unique_name(node_namehint)
        gslice_node = ir.Node(
            domain="",
            op_type="GroupSlice",
            inputs=[data],
            num_outputs=gslice_attrs.num_outputs(),
            name=unqiue_name,
        )
        gslice_node.attributes["starts"] = ir.AttrInt64s("starts", gslice_attrs.starts)
        gslice_node.attributes["ends"] = ir.AttrInt64s("ends", gslice_attrs.ends)
        gslice_node.attributes["axis"] = ir.AttrInt64("axis", gslice_attrs.axis)
        gslice_node.attributes["batch_slice_ids"] = ir.AttrInt64s(
            "batch_slice_ids", gslice_attrs.batch_slice_ids
        )
        gslice_node.attributes["head_slice_ids"] = ir.AttrInt64s(
            "head_slice_ids", gslice_attrs.head_slice_ids
        )

        for i, out in enumerate(gslice_node.outputs):
            batch_slice_id = gslice_attrs.batch_slice_ids[i]
            head_slice_id = gslice_attrs.head_slice_ids[i]
            out_name = get_slice_out_name(graph, data.name, batch_slice_id, head_slice_id)
            out.name = out_name

            self.mark_value_as_slice(  # type: ignore[attr-defined]
                graph,
                data,
                out,
                gslice_attrs.axis,
                gslice_attrs.starts[i],
                gslice_attrs.ends[i],
                batch_slice_id,
                head_slice_id,
            )

        return gslice_node
