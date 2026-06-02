# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module extract lora alpha as an additional graph input
"""

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import (
    OpTypePredicate,
    get_constant_np,
    is_constant,
    scan_previous_nearest_candidate,
)
from qairt.optimizer.utils.logger import logger


class LoraAlphaExtractor(BasePass):
    """
    Pass to capture lora alpha and then set it as an additional input
    """

    @staticmethod
    def get_lora_alpha_np_value(ctx: GraphContext):
        """
        Get lora alpha constant value in the original graph

        Returns:
            numpy.ndarray or None: The lora alpha value if found, None if no LoRA structure exists

        Raises:
            RuntimeError: If LoraAlphaExtractor has not been applied to the graph
        """
        if "lora_alpha_np_value" not in ctx.graph_ir.meta:
            raise RuntimeError(
                "LoraAlphaExtractor must be applied to the graph before calling get_lora_alpha_np_value(). "
                "Please ensure LoraAlphaExtractor.apply(ctx) is called first."
            )
        return ctx.graph_ir.meta["lora_alpha_np_value"]

    def apply(self, ctx: GraphContext) -> int:  # pylint: disable=[too-many-locals,too-many-branches,too-many-statements]
        """
        Find base linear, lora_b and lora_b.
            x-----------
            |           |
            |         lora_a
            |           |
        base linear   Mul alpha
            |           |
            |         lora_b
            |           |
            Add--------/

        Args:
            ctx: The model context containing the graph and metadata

        Returns:
            Number of LoRA structures found and processed
        """
        graph = ctx.graph_ir
        rewrite_count = 0
        n_list = list(graph)[:]
        all_lora_muls = []  # list of (lora_mul, alpha_input_i)

        lora_alpha_np_value = None
        for n in n_list:  # pylint: disable=[too-many-nested-blocks]
            if n.op_type == "Add":
                lora_add = n

                for add_input_i_for_base_proj, add_input_i_for_lora_b in [
                    (0, 1),
                    (1, 0),
                ]:
                    for base_proj_out in scan_previous_nearest_candidate(
                        [lora_add.inputs[add_input_i_for_base_proj]],
                        check_fn=OpTypePredicate(["Conv", "MatMul"]),
                        ignore_fn=OpTypePredicate(["Reshape", "Transpose", "Squeeze", "Unsqueeze"]),
                    ):
                        for lora_b_proj_out in scan_previous_nearest_candidate(
                            [lora_add.inputs[add_input_i_for_lora_b]],
                            check_fn=OpTypePredicate(["Conv", "MatMul"]),
                            ignore_fn=OpTypePredicate(["Reshape", "Transpose", "Squeeze", "Unsqueeze"]),
                        ):
                            for lora_alpha_mul_out in scan_previous_nearest_candidate(
                                [lora_b_proj_out.producer().inputs[0]],
                                check_fn=OpTypePredicate(["Mul"]),
                                ignore_fn=OpTypePredicate(["Reshape", "Transpose", "Squeeze", "Unsqueeze"]),
                            ):
                                for mul_input_i_for_alpha, mul_input_i_for_lora_b in [
                                    (1, 0),
                                    (0, 1),
                                ]:
                                    for lora_a_proj_out in scan_previous_nearest_candidate(
                                        [lora_alpha_mul_out.producer().inputs[mul_input_i_for_lora_b]],
                                        check_fn=OpTypePredicate(["Conv", "MatMul"]),
                                        ignore_fn=OpTypePredicate(
                                            ["Reshape", "Transpose", "Squeeze", "Unsqueeze"]
                                        ),
                                    ):
                                        # lora_a_proj should share same input with base_proj
                                        # get lora_a_proj ancestors
                                        lora_a_ancestors = scan_ancestors(
                                            lora_a_proj_out.producer().inputs[0],
                                            ["Reshape", "Transpose", "Squeeze", "Unsqueeze"],
                                        )
                                        base_ancestors = scan_ancestors(
                                            base_proj_out.producer().inputs[0],
                                            ["Reshape", "Transpose", "Squeeze", "Unsqueeze"],
                                        )

                                        # accept lca node that may not be reshape/transpose/sequnce,
                                        # for example the lca is Mul
                                        #   Mul ------------------------
                                        #    |                           \
                                        #  Transpose/Reshape/Seq          \
                                        #    |                            lora a
                                        #    |                              |
                                        #   BaseConv                Transpose/Reshape/Seq
                                        #    |                              |
                                        #    |                            lora b
                                        #    |                              |
                                        #    |                         Transpose/Reshape/Seq
                                        #  Transpose/Reshape/Seq            |
                                        #   Mul------------------------------
                                        lora_a_ancestors += lora_a_ancestors[-1].predecessors()
                                        base_ancestors += base_ancestors[-1].predecessors()

                                        base_ancestors_set = set(x.name for x in base_ancestors)
                                        # find least common ancestor
                                        lca = None
                                        for ancestor_n in lora_a_ancestors:
                                            if ancestor_n.name in base_ancestors_set:
                                                lca = ancestor_n
                                                break

                                        if lca is None:
                                            logger.debug(
                                                "Ignore candidate lora structure, reason: LCA not found\n"
                                                + "lora_add: '%s', "
                                                + "base_linear:'%s', "
                                                + "lora_a:'%s', "
                                                + "lora_b:'%s'",
                                                lora_add.name,
                                                base_proj_out.producer().name,
                                                lora_a_proj_out.producer().name,
                                                lora_b_proj_out.producer().name,
                                            )
                                            continue

                                        # check if lora alpha is constant
                                        alpha = lora_alpha_mul_out.producer().inputs[mul_input_i_for_alpha]
                                        if not is_constant(alpha):
                                            logger.debug(
                                                "Ignore candidate lora structure, "
                                                + "reason: alpha is not constant\n"
                                                + "lora_add: '%s', "
                                                + "base_linear:'%s', "
                                                + "lora_a:'%s', "
                                                + "alpha:'%s', "
                                                + "lora_b:'%s'",
                                                lora_add.name,
                                                base_proj_out.producer().name,
                                                lora_a_proj_out.producer().name,
                                                alpha.name,
                                                lora_b_proj_out.producer().name,
                                            )
                                            continue

                                        # Captured lora structure
                                        all_lora_muls.append(
                                            (lora_alpha_mul_out.producer(), mul_input_i_for_alpha)
                                        )

        if len(all_lora_muls) == 0:
            # no lora structure, store None and return
            graph.meta["lora_alpha_np_value"] = None
            return 0

        # verify all lora alpha is same value
        lora_alpha_mul_out_ranks_set = set()
        lora_alpha_extra_info = VariableExtraInfo()
        for lora_mul, mul_input_i_for_alpha in all_lora_muls:
            curr_lora_alpha_np_value = get_constant_np(lora_mul.inputs[mul_input_i_for_alpha])
            if lora_alpha_np_value is None:
                lora_alpha_np_value = curr_lora_alpha_np_value
            else:
                if not np.allclose(lora_alpha_np_value, curr_lora_alpha_np_value):
                    raise ValueError(
                        "All lora alpha should be same value, got "
                        + f"{lora_alpha_np_value} at {all_lora_muls[0][0].name}"
                        + f", {curr_lora_alpha_np_value} at {lora_mul.name}"
                    )
            curr_lora_alpha_mul_out_shape = lora_mul.outputs[0].shape.numpy()
            lora_alpha_mul_out_ranks_set.add(len(curr_lora_alpha_mul_out_shape))
            curr_extra_info = lora_mul.inputs[mul_input_i_for_alpha].meta["extra_info"]
            if not lora_alpha_extra_info.defined_encodings():
                lora_alpha_extra_info.merge(curr_extra_info)
            elif lora_alpha_extra_info.named_encodings != curr_extra_info.named_encodings:
                raise ValueError(
                    "All lora alpha should have same encodings, "
                    + f"got {lora_alpha_extra_info.named_encodings} "
                    + f"at {all_lora_muls[0][0].name}"
                    + f", {curr_extra_info.named_encodings} at {lora_mul.name}"
                )

        assert lora_alpha_np_value is not None  # check for mypy, definitely true

        # to align with legacy mha2sha
        lora_alpha_mul_out_ranks = list(lora_alpha_mul_out_ranks_set)
        if len(lora_alpha_mul_out_ranks) == 1 and lora_alpha_mul_out_ranks[0] == 4:
            if isinstance(lora_alpha_np_value, np.ndarray) and lora_alpha_np_value.shape:
                lora_alpha_np_value = lora_alpha_np_value.reshape(1, -1, 1, 1)

        lora_alpha_np_value = lora_alpha_np_value.astype(np.float32)

        # Store lora alpha value in graph metadata
        graph.meta["lora_alpha_np_value"] = lora_alpha_np_value

        # create lora alpha as input
        graph_input_alpha = ir.Value(
            None,
            name="lora_alpha",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(lora_alpha_np_value.shape),
        )
        graph_input_alpha.meta["extra_info"] = lora_alpha_extra_info
        graph.inputs.append(graph_input_alpha)

        # replace lora alpha with graph input
        for lora_mul, mul_input_i_for_alpha in all_lora_muls:
            origin_alpha_name = lora_mul.inputs[mul_input_i_for_alpha].name
            lora_mul.replace_input_with(mul_input_i_for_alpha, graph_input_alpha)
            logger.debug("replace lorav2 alpha '%s' as input '%s'", origin_alpha_name, graph_input_alpha.name)

        rewrite_count = len(all_lora_muls)
        return rewrite_count


def scan_ancestors(start_v: ir.Value | None, acceptable_op_types):
    """
    Scan ancestors of a value, return a list of ancestors
    Stop at any op which op_type is not in acceptable_op_types
    """
    # only scan first inputs of op
    ancestors = []
    while True:
        if start_v is None:
            break
        producer = start_v.producer()
        if producer is None:
            break
        if producer.op_type not in acceptable_op_types:
            break
        ancestors.append(producer)
        start_v = producer.inputs[0]
    return ancestors
