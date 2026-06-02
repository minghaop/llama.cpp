# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides passes for eliminating redundant reshapes in MHA patterns
and clearing intermediate shapes.
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePass, BasePredicatePass
from qairt.optimizer.onnx.passes.shape_infer import ShapeInference
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class ClearIntermediateShapes(BasePredicatePass):
    """
    Clear shapes from all tensors except inputs/outputs
    This prepares the graph for shape inference by removing intermediate
    shape information that may be stale after updates
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        """Check if node has intermediate outputs with shapes"""
        for out in node.outputs:
            if out not in graph.outputs and out.shape is not None:
                return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        """Clear shapes from intermediate outputs"""
        modified = False
        for out in node.outputs:
            if out not in graph.outputs and out.shape is not None:
                out.shape = None
                modified = True
        return modified


class EliminateMHAReshapes(BasePass):
    """
    Eliminate redundant reshapes in MHA patterns

    Pattern:
        Multiple parallel paths feeding into Concat:
        Path_i: [Reshape(s)] -> MatMul1 -> Softmax -> MatMul2 -> Concat

        After Concat: Concat -> [Transpose] -> Reshape

    Goal: Remove reshapes before MatMuls and after Concat if:
        1. MatMuls remain compatible
        2. Final output shape is preserved
    """

    def apply(self, ctx) -> int:
        total_optimized = 0

        # Find all Concat nodes
        concat_nodes = [n for n in ctx.graph_ir if n.op_type == "Concat"]
        logger.debug(f"EliminateMHAReshapes: Found {len(concat_nodes)} Concat nodes")

        for idx, concat in enumerate(concat_nodes):
            logger.debug(f"[{idx + 1}/{len(concat_nodes)}] Analyzing Concat: {concat.name}")

            if self._try_optimize_concat_pattern(ctx.graph_ir, concat):
                total_optimized += 1
            else:
                logger.debug(f"Could not optimize pattern at {concat.name}")

        logger.debug(f"EliminateMHAReshapes: Optimized {total_optimized} patterns")

        # After eliminating reshapes, clear intermediate shapes and run shape inference
        if total_optimized > 0:
            logger.debug("Clearing intermediate shapes and running shape inference")
            ClearIntermediateShapes().apply(ctx)
            ShapeInference().apply(ctx)

        return total_optimized

    def _try_optimize_concat_pattern(self, graph: ir.Graph, concat: ir.Node) -> bool:
        """Try to optimize reshapes around this concat"""

        # Find reshapes before and after concat
        initial_reshapes = self._find_reshapes_before_concat(concat)
        if not initial_reshapes:
            return False

        final_reshapes = self._find_reshapes_after_concat(concat)
        if not final_reshapes:
            return False

        # Get original shape
        original_shape = tuple(get_value_numeric_shape(initial_reshapes[0].inputs[0]))

        # Find which subset can be eliminated
        reshapes_to_eliminate, final_to_eliminate = self._find_eliminable_reshapes(
            concat, initial_reshapes, final_reshapes, original_shape
        )

        if not reshapes_to_eliminate and not final_to_eliminate:
            logger.debug("Cannot eliminate any reshapes")
            return False

        # Eliminate the selected reshapes
        for reshape in reshapes_to_eliminate:
            safe_replace_all_uses_with(graph, reshape.outputs[0], reshape.inputs[0])

        for reshape in final_to_eliminate:
            safe_replace_all_uses_with(graph, reshape.outputs[0], reshape.inputs[0])

        logger.debug(
            f"Eliminated {len(reshapes_to_eliminate)}/{len(initial_reshapes)} initial + "
            f"{len(final_to_eliminate)}/{len(final_reshapes)} final reshapes"
        )
        return True

    def _find_reshapes_before_concat(self, concat: ir.Node) -> List[ir.Node]:
        """Find all reshapes feeding into concat (through MatMuls)"""
        reshapes = []

        for concat_input in concat.inputs:
            if concat_input is None:
                continue

            # Trace back through MatMul2 -> Softmax -> MatMul1
            matmul2 = concat_input.producer()
            if not matmul2 or matmul2.op_type != "MatMul":
                continue

            # Find reshapes before MatMul2's inputs
            for matmul_input in matmul2.inputs:
                if matmul_input is None:
                    continue
                reshape = self._find_reshape_before(matmul_input)
                if reshape:
                    reshapes.append(reshape)

            # Trace back to MatMul1
            matmul1 = self._find_matmul1_before_matmul2(matmul2)
            if matmul1:
                # Find reshapes before MatMul1's inputs
                for matmul_input in matmul1.inputs:
                    if matmul_input is None:
                        continue
                    reshape = self._find_reshape_before(matmul_input)
                    if reshape:
                        reshapes.append(reshape)

        # Remove duplicates
        return list(set(reshapes))

    def _find_matmul1_before_matmul2(self, matmul2: ir.Node) -> ir.Node | None:
        """Find MatMul1 before MatMul2 (through Softmax)"""
        # MatMul2's first input should come from Softmax
        if not matmul2.inputs or not matmul2.inputs[0]:
            return None

        current = matmul2.inputs[0]
        for _ in range(5):  # Max 5 hops
            producer = current.producer()
            if not producer:
                return None

            if producer.op_type == "MatMul":
                return producer

            # Continue through Softmax and other ops
            if producer.inputs and producer.inputs[0]:
                current = producer.inputs[0]
            else:
                return None

        return None

    def _find_reshape_before(self, value: ir.Value) -> ir.Node | None:
        """Find reshape immediately before this value (through elementwise ops only)"""
        elementwise_ops = {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Relu",
            "Sigmoid",
            "Tanh",
            "Gelu",
            "Silu",
            "Transpose",
            "Identity",
            "Cast",
        }

        # Stop ops - if we hit these, there's no reshape
        stop_ops = {"Conv", "ConvTranspose", "MatMul", "Gemm"}

        current = value
        for _ in range(5):  # Max 5 hops
            producer = current.producer()
            if not producer:
                return None

            if producer.op_type == "Reshape" and have_static_shape_on_node_io(producer):
                return producer

            if producer.op_type in stop_ops:
                # Hit a stop op (like Conv), no reshape before this
                return None

            if producer.op_type in elementwise_ops:
                if producer.inputs and producer.inputs[0]:
                    current = producer.inputs[0]
                else:
                    return None
            else:
                # Hit non-elementwise op, stop
                return None

        return None

    def _find_reshapes_after_concat(self, concat: ir.Node) -> List[ir.Node]:
        """Find reshapes after concat (through Transpose)"""
        reshapes = []

        def trace_forward(value: ir.Value, depth: int = 0):
            if depth > 5:
                return

            for use in value.uses():
                consumer = use.node
                if consumer.op_type == "Reshape" and have_static_shape_on_node_io(consumer):
                    reshapes.append(consumer)
                    trace_forward(consumer.outputs[0], depth + 1)
                elif consumer.op_type == "Transpose":
                    trace_forward(consumer.outputs[0], depth + 1)

        trace_forward(concat.outputs[0])
        return reshapes

    def _find_eliminable_reshapes(
        self,
        concat: ir.Node,
        initial_reshapes: List[ir.Node],
        final_reshapes: List[ir.Node],
        original_shape: tuple,
    ) -> Tuple[List[ir.Node], List[ir.Node]]:
        """
        Find which reshapes can be eliminated

        Strategy:
        1. Get ONE representative path
        2. Find which reshapes are in that path
        3. Try different subsets of THOSE reshapes
        4. Apply the same elimination to ALL paths

        Returns: (initial_reshapes_to_eliminate, final_reshapes_to_eliminate)
        """
        # Get one representative path
        first_path = self._get_first_path(concat)
        if not first_path:
            return [], []

        # Find which reshapes are in this ONE path
        path_reshapes = self._get_reshapes_in_path(first_path, initial_reshapes)
        if not path_reshapes:
            return [], []

        # Try eliminating different combinations of reshapes in this path
        # Start with all, then try subsets
        for num_to_eliminate in range(len(path_reshapes), 0, -1):
            for reshape_subset in combinations(path_reshapes, num_to_eliminate):
                reshape_subset_list = list(reshape_subset)

                # Check if this subset can be eliminated (only checks the ONE path)
                if self._can_eliminate_reshapes_in_path(first_path, reshape_subset_list, original_shape):
                    # This works for one path, so apply to ALL paths
                    # Find all reshapes in the same positions across all paths
                    all_reshapes_to_eliminate = self._find_matching_reshapes_in_all_paths(
                        concat, reshape_subset_list, initial_reshapes
                    )

                    # Check which final reshapes can be eliminated
                    final_to_eliminate = self._find_eliminable_final_reshapes(
                        concat, all_reshapes_to_eliminate, final_reshapes, original_shape
                    )

                    return all_reshapes_to_eliminate, final_to_eliminate

        return [], []

    def _can_eliminate_reshapes_in_path(
        self, path: Dict, reshapes: List[ir.Node], original_shape: tuple
    ) -> bool:
        """Check if reshapes can be eliminated in ONE specific path"""
        matmul1 = path["matmul1"]
        matmul2 = path["matmul2"]

        # Check if MatMul1 will work
        if not self._check_matmul_works_with_shape(matmul1, original_shape, reshapes):
            return False

        # Check MatMul1 output has same number of elements (reshapeable)
        current_mm1_out = tuple(get_value_numeric_shape(matmul1.outputs[0]))
        new_mm1_in0 = self._get_shape_after_reshape_removal(matmul1.inputs[0], original_shape, reshapes)
        new_mm1_in1 = self._get_shape_after_reshape_removal(matmul1.inputs[1], original_shape, reshapes)
        new_mm1_out = self._compute_matmul_output(new_mm1_in0, new_mm1_in1)

        # Check same number of elements (can be reshaped)
        current_elements = math.prod(current_mm1_out)
        new_elements = math.prod(new_mm1_out)
        if current_elements != new_elements:
            return False  # Different number of elements, can't reshape!

        # Check if MatMul2 will work
        if not self._check_matmul2_works(matmul1, matmul2, original_shape, reshapes):
            return False

        # Check MatMul2 output has same number of elements (reshapeable)
        current_mm2_out = tuple(get_value_numeric_shape(matmul2.outputs[0]))
        new_mm2_in1 = self._get_shape_after_reshape_removal(matmul2.inputs[1], original_shape, reshapes)
        new_mm2_out = self._compute_matmul_output(new_mm1_out, new_mm2_in1)

        current_elements = math.prod(current_mm2_out)
        new_elements = math.prod(new_mm2_out)
        if current_elements != new_elements:
            return False  # Different number of elements, can't reshape!

        return True

    def _find_matching_reshapes_in_all_paths(
        self, concat: ir.Node, representative_reshapes: List[ir.Node], all_reshapes: List[ir.Node]
    ) -> List[ir.Node]:
        """
        Given reshapes that work in one path, find matching reshapes in all paths

        E.g., if we can eliminate "reshape before MatMul1 input0" in path 1,
        find the same reshape in all other paths
        """
        # Determine which positions these reshapes are at
        positions = set()

        # Get the representative path
        first_path = self._get_first_path(concat)
        if not first_path:
            return representative_reshapes

        matmul1 = first_path["matmul1"]
        matmul2 = first_path["matmul2"]

        # Check which positions the representative reshapes are at
        for r_reshape in representative_reshapes:
            if matmul1.inputs and len(matmul1.inputs) > 0 and matmul1.inputs[0]:
                if self._find_reshape_before(matmul1.inputs[0]) == r_reshape:
                    positions.add("matmul1_input0")

            if matmul1.inputs and len(matmul1.inputs) > 1 and matmul1.inputs[1]:
                if self._find_reshape_before(matmul1.inputs[1]) == r_reshape:
                    positions.add("matmul1_input1")

            if matmul2.inputs and len(matmul2.inputs) > 1 and matmul2.inputs[1]:
                if self._find_reshape_before(matmul2.inputs[1]) == r_reshape:
                    positions.add("matmul2_input1")

        # Now find all reshapes at these positions across ALL paths
        matching_reshapes = []

        for concat_input in concat.inputs:
            if concat_input is None:
                continue

            matmul2 = concat_input.producer()
            if not matmul2 or matmul2.op_type != "MatMul":
                continue

            matmul1 = self._find_matmul1_before_matmul2(matmul2)
            if not matmul1:
                continue

            # Check each position
            if "matmul1_input0" in positions:
                if matmul1.inputs and len(matmul1.inputs) > 0 and matmul1.inputs[0]:
                    reshape = self._find_reshape_before(matmul1.inputs[0])
                    if reshape and reshape in all_reshapes:
                        matching_reshapes.append(reshape)

            if "matmul1_input1" in positions:
                if matmul1.inputs and len(matmul1.inputs) > 1 and matmul1.inputs[1]:
                    reshape = self._find_reshape_before(matmul1.inputs[1])
                    if reshape and reshape in all_reshapes:
                        matching_reshapes.append(reshape)

            if "matmul2_input1" in positions:
                if matmul2.inputs and len(matmul2.inputs) > 1 and matmul2.inputs[1]:
                    reshape = self._find_reshape_before(matmul2.inputs[1])
                    if reshape and reshape in all_reshapes:
                        matching_reshapes.append(reshape)

        return list(set(matching_reshapes))

    def _get_first_path(self, concat: ir.Node) -> Dict | None:
        """Get the first valid path from concat"""
        for concat_input in concat.inputs:
            if concat_input is None:
                continue

            matmul2 = concat_input.producer()
            if not matmul2 or matmul2.op_type != "MatMul":
                continue

            matmul1 = self._find_matmul1_before_matmul2(matmul2)
            if matmul1:
                return {"matmul1": matmul1, "matmul2": matmul2}

        return None

    def _get_reshapes_in_path(self, path: Dict, all_reshapes: List[ir.Node]) -> List[ir.Node]:
        """Get which reshapes are in this specific path"""
        path_reshapes = []

        matmul1 = path["matmul1"]
        matmul2 = path["matmul2"]

        # Check reshapes before MatMul1
        for matmul_input in matmul1.inputs:
            if matmul_input is None:
                continue
            reshape = self._find_reshape_before(matmul_input)
            if reshape and reshape in all_reshapes:
                path_reshapes.append(reshape)

        # Check reshapes before MatMul2
        for matmul_input in matmul2.inputs:
            if matmul_input is None:
                continue
            reshape = self._find_reshape_before(matmul_input)
            if reshape and reshape in all_reshapes:
                path_reshapes.append(reshape)

        return list(set(path_reshapes))

    def _find_eliminable_final_reshapes(
        self,
        concat: ir.Node,
        initial_reshapes: List[ir.Node],
        final_reshapes: List[ir.Node],
        original_shape: tuple,
    ) -> List[ir.Node]:
        """Find which final reshapes can be eliminated given the initial reshapes being removed"""
        eliminable = []

        for final_reshape in final_reshapes:
            # Check if this final reshape would become identity
            current_final_shape = tuple(get_value_numeric_shape(final_reshape.outputs[0]))
            new_concat_output = self._compute_concat_output_after_removal(
                concat, initial_reshapes, original_shape
            )

            if new_concat_output == current_final_shape:
                eliminable.append(final_reshape)

        return eliminable

    def _check_matmul_works_with_shape(
        self, matmul: ir.Node, original_shape: tuple, reshapes: List[ir.Node]
    ) -> bool:
        """Check if MatMul will work when reshapes are removed"""

        if not matmul.inputs or len(matmul.inputs) < 2:
            return False

        # Get what the input shapes will be
        input0_shape = self._get_shape_after_reshape_removal(matmul.inputs[0], original_shape, reshapes)
        input1_shape = self._get_shape_after_reshape_removal(matmul.inputs[1], original_shape, reshapes)

        # Check MatMul compatibility: shape0[..., M, K] × shape1[..., K, N]
        if len(input0_shape) < 2 or len(input1_shape) < 2:
            return False

        # Contraction dimension must match: input0[-1] == input1[-2]
        if input0_shape[-1] != input1_shape[-2]:
            return False

        # Check batch dimensions are broadcastable
        batch0 = input0_shape[:-2]
        batch1 = input1_shape[:-2]

        # Pad to same length
        max_len = max(len(batch0), len(batch1))
        batch0 = (1,) * (max_len - len(batch0)) + batch0
        batch1 = (1,) * (max_len - len(batch1)) + batch1

        # Check each batch dim is broadcastable
        for b0, b1 in zip(batch0, batch1):
            if b0 != b1 and b0 != 1 and b1 != 1:
                return False

        return True

    def _check_matmul2_works(
        self, matmul1: ir.Node, matmul2: ir.Node, original_shape: tuple, reshapes: List[ir.Node]
    ) -> bool:
        """Check if MatMul2 will work after MatMul1's output changes"""

        if not matmul1.inputs or len(matmul1.inputs) < 2:
            return False
        if not matmul2.inputs or len(matmul2.inputs) < 2:
            return False

        # Compute MatMul1's new output shape
        input0_shape = self._get_shape_after_reshape_removal(matmul1.inputs[0], original_shape, reshapes)
        input1_shape = self._get_shape_after_reshape_removal(matmul1.inputs[1], original_shape, reshapes)

        matmul1_output = self._compute_matmul_output(input0_shape, input1_shape)

        # MatMul2's input0 will be matmul1_output (through Softmax)
        # MatMul2's input1 might have a reshape before it
        matmul2_input1 = self._get_shape_after_reshape_removal(matmul2.inputs[1], original_shape, reshapes)

        # Check compatibility: matmul1_output[..., M, K] × matmul2_input1[..., K, N]
        if len(matmul1_output) < 2 or len(matmul2_input1) < 2:
            return False

        # Contraction dimension must match
        if matmul1_output[-1] != matmul2_input1[-2]:
            return False

        # Check batch dimensions are broadcastable
        batch0 = matmul1_output[:-2]
        batch1 = matmul2_input1[:-2]

        # Pad to same length
        max_len = max(len(batch0), len(batch1))
        batch0 = (1,) * (max_len - len(batch0)) + batch0
        batch1 = (1,) * (max_len - len(batch1)) + batch1

        # Check each batch dim is broadcastable
        for b0, b1 in zip(batch0, batch1):
            if b0 != b1 and b0 != 1 and b1 != 1:
                return False

        return True

    def _get_shape_after_reshape_removal(
        self, value: ir.Value | None, original_shape: tuple, reshapes: List[ir.Node]
    ) -> tuple:
        """Get what the shape will be after removing reshapes"""

        if value is None:
            return tuple()

        # Check if there's a reshape before this value
        reshape = self._find_reshape_before(value)

        if reshape and reshape in reshapes:
            # This reshape will be removed, so we need to trace back further
            # to find what the actual input shape will be

            # The reshape's input might have Transpose or other ops
            reshape_input = reshape.inputs[0] if reshape.inputs else None
            if reshape_input:
                # Check if there's a Transpose before the reshape
                transpose = reshape_input.producer()
                if transpose and transpose.op_type == "Transpose":
                    # Compute transpose output shape
                    transpose_input_shape = tuple(get_value_numeric_shape(transpose.inputs[0]))
                    perm = transpose.attributes.get("perm")
                    if perm:
                        perm_list = list(perm.value)
                        return tuple(transpose_input_shape[i] for i in perm_list)
                    else:
                        return tuple(reversed(transpose_input_shape))
                else:
                    # No transpose, use the reshape's input shape
                    return tuple(get_value_numeric_shape(reshape_input))
            else:
                return original_shape
        else:
            # No reshape or reshape not being removed, keep current shape
            return tuple(get_value_numeric_shape(value))

    def _compute_matmul_output(self, shape1: tuple, shape2: tuple) -> tuple:
        """Compute MatMul output shape: shape1[..., M, K] × shape2[..., K, N] = [..., M, N]"""

        # Get batch dims
        batch1 = shape1[:-2]
        batch2 = shape2[:-2]

        # Broadcast batch dims
        max_len = max(len(batch1), len(batch2))
        batch1 = (1,) * (max_len - len(batch1)) + batch1
        batch2 = (1,) * (max_len - len(batch2)) + batch2
        batch_out = tuple(max(b1, b2) for b1, b2 in zip(batch1, batch2))

        # Output shape
        m, n = shape1[-2], shape2[-1]
        return batch_out + (m, n)

    def _compute_concat_output_after_removal(
        self, concat: ir.Node, initial_reshapes: List[ir.Node], original_shape: tuple
    ) -> tuple:
        """Compute what concat output will be after removing initial reshapes"""

        # For each concat input, compute its shape after reshape removal
        input_shapes = []

        for concat_input in concat.inputs:
            if concat_input is None:
                continue

            # This is MatMul2's output
            matmul2 = concat_input.producer()
            if not matmul2 or matmul2.op_type != "MatMul":
                # Fallback to current shape
                input_shapes.append(tuple(get_value_numeric_shape(concat_input)))
                continue

            # Find MatMul1
            matmul1 = self._find_matmul1_before_matmul2(matmul2)
            if not matmul1:
                input_shapes.append(tuple(get_value_numeric_shape(concat_input)))
                continue

            # Compute MatMul1 output with new shapes
            matmul1_in0 = self._get_shape_after_reshape_removal(
                matmul1.inputs[0], original_shape, initial_reshapes
            )
            matmul1_in1 = self._get_shape_after_reshape_removal(
                matmul1.inputs[1], original_shape, initial_reshapes
            )
            matmul1_out = self._compute_matmul_output(matmul1_in0, matmul1_in1)

            # Compute MatMul2 output
            matmul2_in1 = self._get_shape_after_reshape_removal(
                matmul2.inputs[1], original_shape, initial_reshapes
            )
            matmul2_out = self._compute_matmul_output(matmul1_out, matmul2_in1)

            input_shapes.append(matmul2_out)

        if not input_shapes:
            return tuple()

        # Concat along axis
        concat_axis = concat.attributes.get("axis", ir.AttrInt64("axis", 0)).value
        if concat_axis < 0:
            concat_axis = len(input_shapes[0]) + concat_axis

        # Build output shape
        output_shape = list(input_shapes[0])
        output_shape[concat_axis] = sum(shape[concat_axis] for shape in input_shapes)

        return tuple(output_shape)
