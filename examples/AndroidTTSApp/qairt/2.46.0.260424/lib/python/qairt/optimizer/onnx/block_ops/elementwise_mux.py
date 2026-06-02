# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

from qairt.optimizer.onnx.block_ops.base import QnnOnnxBlockOp, qcom_block_op_domain

"""ONNX ElementWiseMux BlockOperator implementation."""


class ElementWiseMux(QnnOnnxBlockOp):
    """ONNX ElementWiseMux Block Operator (strict, 2-input form)."""

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1) -> None:
        self.name = "ElementWiseMux"
        self.min_opset = 15
        self.max_opset = 22
        if aisw_opset_version != 1:
            raise ValueError(
                f"Operator {self.name} is only supported in AISW opset "
                f"version 1 (domain: {qcom_block_op_domain()})."
            )
        super().__init__(onnx_opset_version, aisw_opset_version)

    def get_onn_func_proto(self, num_inputs: int, simplified_version=False) -> onnx.FunctionProto:
        """

        ElementWiseMux(cond, x0, x1, ..., x{m-1}) -> out

        There are two versions of this block function, the simplified one and the full one.

        The Simplified version has only several internal nodes, which can dramatically accelerate onnxruntime preparation.
        But it assumes:
            - x0/x1/x... are tensors having same shape
            - 'cond' is a int tensor with shape [1], and its value should be in [0, m] (m is not included)

        If anyone of this assumption cannot be satisfied, please use the full version.
        """
        if simplified_version:
            return self._get_onn_func_proto_version_simplified(num_inputs)
        else:
            return self._get_onn_func_proto_version_complete(num_inputs)

    def _get_onn_func_proto_version_simplified(self, num_inputs: int) -> onnx.FunctionProto:
        """

        ElementWiseMux(cond, x0, x1, ..., x{m-1}) -> out

        - 'cond' is a int tensor with shape [1]
        - assume x0/x1/x... are tensors having same shape
        - the output equals to x{i}, where 'i' is the value inside of 'cond'

        """

        from onnx import TensorProto, helper

        m = num_inputs
        assert m >= 1, "Need at least one candidate input"

        inputs = ["cond"] + [f"x{i}" for i in range(m)]
        outputs = ["out"]
        nodes: list = []

        # ── Step 1: Concat all xi along axis 0 (no Unsqueeze needed).
        #   raw_concat shape: [m * S[0], S[1], ...]
        nodes.append(
            helper.make_node(
                "Concat",
                inputs=[f"x{i}" for i in range(m)],
                outputs=["raw_concat"],
                axis=0,
            )
        )

        # ── Step 2: Build target shape [m, *S] dynamically.
        #   const_m  : 1-D INT64 tensor [m]
        #   shape_x0 : 1-D INT64 tensor [S[0], S[1], ...] (from Shape op)
        #   target_shape = Concat([m], shape_x0) → [m, S[0], S[1], ...]
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["const_m"],
                value=helper.make_tensor("", TensorProto.INT64, [1], [m]),
            )
        )
        nodes.append(
            helper.make_node(
                "Shape",
                inputs=["x0"],
                outputs=["shape_x0"],
            )
        )
        nodes.append(
            helper.make_node(
                "Concat",
                inputs=["const_m", "shape_x0"],
                outputs=["target_shape"],
                axis=0,
            )
        )

        # ── Step 3: Reshape raw_concat to [m, *S].
        nodes.append(
            helper.make_node(
                "Reshape",
                inputs=["raw_concat", "target_shape"],
                outputs=["stacked"],
            )
        )

        # ── Step 4: Squeeze cond from shape [1] to a scalar so that
        #   Gather returns shape [*S] instead of [1, *S].
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["cond_squeeze_axes"],
                value=helper.make_tensor("", TensorProto.INT64, [1], [0]),
            )
        )
        nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=["cond", "cond_squeeze_axes"],
                outputs=["cond_scalar"],
            )
        )

        # ── Step 5: Gather along axis 0 with scalar index → shape [*S].
        nodes.append(
            helper.make_node(
                "Gather",
                inputs=["stacked", "cond_scalar"],
                outputs=["out"],
                axis=0,
            )
        )

        return helper.make_function(
            domain="qti_aisw",
            fname="ElementWiseMux",
            inputs=inputs,
            outputs=outputs,
            nodes=nodes,
            opset_imports=[helper.make_opsetid("", self.opset.version), helper.make_opsetid("qti_aisw", 1)],
        )

    def _get_onn_func_proto_version_complete(self, num_inputs: int) -> onnx.FunctionProto:
        """
        ElementWiseMux(cond, x0, x1, ..., x{m-1}) -> out

        - Broadcasting: cond and all x_i may be broadcastable to a common shape S.
        - Out-of-range 'cond' indices produce zeros in the output.
        - Output dtype matches the candidates' dtype (derived from x0).
        """
        from onnx import TensorProto, helper

        assert num_inputs >= 1, "Need at least one candidate input"

        input_names = ["cond"] + [f"x{i}" for i in range(num_inputs)]
        nodes = []

        # === Constants ===
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["depth"],
                value=helper.make_tensor("depth", TensorProto.INT64, [], [num_inputs]),
                name="depth_const",
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["values_f32"],
                value=helper.make_tensor("values_f32", TensorProto.FLOAT, [2], [0.0, 1.0]),
                name="values_const",
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["axis0_tensor"],
                value=helper.make_tensor("axis0_tensor", TensorProto.INT64, [1], [0]),
                name="axis0_tensor_const",
            )
        )

        # === Zeros for broadcast-shape discovery ===
        # zc: INT64 zeros
        nodes.append(helper.make_node("Sub", inputs=["cond", "cond"], outputs=["zc"], name="zeros_cond"))
        # Cast zc to the dtype of x0 to avoid dtype mismatch in Add()
        nodes.append(
            helper.make_node("CastLike", inputs=["zc", "x0"], outputs=["zc_cast"], name="cast_zc_like_x0")
        )

        # zi: zeros for each candidate (original dtype of xi)
        z_names = []
        for i in range(num_inputs):
            zi = f"z{i}"
            nodes.append(helper.make_node("Sub", inputs=[f"x{i}", f"x{i}"], outputs=[zi], name=f"zeros_x{i}"))
            # (Optional safety) cast each zi to dtype of x0 to guarantee Add type-match
            zi_cast = f"z{i}_cast"
            nodes.append(
                helper.make_node("CastLike", inputs=[zi, "x0"], outputs=[zi_cast], name=f"cast_z{i}_like_x0")
            )
            z_names.append(zi_cast)

        # Accumulate zeros to force full broadcast shape across (cond and all candidates)
        prev = "zc_cast"
        for i in range(num_inputs):
            nxt = f"zb_{i}"
            nodes.append(
                helper.make_node("Add", inputs=[prev, z_names[i]], outputs=[nxt], name=f"bcast_accum_{i}")
            )
            prev = nxt
        zb = prev

        # Shape S = broadcasted shape across cond + all x_i
        nodes.append(helper.make_node("Shape", inputs=[zb], outputs=["S"], name="shape_S"))

        # === Expand cond and candidates to S ===
        nodes.append(
            helper.make_node("Expand", inputs=["cond", "S"], outputs=["cond_exp"], name="expand_cond")
        )

        expanded = []
        for i in range(num_inputs):
            xi_exp = f"x{i}_exp"
            nodes.append(
                helper.make_node("Expand", inputs=[f"x{i}", "S"], outputs=[xi_exp], name=f"expand_x{i}")
            )
            expanded.append(xi_exp)

        # === Stack expanded candidates along a new leading axis 0: [m, *S] ===
        unsq = []
        for i, xi_exp in enumerate(expanded):
            ui = f"{xi_exp}_u"
            nodes.append(
                helper.make_node("Unsqueeze", inputs=[xi_exp, "axis0_tensor"], outputs=[ui], name=f"unsq_{i}")
            )
            unsq.append(ui)
        nodes.append(
            helper.make_node("Concat", inputs=unsq, outputs=["stacked"], name="stack", axis=0)
        )  # [m, *S]

        # === OneHot over axis=0 (out-of-range -> all-zero rows) ===
        nodes.append(
            helper.make_node(
                "OneHot",
                inputs=["cond_exp", "depth", "values_f32"],
                outputs=["oh_f32"],
                name="one_hot",
                axis=0,
            )
        )

        # Cast mask to candidate dtype so Mul type-checks regardless of (float16/float32/int…)
        nodes.append(
            helper.make_node(
                "CastLike", inputs=["oh_f32", expanded[0]], outputs=["oh"], name="cast_mask_like_x0"
            )
        )

        # === Multiply and sum along axis=0 ===
        nodes.append(helper.make_node("Mul", inputs=["oh", "stacked"], outputs=["mul"], name="mul"))
        # In opset >= 13, axes go in as a tensor input; keepdims can stay as an attribute.
        nodes.append(
            helper.make_node(
                "ReduceSum", inputs=["mul", "axis0_tensor"], outputs=["out"], name="sum", keepdims=0
            )
        )

        func = helper.make_function(
            domain=qcom_block_op_domain(),
            fname="ElementWiseMux",
            inputs=input_names,
            outputs=["out"],
            nodes=nodes,
            opset_imports=[
                helper.make_opsetid("", self.opset.version),  # e.g., 15
                # helper.make_opsetid(qcom_block_op_domain(), 1)
            ],
        )
        return func


def add_element_wise_mux_op(model: ir.Model, expert_slot_num: int, simplified_version=False):
    onnx_opset_version = None
    for domain in ["", "ai.onnx", "main"]:
        onnx_opset_version = model.opset_imports[domain]
        break
    if (qcom_block_op_domain(), "ElementWiseMux", "") in model.functions:
        return
    assert onnx_opset_version is not None
    mux_op = ElementWiseMux(onnx_opset_version=onnx_opset_version)
    func_proto = mux_op.get_onn_func_proto(num_inputs=expert_slot_num, simplified_version=simplified_version)
    # func_proto.overload = str(expert_slot_num)
    func_ir = ir.serde.deserialize_function(func_proto)

    if func_ir.identifier() in model.functions:
        return
    model.functions[func_ir.identifier()] = func_ir
    if qcom_block_op_domain() not in model.graph.opset_imports:
        model.graph.opset_imports[qcom_block_op_domain()] = 1
    return
