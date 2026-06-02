# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Squeeze->GroupSlice) -> (GroupSlice->Squeeze)
"""

from qairt.optimizer.onnx.passes.mha2sha.reorder_unsqueeze_gslice import (
    _ReorderUnSqueezeGroupsliceOrSqueezeGroupslice,
)


class ReorderSqueezeGroupslice(_ReorderUnSqueezeGroupsliceOrSqueezeGroupslice):
    """
    Transform subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Squeeze(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {

            a0,a1,a2... = GroupSlice(in_a)

            b0 = Squeeze(a0)
            b1 = Squeeze(a1)
            b2 = Squeeze(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Squeeze(in_a)
        }

    Also, encodings are updated
    """

    def __init__(self):
        super().__init__("Squeeze")

    @classmethod
    def get_output_axis_to_input_axis_map(cls, op_axes, output_rank):
        # for squeeze
        input_rank = output_rank + len(op_axes)
        axis_map = {}  # key is the output axis id, value is the input axis id
        output_i = 0
        for input_i in range(input_rank):
            if input_i in op_axes:
                continue
            axis_map[output_i] = input_i
            output_i += 1
        return axis_map
