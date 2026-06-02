# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
DLC connected graph.

This module builds a connected graph (internal representation) from a DLC file
using the debugger's established TargetOp-based pattern. It lives in graph_op/
so it can be reused by both the validate_encodings component and the encodings
converter.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set

from qti.aisw.accuracy_debugger.graph_op.target_op import TargetOp


class DLCConnectedGraph:
    """Builds and exposes a connected graph from a DLC file.

    Instead of exposing raw DLC IR objects, this class constructs an internal
    TargetOp-based connected graph consistent with the rest of the debugger and
    exposes query methods used by context-aware validation rules and the
    encodings converter.

    The connected graph is represented as an activation_op_map:
        { output_tensor_name -> TargetOp }
    where each TargetOp holds its input/output tensor names, op type, and
    references to parent/children TargetOp objects.
    """

    def __init__(self, dlc_path: Path | str, logger: logging.Logger | None = None):
        """Initialize and build the connected graph from a DLC file.

        Args:
            dlc_path: Path to the DLC file.
            logger: Optional logger instance.
        """
        self.dlc_path = Path(dlc_path)
        self._logger = logger

        # activation_op_map: output_tensor_name -> TargetOp
        # This is the canonical internal representation used across the debugger.
        self._activation_op_map: Dict[str, TargetOp] = {}

        self._build_connected_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_connected_graph(self) -> None:
        """Build the connected TargetOp graph from the DLC IR.

        Uses the same IrDlcReader that ModelEncoding._load_from_dlc uses so we
        stay consistent with how the rest of the debugger opens DLC files.
        """
        if self._logger:
            self._logger.info(f"Building connected graph from DLC: {self.dlc_path}")

        from qti.aisw.converters.common import modeltools

        model_reader = modeltools.IrDlcReader()
        model_reader.open(str(self.dlc_path))
        ir_graph = model_reader.get_ir_graph()

        # Pass 1: create one TargetOp per IR op, recording input/output tensor names
        # and collecting static (param) tensor names for the data_type field.
        op_name_to_op: Dict[str, TargetOp] = {}
        for ir_op in ir_graph.get_ops():
            output_names = [t.name() for t in ir_op.outputs()]
            input_names = [t.name() for t in ir_op.inputs()]
            static_tensor_names = [
                t.name() for t in ir_op.inputs() if t.is_static_tensor()
            ]
            target_op = TargetOp(
                name=ir_op.name,
                op_type=ir_op.type,
                outputs=output_names,
                inputs=input_names,
                static_tensors=static_tensor_names,
            )
            op_name_to_op[ir_op.name] = target_op
            # Register every output tensor -> TargetOp in the activation map.
            for out_name in output_names:
                self._activation_op_map[out_name] = target_op

        # Pass 2: wire parent/children relationships between TargetOp objects.
        for target_op in op_name_to_op.values():
            for in_name in target_op.inputs:
                parent_op = self._activation_op_map.get(in_name)
                if parent_op and parent_op is not target_op:
                    if target_op not in parent_op.children_ops:
                        parent_op.children_ops = [target_op]
                    if parent_op not in target_op.parent_ops:
                        target_op.parent_ops = [parent_op]

        if self._logger:
            self._logger.info(
                f"Connected graph built: {len(op_name_to_op)} ops, "
                f"{len(self._activation_op_map)} activation tensors"
            )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @property
    def activation_op_map(self) -> Dict[str, TargetOp]:
        """Return the full activation -> TargetOp mapping.

        Returns:
            Dict[str, TargetOp]: Maps each output tensor name to its producing TargetOp.
        """
        return self._activation_op_map

    def get_op_type(self, tensor_name: str) -> str | None:
        """Return the op type of the op that produces *tensor_name*.

        Args:
            tensor_name: Output tensor name to look up.

        Returns:
            str | None: Op type string, or None if not found.
        """
        op = self._activation_op_map.get(tensor_name)
        return op.op_type if op else None

    def find_predecessor_tensors(self, tensor_name: str) -> List[str]:
        """Return the input tensor names of the op that produces *tensor_name*.

        Args:
            tensor_name: Output tensor name to look up.

        Returns:
            List[str]: Input tensor names of the producing op, or [] if not found.
        """
        op = self._activation_op_map.get(tensor_name)
        return op.inputs if op else []

    def find_concat_inputs(self, tensor_name: str) -> List[str]:
        """Return input tensor names when *tensor_name* is produced by a concat op.

        Args:
            tensor_name: Output tensor name of the potential concat op.

        Returns:
            List[str]: Input tensor names, or [] if not a concat op.
        """
        op = self._activation_op_map.get(tensor_name)
        if op and op.op_type and "concat" in op.op_type.lower():
            return op.inputs
        return []

    def is_reshape_op(self, tensor_name: str) -> bool:
        """Return True if *tensor_name* is produced by a reshape/view/flatten op.

        Args:
            tensor_name: Output tensor name to check.
        """
        op = self._activation_op_map.get(tensor_name)
        if op and op.op_type:
            t = op.op_type.lower()
            return "reshape" in t or "view" in t or "flatten" in t
        return False

    def is_transpose_op(self, tensor_name: str) -> bool:
        """Return True if *tensor_name* is produced by a transpose/permute op.

        Args:
            tensor_name: Output tensor name to check.
        """
        op = self._activation_op_map.get(tensor_name)
        if op and op.op_type:
            t = op.op_type.lower()
            return "transpose" in t or "permute" in t
        return False

    def is_concat_op(self, tensor_name: str) -> bool:
        """Return True if *tensor_name* is produced by a concat op.

        Args:
            tensor_name: Output tensor name to check.
        """
        op = self._activation_op_map.get(tensor_name)
        return bool(op and op.op_type and "concat" in op.op_type.lower())

    def get_all_tensor_names(self) -> Set[str]:
        """Return all output tensor names registered in the connected graph.

        Returns:
            Set[str]: Set of all tensor names.
        """
        return set(self._activation_op_map.keys())

    def get_all_op_names(self) -> Set[str]:
        """Return all unique op names in the connected graph.

        Returns:
            Set[str]: Set of all op names.
        """
        return {op.name for op in self._activation_op_map.values()}
