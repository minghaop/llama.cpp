# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" IR graph utility """
import os
import logging
from typing import Optional, Dict, List

# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib
import qti.aisw.emitter.ir_graph_op_handler as op_handler
from qti.aisw.emitter.utils.config import is_custom_ir_op

IrGraph, IrOp, IrTraceTargetType = ir_graph_lib.IrGraph, ir_graph_lib.IrOp, ir_graph_lib.IrTraceTargetType
_logger = logging.getLogger('TorchEmitter')

try:
    from qti.aisw.converters.common import modeltools
except ImportError:
    try:
        # May need this import for QAIRT
        from qti.aisw.dlc_utils import modeltools
    except ImportError:
        raise ImportError("Unable to import DLC utilities")


def serialize_ir_graph_to_dlc(ir_graph: IrGraph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    _serialize_ir_graph_to_dlc(ir_graph, path, filename)


def _serialize_ir_graph_to_dlc(ir_graph: IrGraph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    dlc_serializer = modeltools.IrDlcSerializer(os.path.join(path, filename + ".dlc"))
    dlc_serializer.initialize()
    dlc_serializer.serialize(ir_graph)
    dlc_serializer.finish()


def get_ir_graph_from_dlc(dlc_path: str):
    """
    Obtain IR Graph from DLC.
    :param dlc_path: Path where dlc is located
    """
    dlc_reader_obj = get_dlc_reader(dlc_path)
    ir_graph = dlc_reader_obj.get_ir_graph()
    return ir_graph, dlc_reader_obj


def get_dlc_reader(dlc_path: str):
    """
    Obtain IR Graph from DLC.
    :param dlc_path: Path where dlc is located
    """
    dlc_reader = modeltools.IrDlcReader()
    dlc_reader.open(dlc_path)
    return dlc_reader


def validate_ir_graph(ir_graph: IrGraph, custom_op_type_to_module: Optional[Dict[str, str]] = None):
    """
    Validate that model preparer pro can parse the IR graph and generate a new PyTorch model.

    :param ir_graph: IR graph to validate
    """
    if custom_op_type_to_module is None:
        custom_op_type_to_module = {}

    all_ops = {op.name: (op.type, is_custom_ir_op(op)) for op in ir_graph.get_ops()}
    unknown_op_types = []

    for name, (op_type, is_custom_op) in all_ops.items():
        if (
                not (not is_custom_op and op_type in op_handler.ir_to_handler_dict.keys())
                and (is_custom_op and op_type not in custom_op_type_to_module.keys())
        ):
            unknown_op_types.append((name, op_type))

    if unknown_op_types:
        error_msg = (
            'The following "ops" have types not supported by model preparer or not registered in QNN converter parameters '
            "Unable to proceed with model preparation.\n"
        )
        for name, op_type in unknown_op_types:
            error_msg += name + "\t\t\t" + op_type + "\n"
        _logger.error(error_msg)
        raise RuntimeError(error_msg)

class IrOpTraceInfo:
    """ Creates an optrace info map for the ir_graph. """
    def __init__(self, ir_graph: IrGraph):
        """
        Initializes the Op trace map
        :param ir_graph: Ir Graph whose op trace map needs to be created
        """
        self._op_trace_map = {}
        for op_trace in ir_graph.get_trace_info().get_op_trace_info():
            self._op_trace_map[op_trace.get_name()] = op_trace

    def __getitem__(self, op_name) -> Optional[List[str]]:
        """
        Returns a list of op name of the src graph this op has been mapped to.
        :param op_name: op name to find trace for.
        :return: List of src op names or None
        """
        if op_name not in self._op_trace_map:
            return None

        op_trace = self._op_trace_map[op_name]
        src_op_names = []
        for trace_pair in op_trace.get_trace_pair():
            if trace_pair.get_type() == IrTraceTargetType.OP:
                src_op_names.append(trace_pair.get_name())

        return src_op_names
