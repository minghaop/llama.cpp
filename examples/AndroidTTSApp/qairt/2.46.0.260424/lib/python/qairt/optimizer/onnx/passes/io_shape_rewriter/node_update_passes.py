# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Registry for node update passes used by IOShapeRewriter.

Any pass that updates node constants (shape inputs, index arrays, etc.) when
AR/CL changes should be decorated with ``@register_node_update_pass``.  The
decorator can be applied from any file — the pass does not need to live in
``io_shape_rewriter.py``.

Example::

    from qairt.optimizer.onnx.passes.io_shape_rewriter.node_update_passes import (
        register_node_update_pass,
    )

    @register_node_update_pass
    class UpdateMyOpNodes(BasePredicatePass):
        def __init__(self, config: IOShapeRewriterConfig): ...
        def match(self, graph, node): ...
        def rewrite(self, graph, node, match_info=None): ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qairt.optimizer.onnx.passes.base import BasePredicatePass
    from qairt.optimizer.onnx.passes.io_shape_rewriter.config import IOShapeRewriterConfig

_NODE_UPDATE_REGISTRY: list[type[BasePredicatePass]] = []


def register_node_update_pass(cls: type[BasePredicatePass]) -> type[BasePredicatePass]:
    """Class decorator that registers a node update pass.

    Apply to every ``Update*`` pass so that :func:`get_node_update_passes`
    automatically includes it in the Step 5 pipeline without manual list
    maintenance.  The decorated class can live in any module — it is registered
    at import time as long as the module is imported before
    ``IOShapeRewriter.apply`` runs.
    """
    _NODE_UPDATE_REGISTRY.append(cls)
    return cls


def get_node_update_passes(config: IOShapeRewriterConfig) -> list[BasePredicatePass]:
    """Return instantiated node update passes in registration order."""
    return [cls(config) for cls in _NODE_UPDATE_REGISTRY]
