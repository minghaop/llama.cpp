# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the base ir rewriter class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.utils.ir_extra_info import (
    GraphExtraInfo,
    VariableExtraInfo,
)
from qairt.optimizer.onnx.utils.utils import (
    get_shape_of_slice,
    get_value_numeric_shape,
    has_static_shape_on_value,
)


@dataclass
class PassConfig(ABC):
    """Abstract base class for pass-specific configuration"""

    pass


class BasePass(ABC):
    """
    Base class for stateless graph rewriters

    This class provides the foundation for all graph rewriting passes
    Passes should be stateless, taking a graph context as input and
    returning a result without storing graph state in the pass instance
    """

    def __init__(self, config: PassConfig | None = None):
        """
        Initialize the rewriter with optional configuration.

        Args:
            config: Optional pass-specific configuration
        """
        if config is None and hasattr(self, "Config"):
            config_cls = getattr(self, "Config")
            if issubclass(config_cls, PassConfig):
                config = config_cls()

        self.config = config
        self.curr_pass_rewrite_uid = 0

    @abstractmethod
    def apply(self, ctx: GraphContext) -> int:
        """
        Apply the rewriter on the model context.

        Args:
            ctx: The model context containing the ir.Graph and metadata

        Returns: The number of nodes that were rewritten
        """
        # curr_pass_rewrite_uid should be incremented in the overriding method

    def get_curr_pass_name(self):
        """
        Get name of the current pass
        every pass has a unique name
        """
        return self.__class__.__name__ + f"[{self.curr_pass_rewrite_uid}]"

    def mark_value_as_copy(self, graph: ir.Graph, copy_from: ir.Value, value: ir.Value):
        """
        Record the value is copied from copy_from,
        which means they should have
        - same numerical value
        - shape
        - dtype
        - same encodings
        - same safetensors if they have
        - same updatable attribute if they have

        Args:
            graph: ir.Graph instance
            copy_from: the tensor copied from
            value: the tensor copied to
        """
        assert isinstance(copy_from.meta["extra_info"], VariableExtraInfo)
        assert isinstance(graph.meta["extra_info"], GraphExtraInfo)
        value.meta["extra_info"] = copy_from.meta["extra_info"].copy()
        graph.meta["extra_info"].record_copy(copy_from.name, value.name, self.get_curr_pass_name())
        value.shape = copy_from.shape
        if copy_from.dtype is not None:
            value.dtype = copy_from.dtype

    # pylint: disable=[too-many-arguments, too-many-positional-arguments]
    def mark_value_as_slice(
        self,
        graph: ir.Graph,
        slice_from: ir.Value,
        value: ir.Value,
        axis,
        start,
        end,
        batch_slice_id,
        head_slice_id,
    ):
        """
        Record the value is a slice of slice_from,
        shape/dtype of value will be infered automatically

        Args:
            graph: ir.Graph instance
            slice_from: the tensor slice from
            value: the tensor sliced to
            axis: the slice axis
            start: start of the slice
            end: end of the slice
            batch_slice_id: the batch slice id
            head_slice_id: the head slice id
        """
        value.meta["extra_info"] = slice_from.meta["extra_info"].slice(axis, start, end)
        graph.meta["extra_info"].record_slicing(
            slice_from.name,
            value.name,
            self.get_curr_pass_name(),
            axis,
            start,
            end,
            batch_slice_id,
            head_slice_id,
        )
        # shape infer
        if value.shape is None and has_static_shape_on_value(slice_from):
            value.shape = ir.Shape(
                get_shape_of_slice(get_value_numeric_shape(slice_from), [axis], [start], [end])
            )
        if slice_from.dtype is not None:
            value.dtype = slice_from.dtype


@dataclass
class PredicatePassExtraInfo(ABC):
    model_ir: ir.Model


@dataclass
class MatchInfoProtocol(ABC):
    """Interface for match information passed between match and rewrite"""

    @property
    def extra_info(self) -> PredicatePassExtraInfo:
        # PredicatePass preserved variable
        return getattr(self, "_internal_extra_info")

    @extra_info.setter
    def extra_info(self, info: PredicatePassExtraInfo):
        # PredicatePass preserved variable
        if hasattr(self, "_internal_extra_info"):
            assert isinstance(getattr(self, "_internal_extra_info"), PredicatePassExtraInfo), (
                "field name '_internal_extra_info' is preserved"
            )
        setattr(self, "_internal_extra_info", info)


class BasePredicatePass(BasePass):
    """Base rewriter for match->rewrite pattern"""

    TRAVERSAL_ORDER: Literal["top_down", "bottom_up"] = "top_down"

    @abstractmethod
    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfoProtocol:
        """Check if the node can be rewritten"""
        pass

    @abstractmethod
    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        """Rewrite the node, return True if rewritten"""
        pass

    def apply(self, ctx: GraphContext) -> int:
        self.pre_rewrite_hook(ctx)
        count = 0
        if self.TRAVERSAL_ORDER == "top_down":
            nodes = list(ctx.graph_ir)
        elif self.TRAVERSAL_ORDER == "bottom_up":
            nodes = list(reversed(ctx.graph_ir))
        else:
            raise ValueError(f"unknown TRAVERSAL_ORDER '{self.TRAVERSAL_ORDER}'")

        for n in nodes:
            if n.graph is not ctx.graph_ir:
                # since we are traversing nodes in a cached list
                # the cache maybe out-dated, so we need to check whether the node exist in the graph
                continue
            match_result = self.match(ctx.graph_ir, n)

            if match_result:
                match_info = match_result if isinstance(match_result, MatchInfoProtocol) else None
                if match_info:
                    match_info.extra_info = PredicatePassExtraInfo(model_ir=ctx.model_ir)

                self.pre_each_rewrite_hook(ctx)
                rewrite_success = self.rewrite(ctx.graph_ir, n, match_info)
                if rewrite_success:
                    count += 1
                    self.curr_pass_rewrite_uid += 1
                self.post_each_rewrite_hook(ctx, rewrite_success)

        self.post_rewrite_hook(ctx)
        return count

    def pre_each_rewrite_hook(self, ctx: GraphContext):
        """Hook to execute before each rewrite"""
        pass

    def post_each_rewrite_hook(self, ctx: GraphContext, rewrite_success: bool):
        """Hook to execute after each rewrite is finished"""
        pass

    def pre_rewrite_hook(self, ctx: GraphContext):
        """Hook to execute before any rewrite is started"""
        pass

    def post_rewrite_hook(self, ctx: GraphContext):
        """Hook to execute after all possible rewrite is finished"""
        pass
