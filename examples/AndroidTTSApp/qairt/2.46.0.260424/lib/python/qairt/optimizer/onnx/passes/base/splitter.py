# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Base Graph Splitter Classes
This module contains the base classes for graph splitting operations
"""

from abc import ABC, abstractmethod

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import PassConfig


class BaseGraphSplitter(ABC):
    """
    Base class for stateless graph splitters

    This class provides the foundation for all graph splitting passes.
    Splitters should be stateless, taking a model context as input and
    returning a list of model contexts without storing graph state in the splitter instance.
    """

    def __init__(self, config: PassConfig):
        """
        Initialize the splitter with configuration.

        Args:
            config: Pass-specific configuration for the splitter
        """
        self.config = config

    @abstractmethod
    def split(self, ctx: GraphContext) -> list[GraphContext]:
        """
        Apply the splitter to split the model into multiple models

        Args:
            ctx: The input context context containing the full model

        Returns:
            List of GraphContext objects, each containing a split of the original model
        """
        pass
