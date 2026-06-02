# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" File contains the definition of the QcModule, base class for writing any graph module to be parsed using MAD """

import abc
import os
import logging
from abc import ABC

import numpy as np

from qti.aisw.graph_gen.mad.graph import ir_graph_gen as ir_graph_gen
from qti.aisw.graph_gen.mad.graph import parser as parser
from qti.aisw.graph_gen.mad.lib.op import QcOp
from qti.aisw.graph_gen.mad import utils as mad_utils


logger = logging.getLogger(__name__)

class QcModule(ABC):
    """
    Abstract base class for a Quantized Compute Module.

    This class provides the basic structure for defining modules that can be
    parsed and exported as computational graphs. Subclasses must implement
    the `__init__` and `forward` methods.

    Attributes:
        leaf_node (bool): Indicates if the module is a leaf node in the module hierarchy.
        list_of_ops (list): A list to store operations within the module.
    """
    leaf_node = False
    list_of_ops = []

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the QcModule. Subclasses must implement this method.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abc.abstractmethod
    def forward(self, *input, **kwargs):
        """
        Defines the computation performed at every call. Subclasses must implement this method.

        Args:
            *input: Variable positional arguments representing inputs.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def __call__(self, *input, **kwargs):
        """
        Enables the module instance to be called as a function, executing its forward pass.

        Args:
            *input: Variable positional arguments representing inputs.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.forward(*input, **kwargs)

    def export(self, input_shapes=None, weight_dict=None, pre_serialization_callback=None, save_dir = './', filename_prefix = 'sample_model', encoding_file_path=None):
        """
        Exports the module's computational graph to a DLC file.

        This method parses the module's `forward` method, generates an intermediate
        representation (IR) graph, and then serializes it. It also supports
        applying quantization encodings and splitting the graph.

        Args:
            input_shapes (list, optional): List of input tensor shapes. Defaults to None.
            weight_dict (dict, optional): Dictionary of weights for the graph. Defaults to None.
            pre_serialization_callback (callable, optional): A callback function to
                                                                manipulate the IR graph
                                                                before serialization.

            save_dir (str, optional): Directory to save the exported DLC file. Defaults to './'.
            filename_prefix (str, optional): Prefix for the exported DLC filename. Defaults to 'sample_model'.
            encoding_file_path (str, optional): Path to a JSON file containing quantization encodings. Defaults to None.

        Returns:
            Graph: The parsed computational graph.
        """
        parser_instance = parser.DagParser(self, input_tensors_data_info=input_shapes)
        graph = parser_instance.parse_model(self)
        graphGen = ir_graph_gen.IrGraphGenerator(graph, weight_dict, encoding_file_path, encoding_file_path is not None, graph_name = filename_prefix)
        graphGen.generate()

        # TODO: Pre-serialization for online graph to update tensor will not work.
        # As the input and output tensors are not getting reflected in the graph.
        # Need to discuss this with team and come up with a graph level update
        # function to recompile the information from the tensors.
        upd_input = graph.inputs
        upd_output = graph.outputs
        if pre_serialization_callback is not None:
            logger.info("Pre serialization invoked using provided callback..")
            pre_serialization_callback(graphGen.ir_graph)

        # Overrides are applied post per_serialization call to get the effect of
        # the updated graph names while applying encoding.
        graphGen.process_overrides()

        if len(graph.split_info) > 1:
            logger.info("Processing model splits.")
            mad_utils.split_and_serialize(graphGen.ir_graph, graph.split_info, save_dir, filename_prefix)
        else:
            save_path = os.path.join(save_dir, filename_prefix + '.dlc')
            logger.info(f"Serializing the ir_graph{save_path}")
            graphGen.save(save_path=save_path)
        return graph

    def _get_attr_dict(self):
        """
        Returns a dictionary of attributes for the module.

        Returns:
            dict: A dictionary of attributes.
        """
        return {}

    @property
    def layers(self):
        """
        Collects and returns a dictionary of layers (QcOp instances) and submodules (QcModule instances)
        contained within this module.

        Returns:
            dict: A dictionary where keys are layer/submodule names and values are the corresponding objects.
        """
        layers = {}
        for name, value in self.__dict__.items():
            if isinstance(value, QcModule):
                # Submodule
                layers[name] = value.layers
            elif isinstance(value, QcOp):
                # Op
                layers[name] = value

            elif isinstance(value, list):
                for i, obj in enumerate(value):
                    item_name = f'{name}.{i}'
                    if isinstance(obj, QcModule):
                        layers[item_name] = value[i].layers
                    elif isinstance(obj, QcOp):
                        layers[item_name] = value[i]

        return layers

    @property
    def attributes(self):
        """
        Collects and returns a dictionary of basic attributes (int, float, numpy array, list)
        defined within this module.

        Returns:
            dict: A dictionary where keys are attribute names and values are the corresponding data.
        """
        attributes = {}
        for name, value in self.__dict__.items():
            if isinstance(value, (int, float, np.ndarray, list)):
                attributes[name] = value

        return attributes

    @property
    def submodules(self):
        """
        Collects and returns a dictionary of QcModule instances directly contained within this module.

        Returns:
            dict: A dictionary where keys are submodule names and values are the QcModule objects.
        """
        submodules = {}
        for name, value in self.__dict__.items():
            if isinstance(value, QcModule):
                submodules[name] = value
            elif isinstance(value, list):
                for i, obj in enumerate(value):
                    item_name = f'{name}.{i}'
                    if isinstance(obj, QcModule):
                        submodules[item_name] = obj

        return submodules


class ModuleMarker:
    """
    A utility class providing static methods to mark specific points in a module's execution flow,
    such as split points for graph partitioning.
    """
    @staticmethod
    def split_at(*args):
        """
        Marks a point in the module where the graph can be split.

        Args:
            *args: Variable positional arguments representing the tensors at the split point.
        """
        logger.info(f"Split marker is called with arguments: {args}")
