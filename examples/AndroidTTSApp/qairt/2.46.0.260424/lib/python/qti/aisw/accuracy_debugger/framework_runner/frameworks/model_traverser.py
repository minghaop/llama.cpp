# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Union

import onnx
from onnx import NodeProto


class ModelTraverser:
    """Class for traversing ONNX models and extracting node information.

    This class provides methods to:
    1. List all node names in topological order
    2. List node names between a start and end node
    3. List all node names of a specific type
    4. List all node names that are not of a specific type
    """

    def __init__(self, model_path: Path):
        """Initialize the ModelTraverser with an ONNX model.

        Args:
            model_path: Path to the ONNX model file

        Note:
            Only the model graph is stored to minimize memory usage.
            The full model object is loaded temporarily and then discarded.
        """
        self.model_path = model_path
        # Load model temporarily, extract graph, then discard model to reduce memory footprint
        model = onnx.load(self.model_path)
        self.graph = model.graph
        # model object will be garbage collected after this point
        del model
        self._topological_order_cache = None  # Cache for topological order
        self._build_graph_structure()

    def _get_node_identifier(self, node: NodeProto, idx: int) -> str:
        """Get identifier for a node (helper to reduce code duplication).

        Args:
            node: The ONNX node
            idx: Index of the node

        Returns:
            Node identifier string
        """
        if node.name:
            return node.name
        elif node.output:
            return node.output[0]
        else:
            return f"node_{idx}"

    def _build_graph_structure(self):
        """Build internal graph structure for efficient traversal."""
        # Initialize all data structures
        self.node_map = {}
        self.node_by_index = {}
        self.output_to_node = {}
        self.output_to_node_idx = {}
        self.input_to_nodes = {}
        self.input_to_node_indices = {}

        # Single pass to build node maps and tensor mappings
        for idx, node in enumerate(self.graph.node):
            node_identifier = self._get_node_identifier(node, idx)

            # Store node mappings
            self.node_map[node_identifier] = node
            self.node_by_index[idx] = node

            # Map output tensors to this node
            for output in node.output:
                self.output_to_node[output] = node_identifier
                self.output_to_node_idx[output] = idx

            # Map input tensors to this node
            for input_tensor in node.input:
                if input_tensor not in self.input_to_nodes:
                    self.input_to_nodes[input_tensor] = []
                    self.input_to_node_indices[input_tensor] = []
                self.input_to_nodes[input_tensor].append(node_identifier)
                self.input_to_node_indices[input_tensor].append(idx)

        # Build parent-child relationships
        self.children_map = {name: [] for name in self.node_map}
        self.parent_map = {name: [] for name in self.node_map}

        for node_identifier, node in self.node_map.items():
            for input_tensor in node.input:
                if input_tensor in self.output_to_node:
                    parent_name = self.output_to_node[input_tensor]
                    self.parent_map[node_identifier].append(parent_name)
                    self.children_map[parent_name].append(node_identifier)

    def get_nodes_in_topological_order(self) -> List[str]:
        """Get all node names in topological order.

        Returns:
            List of node names in topological order
        """
        # Kahn's algorithm for topological sorting
        in_degree = {name: len(parents) for name, parents in self.parent_map.items()}
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        topological_order = []

        while queue:
            # Sort to ensure deterministic ordering when multiple nodes have no dependencies
            queue_list = sorted(queue)
            queue.clear()
            queue.extend(queue_list)
            current = queue.popleft()
            topological_order.append(current)

            # Reduce in-degree for children
            for child in self.children_map[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Check if all nodes were processed (no cycles)
        if len(topological_order) != len(self.node_map):
            raise ValueError("Graph contains a cycle, cannot perform topological sort")

        return topological_order

    def get_nodes_between(
        self, start_node: Optional[str] = None, end_node: Optional[str] = None
    ) -> List[str]:
        """Get all node names between a start node and an end node (inclusive).

        This method finds all nodes that are on any path from start_node to end_node.
        If only start_node is provided, returns all nodes from start_node to graph outputs.
        If only end_node is provided, returns all nodes from graph inputs to end_node.
        If both are provided, returns all nodes on any path between them.

        Args:
            start_node: Name of the starting node (optional). If None, starts from graph inputs.
            end_node: Name of the ending node (optional). If None, goes to graph outputs.

        Returns:
            List of node names between start and end nodes in topological order

        Raises:
            ValueError: If start_node or end_node doesn't exist in the graph, or if both are None
        """
        if start_node is None and end_node is None:
            raise ValueError("At least one of start_node or end_node must be provided")

        # Validate nodes exist if provided
        if start_node is not None and start_node not in self.node_map:
            raise ValueError(f"Start node '{start_node}' not found in graph")
        if end_node is not None and end_node not in self.node_map:
            raise ValueError(f"End node '{end_node}' not found in graph")

        # Case 1: Only start_node provided - get all nodes from start to outputs
        if start_node is not None and end_node is None:
            forward_reachable = self._get_reachable_nodes_forward(start_node)
            all_nodes_topo = self.get_nodes_in_topological_order()
            return [node for node in all_nodes_topo if node in forward_reachable]

        # Case 2: Only end_node provided - get all nodes from inputs to end
        if start_node is None and end_node is not None:
            backward_reachable = self._get_reachable_nodes_backward(end_node)
            all_nodes_topo = self.get_nodes_in_topological_order()
            return [node for node in all_nodes_topo if node in backward_reachable]

        # Case 3: Both provided - get nodes on paths between them
        # Find all nodes reachable from start_node (forward reachability)
        forward_reachable = self._get_reachable_nodes_forward(start_node)

        # Find all nodes that can reach end_node (backward reachability)
        backward_reachable = self._get_reachable_nodes_backward(end_node)

        # Nodes between start and end are in both sets
        nodes_between = forward_reachable.intersection(backward_reachable)

        # Return in topological order
        all_nodes_topo = self.get_nodes_in_topological_order()
        return [node for node in all_nodes_topo if node in nodes_between]

    def get_nodes_between_tensors(self, input_tensor: str, output_tensor: str) -> List[str]:
        """Get all node names between an input tensor and an output tensor.

        This method finds all nodes on paths from nodes consuming input_tensor to
        the node producing output_tensor, operating on intermediate output tensors
        rather than node names (which may not be available before shape inference).

        Args:
            input_tensor: Name of the input tensor (nodes consuming this tensor are the start)
            output_tensor: Name of the output tensor (node producing this tensor is the end)

        Returns:
            List of node identifiers between the tensors in topological order.
            Node identifiers are: node.name if available, otherwise the first output tensor name.

        Raises:
            ValueError: If input_tensor or output_tensor doesn't exist in the graph,
                       or if no nodes are found for the tensors
        """
        # Find all nodes that consume the input_tensor (these are our starting points)
        start_node_indices = self._get_node_indices_consuming_tensor(input_tensor)

        # Find the node that produces the output_tensor (this is our end point)
        end_node_idx = self._get_node_index_producing_tensor(output_tensor)

        # Perform tensor-based traversal from start nodes to end node using the multiple tensors method
        nodes_between_indices = self._get_nodes_between_multiple_tensors(
            start_node_indices, [end_node_idx]
        )

        # Convert indices back to node identifiers and return in topological order
        all_nodes_topo = self.get_nodes_in_topological_order()
        node_identifiers_between = set()
        for idx in nodes_between_indices:
            node = self.node_by_index[idx]
            node_identifier = self._get_node_identifier(node, idx)
            node_identifiers_between.add(node_identifier)

        return [node for node in all_nodes_topo if node in node_identifiers_between]

    def get_subgraph_output_tensors(
        self, input_tensors: Optional[List[str]] = None, output_tensors: Optional[List[str]] = None
    ) -> List[str]:
        """Get all output tensors from nodes in a subgraph between input and output tensors.

        This method finds all nodes on paths from nodes consuming input_tensors to
        nodes producing output_tensors, and returns all output tensors from those nodes.

        Args:
            input_tensors: List of input tensor names (nodes consuming these are starting points).
                          Optional if output_tensors is provided.
            output_tensors: List of output tensor names (nodes producing these are ending points).
                           Optional if input_tensors is provided.

        Returns:
            List of all output tensor names from nodes in the subgraph, in topological order.

        Raises:
            ValueError: If both input_tensors and output_tensors are None, or if any tensor
                       doesn't exist in the graph, or if no nodes are found for the tensors.
        """
        if input_tensors is None and output_tensors is None:
            raise ValueError("At least one of input_tensors or output_tensors must be provided")

        # Convert None to empty list for uniform handling
        input_tensors = input_tensors or []
        output_tensors = output_tensors or []

        # Find all starting node indices (nodes consuming input tensors)
        start_node_indices = []
        for tensor in input_tensors:
            try:
                indices = self._get_node_indices_consuming_tensor(tensor)
                start_node_indices.extend(indices)
            except ValueError as e:
                raise ValueError(f"Error with input tensor '{tensor}': {str(e)}")

        # Find all ending node indices (nodes producing output tensors)
        end_node_indices = []
        for tensor in output_tensors:
            try:
                idx = self._get_node_index_producing_tensor(tensor)
                end_node_indices.append(idx)
            except ValueError as e:
                raise ValueError(f"Error with output tensor '{tensor}': {str(e)}")

        # Remove duplicates while preserving order
        start_node_indices = list(dict.fromkeys(start_node_indices))
        end_node_indices = list(dict.fromkeys(end_node_indices))

        # Get all nodes in the subgraph
        if start_node_indices and end_node_indices:
            # Both provided - find nodes on paths between them
            nodes_in_subgraph = self._get_nodes_between_multiple_tensors(
                start_node_indices, end_node_indices
            )
        elif start_node_indices:
            # Only input tensors provided - get all nodes reachable forward
            nodes_in_subgraph = self._get_nodes_reachable_forward_from_indices(start_node_indices)
        else:
            # Only output tensors provided - get all nodes that can reach the outputs
            nodes_in_subgraph = self._get_nodes_reachable_backward_from_indices(end_node_indices)

        # Convert node indices to identifiers and order by topological sort
        all_nodes_topo = self.get_nodes_in_topological_order()
        node_identifiers_in_subgraph = {
            self._get_node_identifier(self.node_by_index[idx], idx) for idx in nodes_in_subgraph
        }

        # Collect output tensors in topological order
        ordered_output_tensors = []
        for node_id in all_nodes_topo:
            if node_id in node_identifiers_in_subgraph:
                ordered_output_tensors.extend(self.node_map[node_id].output)

        return ordered_output_tensors

    def get_subgraph_nodes_by_type(
        self,
        op_type: Union[str, List[str]],
        input_tensors: Optional[List[str]] = None,
        output_tensors: Optional[List[str]] = None,
    ) -> List[str]:
        """Get output tensors from nodes of specific type(s) within a subgraph defined by input and output tensors.

        This method finds all nodes on paths from nodes consuming input_tensors to
        nodes producing output_tensors, then filters those nodes by operation type(s)
        and returns their output tensors.

        Args:
            op_type: The operation type(s) to filter by. Can be a single string
                    (e.g., 'Conv') or a list of strings (e.g., ['Conv', 'Relu', 'Add'])
            input_tensors: List of input tensor names (nodes consuming these are starting points).
                          Optional if output_tensors is provided.
            output_tensors: List of output tensor names (nodes producing these are ending points).
                           Optional if input_tensors is provided.

        Returns:
            List of output tensor names from nodes with the specified operation type(s) in the subgraph,
            in topological order.

        Raises:
            ValueError: If both input_tensors and output_tensors are None, or if any tensor
                       doesn't exist in the graph, or if no nodes are found for the tensors,
                       or if any of the requested node types don't exist in the graph.
        """
        if input_tensors is None and output_tensors is None:
            raise ValueError("At least one of input_tensors or output_tensors must be provided")

        # Convert None to empty list for uniform handling
        input_tensors = input_tensors or []
        output_tensors = output_tensors or []

        # Convert single string to list for uniform handling
        op_types = [op_type] if isinstance(op_type, str) else op_type

        # Validate that requested types exist in the graph
        self._validate_node_types_exist(op_types)

        # Find all starting node indices (nodes consuming input tensors)
        start_node_indices = []
        for tensor in input_tensors:
            try:
                indices = self._get_node_indices_consuming_tensor(tensor)
                start_node_indices.extend(indices)
            except ValueError as e:
                raise ValueError(f"Error with input tensor '{tensor}': {str(e)}")

        # Find all ending node indices (nodes producing output tensors)
        end_node_indices = []
        for tensor in output_tensors:
            try:
                idx = self._get_node_index_producing_tensor(tensor)
                end_node_indices.append(idx)
            except ValueError as e:
                raise ValueError(f"Error with output tensor '{tensor}': {str(e)}")

        # Remove duplicates while preserving order
        start_node_indices = list(dict.fromkeys(start_node_indices))
        end_node_indices = list(dict.fromkeys(end_node_indices))

        # Get all nodes in the subgraph
        if start_node_indices and end_node_indices:
            # Both provided - find nodes on paths between them
            nodes_in_subgraph = self._get_nodes_between_multiple_tensors(
                start_node_indices, end_node_indices
            )
        elif start_node_indices:
            # Only input tensors provided - get all nodes reachable forward
            nodes_in_subgraph = self._get_nodes_reachable_forward_from_indices(start_node_indices)
        else:
            # Only output tensors provided - get all nodes that can reach the outputs
            nodes_in_subgraph = self._get_nodes_reachable_backward_from_indices(end_node_indices)

        # Convert node indices to identifiers and filter by operation type
        all_nodes_topo = self.get_nodes_in_topological_order()
        node_identifiers_in_subgraph = set()

        for idx in nodes_in_subgraph:
            node = self.node_by_index[idx]
            # Only include nodes that match the specified operation type(s)
            if node.op_type in op_types:
                node_identifier = self._get_node_identifier(node, idx)
                node_identifiers_in_subgraph.add(node_identifier)

        # Collect output tensors in topological order
        result_output_tensors = []
        for node_name in all_nodes_topo:
            if node_name in node_identifiers_in_subgraph:
                result_output_tensors.extend(self.node_map[node_name].output)

        return result_output_tensors

    def _get_nodes_between_multiple_tensors(
        self, start_node_indices: List[int], end_node_indices: List[int]
    ) -> Set[int]:
        """Get all node indices between multiple start and end nodes.

        Args:
            start_node_indices: List of starting node indices
            end_node_indices: List of ending node indices

        Returns:
            Set of node indices between start and end nodes
        """
        # Find all nodes reachable forward from any start node
        forward_reachable = self._get_nodes_reachable_forward_from_indices(start_node_indices)

        # Find all nodes that can reach any end node backward
        backward_reachable = self._get_nodes_reachable_backward_from_indices(end_node_indices)

        # Nodes in the subgraph are in both sets
        return forward_reachable.intersection(backward_reachable)

    def _get_nodes_reachable_forward_from_indices(self, start_indices: List[int]) -> Set[int]:
        """Get all node indices reachable forward from a list of starting nodes.

        Args:
            start_indices: List of starting node indices

        Returns:
            Set of node indices reachable from the start nodes
        """
        reachable = set()
        queue = deque(start_indices)

        while queue:
            current_idx = queue.popleft()
            if current_idx in reachable:
                continue
            reachable.add(current_idx)

            # Get the current node and its output tensors
            current_node = self.node_by_index[current_idx]

            # For each output tensor, find nodes that consume it
            for output_tensor in current_node.output:
                if output_tensor in self.input_to_node_indices:
                    for child_idx in self.input_to_node_indices[output_tensor]:
                        if child_idx not in reachable:
                            queue.append(child_idx)

        return reachable

    def _get_nodes_reachable_backward_from_indices(self, end_indices: List[int]) -> Set[int]:
        """Get all node indices that can reach a list of ending nodes backward.

        Args:
            end_indices: List of ending node indices

        Returns:
            Set of node indices that can reach the end nodes
        """
        reachable = set()
        queue = deque(end_indices)

        while queue:
            current_idx = queue.popleft()
            if current_idx in reachable:
                continue
            reachable.add(current_idx)

            # Get the current node and its input tensors
            current_node = self.node_by_index[current_idx]

            # For each input tensor, find the node that produces it
            for input_tensor in current_node.input:
                if input_tensor in self.output_to_node_idx:
                    parent_idx = self.output_to_node_idx[input_tensor]
                    if parent_idx not in reachable:
                        queue.append(parent_idx)

        return reachable

    def _get_node_index_producing_tensor(self, tensor_name: str) -> int:
        """Get the node index that produces a given tensor.

        Args:
            tensor_name: Name of the tensor

        Returns:
            Index of the node that produces this tensor

        Raises:
            ValueError: If no node produces this tensor
        """
        if tensor_name in self.output_to_node_idx:
            return self.output_to_node_idx[tensor_name]

        # Check if it's a graph input
        for inp in self.graph.input:
            if inp.name == tensor_name:
                raise ValueError(
                    f"Tensor '{tensor_name}' is a graph input, not produced by any node"
                )

        raise ValueError(f"No node found that produces tensor '{tensor_name}'")

    def _get_node_producing_tensor(self, tensor_name: str) -> str:
        """Get the node that produces a given tensor.

        Args:
            tensor_name: Name of the tensor

        Returns:
            Name of the node that produces this tensor

        Raises:
            ValueError: If no node produces this tensor
        """
        if tensor_name in self.output_to_node:
            return self.output_to_node[tensor_name]

        # Check if it's a graph input
        for inp in self.graph.input:
            if inp.name == tensor_name:
                raise ValueError(
                    f"Tensor '{tensor_name}' is a graph input, not produced by any node"
                )

        raise ValueError(f"No node found that produces tensor '{tensor_name}'")

    def _get_node_indices_consuming_tensor(self, tensor_name: str) -> List[int]:
        """Get all node indices that consume a given tensor.

        Args:
            tensor_name: Name of the tensor

        Returns:
            List of indices of nodes that consume this tensor

        Raises:
            ValueError: If no node consumes this tensor
        """
        if tensor_name in self.input_to_node_indices:
            return self.input_to_node_indices[tensor_name]

        # Check if it's a graph output
        for out in self.graph.output:
            if out.name == tensor_name:
                # Find the node that produces this output
                if tensor_name in self.output_to_node_idx:
                    return [self.output_to_node_idx[tensor_name]]
                raise ValueError(
                    f"Tensor '{tensor_name}' is a graph output but no node produces it"
                )

        raise ValueError(f"No node found that consumes tensor '{tensor_name}'")

    def _get_reachable_nodes_forward(self, start_node: str) -> Set[str]:
        """Get all nodes reachable from start_node by following children.

        Args:
            start_node: Name of the starting node

        Returns:
            Set of node names reachable from start_node
        """
        visited = set()
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for child in self.children_map[current]:
                if child not in visited:
                    queue.append(child)

        return visited

    def _get_reachable_nodes_backward(self, end_node: str) -> Set[str]:
        """Get all nodes that can reach end_node by following parents.

        Args:
            end_node: Name of the ending node

        Returns:
            Set of node names that can reach end_node
        """
        visited = set()
        queue = deque([end_node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for parent in self.parent_map[current]:
                if parent not in visited:
                    queue.append(parent)

        return visited

    def _validate_node_types_exist(self, op_types: List[str]) -> None:
        """Validate that the requested node types exist in the graph.

        Args:
            op_types: List of operation types to validate

        Raises:
            ValueError: If any of the requested types don't exist in the graph
        """
        # Get all unique node types in the graph
        existing_types = set(node.op_type for node in self.node_map.values())

        # Check which requested types don't exist
        missing_types = [op_type for op_type in op_types if op_type not in existing_types]

        if missing_types:
            available_types = sorted(existing_types)
            raise ValueError(
                f"Node type(s) not found in graph: {missing_types}. "
                f"Available types: {available_types}"
            )

    def get_nodes_by_type(self, op_type: Union[str, List[str]]) -> List[str]:
        """Get all output tensors from nodes of a specific operation type or types.

        Args:
            op_type: The operation type(s) to filter by. Can be a single string
                    (e.g., 'Conv') or a list of strings (e.g., ['Conv', 'Relu', 'Add'])

        Returns:
            List of output tensor names from nodes with the specified operation type(s) in topological order

        Raises:
            ValueError: If any of the requested node types don't exist in the graph
        """
        # Convert single string to list for uniform handling
        op_types = [op_type] if isinstance(op_type, str) else op_type

        # Validate that requested types exist in the graph
        self._validate_node_types_exist(op_types)

        # Filter nodes by type
        nodes_of_type = [name for name, node in self.node_map.items() if node.op_type in op_types]

        # Collect output tensors in topological order
        all_nodes_topo = self.get_nodes_in_topological_order()
        output_tensors = []
        for node_name in all_nodes_topo:
            if node_name in nodes_of_type:
                output_tensors.extend(self.node_map[node_name].output)

        return output_tensors

    def get_nodes_not_of_type(self, op_type: Union[str, List[str]]) -> List[str]:
        """Get all output tensors from nodes that are NOT of a specific operation type or types.

        Args:
            op_type: The operation type(s) to exclude. Can be a single string
                    (e.g., 'Conv') or a list of strings (e.g., ['Conv', 'Relu', 'Add'])

        Returns:
            List of output tensor names from nodes that are NOT of the specified operation type(s) in topological order

        Raises:
            ValueError: If any of the requested node types don't exist in the graph
        """
        # Convert single string to list for uniform handling
        op_types = [op_type] if isinstance(op_type, str) else op_type

        # Validate that requested types exist in the graph
        self._validate_node_types_exist(op_types)

        # Filter nodes by excluding the specified types
        nodes_not_of_type = [
            name for name, node in self.node_map.items() if node.op_type not in op_types
        ]

        # Collect output tensors in topological order
        all_nodes_topo = self.get_nodes_in_topological_order()
        output_tensors = []
        for node_name in all_nodes_topo:
            if node_name in nodes_not_of_type:
                output_tensors.extend(self.node_map[node_name].output)

        return output_tensors

    def get_node_info(self, node_name: str) -> dict:
        """Get detailed information about a specific node.

        Args:
            node_name: Name of the node

        Returns:
            Dictionary containing node information

        Raises:
            ValueError: If node doesn't exist in the graph
        """
        if node_name not in self.node_map:
            raise ValueError(f"Node '{node_name}' not found in graph")

        node = self.node_map[node_name]
        return {
            "name": node_name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "parents": self.parent_map[node_name],
            "children": self.children_map[node_name],
        }
