import copy
import logging
from collections import deque
from typing import TypeAlias

import onnx_ir as ir
import rich
from onnxscript.optimizer import remove_unused_nodes

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BaseGraphSplitter
from qairt.optimizer.onnx.passes.splitters.config import LLMSplitterConfig
from qairt.optimizer.onnx.passes.splitters.utils.pretty_print import (
    PrettyPrintConstants,
    bold_text,
    create_rich_table,
)
from qairt.optimizer.onnx.utils import get_embedding_node, get_input_ids
from qairt.optimizer.onnx.utils.encodings import EncKind
from qairt.optimizer.utils.logger import logger


class LLMSplitter(BaseGraphSplitter):
    Config: TypeAlias = LLMSplitterConfig
    _EMBEDDING_SPLIT_IO_NAME = "inputs_embeds"

    def __init__(self, config: Config):
        super().__init__(config)
        # Manually assign the config type for mypy
        self.config: LLMSplitter.Config = config

    def split(self, ctx: GraphContext) -> list[GraphContext]:
        """
        Splits the given ONNX model into multiple sub-models.
        This function splits the model in-place into the specified number of sub-models
        It supports splitting embeddings and lm_head
        Args:
            ctx: The GraphContext to split
        Returns:
            List[GraphContext]: List of GraphContext instances for each split
        Model Topology:
                   │ ←─────────  layers[0]  ────────────→ │       │ ←─────────  layers[-1]  ───────────-----─→ │
                   │                                      │       │                                            │
        embed  ────┬─────────── add 0 ─┬────────── add 1 ──  ┄┄ ┄─┬─────────────── add(n-2) ─┬──────────── add(n-1) ─── lmhead
                 ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑       ↑ └─ norm ─ attn ─┘        └─ norm ─ ffn ─┘  ↑
                 │                                        │       │                                            │
                 │                                        │       │                                            │
                valid splitting points
        """
        splits = self._split_model(ctx.model_ir)
        return [GraphContext(model_ir) for model_ir in splits]

    def _split_model(self, model: ir.Model) -> list[ir.Model]:
        """
        Internal method that performs the actual model splitting logic.

        Args:
            model: The ONNX model to be split
        Returns:
            List of split models
        """
        if self.config.num_splits <= 1:
            logger.debug("skip split because num_splits=%d", self.config.num_splits)
            return [model]

        original_num_splits = self.config.num_splits
        num_splits = self.config.num_splits

        # Create node_idx mapping using node id since some nodes may have empty names
        node_idx = {id(node): i for i, node in enumerate(model.graph)}

        def can_visit(src: ir.Node, dst: ir.Node) -> bool:
            """Whether node 'dst' can be visited starting from node 'src'"""
            q = deque([src])

            while q:
                curr = q.popleft()

                if node_idx[id(curr)] > node_idx[id(dst)]:
                    return False

                if curr is src:
                    consumers = [node for node in curr.outputs[0].consumers() if node is not dst]
                else:
                    consumers = [node for node in curr.outputs[0].consumers()]

                if dst in consumers:
                    return True

                q.extend(consumers)

            return False

        def is_residual_add(node: ir.Node) -> bool:
            """
            Return True if 'node' is a residual add
            """

            def is_non_constant_input(value: ir.Value) -> bool:
                """
                Return True if 'value' is neither a constant input nor a graph input
                'value' is Constant if either 'value.const_value' is valid or if the producer op is Constant
                'value' is graph input if value.producer() is None
                """
                if value.const_value is not None:
                    return False

                producer = value.producer()
                return producer is not None and producer.op_type != "Constant"

            if node.op_type == "Add":
                input1 = node.inputs[0]
                input2 = node.inputs[1]

                if input1 is None or input2 is None:
                    return False

                if not is_non_constant_input(input1) or not is_non_constant_input(input2):
                    return False

                # An essential condition for a residual add is that
                # One of its inputs is consumed by multiple nodes
                if len(input1.consumers()) == 1 and len(input2.uses()) > 1:
                    # To satisfy mypy
                    input2_producer = input2.producer()
                    assert input2_producer is not None

                    return can_visit(input2_producer, node)

                elif len(input1.consumers()) > 1 and len(input2.uses()) == 1:
                    # To satisfy mypy
                    input1_producer = input1.producer()
                    assert input1_producer is not None

                    return can_visit(input1_producer, node)

            return False

        def copy_subgraph(start_tensor: ir.Value, end_tensor: ir.Value, name: str = "graph") -> ir.Graph:
            """
            Create a new ir.Graph based on the start_tensor end_tensor
            Given a start_tensor and an end tensor as the input and output of the subgraph,
            this function will replicate all the nodes, its inputs, outputs and initializers in the subgraph
            Any graph inputs/outputs between these nodes will also be the inputs/outputs of the subgraph
            """

            def create_new_value(value: ir.Value) -> ir.Value:
                """
                Create a new ir.Value instance from the passed 'value'
                This is required to create new ir.Value objects(tensors) in the new split graph
                as ir.Value cannot be used across graphs
                """
                new_value = ir.Value(name=value.name, shape=value.shape, type=value.type)

                # If the value is an initializer, copy the constant value
                if value.is_initializer():
                    new_value.const_value = value.const_value

                # If the value object has some meta data, copy it to the new ir.Value object
                # The metadata might include
                # 1. named encodings (base encodings + encodings for each lora adapter)
                # 2. named safetensors (weights for each lora adapter)
                # 3. Whether the tensor is updatable (lora tensor)
                if "extra_info" in value.meta:
                    new_value.meta["extra_info"] = value.meta["extra_info"].copy()

                return new_value

            def create_new_node(node: ir.Node, node_inputs: list[ir.Value], node_outputs: list[ir.Value]):
                """
                Create a new ir.Node instance from the passed 'node' object and its inputs and outputs
                This is required to create new nodes in the new split graph
                as nodes cannot be used across graphs
                """
                new_node = ir.Node(
                    domain=node.domain,
                    inputs=node_inputs,
                    outputs=node_outputs,
                    op_type=node.op_type,
                    attributes=node.attributes,
                    version=node.version,
                    name=node.name,
                )

                return new_node

            value_map = {}
            start_index = node_idx[id(start_tensor.consumers()[0])]

            # To satisfy mypy
            end_tensor_producer = end_tensor.producer()
            assert end_tensor_producer is not None

            end_index = node_idx[id(end_tensor_producer)] + 1

            # Pre-compute set of graph-output tensors whose producer node falls
            # inside [start_index, end_index) but whose inputs all come from an earlier
            # shard (node_idx < start_index, or graph inputs/initializers).
            # Skip these nodes here and process them in third pass
            deferred_graph_outputs: set[str] = set()
            for graph_out in model.graph.outputs:
                producer = graph_out.producer()
                if producer is None:
                    continue
                prod_idx = node_idx.get(id(producer))
                if prod_idx is None or prod_idx < start_index or prod_idx >= end_index:
                    continue
                # This producer is inside our range. Check if ALL of its inputs were
                # produced strictly before start_index (i.e., belong to an earlier shard).
                if all(
                    inp is None
                    or inp.is_graph_input()
                    or inp.is_initializer()
                    or (
                        inp.producer() is not None
                        and node_idx.get(id(inp.producer()), start_index) < start_index
                    )
                    for inp in producer.inputs
                ):
                    if graph_out.name is not None:
                        deferred_graph_outputs.add(graph_out.name)

            subgraph_nodes = []
            subgraph_inputs = []
            subgraph_outputs = []
            subgraph_initializers = []

            # Track mapping from new nodes to their original nodes for sorting
            new_node_to_original = {}

            # Replicate the node and add it to 'subgraph_nodes' to create a split model
            for node in model.graph[start_index:end_index]:
                # Skip nodes deferred to an earlier shard (e.g. in SHA models, the Concat
                # nodes that aggregate per-head key/value tensors into KV output tensors are
                # placed at the end of the ONNX graph but belong to an earlier layer range).
                # Their inputs must not be added to this shard's subgraph_inputs.
                if any(out.name in deferred_graph_outputs for out in node.outputs):
                    continue
                node_inputs: list = []
                for inp in node.inputs:
                    if inp:
                        if inp.name not in value_map:
                            new_value = create_new_value(inp)

                            if inp.is_graph_input() or inp is start_tensor:
                                subgraph_inputs.append(new_value)
                            elif inp.is_initializer():
                                subgraph_initializers.append(new_value)

                            value_map[inp.name] = new_value

                        node_inputs.append(value_map[inp.name])
                    else:
                        # Preserve None for optional inputs to maintain correct input order
                        node_inputs.append(None)

                node_outputs = []
                for out in node.outputs:
                    if out.name not in value_map:
                        new_value = create_new_value(out)

                        if (
                            out.is_graph_output() or out is end_tensor
                        ) and out.name not in deferred_graph_outputs:
                            subgraph_outputs.append(new_value)

                        value_map[out.name] = new_value

                    node_outputs.append(value_map[out.name])

                new_node = create_new_node(node, node_inputs, node_outputs)
                # Track which original node this new node came from
                new_node_to_original[id(new_node)] = node
                subgraph_nodes.append(new_node)

            # If an input to any node in the graph is
            #       1. Neither a graph or split input
            #       2. Nor a part of the split
            # Then we need to rebuild that subgraph in the current split
            # Examples of such instances are
            # 1. ['lora_alpha' -> Pad -> Reshape] -> Gather
            # 2. Alibi positional encodings

            visited = set()
            all_node_outputs = set()

            for node in subgraph_nodes:
                all_node_outputs.update([out.name for out in node.outputs])

            for node in model.graph[start_index:end_index]:
                if any(out.name in deferred_graph_outputs for out in node.outputs):
                    continue
                for inp in node.inputs:
                    if (
                        inp
                        and value_map[inp.name] not in subgraph_inputs
                        and inp.producer()
                        and inp.name not in all_node_outputs
                    ):
                        # Do a breadth-first search in reverse to construct the subgraph
                        q = deque([inp.producer()])

                        while q:
                            curr = q.popleft()

                            if curr and curr not in visited:
                                visited.add(curr)

                                q.extend([inp.producer() for inp in curr.inputs if inp and inp.producer()])

                                # Replicate node 'curr' if it is not in subgraph_nodes
                                # but if its output is consumed by any node in subgraph_nodes list
                                node_inputs: list = []  # type: ignore[no-redef]
                                for curr_inp in curr.inputs:
                                    if curr_inp:
                                        if curr_inp and curr_inp.name not in value_map:
                                            new_value = create_new_value(curr_inp)

                                            if curr_inp.is_graph_input():
                                                subgraph_inputs.append(new_value)
                                            elif curr_inp.is_initializer():
                                                subgraph_initializers.append(new_value)

                                            value_map[curr_inp.name] = new_value

                                        node_inputs.append(value_map[curr_inp.name])
                                    else:
                                        # Preserve None for optional inputs to maintain correct input order
                                        node_inputs.append(None)

                                node_outputs = []
                                for out in curr.outputs:
                                    if out is not inp and out.name not in value_map:
                                        new_value = create_new_value(out)

                                        if out.is_graph_output():
                                            subgraph_outputs.append(new_value)

                                        value_map[out.name] = new_value

                                    node_outputs.append(value_map[out.name])

                                all_node_outputs.update([out.name for out in node_outputs])

                                new_node = create_new_node(curr, node_inputs, node_outputs)
                                # Track which original node this new node came from
                                new_node_to_original[id(new_node)] = curr
                                subgraph_nodes.insert(0, new_node)

            # Third pass: pick up graph-output nodes whose inputs are all resolved in
            # value_map but were missed by the main loop (e.g. SHA Concat/KV nodes
            # placed out-of-order at the end of the graph by the ONNX exporter).

            for graph_out in model.graph.outputs:
                if graph_out.name in value_map:
                    # Already captured by the main loop
                    continue
                producer = graph_out.producer()
                if producer is None:
                    continue
                # Check whether all inputs of this producer are already in value_map
                # (meaning they were produced within this shard's layer range).
                if not all(inp is None or inp.name in value_map for inp in producer.inputs):
                    continue

                # All inputs are resolved — create just this node (no backward traversal).
                node_inputs_bwd: list = []
                for curr_inp in producer.inputs:
                    if curr_inp:
                        node_inputs_bwd.append(value_map[curr_inp.name])
                    else:
                        node_inputs_bwd.append(None)

                node_outputs_bwd = []
                for out in producer.outputs:
                    if out.name not in value_map:
                        new_value = create_new_value(out)
                        if out.is_graph_output():
                            subgraph_outputs.append(new_value)
                        value_map[out.name] = new_value
                    node_outputs_bwd.append(value_map[out.name])

                all_node_outputs.update([out.name for out in node_outputs_bwd if out])

                new_node = create_new_node(producer, node_inputs_bwd, node_outputs_bwd)
                new_node_to_original[id(new_node)] = producer
                subgraph_nodes.append(new_node)

            # When constructing a subgraph, nodes are always inserted at index 0
            # This might result in nodes getting added in non-topological ordering
            # Sort by the original node's index using the mapping we created
            subgraph_nodes.sort(key=lambda _node: node_idx[id(new_node_to_original[id(_node)])])

            split_graph = ir.Graph(
                name=name,
                inputs=subgraph_inputs,
                outputs=subgraph_outputs,
                nodes=subgraph_nodes,
                initializers=subgraph_initializers,
                opset_imports=model.graph.opset_imports,
            )

            # Copy graph-level metadata from the original graph
            # This includes tracing info, encoding_versions, quantizer_args, etc
            if "extra_info" in model.graph.meta:
                split_graph.meta["extra_info"] = copy.deepcopy(model.graph.meta["extra_info"])

            return split_graph

        # Step 1: Identify all residual adds in the model
        residual_adds = [node.outputs[0] for node in model.graph if is_residual_add(node)]

        # Step 2: Filter to keep only post-FFN residual adds (every other one)
        # In an LLM, the structure is: Attention → Add 1 → FFN → Add 2 → Attention → Add 3 → FFN → Add 4
        # We keep Add 2, Add 4, etc. (indices 1, 3, 5, ...) which represent complete layer boundaries
        residual_adds = residual_adds[1::2]

        logger.debug(f"Found {len(residual_adds)} post-FFN residual adds (valid split points)")

        # Step 3: Calculate the maximum number of possible splits
        # Example: 24 layers can create up to 24 + 0 + 1 = 25 possible splits
        #          If split_embedding is set, 24 + 1 + 1 = 26 possible splits
        n_possible_splits = len(residual_adds) + int(self.config.split_embedding) + 1

        # Step 4: Validate that we have enough layers for the requested splits
        if n_possible_splits < self.config.num_splits:
            raise ValueError(
                f"Not enough layers in the model to properly split.\n"
                f"  Available split points: {n_possible_splits}\n"
                f"  Requested splits: {self.config.num_splits}\n"
                f"  Layers in model: {len(residual_adds)}\n"
                f"  split_embedding: {self.config.split_embedding}\n"
                f"  Suggestion: Reduce num_splits to {n_possible_splits} or fewer"
            )

        input_ids = get_input_ids(model)
        embeddings = None
        if self.config.split_embedding:
            if not input_ids:
                raise ValueError(
                    "`split_embedding` set to True, but no input named 'input_ids' or no model input to a Gather op"
                )

            embedding_node = get_embedding_node(input_ids)
            embeddings = embedding_node.outputs[0]
            if embeddings is not None and embeddings.name != self._EMBEDDING_SPLIT_IO_NAME:
                logger.warning(
                    "Renaming embedding split boundary tensor '%s' to '%s'",
                    embeddings.name,
                    self._EMBEDDING_SPLIT_IO_NAME,
                )
                embeddings.name = self._EMBEDDING_SPLIT_IO_NAME
            # NOTE: Hack to keep only the embedding in first split
            if model.graph[0].op_type == "Pad":
                model.graph.insert_before(model.graph[0], [model.graph[1]])
                # Rebuild node_idx after modifying the graph order
                node_idx = {id(node): i for i, node in enumerate(model.graph)}
            num_splits -= 1

        lm_head = None
        if self.config.split_lm_head:
            lm_head = residual_adds.pop()
            num_splits -= 1

        # Step 5: Build the list of split boundary tensors
        split_tensors: list = [input_ids]

        if self.config.split_embedding:
            split_tensors.append(embeddings)

        # Step 6: Distribute layers evenly across splits
        # ------------------------------------------------
        # Goal: Minimize variance by giving each split either floor(n/k) or ceil(n/k) layers
        #
        # Algorithm:
        #   1. Calculate base layers per split: n // k (integer division)
        #   2. Calculate remainder: n % k (how many splits need one extra layer)
        #   3. First 'remainder' splits get (base + 1) layers, rest get base layers
        #
        # Example: 23 layers → 7 splits
        #   - base = 23 // 7 = 3 layers per split
        #   - extra = 23 % 7 = 2 (first 2 splits get one extra)
        #   - Distribution: [4, 4, 3, 3, 3, 3, 3]
        #
        # Why cumulative_idx starts at -1?
        #   - residual_adds[i] represents the output after layer i
        #   - Starting at -1 means "before any layers have been processed"
        #   - After adding layers for split 1: -1 + 4 = 3 → residual_adds[3] (after layer 3)
        #   - This correctly marks the boundary between splits

        layers_per_split = len(residual_adds) // num_splits  # Base layers each split gets
        extra_layers = len(residual_adds) % num_splits  # How many splits get +1 layer

        cumulative_idx = -1  # Start before first layer (conceptually at "layer -1")

        for i in range(1, num_splits):
            # First 'extra_layers' splits get one additional layer
            layers_in_current_split = layers_per_split + (1 if i <= extra_layers else 0)

            # Accumulate to find the split boundary index
            cumulative_idx += layers_in_current_split

            # residual_adds[cumulative_idx] is the output after 'cumulative_idx' layers
            # This becomes the boundary between current split and next split
            split_tensors.append(residual_adds[cumulative_idx])

            logger.debug(
                f"Split {i}: {layers_in_current_split} layers, boundary at {residual_adds[cumulative_idx]}"
            )

        if self.config.split_lm_head:
            split_tensors.append(lm_head)

        # Add the output of the last node
        split_tensors.append(model.graph[-1].outputs[0])

        # Fill missing encodings of boundary tensors
        for boundary_tensor in split_tensors[1:]:
            if "extra_info" in boundary_tensor.meta:
                enc = boundary_tensor.meta["extra_info"].named_encodings

                if not enc:
                    start_tensor: ir.Value | None = boundary_tensor

                    while not enc:
                        # To satisfy mypy
                        assert start_tensor is not None

                        producer = start_tensor.producer()

                        if producer:
                            start_tensor = producer.inputs[0]

                            if start_tensor and "extra_info" in start_tensor.meta:
                                enc = start_tensor.meta["extra_info"].named_encodings

                            else:
                                # No extra_info, and hence, no encodings present for the tensor
                                continue

                            if enc:
                                for encset_name, v_enc in enc.items():
                                    enc_copy = copy.deepcopy(v_enc)

                                    # NOTE: In instances where param encodings are copied to an activation tensor
                                    # it is important to set the enc_kind of the copied tensor to ACTIVATION
                                    # Notable when copying the encodings of "Gather.weights" (param) to "Gather_output" (activation)
                                    # When split_embeddding is True
                                    enc_copy.enc_kind = EncKind.ACTIVATION
                                    boundary_tensor.meta["extra_info"].named_encodings[encset_name] = enc_copy

                                logger.info(
                                    f"Copied the encodings of {start_tensor.name} to {boundary_tensor.name}"
                                )
                                break

                        else:
                            # Graph input reached
                            break

        splits: list[ir.Model] = []

        for i in range(len(split_tensors) - 1):
            subgraph_start_tensor = split_tensors[i]
            subgraph_end_tensor = split_tensors[i + 1]

            split_graph = copy_subgraph(
                subgraph_start_tensor, subgraph_end_tensor, name=f"{i + 1}_of_{original_num_splits}"
            )

            split_model = ir.Model(
                graph=split_graph,
                ir_version=model.ir_version,
                domain=model.domain,
                model_version=model.model_version,
            )

            # In instances where nodes are added to a split but they are not used
            # because they are not used by any nodes in the current split
            # But in other splits (in which case, such subgraphs are re-constructed in the other splits)
            # Remove such nodes in the current split
            remove_unused_nodes(split_model)

            splits.append(split_model)

        # Display rich table only when DEBUG logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            table = create_rich_table(
                title=bold_text("Model Splitting Results:", color=PrettyPrintConstants.Q_BLUE),
                headers=["Split Number", "New Inputs", "New Outputs"],
                positions=[0.18, 0.59, 1.0],
                alignment=["left", "left", "left"],
            )
            for i, _split in enumerate(splits):
                table.add_row(
                    str(i + 1),
                    ", ".join([_input.name for _input in _split.graph.inputs if _input and _input.name]),
                    ", ".join([_output.name for _output in _split.graph.outputs if _output and _output.name]),
                )

            console = rich.console.Console(highlight=True)
            console.print(table, overflow="fold")

        return splits
