# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Pipeline configuration classes for LMM (Large Multimodal Model) execution.
These classes define the structure and behavior of pipeline nodes and their connections
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field
from typing_extensions import Self, TypedDict

from qairt.api.configs.common import AISWBaseModel
from qairt.modules.lora.lora_config import UseCaseRunConfig


class NodeConfig(TypedDict):
    """Node configuration containing Genie Node Specific Configurations"""

    name: str
    path: str


class GenieNodeLoraConfig(UseCaseRunConfig):
    """
    LoRA configuration for a genie pipeline node

    Extends UseCaseRunConfig with node-specific settings needed for Genie App execution.
    Inherits use_case_name and adapters from UseCaseRunConfig.
    """

    alpha_tensor_name: str = "lora_alpha"
    """Name of the alpha tensor in the model"""


class GenieNodeType(str, Enum):
    """
    Types of nodes in the Genie pipeline
    """

    TEXT = "text"
    IMAGE = "image"


class GenieNodeIOType(str, Enum):
    """
    Input/Output types for Genie pipeline nodes
    """

    TEXT_ENCODER_INPUT = "GENIE_NODE_TEXT_ENCODER_TEXT_INPUT"
    TEXT_ENCODER_OUTPUT = "GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT"
    IMAGE_ENCODER_INPUT = "GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT"
    IMAGE_ENCODER_OUTPUT = "GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT"
    TEXT_GENERATOR_INPUT = "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"
    TEXT_GENERATOR_OUTPUT = "GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT"


# Valid node names that map to specific input/output types
GenieNodeName = Literal["lutEncoder", "imageEncoder", "textGenerator"]


class GeniePipelineNode(AISWBaseModel):
    """
    Represents a node in the Genie pipeline with its configuration and properties
    """

    name: GenieNodeName
    """
    The unique name of the node (e.g., 'lutEncoder', 'imageEncoder', 'textGenerator')
    """

    type: GenieNodeType
    """
    The type of node - either 'text' for text processing or 'image' for image processing
    """

    config: NodeConfig
    """
    Node configuration containing both name and path.
    Example: {"name": "lutEncoderConfig", "path": "text_encoder_config.json"}
    """

    _config_data: Optional[Dict[str, Any]] = None
    """Cached configuration data loaded from the node config file"""

    @property
    def model_paths(self) -> List[str]:
        """
        Extract model paths from the config JSON file using known dialog schema structure.

        Returns:
            List[str]: List of model file paths found in the config
        """
        if not self._config_data:
            return []

        try:
            paths = []

            if "text-encoder" in self._config_data and "lut" in self._config_data["text-encoder"]:
                lut_path = self._config_data["text-encoder"]["lut"].get("lut-path")
                if lut_path is None:
                    raise ValueError(
                        f"Missing required 'lut-path' in text-encoder config: {self.config['path']}"
                    )
                paths.append(lut_path)

            if "image-encoder" in self._config_data:
                engine = self._config_data["image-encoder"].get("engine", {})
                model = engine.get("model", {})
                binary = model.get("binary", {})
                ctx_bins = binary.get("ctx-bins")

                if not ctx_bins:
                    raise ValueError(
                        f"Missing required 'ctx-bins' in image-encoder config: {self.config['path']}"
                    )

                paths.extend(ctx_bins)

            if "text-generator" in self._config_data:
                engine = self._config_data["text-generator"].get("engine", {})
                model = engine.get("model", {})
                binary = model.get("binary", {})
                ctx_bins = binary.get("ctx-bins")

                if not ctx_bins:
                    raise ValueError(
                        f"Missing required 'ctx-bins' in text-generator config: {self.config['path']}"
                    )

                paths.extend(ctx_bins)

            return paths

        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse node config file '{self.config['path']}': {e}") from e

    @property
    def lora_adapter_paths(self) -> List[str]:
        """
        Extract all LoRA adapter bin file paths from the config JSON file

        Returns all adapter paths regardless of which adapter is specified in lora_config,
        because Genie App validates all adapter paths in the JSON at node creation time (NEED TO VERIFY THIS)

        If no lora_config is set, returns empty list (no LoRA support for this node)

        Returns:
            List[str]: List of all LoRA adapter binary file paths found in the config
        """
        # If no lora_config is set, this node doesn't use LoRA at all
        if not self.lora_config:
            return []

        if not self._config_data:
            return []

        try:
            paths = []

            # Check all possible node types for LoRA adapters
            for node_key in ["text-encoder", "image-encoder", "text-generator"]:
                if node_key not in self._config_data:
                    continue

                engine = self._config_data[node_key].get("engine", {})
                model = engine.get("model", {})
                binary = model.get("binary", {})
                lora_config_data = binary.get("lora", {})
                adapters = lora_config_data.get("adapters", [])

                # Extract all adapter bin-sections (Genie App validates all paths BUT NEED TO VERIFY WHY THIS IS)
                for adapter in adapters:
                    bin_sections = adapter.get("bin-sections", [])
                    paths.extend([section for section in bin_sections if section])

            return paths

        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Failed to parse LoRA adapters from node config file '{self.config['path']}': {e}"
            ) from e

    input_name: Optional[GenieNodeIOType] = None
    """
    Input type identifier for this node (assigned based on node name)
    """

    output_name: Optional[GenieNodeIOType] = None
    """
    Output type identifier for this node (assigned based on node name)
    """

    output_callback_type: Optional[str] = None
    """
    Optional callback type for handling node output
    """

    lora_config: Optional[GenieNodeLoraConfig] = None
    """
    Optional LoRA configuration for this node
    """

    def model_post_init(self, __context):
        """Set input/output names based on exact node name mapping, and load config JSON data."""
        # Load config data if file exists
        if os.path.exists(self.config["path"]):
            try:
                with open(self.config["path"], "r") as f:
                    self._config_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise ValueError(f"Failed to load node config file '{self.config['path']}': {e}") from e

        node_io_mapping = {
            "lutEncoder": (GenieNodeIOType.TEXT_ENCODER_INPUT, GenieNodeIOType.TEXT_ENCODER_OUTPUT),
            "imageEncoder": (GenieNodeIOType.IMAGE_ENCODER_INPUT, GenieNodeIOType.IMAGE_ENCODER_OUTPUT),
            "textGenerator": (GenieNodeIOType.TEXT_GENERATOR_INPUT, GenieNodeIOType.TEXT_GENERATOR_OUTPUT),
        }

        if self.name in node_io_mapping:
            self.input_name, self.output_name = node_io_mapping[self.name]
        else:
            raise ValueError(
                f"Unknown node name: {self.name}. Valid names are: {list(node_io_mapping.keys())}"
            )


class GeniePipelineConfig(AISWBaseModel):
    """
    Configuration for the entire Genie pipeline, including nodes, connections, and inputs

    Supports building pipelines incrementally as well:
    - Start with minimal set of nodes
    - Add nodes, connections, inputs, and callbacks iteratively
    - Generate script at any point in construction
    - Continue adding nodes and regenerate as needed

    Example:
        config = GeniePipelineConfig(nodes=[node_a])
        config.add_node(node_b).add_connection(node_a.name, node_b.name)
        config.add_input(node_a.name, "sample input text")
        script = config.generate_pipeline_script()
    """

    version: int = 1
    """
    Version number of the pipeline configuration schema
    """

    nodes: Dict[str, GeniePipelineNode] = Field(default_factory=dict)
    """
    Dictionary of pipeline nodes keyed by node name that will be created and configured
    """

    inputs: List[Tuple[str, str]] = Field(default_factory=list)
    """
    Pipeline inputs as (node_name, input_value) tuples.
    For text nodes: input_value is the actual text string to process.
    For image nodes: input_value is the file path to the image file.
    """

    connections: List[Tuple[str, str]] = Field(default_factory=list)
    """
    Node connections as (output_node_name, input_node_name) tuples
    defining how data flows between pipeline nodes
    """

    enable_profiling: bool = True
    """
    Enable profiling for the pipeline. When enabled, generates profile commands.
    """

    profile_name: str = "profile1"
    """
    Name of the profile to create when profiling is enabled
    """

    profile_output_path: str = "profile.json"
    """
    Output file for the profile data when profiling is enabled
    """

    def add_node(self, node: GeniePipelineNode) -> Self:
        """
        Add a node to the pipeline.

        Args:
            node: The pipeline node to add

        Returns:
            Self for method chaining
        """
        self.nodes[node.name] = node
        return self

    def add_connection(self, output_node_name: str, input_node_name: str) -> Self:
        """
        Add a connection between two nodes.

        Args:
            output_node_name: Name of the node providing output
            input_node_name: Name of the node receiving input

        Returns:
            Self for method chaining
        """
        connection = (output_node_name, input_node_name)
        if connection not in self.connections:
            self.connections.append(connection)
        return self

    def add_input(self, node_name: str, input_value: str) -> Self:
        """
        Add an input to a pipeline node.

        Args:
            node_name: Name of the node to receive input
            input_value: The input value (text string or file path)

        Returns:
            Self for method chaining
        """
        pipeline_input = (node_name, input_value)
        if pipeline_input not in self.inputs:
            self.inputs.append(pipeline_input)
        return self

    def remove_node(self, node_name: str) -> Self:
        """
        Remove a node and clean up all references to it.

        Args:
            node_name: Name of the node to remove

        Returns:
            Self for method chaining
        """
        # Remove the node itself
        self.nodes.pop(node_name, None)

        # Remove connections involving this node
        self.connections = [
            (output, input_name)
            for output, input_name in self.connections
            if output != node_name and input_name != node_name
        ]

        # Remove inputs for this node
        self.inputs = [(name, value) for name, value in self.inputs if name != node_name]

        return self

    def validate_pipeline(self):
        """
        Validate that the pipeline configuration is internally consistent.
        Called before script generation to ensure validity.
        """
        if not self.nodes:
            raise ValueError("Pipeline must contain at least one node")

        all_node_names = set(self.nodes.keys())

        # Validate connections
        for output_node_name, input_node_name in self.connections:
            if output_node_name not in all_node_names:
                raise ValueError(
                    f"Connection references unknown output node '{output_node_name}'. "
                    f"Available nodes: {list(all_node_names)}"
                )
            if input_node_name not in all_node_names:
                raise ValueError(
                    f"Connection references unknown input node '{input_node_name}'. "
                    f"Available nodes: {list(all_node_names)}"
                )

        # Validate inputs
        for node_name, _ in self.inputs:
            if node_name not in all_node_names:
                raise ValueError(
                    f"Pipeline input references unknown node '{node_name}'. "
                    f"Available nodes: {list(all_node_names)}"
                )

    def _create_nodes(self) -> List[str]:
        """Generate node creation commands.

        Returns:
            List[str]: List of commands to create pipeline nodes
        """
        node_creation_list = []
        for node in self.nodes.values():
            # Use basename for the config filename since files will be in the container's working directory
            config_filename = Path(node.config["path"]).name
            node_creation_list.append(f"node config create {node.config['name']} {config_filename}")
            node_creation_list.append(f"node create {node.name} {node.config['name']}")

            # Add LoRA commands if configured
            if node.lora_config:
                lora = node.lora_config
                # Apply each adapter in the use case
                for adapter in lora.adapters:
                    node_creation_list.append(
                        f"node applyLora {node.name} {lora.use_case_name} {adapter.adapter_name}"
                    )
                    node_creation_list.append(
                        f"node setLoraStrength {node.name} {lora.use_case_name} {lora.alpha_tensor_name} {adapter.alpha}"
                    )

        return node_creation_list

    def _create_output_callbacks(self) -> List[str]:
        """Generate output callback configuration commands.

        Returns:
            List[str]: List of commands to configure pipeline output callbacks
        """
        pipeline_output_callbacks_list = []
        for node in self.nodes.values():
            if node.output_callback_type is not None:
                assert node.output_name is not None, (
                    f"Node {node.name} missing output_name after initialization"
                )
                pipeline_output_callbacks_list.append(
                    f"node set {node.output_callback_type} {node.name} {node.output_name.value}"
                )
        return pipeline_output_callbacks_list

    def _add_nodes(self) -> List[str]:
        """Generate commands to add nodes to the pipeline.

        Returns:
            List[str]: List of commands to add nodes to the pipeline
        """
        return [f"pipeline add GeniePipeline {node.name}" for node in self.nodes.values()]

    def _connect_nodes(self) -> List[str]:
        """Generate node interconnection commands.

        Returns:
            List[str]: List of commands to connect pipeline nodes
        """
        node_interconnections_list = []

        for output_node_name, input_node_name in self.connections:
            output_node = self.nodes[output_node_name]
            input_node = self.nodes[input_node_name]

            assert output_node.output_name is not None, (
                f"Node {output_node.name} missing output_name after initialization"
            )
            assert input_node.input_name is not None, (
                f"Node {input_node.name} missing input_name after initialization"
            )
            node_interconnections_list.append(
                f"pipeline connect GeniePipeline {output_node.name} {output_node.output_name.value} "
                f"{input_node.name} {input_node.input_name.value}"
            )

        return node_interconnections_list

    def _set_inputs(self) -> List[str]:
        """Generate pipeline input configuration commands.

        Returns:
            List[str]: List of commands to configure pipeline inputs
        """
        pipeline_inputs_list = []

        for node_name, input_value in self.inputs:
            node = self.nodes[node_name]
            assert node.input_name is not None, f"Node {node.name} missing input_name after initialization"
            pipeline_inputs_list.append(
                f'node set {node.type.value} {node.name} {node.input_name.value} "{input_value}"'
            )

        return pipeline_inputs_list

    def generate_pipeline_script(self) -> str:
        """
        Generate the complete pipeline script.

        Validates pipeline before generation to ensure all references are valid.

        Returns:
            str: Complete pipeline script with all nodes, connections, and configurations

        Raises:
            ValueError: If pipeline configuration is invalid or inconsistent
        """
        # Validate pipeline before generating script
        self.validate_pipeline()

        pipeline_script_template = (
            "version\n"
            "{profile_create}"
            "pipeline config create pipelineConfig\n"
            "{profile_bind}"
            "pipeline create GeniePipeline pipelineConfig\n"
            "{node_creation}\n"
            "{pipeline_output_callbacks}\n"
            "{node_addition_to_pipeline}\n"
            "{node_interconnections}\n"
            "{pipeline_inputs}\n"
            "pipeline execute GeniePipeline\n"
            "{profile_save}"
            "{free_nodes}\n"
            "pipeline free GeniePipeline"
            "{profile_free}"
        )

        return pipeline_script_template.format(
            profile_create=f"profile create {self.profile_name}\n" if self.enable_profiling else "",
            profile_bind=f"pipeline config bind profile pipelineConfig {self.profile_name}\n"
            if self.enable_profiling
            else "",
            profile_save=f"profile save {self.profile_name} {self.profile_output_path}\n"
            if self.enable_profiling
            else "",
            profile_free=f"\nprofile free {self.profile_name}" if self.enable_profiling else "",
            profile_name=self.profile_name,
            profile_output_path=self.profile_output_path,
            node_creation="\n".join(self._create_nodes()),
            pipeline_output_callbacks="\n".join(self._create_output_callbacks()),
            node_addition_to_pipeline="\n".join(self._add_nodes()),
            node_interconnections="\n".join(self._connect_nodes()),
            pipeline_inputs="\n".join(self._set_inputs()),
            free_nodes="\n".join([f"node free {node_name}" for node_name in self.nodes.keys()]),
        )

    def export_pipeline_script(self, output_filepath: str = "LMMScript.cfg") -> str:
        """Export pipeline script to file and return the script content.

        Args:
            output_filepath: Path where the script will be saved

        Returns:
            str: The generated pipeline script content
        """
        script_content = self.generate_pipeline_script()
        with open(output_filepath, "w") as f:
            f.write(script_content)
        return script_content
