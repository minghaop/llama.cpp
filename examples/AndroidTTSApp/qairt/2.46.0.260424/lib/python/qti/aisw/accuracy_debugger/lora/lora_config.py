# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""Data classes for LoRA Model Creator and LoRA Importer configurations.
These classes map to the arguments of qairt-lora-model-creator and qairt-lora-importer tools.
"""

from typing import Dict, Optional

from pydantic import DirectoryPath, Field, FilePath
from qti.aisw.accuracy_debugger.utils.helper import DebuggerConfig


class LoRAModelCreatorInputConfig(DebuggerConfig):
    """Input configuration for LoRA Model Creator.

    Maps to qairt-lora-model-creator arguments.

    Attributes:
        lora_config: Path to the YAML config file for LoRA (required)
        output_dir: Path to store the output artifacts (optional)
        quant_updatable_mode: Quantization updatable mode ("none", "adapter_only", "all")
        debug: Debug level for logging
        skip_validation: Skip validation checks
        dump_usecase_onnx: Dump per-usecase ONNX models
        transforms_metadata: Path to JSON file for transformation metadata
    """

    lora_config: FilePath
    output_dir: Optional[DirectoryPath] = None
    quant_updatable_mode: str = Field(default="adapter_only", pattern="^(none|adapter_only|all)$")
    debug: int = -1
    skip_validation: bool = False
    dump_usecase_onnx: bool = False
    transforms_metadata: Optional[FilePath] = None


class LoRAModelCreatorOutputConfig(DebuggerConfig):
    """Output configuration from LoRA Model Creator.

    Attributes:
        base_model_path: Path to the generated base model (base_model.onnx)
        base_encodings_path: Path to base encodings file (base_encodings.json)
        lora_tensor_names_file: Path to LoRA tensor names file (lora_tensor_names.txt)
        importer_config: Path to LoRA importer config (lora_importer_config.yaml)
        use_case_artifacts: Dictionary of per-usecase artifacts (safetensors, encodings)
        output_directory: Directory where all outputs are stored
    """

    base_model_path: FilePath
    base_encodings_path: Optional[FilePath] = None
    lora_tensor_names_file: FilePath
    importer_config: FilePath
    use_case_artifacts: Dict[str, Dict[str, FilePath]] = Field(default_factory=dict)
    output_directory: DirectoryPath


class LoRAImporterInputConfig(DebuggerConfig):
    """Input configuration for LoRA Importer.

    Maps to qairt-lora-importer arguments.

    Attributes:
        lora_config: Path to the YAML config file for LoRA
        input_dlc: Path to the Float or Quantized DLC
        input_network: Path to the source ONNX model
        input_list: Path to file specifying input data for quantization
        output_dir: Directory to store all output artifacts
        float_fallback: Enable fallback to floating point
        debug: Debug level for logging
        skip_validation: Skip validation checks
        dump_usecase_dlc: Dump per-usecase DLC files
        dump_usecase_onnx: Dump per-usecase ONNX models
        skip_apply_graph_transforms: Skip applying graph transforms
    """

    # Required fields (when not auto-populated from model creator)
    lora_config: Optional[FilePath] = None
    input_dlc: Optional[FilePath] = None
    input_network: Optional[FilePath] = None
    input_list: Optional[FilePath] = None

    # Optional fields
    output_dir: Optional[DirectoryPath] = None
    float_fallback: bool = False
    debug: int = -1
    skip_validation: bool = False
    dump_usecase_dlc: bool = False
    dump_usecase_onnx: bool = False
    skip_apply_graph_transforms: bool = False


class UseCaseArtifacts(DebuggerConfig):
    """Artifacts for a single LoRA use case.

    Attributes:
        name: Use case name
        graph_name: Graph name from DLC
        weights: Path to safetensors file
        encodings: Path to encodings file (optional)
        binary: Path to context binary file (set after context binary generation)
    """

    name: str
    graph_name: str
    weights: FilePath
    encodings: Optional[FilePath] = None
    binary: Optional[FilePath] = None


class LoRAImporterOutputConfig(DebuggerConfig):
    """Output configuration from LoRA Importer.

    Attributes:
        lora_output_files: Path to main LoRA output config (lora_output_files.yaml)
        use_case_artifacts: Dictionary of per-usecase artifacts
        output_directory: Directory where all outputs are stored
    """

    lora_output_files: FilePath
    use_case_artifacts: Dict[str, UseCaseArtifacts] = Field(default_factory=dict)
    output_directory: DirectoryPath
