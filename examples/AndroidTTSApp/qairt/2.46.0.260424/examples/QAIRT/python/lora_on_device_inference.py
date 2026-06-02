"""
..  # ==============================================================================
    #
    # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
    # All Rights Reserved.
    # Confidential and Proprietary - Qualcomm Technologies, Inc.
    #
    # ==============================================================================

===================================
Low-Rank Adaptation (LoRA) Tutorial
===================================
This guide will walk you through the process of deploying an LLM with LoRA adapters on a Snapdragon device using the QAIRT Tools Python API. For more details, see `LoRA documentation <https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/lora_intro.html>`.


Overview
--------------
Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that allows you to adapt
large language models for specific tasks without modifying the entire model. This tutorial demonstrates:

1. Creating QAIRT assets with LoRA adapters
2. Building a LoRA-enabled model for HTP backend
3. Performing inference with different LoRA adapters on device

Prerequisites
----------------------
   - This guide uses a base LLM model (e.g., Llama 3.2-3B) with LoRA adapters. You can download the base model from Hugging Face: `Llama-3.2-3B <https://huggingface.co/meta-llama/Llama-3.2-3B>`_
     using a valid license.
   - The guide assumes you have obtained:

     * N+1 ONNX models - Quantized ONNX base model (without any LoRA branches) and N ONNX graphs
        for each use case.
        * For example, if there are three adapters (A, B, and C) and the expected use cases are
            A, B, C, and A+C, then N equals 4.
     * M LoRA adapter weights in SafeTensors format - 1 per adapter
     * LoRA configuration files (adapter configs and attach point mappings)
     * 1 PyTorch to ONNX names mapping for the base graph
     * N+1 Quantization encodings - For both base model and LoRA use cases

     **Example input artifacts (before build):**

     ::

         <model_exports>/
            Llama-3.2-3B_base/
               onnx/
                  llama3_2_base.onnx
                  llama3_2_base.encodings
                  llama3_2_base.data
                  config.json
                  tokenizer.json
                  llama3_2_base_node_mapping.json
            Llama-3.2-3B_elementary/
               onnx/
                  llama3_2_elementary.onnx
                  llama3_2_elementary.encodings
                  llama3_2_elementary.data
                  elementary.json
            Llama-3.2-3B_long/
               onnx/
                  llama3_2_long.onnx
                  llama3_2_long.encodings
                  llama3_2_long.data
                  long.json
            Llama-3.2-3B_elementary+long/
               onnx/
                  llama3_2_elementary+long.onnx
                  llama3_2_elementary+long.encodings
                  llama3_2_elementary+long.data
                  elementary.json
                  long.json
            top_level_lora_meta.yaml

     **Key points about input structure:**

     - Base model contains the unmodified model without LoRA branches
     - Each LoRA use case directory contains the model with LoRA branches integrated
     - Multi-adapter use cases (e.g., ``elementary+long``) contain combined LoRA branches
     - ``.data`` files contain external weight data for the ONNX models
     - ``top_level_lora_meta.yaml`` defines all adapters and use cases

    - Upon completion of model preparation and building, the following directory structure is expected:

     **Example output structure (after build):**

     ::

         <container_output>/
            models/
               split_0/
                  model.bin
               split_1/
                  model.bin
                  elementary.bin
                  long.bin
                  function.bin
                  elementary+long.bin
               split_2/
                  model.bin
                  elementary.bin
                  long.bin
                  function.bin
                  elementary+long.bin
               split_3/
                  model.bin
                  elementary.bin
                  long.bin
                  function.bin
                  elementary+long.bin
               split_4/
                  model.bin
               use_cases.json

     **Key points about output structure:**

     - ``split_N/`` directories contain the compiled model splits
     - ``model.bin`` is the compiled base model for each split
     - ``<use_case>.bin`` files are the compiled LoRA adapters for each use case
     - Not all splits contain LoRA adapters (e.g., split_0 and split_4 may only have base model)
     - ``use_cases.json`` defines the available LoRA use cases and their configurations
     - Use case names like ``elementary+long`` represent multi-adapter combinations

   - This guide uses a **Snapdragon SD 8 Elite (SM8750) Android device** to demonstrate the workflow.
   - We recommend a machine with at least 64 GB of RAM for timely completion of the workflow.

**Important:** Set the environment variable **QAIRT_TMP_DIR** to define an alternative default temporary directory path.
              This is recommended because temporary artifacts are created during build process which may consume temp memory entirely.
"""

############################################################
# Setup
# ------------------------
import os
from typing import cast

import qairt
from qairt import Device, DevicePlatformType
from qairt.gen_ai_api.builders.llama.builder import LlamaBuilderHTP
from qairt.gen_ai_api.containers.gen_ai_container import GenAIContainer
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
from qairt.modules.lora.lora_config import (
    AdapterRunConfig,
    LoraBuilderInputConfig,
    UseCaseRunConfig,
)

############################################################
# Define paths
# ------------------------
# Path to the base model exports directory
llama3_exports = "./llama_3.2_3b/<your_path>"

# Path to LoRA configuration file (or you can create config programmatically)
lora_config_path = "./llama_3.2_3b/lora_config.yaml"

# Set QAIRT_TMP_DIR
os.environ["QAIRT_TMP_DIR"] = "./llm_scratch/"

# Set a cache directory for intermediate artifacts
CACHE_ROOT = "./llama3_cache"

############################################################
# Understanding LoRA configuration
# ------------------------
# LoRA configuration consists of three main components:
#
# 1. **Adapter configuration**: Defines the LoRA adapters with their parameters
#    - name: Identifier for the adapter
#    - rank: Rank of the low-rank matrices
#    - alpha: Scaling factor for the adapter
#    - target_modules: To which model layers to apply the adapter
#
# 2. **Use case configuration**: Defines how adapters are combined for specific tasks
#    - name: Identifier for the use case
#    - adapter_names: List of adapters to use
#    - adapter_alphas: Scaling factors for each adapter
#    - model_name: Path to the base model
#    - encodings: Path to quantization encodings
#
# 3. **Attach point mapping**: Maps framework module names to ONNX node names

############################################################
# Option 1: Load LoRA configuration from file
# ------------------------
# If you have a pre-existing lora_config.yaml file:

lora_input_config = LoraBuilderInputConfig(
    lora_config_path=lora_config_path,
    create_lora_graph=True,  # Set to True to create LoRA max rank-concatenated graph
    quant_updatable_mode="adapter_only",  # Options: "none", "adapter_only", "all"
    alpha_tensor_name="lora_alpha",
)

############################################################
# Building a LoRA-enabled model for HTP
# ------------------------
# The process is similar to building a standard LLM, but with LoRA configuration:

# Create a builder instance
llama_builder = GenAIBuilderFactory.create(llama3_exports, "HTP", cache_root=CACHE_ROOT)

# Ensure the appropriate builder is returned
assert isinstance(llama_builder, LlamaBuilderHTP)

# Customize the builder for the target device
llama_builder.set_targets([f"chipset:SM8750"])

# Set the LoRA configuration
llama_builder.lora_config = lora_input_config

############################################################
# What happens during a LoRA build?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When you set lora_config and build, the following steps occur:
#
# 1. **LoRA graph creation**: A max-rank concatenated LoRA graph is created
#    - All adapters are combined into a single graph structure
#    - The graph supports dynamic adapter selection at runtime
#
# 2. **Model transformation**: The base model is transformed with LoRA support
#    - Embedding and LM head are split (required for LoRA)
#    - Multi-head attention is converted to single-head attention (MHA v2)
#
# 3. **Conversion**: Both base model and LoRA adapters are converted
#    - Base model uses provided encodings
#    - LoRA adapters use separate encodings for fine-grained control
#
# 4. **Compilation**: The model is compiled for HTP backend
#    - Adapter weights are prepared for efficient on-device switching

############################################################
# Build the LoRA-enabled model
llama_lora_container: GenAIContainer = llama_builder.build()

############################################################
# Understanding Quant Updatable mode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The quant_updatable_mode parameter controls which tensors can be updated:
#
# - "none": No quantization encodings are updatable
# - "adapter_only": Quantization encodings for only lora/adapter branch (Conv->Mul->Conv)
#                   change across use-case. The base branch quantization encodings remain the same.
# - "all": All quantization encodings are updatable.

############################################################
# Caching with LoRA
# ^^^^^^^^^^^^^^^^^^^
# LoRA builds can be time-consuming. Use caching to resume builds:

existing_cache_dir = CACHE_ROOT
genai_builder = GenAIBuilderFactory.create(llama3_exports, "HTP", cache_root=existing_cache_dir)


genai_builder.lora_config = lora_input_config

# The builder resumes from the last completed stage if configuration hasn't changed

############################################################
# Set up an Android device
# ------------------------
android_serial = os.getenv("ANDROID_SERIAL")
android_hostname = os.getenv("ANDROID_HOSTNAME")

device_id = f"{android_serial}@{android_hostname}" if android_hostname else android_serial
android_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

###########################################################################
# Generate text with LoRA adapters
# ---------------------------------
# Create an executor from the container
from qairt.gen_ai_api.executors.gen_ai_executor import GenAIExecutor, TextGenerationResult

llm: GenAIExecutor = llama_lora_container.get_executor(android_device, clean_up=False)

# Ensure the appropriate executor is returned
assert isinstance(llm, T2TExecutor)

###########################################################################
# Using different LoRA adapters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can switch between different LoRA use cases at runtime:

# Define the prompt template
prompt_template = (
    "<|begin_of_text|>"
    "<|start_header_id|>{system}<|end_header_id|>{system_prompt}<|eot_id|>"
    "<|start_header_id|>{user}<|end_header_id|>{user_prompt}<|eot_id|>"
    "<|start_header_id|>{assistant}<|end_header_id|>"
)

###########################################################################
# Example 1: Generate with first use case (long)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Configure the executor to use the first use case
use_case_config_1 = UseCaseRunConfig(
    use_case_name="long",
    adapters=[AdapterRunConfig(adapter_name="long", alpha=1.0)],
)

prompt_1 = prompt_template.format(
    system="system",
    system_prompt="You are a helpful coding assistant.",
    user="user",
    user_prompt="Write a Python function to calculate fibonacci numbers.",
    assistant="assistant",
)

# Generate text with the long use case
result_1: TextGenerationResult = llm.generate(prompt_1, lora_config=use_case_config_1)
print("Response with long use case:")
print(result_1.generated_text)
print("\nMetrics:")
print(result_1.metrics)

###########################################################################
# Example 2: Generate with second use case (elementary+long)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Configure the executor to use multiple adapters
use_case_config_2 = UseCaseRunConfig(
    use_case_name="elementary+long",
    adapters=[
        AdapterRunConfig(adapter_name="elementary", alpha=1.0),
        AdapterRunConfig(adapter_name="long", alpha=0.5),
    ],
)

prompt_2 = prompt_template.format(
    system="system",
    system_prompt="You are a medical coding assistant.",
    user="user",
    user_prompt="Explain how to implement a patient data validation function.",
    assistant="assistant",
)

# Generate text with multiple adapters
result_2: TextGenerationResult = llm.generate(prompt_2, lora_config=use_case_config_2)
print("\nResponse with elementary+long use case:")
print(result_2.generated_text)
print("\nMetrics:")
print(result_2.metrics)

###########################################################################
# Example 3: Dynamic alpha adjustment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can adjust adapter influence by changing alpha values:

use_case_config_3 = UseCaseRunConfig(
    use_case_name="elementary+long",
    adapters=[
        AdapterRunConfig(adapter_name="elementary", alpha=0.2),  # Reduced influence
        AdapterRunConfig(adapter_name="long", alpha=1.0),  # Increased influence
    ],
)

prompt_3 = prompt_template.format(
    system="system",
    system_prompt="You are a helpful assistant.",
    user="user",
    user_prompt="What are best practices for medical software development?",
    assistant="assistant",
)

result_3: TextGenerationResult = llm.generate(prompt_3, lora_config=use_case_config_3)
print("\nResponse with adjusted alphas:")
print(result_3.generated_text)

###########################################################################
# Performance considerations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# - Switching between use cases has minimal overhead
# - The first inference after switching may be slightly slower
# - Multiple adapters in a use case have additive computational cost
# - Alpha values can be tuned for optimal task performance

# Clean up device artifacts
llm.clean_environment()

######################################################################
# Export and reload
# -----------------
# Save the LoRA-enabled container for later use:

llama_lora_container.save("./llama3_lora_container", exist_ok=True)

######################################################################
# Container contents
# ^^^^^^^^^^^^^^^^^^
# The saved container includes:
#   - Compiled base model binaries
#   - LoRA adapter weights for all use cases
#   - Configuration files for adapter management
#   - Quantization parameters
#
# Reload the container:

from qairt.gen_ai_api.containers.llm_container import LLMContainer

# Load a previously saved LoRA container
loaded_container = LLMContainer.load("./llama3_lora_container")

# Create a new executor from the loaded container
llm_reloaded = loaded_container.get_executor(android_device, clean_up=False)

result_reloaded = llm_reloaded.generate(prompt_1, lora_config=use_case_config_1)
print("\nResponse from reloaded container:")
print(result_reloaded.generated_text)

llm_reloaded.clean_environment()
