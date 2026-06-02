# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Speculative Decoding Tutorial (single-script version)
===================================================

This script demonstrates how to enable one of the three supported speculative
decoding methods (LADE, SSD, or Eaglet) by setting the ``speculative_config``
property on a :class:`GenAIBuilderHTP` instance.  It follows the same
structure as ``core/examples/tutorials/llm_on_device_inference.py`` but adds the
speculative configuration step.

For eaglet, the model specified in draft_model_path must also have the same
directory structure as the base model, as detailed in
``core/examples/tutorials/llm_on_device_inference.py``

Edit the variables in the **Configuration** section below to point to your
exported model and choose the desired speculative decoding type
"""

import os
from pathlib import Path
from typing import Optional, cast

from qairt.api.configs.common import BackendType
from qairt.api.configs.device import Device
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP
from qairt.gen_ai_api.configs.eaglet_config import EagletBuilderConfig
from qairt.gen_ai_api.configs.lade_config import LadeBuilderConfig
from qairt.gen_ai_api.configs.ssd_config import SsdBuilderConfig
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Path to the exported model directory (replace <your_path> with the actual path)
MODEL_EXPORTS = "./llama_3_8b/<your_path>"

# Prompt to generate (you can replace this with any text you like)
PROMPT = "briefly explain speculative decoding and its benefits."

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
# Temporary directory for intermediate artifacts
os.environ["QAIRT_TMP_DIR"] = os.environ.get("QAIRT_TMP_DIR", "./llm_scratch/")

# Cache directory – reuse across builds to speed up subsequent runs
CACHE_ROOT = "./llama3_cache"

############################################################
# Set up an Android device
# ------------------------
# Reusing the android setup instructions from the `:ref:quick_start` guide:

android_serial = os.getenv("ANDROID_SERIAL")
android_hostname = os.getenv("ANDROID_HOSTNAME")

device_id = f"{android_serial}@{android_hostname}" if android_hostname else android_serial
android_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)


# ----------------------------------------------------------------------
# Build the GenAI container with speculative decoding enabled
# ----------------------------------------------------------------------
cache_root = Path(CACHE_ROOT)
cache_root.mkdir(parents=True, exist_ok=True)

# Create the builder
builder: GenAIBuilderHTP = GenAIBuilderFactory.create(
    Path(MODEL_EXPORTS),
    BackendType.HTP,
    cache_root=cache_root,
)

# Target device – example uses Snapdragon SM8750
soc_details = f"chipset:{android_device.get_chipset()}"
builder.set_targets([soc_details])

# ------------------------------------------------------------------
# Speculative decoding configuration
# ------------------------------------------------------------------
# To try different speculative decoding methods:
# 1. Comment out the active configuration below
# 2. Uncomment one of the alternative configurations (SSD or Eaglet)
# 3. Ensure only ONE configuration is active at a time

# LADE (Look-Ahead Decoding) – fast speculative decoding using a small
# window of future tokens. It predicts multiple tokens ahead to reduce
# latency. Example configuration:
builder.speculative_config = LadeBuilderConfig(
    window=8,
    ngram=5,
    gcap=8,
)

# SSD (Speculative Sampling Decoding) – uses a forecast model to
# generate multiple candidate tokens. Example configuration (commented out):
# builder.speculative_config = SsdBuilderConfig(
#     forecast_token_count=4,
#     forecast_prefix=16,
#     branches=[4, 4],
#     ssd_tensor_file="./ssd_tensor.pt",
#     n_streams=1,
# )

# Eaglet – speculative decoding with a draft model that runs in parallel.
# Example configuration (commented out):
# builder.speculative_config = EagletBuilderConfig(
#     draft_len=6,
#     n_branches=6,
#     max_tokens_target_can_evaluate=32,
#     draft_kv_cache=True,
#     draft_model_path="./draft_model.onnx",
#     draft_token_map="./draft_token_map.json",
# )

# Build the container (returns a GenAIContainer)
container = builder.build()

# ------------------------------------------------------------------
# Inference on the Android device
# ------------------------------------------------------------------
device = android_device  # Change if you need a different target
executor = cast(T2TExecutor, container.get_executor(device))

# Generate text
result = executor.generate(PROMPT)

# Output
print("\n--- Generated Text ---")
print(result.generated_text)

if result.metrics:
    print("\n--- Metrics ---")
    print(result.metrics)

# Clean up device artifacts
executor.clean_environment()

# Save the container for later reuse
container.save("./speculative_container", exist_ok=True)

logger.info("Speculative decoding tutorial completed successfully.")
