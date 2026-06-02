"""
..  # ==============================================================================
    #
    # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
    # All Rights Reserved.
    # Confidential and Proprietary - Qualcomm Technologies, Inc.
    #
    # ==============================================================================

.. _tuning_models:


Tuning Models
=============

This tutorial shows how to optimize a model for performance using the Tuner API. It includes a
step-by-step breakdown of the process. You can copy each snippet into a single script to run the tutorial end to end.


.. note::

    If you would like to skip the breakdown, you can obtain a simplified version of the tutorial in the QAIRT SDK from
    the following path:

        - ``examples/QAIRT/python/tuning_tutorial.py``

The parameters for this tutorial are as follows:

 - Framework: PyTorch
 - Configurations:

   - Host OS: Linux (x86_64)
   - Model: `ResNet18 <https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>`_
   - Target Devices: Snapdragon Android Device SM8750
   - Processor: Qualcomm NPU
   - Backend: HTP

.. tip::
       This tutorial creates some temporary files as part of the workflow. To customize the temporary file
       location, set the environment variable *QAIRT_TMP_DIR* to a location of your choosing.
"""

# -----------------------------------
# Step 1: Setup
# ----------------------------------
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import models

import qairt
from qairt import CompileConfig, Device, DevicePlatformType

# This example uses `qairt-visualizer <https://pypi.org/project/qairt-visualizer/>`_. See the profiling section for more details.
# We strongly recommend using `QAIRT Visualizer <https://docs.qualcomm.com/bundle/publicresource/topics/80-87189-1/overview.html>`_
# to visualize the reports generated.
ENABLE_QAIRT_VISUALIZER = False

# Define an artifacts directory
tuned_artifacts_dir = Path("./tuned_artifacts")
tuned_artifacts_dir.mkdir(exist_ok=True)

# -----------------------------
# Step 2: Obtain a Resnet model
# -----------------------------
# Load a pre-trained ResNet model (e.g., ResNet18)
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Set the model to evaluation mode
resnet_model.eval()

# Create a dummy input tensor with the shape
dummy_input = torch.randn(1, 3, 32, 32)

# Export the model to ONNX format
resnet_model_path = "resnet18.onnx"
torch.onnx.export(
    resnet_model,
    dummy_input,
    resnet_model_path,
    input_names=["input"],
)


# ---------------------------------
# Step 3: Set up an Android device
# --------------------------------

# Reusing the Android setup instructions from the `:ref:quick_start` guide:
android_serial = os.getenv("ANDROID_SERIAL")
android_hostname = os.getenv("ANDROID_HOSTNAME")

android_device = None

if android_serial:
    print(f"INFO: ANDROID_SERIAL : {android_serial} was set. Enabling tutorial for Android")
    device_id = f"{android_serial}@{android_hostname}" if android_hostname else android_serial
    android_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

if not android_device:
    print("INFO: ANDROID_SERIAL was not set. Exiting")
    exit(1)

# ------------------------------------------
# Step 4: Generate the initial QHAS reports
# -----------------------------------------
# Following on from the profiling tutorial, you can convert, compile, and produce
# a QHAS report for resnet_model

resnet_model_converted = qairt.convert(
    resnet_model_path, input_tensor_config=[dict(name="input", shape=(1, 3, 32, 32))]
)
resnet_model_converted.save(tuned_artifacts_dir / "resnet_model_converted.dlc")

# Create a SoC specification to compile for your Android device
# Setting the SoC details will create a configuration for the SoC model and DSP architecture.
compile_config = CompileConfig(backend="HTP", soc_details=f"chipset:{android_device.get_chipset()}")

# .. note::
#
#   In some cases, the get_chipset command may not return the correct chipset.
#   In this case, you may obtain the chipset using the following command from your host terminal:
#   `adb -H ${ANDROID_HOSTNAME} -s $ANDROID_SERIAL shell getprop ro.soc.model`


with qairt.Profiler(context=dict(level="detailed", option="optrace")) as pf:
    compiled_model = qairt.compile(resnet_model_converted, config=compile_config)

    input_array = np.random.randn(1, 3, 32, 32).astype(np.float32)
    _ = compiled_model(inputs=input_array, device=android_device)

    op_trace_report = pf.generate_report()
    qhas_report_path = tuned_artifacts_dir / "qhas_report.json"
    qhas_report = op_trace_report.summary.dump(qhas_report_path)


def print_or_view_qhas(report_path):
    if ENABLE_QAIRT_VISUALIZER:
        from qairt_visualizer import view

        view(reports=str(qhas_report_path))
    else:
        qhas_report = json.load(open(report_path))
        data = qhas_report["data"]["htp_overall_summary"]["data"][0]

        qhas_summary = {}
        for key, value in data.items():
            if key != "htp_resources":
                qhas_summary[key] = value

        print("QHAS Summary: \n")
        print(json.dumps(qhas_summary, indent=4))


print_or_view_qhas(qhas_report_path)

######################################################################
# The data above shows *total dram* as a measure of DDR bandwidth in bytes, and *time_us* as a measure of latency
# in micro-seconds.
#
# The section below introduces the Tuner API, which can help reduce both bandwidth
# and latency for some models.

# The Tuner API performs hardware-in-the-loop optimization to obtain the best performant models in the
# aforementioned criteria. The tuning process encapsulates compilation and execution in order to determine
# the best performing model.

# -----------------------
# Step 5: Tuning a model
# ----------------------
# Tuning can be performed to identify a reduction in latency - *time_us* metrics. The following example imports
# the tuner and uses the converted resnet-18 model to demonstrate the API.

# Import the tuner API

from qairt.api.common.backends.common import tuner

# ##############################
# # Tuning for latency reduction
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can now tune the model by passing in a compilation configuration and execution arguments containing the
# input data and the device on which to execute. The tuner exposes an `optimize` API that consumes
# compilation and execution args formulated as dictionaries.

# Use the same compile configuration as before

# Define input data
input_array = np.random.randn(1, 3, 32, 32).astype(np.float32)
best_compiled_model, report = tuner.optimize(
    resnet_model_converted,
    criteria="latency",
    compile_args=dict(config=compile_config),
    execution_args=dict(inputs=input_array, device=android_device),
)

# .. note::

#    compile_args and execution_args are dictionaries that are passed to the `:func:`qairt.compile` and
#    :func:`qairt.api.compiled_model.CompiledModel.__call__` functions. The tuner API simply forwards these arguments to the
#     underlying functions. You can pass in any arguments that are valid for the underlying functions.

#    For more information on the arguments and calling conventions, see the `qairt.compile` and
#    :func:`qairt.api.compiled_model.CompiledModel.__call__` docstrings respectively.


# ###############################################################################################
# Save the compiled model
best_compiled_model.save(tuned_artifacts_dir / "latency_tuned_model.bin")
#
# ###############################################################################################
# # You should see iterative execution and compilation steps printed in your console, along with an improvement
# # in latency per step.

# The expected reduction output is shown below:

# 2025-07-18 13:18:15,005 - qairt.tuner - INFO - Improvement in criteria (latency) observed: 10.67%

# # Once the process is complete, view the tuned QHAS report for more details.
#
# Validate and save the report summary
assert report.summary is not None
bw_tuned_qhas_report_path = Path(tuned_artifacts_dir / "latency_tuned_model_qhas.json")
bw_qhas = report.summary.dump(bw_tuned_qhas_report_path)
#
# ###############################################################################################
# # Print or view the QHAS report
print_or_view_qhas(bw_tuned_qhas_report_path)
#

# Using a Snapdragon SM8750 device, you should see a reduction in *total_time (us)* metric of about 10%. Note that
# this number will vary depending on the device used for tuning, and may show slight variations on different
# execution runs. ** You may require multiple tuning runs (or a warm start) to achieve a tangible
# reduction in latency.**

# .. note::
#   The tuner API can also be used to reduce bandwidth - *total_dram* metrics. To do so, set the
#   `criteria` argument to `bandwidth`.

# .. warning::
#
#    In some cases, no measurable improvement can be reached. It is recommended to use the Tuner API for smaller
#    models to understand its impact on your use case.
