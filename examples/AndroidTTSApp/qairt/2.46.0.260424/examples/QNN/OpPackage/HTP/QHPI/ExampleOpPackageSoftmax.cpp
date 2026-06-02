// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#include "HTP/core/intrinsics.h"
#include "HTP/core/qhpi.h"
#include "qhpi_example_utils.h"
#include <cassert>

extern "C" {
uint32_t softmax_impl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                      QHPI_Tensor **outputs, uint32_t num_inputs,
                      const QHPI_Tensor *const *inputs) {
  Flat4<float> in(inputs[0]);
  Flat4<float> out(outputs[0]);

  QHPI_Shape in0_shape = qhpi_tensor_shape(inputs[0]);

  float beta = *((float *)qhpi_tensor_raw_data(inputs[1]));

  size_t b_in = in0_shape.dims[0];
  size_t h_in = in0_shape.dims[1];
  size_t w_in = in0_shape.dims[2];
  size_t d_in = in0_shape.dims[3];

  for (size_t b = 0; b < b_in; b++) {
    for (size_t h = 0; h < h_in; h++) {
      for (size_t w = 0; w < w_in; w++) {
        float max = in(b, h, w, 0);
        for (size_t d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          max = fmaxf(inval, max);
        }
        float sum = 0;
        for (size_t d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          sum += (out(b, h, w, d) = expf(beta * (inval - max)));
        }
        float const sum_recip = 1.0f / sum;
        for (size_t d = 0; d < d_in; d++) {
          float const outval = out(b, h, w, d);
          out(b, h, w, d) = outval * sum_recip;
        }
      }
    }
  }

  return QHPI_Success;
}
}

static const QHPI_Op *softmax_to_ref(const QHPI_Op *op) {
  uint32_t num_inputs = qhpi_op_num_inputs(op);

  QHPI_OpRef in = qhpi_op_input(op, 0);

  QHPI_OutputDef beta_def = {.type = QHPI_Float32,
                             .quant_parameters = {0, 0.0f},
                             .shape = {.rank = 4, .dims = {1, 1, 1, 1}}};

  // Get or create beta parameter
  QHPI_OpRef beta;
  float beta_value = 1.0f;
  if (num_inputs >= 2) {
    // Use beta value given
    beta = qhpi_op_input(op, 1);
    const void* beta_data = qhpi_op_constant_data(beta.op);
    if (beta_data) {
      beta_value = *((const float *)beta_data);  // Direct float cast, not reinterpret
    }
  }

  const QHPI_Op *beta_const =
      qhpi_op_create_constant(op, &beta_def, sizeof(float), &beta_value);
  beta = qhpi_op_reference(beta_const, 0);

  QHPI_OutputDef out_def = qhpi_op_output(op, 0);

  QHPI_OpRef inputs[2] = {in, beta};
  const QHPI_Op *new_op = qhpi_op_create(op, THIS_PKG_NAME_STR "::Softmax_ref",
                                         2, inputs, 1, &out_def);

  return new_op;
}

static QHPI_Tensor_Signature_v1 in_sigs[] = {
    {QHPI_Any_Element_Type, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 out_sigs[] = {
    {QHPI_Any_Element_Type, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_TCM_Only},
};

// Kernel instances
static QHPI_Kernel_v1 op_instances[] = {
    {.function_name = "softmax_impl",
     .function = softmax_impl,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = false,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 2,
     .input_signature = in_sigs,
     .min_outputs = 1,
     .output_signature = out_sigs,
     .cost_function = nullptr,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr},
};

// Op registration
static QHPI_OpInfo_v1 softmax_ops[] = {
    {.name = THIS_PKG_NAME_STR "::Softmax",
     .num_kernels = 0,
     .kernels = nullptr,
     .early_rewrite = softmax_to_ref,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::Softmax_ref",
     .num_kernels = 1,
     .kernels = op_instances,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr}};

void register_softmax_ops() {
  qhpi_register_ops_v1(2, softmax_ops, THIS_PKG_NAME_STR);
  return;
}
