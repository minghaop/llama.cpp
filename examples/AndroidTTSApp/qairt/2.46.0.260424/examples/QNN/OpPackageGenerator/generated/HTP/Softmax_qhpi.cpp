// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#include <array>
#include <cassert>
#include <cmath>
#include <functional>

#include "HTP/core/qhpi.h"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X)        STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR   STRINGIZE(THIS_PKG_NAME)

// Utility template for tensor access
template <typename Elt>
struct Flat4 {
  uint32_t multipliers[3];
  Elt *data;
  QHPI_Quant_Parameters params;
  QHPI_Shape shape;

  Flat4(const QHPI_Tensor *tensor) {
    shape          = qhpi_tensor_shape(tensor);
    data           = reinterpret_cast<Elt *>(qhpi_tensor_raw_data(tensor));
    multipliers[2] = shape.dims[3];
    multipliers[1] = multipliers[2] * shape.dims[2];
    multipliers[0] = multipliers[1] * shape.dims[1];
    params         = qhpi_tensor_quant_parameters(tensor);
  }

  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    uint32_t offset = b * multipliers[0] + h * multipliers[1] + w * multipliers[2] + d;
    return data[offset];
  }

  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    uint32_t offset = b * multipliers[0] + h * multipliers[1] + w * multipliers[2] + d;
    return data[offset];
  }

  float operator()(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    Elt value = get_raw(b, h, w, d);
    if constexpr (std::is_same_v<Elt, float>) {
      return value;
    } else {
      return (value - params.zero_offset) * params.stepsize;
    }
  }

  void set(uint32_t b, uint32_t h, uint32_t w, uint32_t d, float value) {
    if constexpr (std::is_same_v<Elt, float>) {
      get_raw(b, h, w, d) = value;
    } else {
      get_raw(b, h, w, d) =
          static_cast<Elt>(std::round(value / params.stepsize + params.zero_offset));
    }
  }
};

// Softmax implementation
template <typename TensorType>
static uint32_t softmaxImpl(QHPI_RuntimeHandle *,
                            uint32_t num_outputs,
                            QHPI_Tensor **outputs,
                            uint32_t num_inputs,
                            const QHPI_Tensor *const *inputs) {
  if (num_inputs < 1 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorType out(outputs[0]);
  const TensorType in(inputs[0]);

  // Get axis parameter if provided (default to last dimension)
  int32_t axis = -1;  // Default to last dimension
  if (num_inputs >= 2) {
    // Axis is provided as a tensor parameter
    const Flat4<int32_t> axis_tensor(inputs[1]);
    axis = axis_tensor(0, 0, 0, 0);
  }

  auto b_in = in.shape.dims[0];
  auto h_in = in.shape.dims[1];
  auto w_in = in.shape.dims[2];
  auto d_in = in.shape.dims[3];

  // Handle negative axis
  if (axis < 0) {
    axis = 4 + axis;  // Convert to positive index (assuming 4D tensor)
  }

  // For simplicity, implement softmax along the depth dimension (axis=3)
  // This matches the original legacy implementation
  for (uint32_t b = 0; b < b_in; b++) {
    for (uint32_t h = 0; h < h_in; h++) {
      for (uint32_t w = 0; w < w_in; w++) {
        // Get maximum element for numerical stability
        float max = in(b, h, w, 0);
        for (uint32_t d = 1; d < d_in; d++) {
          float inval = in(b, h, w, d);
          max         = fmaxf(inval, max);
        }

        // Sum of exponentials
        float sum = 0;
        for (uint32_t d = 0; d < d_in; d++) {
          float inval = in(b, h, w, d);
          sum += expf(inval - max);
        }

        // Normalization
        float sum_recip = 1.0f / sum;
        for (uint32_t d = 0; d < d_in; d++) {
          float inval  = in(b, h, w, d);
          float outval = expf(inval - max);
          out.set(b, h, w, d, outval * sum_recip);
        }
      }
    }
  }

  return QHPI_Success;
}

// Early rewrite function to add default axis parameter
static const QHPI_Op *softmax_add_axis(const QHPI_Op *op) {
  QHPI_OutputDef axis_def = {.type = QHPI_Int32, .shape = {4, {1, 1, 1, 1}}};
  int32_t axis_value      = -1;

  uint32_t num_inputs = qhpi_op_num_inputs(op);
  if (num_inputs >= 2) {
    // Axis already provided
    axis_value = *((int32_t*)qhpi_op_constant_data(qhpi_op_input(op, 1).op));
  }

  // Create default axis parameter (axis = -1, which means last dimension)
  const QHPI_Op *axis_const =
      qhpi_op_create_constant(op, &axis_def, sizeof(axis_value), &axis_value);
  QHPI_OpRef axis_ref = {axis_const, 0};

  // Get original input
  QHPI_OpRef input = qhpi_op_input(op, 0);

  // Create new op with axis parameter
  QHPI_OpRef inputs[]   = {input, axis_ref};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::Softmax_ref", 2, inputs, 1, &output);
}

// Cost function
static float softmaxCostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
  if (num_inputs < 1) return 0.0f;

  QHPI_Shape shape        = qhpi_tensor_shape(inputs[0]);
  uint32_t total_elements = 1;
  for (uint32_t i = 0; i < shape.rank; i++) {
    total_elements *= shape.dims[i];
  }

  // Softmax involves exp, sum, and division operations
  return total_elements * 10.0f;  // Higher cost due to exp operations
}

// Template-based kernel wrapper function to avoid code duplication
template <typename ElementType>
static uint32_t softmaxKernelWrapper(QHPI_RuntimeHandle *handle,
                                     uint32_t num_outputs,
                                     QHPI_Tensor **outputs,
                                     uint32_t num_inputs,
                                     const QHPI_Tensor *const *inputs) {
  return softmaxImpl<Flat4<ElementType>>(handle, num_outputs, outputs, num_inputs, inputs);
}

// Tensor signatures
static QHPI_Tensor_Signature_v1 softmax_float_in[] = {{.element_type  = QHPI_Float32,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only},
                                                      {.element_type  = QHPI_Int32,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 softmax_float_out[] = {{.element_type  = QHPI_Float32,
                                                        .layout        = QHPI_Layout_Flat4,
                                                        .storage       = QHPI_Storage_Direct,
                                                        .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 softmax_quint8_in[] = {{.element_type  = QHPI_QUInt8,
                                                        .layout        = QHPI_Layout_Flat4,
                                                        .storage       = QHPI_Storage_Direct,
                                                        .mem_placement = QHPI_MemLoc_DDR_Only},
                                                       {.element_type  = QHPI_Int32,
                                                        .layout        = QHPI_Layout_Flat4,
                                                        .storage       = QHPI_Storage_Direct,
                                                        .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 softmax_quint8_out[] = {{.element_type  = QHPI_QUInt8,
                                                         .layout        = QHPI_Layout_Flat4,
                                                         .storage       = QHPI_Storage_Direct,
                                                         .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 softmax_quint16_in[] = {{.element_type  = QHPI_QUInt16,
                                                         .layout        = QHPI_Layout_Flat4,
                                                         .storage       = QHPI_Storage_Direct,
                                                         .mem_placement = QHPI_MemLoc_DDR_Only},
                                                        {.element_type  = QHPI_Int32,
                                                         .layout        = QHPI_Layout_Flat4,
                                                         .storage       = QHPI_Storage_Direct,
                                                         .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 softmax_quint16_out[] = {{.element_type  = QHPI_QUInt16,
                                                          .layout        = QHPI_Layout_Flat4,
                                                          .storage       = QHPI_Storage_Direct,
                                                          .mem_placement = QHPI_MemLoc_DDR_Only}};

// Kernel definitions
static QHPI_Kernel_v1 softmax_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::softmaxKernelWrapper<float>",
    .function                       = softmaxKernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = true,  // Allow variable number of inputs (1 or 2)
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = softmax_float_in,
    .min_outputs                    = 1,
    .output_signature               = softmax_float_out,
    .cost_function                  = softmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 softmax_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::softmaxKernelWrapper<uint8_t>",
    .function                       = softmaxKernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = true,  // Allow variable number of inputs (1 or 2)
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = softmax_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = softmax_quint8_out,
    .cost_function                  = softmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 softmax_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::softmaxKernelWrapper<uint16_t>",
    .function                       = softmaxKernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = true,  // Allow variable number of inputs (1 or 2)
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = softmax_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = softmax_quint16_out,
    .cost_function                  = softmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

// Kernel array
static QHPI_Kernel_v1 softmax_kernels[] = {
    softmax_float_kernel, softmax_quint8_kernel, softmax_quint16_kernel};

// OpInfo definition
QHPI_OpInfo_v1 softmaxOpInfo[] = {{.name            = THIS_PKG_NAME_STR "::Softmax",
                                   .early_rewrite   = softmax_add_axis,
                                   },

                                   {.name = THIS_PKG_NAME_STR "::Softmax_ref",
                                   .num_kernels     = 3,
                                   .kernels         = softmax_kernels,
                                   }
                                   };

// Registration function for Softmax operations
extern "C" void register_softmax_ops() {
  qhpi_register_ops_v1(
      sizeof(softmaxOpInfo) / sizeof(softmaxOpInfo[0]), softmaxOpInfo, THIS_PKG_NAME_STR);
}
