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

// ArgMax implementation for different tensor types
template <typename TensorType>
static uint32_t argmaxImpl(QHPI_RuntimeHandle *,
                           uint32_t num_outputs,
                           QHPI_Tensor **outputs,
                           uint32_t num_inputs,
                           const QHPI_Tensor *const *inputs) {
  if (num_inputs != 1 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorType out(outputs[0]);
  const TensorType in(inputs[0]);

  auto b_in = in.shape.dims[0];
  auto h_in = in.shape.dims[1];
  auto w_in = in.shape.dims[2];
  auto d_in = in.shape.dims[3];

  for (uint32_t b = 0; b < b_in; b++) {
    for (uint32_t h = 0; h < h_in; h++) {
      for (uint32_t w = 0; w < w_in; w++) {
        float max     = in(b, h, w, 0);
        float max_idx = 0;
        for (uint32_t d = 1; d < d_in; d++) {
          float inval = in(b, h, w, d);
          if (inval > max) {
            max     = inval;
            max_idx = d;
          }
        }
        out.set(b, h, w, 0, max_idx);
      }
    }
  }

  return QHPI_Success;
}

// Cost function
static float argmaxCostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
  if (num_inputs < 1) return 0.0f;

  QHPI_Shape shape        = qhpi_tensor_shape(inputs[0]);
  uint32_t total_elements = 1;
  for (uint32_t i = 0; i < shape.rank; i++) {
    total_elements *= shape.dims[i];
  }

  return total_elements * 2.0f;  // Approximate cost
}

// Template-based kernel wrapper function to avoid code duplication
template <typename ElementType>
static uint32_t argmaxKernelWrapper(QHPI_RuntimeHandle *handle,
                                    uint32_t num_outputs,
                                    QHPI_Tensor **outputs,
                                    uint32_t num_inputs,
                                    const QHPI_Tensor *const *inputs) {
  return argmaxImpl<Flat4<ElementType>>(handle, num_outputs, outputs, num_inputs, inputs);
}

// Tensor signatures
static QHPI_Tensor_Signature_v1 argmax_float_in[] = {{.element_type  = QHPI_Float32,
                                                      .layout        = QHPI_Layout_Flat4,
                                                      .storage       = QHPI_Storage_Direct,
                                                      .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 argmax_float_out[] = {{.element_type  = QHPI_Float32,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 argmax_quint8_in[] = {{.element_type  = QHPI_QUInt8,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 argmax_quint8_out[] = {{.element_type  = QHPI_QUInt8,
                                                        .layout        = QHPI_Layout_Flat4,
                                                        .storage       = QHPI_Storage_Direct,
                                                        .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 argmax_quint16_in[] = {{.element_type  = QHPI_QUInt16,
                                                        .layout        = QHPI_Layout_Flat4,
                                                        .storage       = QHPI_Storage_Direct,
                                                        .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 argmax_quint16_out[] = {{.element_type  = QHPI_QUInt16,
                                                         .layout        = QHPI_Layout_Flat4,
                                                         .storage       = QHPI_Storage_Direct,
                                                         .mem_placement = QHPI_MemLoc_DDR_Only}};

// Kernel definitions
static QHPI_Kernel_v1 argmax_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::argmaxKernelWrapper<float>",
    .function                       = argmaxKernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = argmax_float_in,
    .min_outputs                    = 1,
    .output_signature               = argmax_float_out,
    .cost_function                  = argmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 argmax_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::argmaxKernelWrapper<uint8_t>",
    .function                       = argmaxKernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = argmax_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = argmax_quint8_out,
    .cost_function                  = argmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 argmax_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::argmaxKernelWrapper<uint16_t>",
    .function                       = argmaxKernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = argmax_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = argmax_quint16_out,
    .cost_function                  = argmaxCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

// Kernel array
static QHPI_Kernel_v1 argmax_kernels[] = {
    argmax_float_kernel, argmax_quint8_kernel, argmax_quint16_kernel};

// OpInfo definition
QHPI_OpInfo_v1 argmaxOpInfo[] = {{.name            = THIS_PKG_NAME_STR "::ArgMax",
                                  .num_kernels     = 3,
                                  .kernels         = argmax_kernels,
                                  .early_rewrite   = nullptr,
                                  .shape_required  = nullptr,
                                  .shape_legalized = nullptr,
                                  .build_tile      = nullptr,
                                  .late_rewrite    = nullptr}};

// Registration function for ArgMax operations
extern "C" void register_argmax_ops() {
  qhpi_register_ops_v1(
      sizeof(argmaxOpInfo) / sizeof(argmaxOpInfo[0]), argmaxOpInfo, THIS_PKG_NAME_STR);
}
