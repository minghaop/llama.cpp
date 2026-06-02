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

// Create a constant scalar
static QHPI_OpRef gen_const_scalar_f32(const QHPI_Op *op, float value) {
  QHPI_OutputDef scalar_def = {.type = QHPI_Float32, .shape = {4, {1, 1, 1, 1}}};
  return {qhpi_op_create_constant(op, &scalar_def, sizeof(value), &value), 0};
}

// ReLU implementation
template <typename TensorType>
static uint32_t reluImpl(QHPI_RuntimeHandle *,
                         uint32_t num_outputs,
                         QHPI_Tensor **outputs,
                         uint32_t num_inputs,
                         const QHPI_Tensor *const *inputs) {
  if (num_inputs != 1 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorType out(outputs[0]);
  const TensorType in(inputs[0]);

  auto b = in.shape.dims[0];
  auto h = in.shape.dims[1];
  auto w = in.shape.dims[2];
  auto d = in.shape.dims[3];

  for (uint32_t batch = 0; batch < b; batch++) {
    for (uint32_t height = 0; height < h; height++) {
      for (uint32_t width = 0; width < w; width++) {
        for (uint32_t depth = 0; depth < d; depth++) {
          float inval = in(batch, height, width, depth);
          out.set(batch, height, width, depth, fmaxf(inval, 0.0f));
        }
      }
    }
  }

  return QHPI_Success;
}

// ReluX implementation (ReLU with upper bound)
template <typename TensorTypeI, typename TensorTypeX>
static uint32_t reluXImpl(QHPI_RuntimeHandle *,
                          uint32_t num_outputs,
                          QHPI_Tensor **outputs,
                          uint32_t num_inputs,
                          const QHPI_Tensor *const *inputs) {
  if (num_inputs != 2 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorTypeI out(outputs[0]);
  const TensorTypeI in(inputs[0]);
  const TensorTypeX inX(inputs[1]);

  float x = inX(0, 0, 0, 0);

  if (!(x > 0.0f)) {
    return QHPI_ErrorFatal;
  }

  auto b = in.shape.dims[0];
  auto h = in.shape.dims[1];
  auto w = in.shape.dims[2];
  auto d = in.shape.dims[3];

  for (uint32_t batch = 0; batch < b; batch++) {
    for (uint32_t height = 0; height < h; height++) {
      for (uint32_t width = 0; width < w; width++) {
        for (uint32_t depth = 0; depth < d; depth++) {
          float inval = in(batch, height, width, depth);
          out.set(batch, height, width, depth, fminf(fmaxf(inval, 0.0f), x));
        }
      }
    }
  }

  return QHPI_Success;
}

// Relu1 implementation (ReLU with range [-1, 1])
template <typename TensorType>
static uint32_t relu1Impl(QHPI_RuntimeHandle *,
                          uint32_t num_outputs,
                          QHPI_Tensor **outputs,
                          uint32_t num_inputs,
                          const QHPI_Tensor *const *inputs) {
  if (num_inputs != 1 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorType out(outputs[0]);
  const TensorType in(inputs[0]);

  auto b = in.shape.dims[0];
  auto h = in.shape.dims[1];
  auto w = in.shape.dims[2];
  auto d = in.shape.dims[3];

  for (uint32_t batch = 0; batch < b; batch++) {
    for (uint32_t height = 0; height < h; height++) {
      for (uint32_t width = 0; width < w; width++) {
        for (uint32_t depth = 0; depth < d; depth++) {
          float inval = in(batch, height, width, depth);
          out.set(batch, height, width, depth, fminf(fmaxf(inval, -1.0f), 1.0f));
        }
      }
    }
  }

  return QHPI_Success;
}

// Early rewrite functions to convert variants to ReluMinMax
static const QHPI_Op *relu_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);

  // Create ReluMinMax with min=0.0f, max=INF
  QHPI_OpRef min_const = gen_const_scalar_f32(op, 0.0f);
  QHPI_OpRef max_const = gen_const_scalar_f32(op, INFINITY);

  QHPI_OpRef inputs[]   = {input, min_const, max_const};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1, &output);
}

static const QHPI_Op *relux_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input     = qhpi_op_input(op, 0);
  QHPI_OpRef max_input = qhpi_op_input(op, 1);

  // Create ReluMinMax with min=0.0f, max=Max
  QHPI_OpRef min_const = gen_const_scalar_f32(op, 0.0f);

  QHPI_OpRef inputs[]   = {input, min_const, max_input};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1, &output);
}

static const QHPI_Op *relu1_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);

  // Create ReluMinMax with min=-1.0f, max=1.0f
  QHPI_OpRef min_const = gen_const_scalar_f32(op, -1.0f);
  QHPI_OpRef max_const = gen_const_scalar_f32(op, 1.0f);

  QHPI_OpRef inputs[]   = {input, min_const, max_const};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1, &output);
}

// ReluMinMax implementation
template <typename TensorTypeI, typename TensorTypeX>
static uint32_t reluMinMaxImpl(QHPI_RuntimeHandle *,
                               uint32_t num_outputs,
                               QHPI_Tensor **outputs,
                               uint32_t num_inputs,
                               const QHPI_Tensor *const *inputs) {
  if (num_inputs != 3 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  TensorTypeI out(outputs[0]);
  const TensorTypeI in(inputs[0]);
  const TensorTypeX inMin(inputs[1]);
  const TensorTypeX inMax(inputs[2]);

  float min_val = inMin(0, 0, 0, 0);
  float max_val = inMax(0, 0, 0, 0);

  if (!(max_val > min_val)) {
    return QHPI_ErrorFatal;
  }

  auto b = in.shape.dims[0];
  auto h = in.shape.dims[1];
  auto w = in.shape.dims[2];
  auto d = in.shape.dims[3];

  for (uint32_t batch = 0; batch < b; batch++) {
    for (uint32_t height = 0; height < h; height++) {
      for (uint32_t width = 0; width < w; width++) {
        for (uint32_t depth = 0; depth < d; depth++) {
          float inval = in(batch, height, width, depth);
          out.set(batch, height, width, depth, fminf(fmaxf(inval, min_val), max_val));
        }
      }
    }
  }

  return QHPI_Success;
}

// Cost functions
static float reluCostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
  if (num_inputs < 1) return 0.0f;

  QHPI_Shape shape        = qhpi_tensor_shape(inputs[0]);
  uint32_t total_elements = 1;
  for (uint32_t i = 0; i < shape.rank; i++) {
    total_elements *= shape.dims[i];
  }

  return total_elements * 1.0f;  // Simple cost
}

// Template-based kernel wrapper functions to avoid code duplication
template <typename ElementType>
static uint32_t reluKernelWrapper(QHPI_RuntimeHandle *handle,
                                  uint32_t num_outputs,
                                  QHPI_Tensor **outputs,
                                  uint32_t num_inputs,
                                  const QHPI_Tensor *const *inputs) {
  return reluImpl<Flat4<ElementType>>(handle, num_outputs, outputs, num_inputs, inputs);
}

template <typename ElementType>
static uint32_t relu1KernelWrapper(QHPI_RuntimeHandle *handle,
                                   uint32_t num_outputs,
                                   QHPI_Tensor **outputs,
                                   uint32_t num_inputs,
                                   const QHPI_Tensor *const *inputs) {
  return relu1Impl<Flat4<ElementType>>(handle, num_outputs, outputs, num_inputs, inputs);
}

template <typename ElementType>
static uint32_t reluXKernelWrapper(QHPI_RuntimeHandle *handle,
                                   uint32_t num_outputs,
                                   QHPI_Tensor **outputs,
                                   uint32_t num_inputs,
                                   const QHPI_Tensor *const *inputs) {
  return reluXImpl<Flat4<ElementType>, Flat4<float>>(
      handle, num_outputs, outputs, num_inputs, inputs);
}

template <typename ElementType>
static uint32_t reluMinMaxKernelWrapper(QHPI_RuntimeHandle *handle,
                                        uint32_t num_outputs,
                                        QHPI_Tensor **outputs,
                                        uint32_t num_inputs,
                                        const QHPI_Tensor *const *inputs) {
  return reluMinMaxImpl<Flat4<ElementType>, Flat4<float>>(
      handle, num_outputs, outputs, num_inputs, inputs);
}

// Tensor signatures
static QHPI_Tensor_Signature_v1 relu_float_in[] = {{.element_type  = QHPI_Float32,
                                                    .layout        = QHPI_Layout_Flat4,
                                                    .storage       = QHPI_Storage_Direct,
                                                    .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_float_out[] = {{.element_type  = QHPI_Float32,
                                                     .layout        = QHPI_Layout_Flat4,
                                                     .storage       = QHPI_Storage_Direct,
                                                     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_quint8_in[] = {{.element_type  = QHPI_QUInt8,
                                                     .layout        = QHPI_Layout_Flat4,
                                                     .storage       = QHPI_Storage_Direct,
                                                     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_quint8_out[] = {{.element_type  = QHPI_QUInt8,
                                                      .layout        = QHPI_Layout_Flat4,
                                                      .storage       = QHPI_Storage_Direct,
                                                      .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_quint16_in[] = {{.element_type  = QHPI_QUInt16,
                                                      .layout        = QHPI_Layout_Flat4,
                                                      .storage       = QHPI_Storage_Direct,
                                                      .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_quint16_out[] = {{.element_type  = QHPI_QUInt16,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relux_float_in[] = {{.element_type  = QHPI_Float32,
                                                     .layout        = QHPI_Layout_Flat4,
                                                     .storage       = QHPI_Storage_Direct,
                                                     .mem_placement = QHPI_MemLoc_DDR_Only},
                                                    {.element_type  = QHPI_Float32,
                                                     .layout        = QHPI_Layout_Flat4,
                                                     .storage       = QHPI_Storage_Direct,
                                                     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relux_quint8_in[] = {{.element_type  = QHPI_QUInt8,
                                                      .layout        = QHPI_Layout_Flat4,
                                                      .storage       = QHPI_Storage_Direct,
                                                      .mem_placement = QHPI_MemLoc_DDR_Only},
                                                     {.element_type  = QHPI_Float32,
                                                      .layout        = QHPI_Layout_Flat4,
                                                      .storage       = QHPI_Storage_Direct,
                                                      .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relux_quint16_in[] = {{.element_type  = QHPI_QUInt16,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only},
                                                      {.element_type  = QHPI_Float32,
                                                       .layout        = QHPI_Layout_Flat4,
                                                       .storage       = QHPI_Storage_Direct,
                                                       .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_minmax_float_in[] = {{.element_type  = QHPI_Float32,
                                                           .layout        = QHPI_Layout_Flat4,
                                                           .storage       = QHPI_Storage_Direct,
                                                           .mem_placement = QHPI_MemLoc_DDR_Only},
                                                          {.element_type  = QHPI_Float32,
                                                           .layout        = QHPI_Layout_Flat4,
                                                           .storage       = QHPI_Storage_Direct,
                                                           .mem_placement = QHPI_MemLoc_DDR_Only},
                                                          {.element_type  = QHPI_Float32,
                                                           .layout        = QHPI_Layout_Flat4,
                                                           .storage       = QHPI_Storage_Direct,
                                                           .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_minmax_quint8_in[] = {{.element_type  = QHPI_QUInt8,
                                                            .layout        = QHPI_Layout_Flat4,
                                                            .storage       = QHPI_Storage_Direct,
                                                            .mem_placement = QHPI_MemLoc_DDR_Only},
                                                           {.element_type  = QHPI_Float32,
                                                            .layout        = QHPI_Layout_Flat4,
                                                            .storage       = QHPI_Storage_Direct,
                                                            .mem_placement = QHPI_MemLoc_DDR_Only},
                                                           {.element_type  = QHPI_Float32,
                                                            .layout        = QHPI_Layout_Flat4,
                                                            .storage       = QHPI_Storage_Direct,
                                                            .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_minmax_quint16_in[] = {
    {.element_type  = QHPI_QUInt16,
     .layout        = QHPI_Layout_Flat4,
     .storage       = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type  = QHPI_Float32,
     .layout        = QHPI_Layout_Flat4,
     .storage       = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type  = QHPI_Float32,
     .layout        = QHPI_Layout_Flat4,
     .storage       = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

// Kernel definitions
static QHPI_Kernel_v1 relu_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluKernelWrapper<float>",
    .function                       = reluKernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_float_in,
    .min_outputs                    = 1,
    .output_signature               = relu_float_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluKernelWrapper<uint8_t>",
    .function                       = reluKernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint8_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluKernelWrapper<uint16_t>",
    .function                       = reluKernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint16_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relux_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluXKernelWrapper<float>",
    .function                       = reluXKernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 2,
    .input_signature                = relux_float_in,
    .min_outputs                    = 1,
    .output_signature               = relu_float_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relux_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluXKernelWrapper<uint8_t>",
    .function                       = reluXKernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 2,
    .input_signature                = relux_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint8_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relux_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluXKernelWrapper<uint16_t>",
    .function                       = reluXKernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 2,
    .input_signature                = relux_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint16_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu1_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::relu1KernelWrapper<float>",
    .function                       = relu1KernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_float_in,
    .min_outputs                    = 1,
    .output_signature               = relu_float_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu1_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::relu1KernelWrapper<uint8_t>",
    .function                       = relu1KernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint8_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu1_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::relu1KernelWrapper<uint16_t>",
    .function                       = relu1KernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 1,
    .input_signature                = relu_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint16_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu_minmax_float_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluMinMaxKernelWrapper<float>",
    .function                       = reluMinMaxKernelWrapper<float>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 3,
    .input_signature                = relu_minmax_float_in,
    .min_outputs                    = 1,
    .output_signature               = relu_float_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu_minmax_quint8_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluMinMaxKernelWrapper<uint8_t>",
    .function                       = reluMinMaxKernelWrapper<uint8_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 3,
    .input_signature                = relu_minmax_quint8_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint8_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

static QHPI_Kernel_v1 relu_minmax_quint16_kernel = {
    .function_name                  = THIS_PKG_NAME_STR "::reluMinMaxKernelWrapper<uint16_t>",
    .function                       = reluMinMaxKernelWrapper<uint16_t>,
    .resources                      = QHPI_RESOURCE_HVX,
    .source_destructive             = false,
    .multithreaded                  = false,
    .variable_inputs                = false,
    .variable_outputs               = false,
    .min_inputs                     = 3,
    .input_signature                = relu_minmax_quint16_in,
    .min_outputs                    = 1,
    .output_signature               = relu_quint16_out,
    .cost_function                  = reluCostFunc,
    .sync_block_size                = 0,
    .precomputed_data_size          = 0,
    .do_precomputation_function     = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate                      = nullptr};

// Kernel arrays
static QHPI_Kernel_v1 relu_kernels[] = {relu_float_kernel, relu_quint8_kernel, relu_quint16_kernel};
static QHPI_Kernel_v1 relux_kernels[] = {
    relux_float_kernel, relux_quint8_kernel, relux_quint16_kernel};
static QHPI_Kernel_v1 relu1_kernels[] = {
    relu1_float_kernel, relu1_quint8_kernel, relu1_quint16_kernel};
static QHPI_Kernel_v1 relu_minmax_kernels[] = {
    relu_minmax_float_kernel, relu_minmax_quint8_kernel, relu_minmax_quint16_kernel};

// OpInfo definitions
QHPI_OpInfo_v1 reluOpInfo[] = {{.name            = THIS_PKG_NAME_STR "::Relu",
                                .num_kernels     = 3,
                                .kernels         = relu_kernels,
                                .early_rewrite   = relu_to_relu_minmax,
                                .shape_required  = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile      = nullptr,
                                .late_rewrite    = nullptr},
                               {.name            = THIS_PKG_NAME_STR "::ReluX",
                                .num_kernels     = 3,
                                .kernels         = relux_kernels,
                                .early_rewrite   = relux_to_relu_minmax,
                                .shape_required  = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile      = nullptr,
                                .late_rewrite    = nullptr},
                               {.name            = THIS_PKG_NAME_STR "::Relu1",
                                .num_kernels     = 3,
                                .kernels         = relu1_kernels,
                                .early_rewrite   = relu1_to_relu_minmax,
                                .shape_required  = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile      = nullptr,
                                .late_rewrite    = nullptr},
                               {.name            = THIS_PKG_NAME_STR "::ReluMinMax",
                                .num_kernels     = 3,
                                .kernels         = relu_minmax_kernels,
                                .early_rewrite   = nullptr,
                                .shape_required  = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile      = nullptr,
                                .late_rewrite    = nullptr}};

// Registration function for ReLU operations
extern "C" void register_relu_ops() {
  qhpi_register_ops_v1(sizeof(reluOpInfo) / sizeof(reluOpInfo[0]), reluOpInfo, THIS_PKG_NAME_STR);
}
