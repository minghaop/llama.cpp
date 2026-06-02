// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#include "HTP/core/log.h"
#include "HTP/core/qhpi.h"
#include "qhpi_example_utils.h"
#include <array>
#include <cassert>
#include <cmath>
#include <functional>

#define RELU_TILE_HEIGHT 8
#define RELU_CHANNEL_SPLIT_SIZE 256

// Create a constant scalar
static QHPI_OpRef gen_const_scalar_f32(const QHPI_Op *op, float value) {
  QHPI_OutputDef scalar_def = {.type = QHPI_Float32,
                               .shape = {4, {1, 1, 1, 1}}};
  return {qhpi_op_create_constant(op, &scalar_def, sizeof(value), &value), 0};
}

static QHPI_OpRef gen_const_scalar_i32(const QHPI_Op *op, int32_t value) {
  QHPI_OutputDef scalar_def = {.type = QHPI_Int32, .shape = {4, {1, 1, 1, 1}}};
  return {qhpi_op_create_constant(op, &scalar_def, sizeof(value), &value), 0};
}

/*======================================  End C++ Plugin Helpers
 * =============================================*/

// op execute function declarations
template <typename T_Ttype>
static uint32_t reluImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                         QHPI_Tensor **outputs, uint32_t num_inputs,
                         const QHPI_Tensor *const *inputs);

template <typename T_TtypeI, typename T_TtypeX>
static uint32_t reluMinMaxImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                               QHPI_Tensor **outputs, uint32_t num_inputs,
                               const QHPI_Tensor *const *inputs);

static uint32_t reluTablegenImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                                 QHPI_Tensor **outputs, uint32_t num_inputs,
                                 const QHPI_Tensor *const *inputs);

static uint32_t tableLookupImpl(QHPI_RuntimeHandle *handle,
                                uint32_t num_outputs, QHPI_Tensor **outputs,
                                uint32_t num_inputs,
                                const QHPI_Tensor *const *inputs);

// Early rewrite functions (equivalent to DEF_PACKAGE_OPTIMIZATION)
static const QHPI_Op *relu_to_relu_minmax_quant(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);

  // Check if input is quantized type
  if (input_output.type != QHPI_QUInt8 && input_output.type != QHPI_QUInt16 &&
      input_output.type != QHPI_QInt8 && input_output.type != QHPI_QInt16) {
    return op;
  }

  // Create ReluMinMax with min=0.0f, max=INF
  QHPI_OpRef min_const = gen_const_scalar_f32(op, 0.0f);
  QHPI_OpRef max_const = gen_const_scalar_f32(op, INFINITY);

  QHPI_OpRef inputs[] = {input, min_const, max_const};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1,
                        &output);
}

static const QHPI_Op *relu6_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);

  // Create ReluMinMax with min=0.0f, max=6.0f
  QHPI_OpRef min_const = gen_const_scalar_f32(op, 0.0f);
  QHPI_OpRef max_const = gen_const_scalar_f32(op, 6.0f);

  QHPI_OpRef inputs[] = {input, min_const, max_const};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1,
                        &output);
}

static const QHPI_Op *relu1_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);

  // Create ReluMinMax with min=-1.0f, max=1.0f
  QHPI_OpRef min_const = gen_const_scalar_f32(op, -1.0f);
  QHPI_OpRef max_const = gen_const_scalar_f32(op, 1.0f);

  QHPI_OpRef inputs[] = {input, min_const, max_const};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1,
                        &output);
}

static const QHPI_Op *relux_to_relu_minmax(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OpRef max_input = qhpi_op_input(op, 1);

  // Create ReluMinMax with min=0.0f, max=Max
  QHPI_OpRef min_const = gen_const_scalar_f32(op, 0.0f);

  QHPI_OpRef inputs[] = {input, min_const, max_input};
  QHPI_OutputDef output = qhpi_op_output(op, 0);

  return qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluMinMax", 3, inputs, 1,
                        &output);
}

// Helper function to check quantization parameters
static bool same_quant_params(const QHPI_OutputDef &out1,
                              const QHPI_OutputDef &out2) {
  return (out1.quant_parameters.stepsize == out2.quant_parameters.stepsize &&
          out1.quant_parameters.zero_offset ==
              out2.quant_parameters.zero_offset);
}

static float min_qu8_range(const QHPI_OutputDef &output) {
  return output.quant_parameters.stepsize *
         (-1.0f * output.quant_parameters.zero_offset);
}

static float max_qu8_range(const QHPI_OutputDef &output) {
  return output.quant_parameters.stepsize *
         (255.0f - output.quant_parameters.zero_offset);
}

static const QHPI_Op *relu_minmax_to_tablelookup(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OpRef min_input = qhpi_op_input(op, 1);
  QHPI_OpRef max_input = qhpi_op_input(op, 2);

  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);
  QHPI_OutputDef op_output = qhpi_op_output(op, 0);

  // Check if input is QUint8 and output is QUint8 with different quantization
  if (input_output.type == QHPI_QUInt8 && op_output.type == QHPI_QUInt8 &&
      !same_quant_params(input_output, op_output)) {

    // Create ReluTableGen
    QHPI_OpRef stepsize_const =
        gen_const_scalar_f32(op, input_output.quant_parameters.stepsize);
    QHPI_OpRef offset_const =
        gen_const_scalar_i32(op, input_output.quant_parameters.zero_offset);

    QHPI_OutputDef table_output = {
        .type = QHPI_QUInt8,
        .quant_parameters = {op_output.quant_parameters.zero_offset,
                             op_output.quant_parameters.stepsize},
        .shape = {4, {1, 1, 1, 256}}};

    QHPI_OpRef table_inputs[] = {stepsize_const, offset_const, min_input,
                                 max_input};
    QHPI_OpRef table_op = {qhpi_op_create(op,
                                          THIS_PKG_NAME_STR "::ReluTableGen", 4,
                                          table_inputs, 1, &table_output),
                           0};

    // Create TableLookup
    QHPI_OpRef lookup_inputs[] = {input, table_op};
    return qhpi_op_create(op, THIS_PKG_NAME_STR "::TableLookup", 2,
                          lookup_inputs, 1, &op_output);
  }

  return op;
}

static const QHPI_Op *relu_minmax_quant_passthrough(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OpRef min_input = qhpi_op_input(op, 1);
  QHPI_OpRef max_input = qhpi_op_input(op, 2);

  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);
  QHPI_OutputDef op_output = qhpi_op_output(op, 0);

  // Check if both are QUint8 and have same quantization parameters
  if (input_output.type == QHPI_QUInt8 && op_output.type == QHPI_QUInt8 &&
      same_quant_params(input_output, op_output)) {

    // Get min/max constant values (assuming they are constants)
    if (qhpi_op_is_constant(min_input.op) &&
        qhpi_op_is_constant(max_input.op)) {
      const float *min_data =
          (const float *)qhpi_op_constant_data(min_input.op);
      const float *max_data =
          (const float *)qhpi_op_constant_data(max_input.op);

      float input_min = min_qu8_range(input_output);
      float input_max = max_qu8_range(input_output);

      // Check if input range is within the relu range
      if (input_min >= *min_data && input_max <= *max_data) {
        // Return input directly (passthrough)
        return input.op;
      }
    }
  } else {
    // Fall back to tablelookup
    return relu_minmax_to_tablelookup(op);
  }

  return op;
}

// Tiling support functions (equivalent to AUTOSPLIT)
static QHPI_Shape relu_shape_required(const QHPI_Op *op) {
  // Define tiling requirements - split on height dimension
  static QHPI_Shape required = {
      .rank = 4, .dims = {1, RELU_TILE_HEIGHT, 0, RELU_CHANNEL_SPLIT_SIZE}};
  return required;
}

static const QHPI_Op *relu_build_tile(const QHPI_Op *op,
                                      const QHPI_Shape *out_start,
                                      const QHPI_Shape *out_extent) {
  // Get input reference
  QHPI_OpRef input_ref = qhpi_op_input(op, 0);

  // For ReLU, input and output have same dimensions, so input slice = output
  // slice
  QHPI_Shape in_start = *out_start;
  QHPI_Shape in_extent = *out_extent;

  // Create input slice
  QHPI_OpRef input_slice = qhpi_op_slice(input_ref, &in_start, &in_extent);

  // Build tiled operator with sliced input
  QHPI_OpRef inputs[] = {input_slice};

  QHPI_OutputDef outputs[] = {
      {.type = qhpi_op_output(op, 0).type,
       .quant_parameters = qhpi_op_output(op, 0).quant_parameters,
       .shape = *out_extent}};

  return qhpi_op_create(op, qhpi_op_name(op), 1, inputs, 1, outputs);
}

static QHPI_Shape relu_minmax_shape_required(const QHPI_Op *op) {
  static QHPI_Shape required = {
      .rank = 4, .dims = {1, RELU_TILE_HEIGHT, 0, RELU_CHANNEL_SPLIT_SIZE}};
  return required;
}

static const QHPI_Op *relu_minmax_build_tile(const QHPI_Op *op,
                                             const QHPI_Shape *out_start,
                                             const QHPI_Shape *out_extent) {
  QHPI_OpRef input_ref = qhpi_op_input(op, 0);
  QHPI_OpRef min_ref = qhpi_op_input(op, 1);
  QHPI_OpRef max_ref = qhpi_op_input(op, 2);

  QHPI_Shape in_start = *out_start;
  QHPI_Shape in_extent = *out_extent;
  QHPI_OpRef input_slice = qhpi_op_slice(input_ref, &in_start, &in_extent);

  // Min/max inputs are typically scalars, so no slicing needed
  QHPI_OpRef inputs[] = {input_slice, min_ref, max_ref};
  QHPI_OutputDef outputs[] = {
      {.type = qhpi_op_output(op, 0).type,
       .quant_parameters = qhpi_op_output(op, 0).quant_parameters,
       .shape = *out_extent}};

  return qhpi_op_create(op, qhpi_op_name(op), 3, inputs, 1, outputs);
}

/* execute functions for ops */
template <typename T_Ttype>
static uint32_t reluImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                         QHPI_Tensor **outputs, uint32_t num_inputs,
                         const QHPI_Tensor *const *inputs) {
  T_Ttype out(outputs[0]);
  const T_Ttype in(inputs[0]);

  auto b = in.shape.dims[0], h = in.shape.dims[1], w = in.shape.dims[2],
       d = in.shape.dims[3];
  debuglog("relu execute... dims=(%zdx%zdx%zdx%zd)", in.shape.dims[0],
           in.shape.dims[1], in.shape.dims[2], in.shape.dims[3]);
  debuglog("in=%p out=%p", &in, &out);

  for (uint32_t batch = 0; batch < b; batch++) {
    for (uint32_t height = 0; height < h; height++) {
      for (uint32_t width = 0; width < w; width++) {
        for (uint32_t depth = 0; depth < d; depth++) {
          float inval = in(batch, height, width, depth);
          out(batch, height, width, depth) = fmaxf(inval, 0.0f);
        }
      }
    }
  }

  return QHPI_Success;
}

template <typename T_TtypeI, typename T_TtypeX>
static uint32_t reluMinMaxImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                               QHPI_Tensor **outputs, uint32_t num_inputs,
                               const QHPI_Tensor *const *inputs) {
  T_TtypeI out(outputs[0]);
  const T_TtypeI in(inputs[0]);
  const T_TtypeX inX(inputs[1]);
  const T_TtypeX inY(inputs[2]);

  float x = inX(0, 0, 0, 0);
  float y = inY(0, 0, 0, 0);

  if (!(y > x)) {
    return QHPI_ErrorFatal;
  }

  auto bIn = in.shape.dims[0], hIn = in.shape.dims[1], wIn = in.shape.dims[2],
       dIn = in.shape.dims[3];
  debuglog("reluMinMax execute... dims=(%zdx%zdx%zdx%zd)", in.shape.dims[0],
           in.shape.dims[1], in.shape.dims[2], in.shape.dims[3]);
  debuglog("in=%p out=%p", &in, &out);

  for (uint32_t b = 0; b < bIn; b++) {
    for (uint32_t h = 0; h < hIn; h++) {
      for (uint32_t w = 0; w < wIn; w++) {
        for (uint32_t d = 0; d < dIn; d++) {
          float inval = in(b, h, w, d);
          out(b, h, w, d) = fminf(fmaxf(inval, x), y);
        }
      }
    }
  }

  return QHPI_Success;
}

inline size_t flatToVlut(size_t index) {
  return ((index & 63) << 1) | ((index >> 6) & 1) | (index & -128);
}

static uint32_t reluTablegenImpl(QHPI_RuntimeHandle *, uint32_t num_outputs,
                                 QHPI_Tensor **outputs, uint32_t num_inputs,
                                 const QHPI_Tensor *const *inputs) {
  if (num_inputs != 4 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  Flat4<uint8_t> out(outputs[0]);
  const Flat4<float> inStepsize(inputs[0]);
  const Flat4<int32_t> inOffset(inputs[1]);
  const Flat4<float> min(inputs[2]);
  const Flat4<float> max(inputs[3]);

  float inMin = min(0, 0, 0, 0);
  float inMax = max(0, 0, 0, 0);
  const float inStepsizeVal = inStepsize(0, 0, 0, 0);
  float inOffsetVal = inOffset(0, 0, 0, 0);

  for (int i = 0; i < 256; i++) {
    /* Calculate what input equal to i would mean */
    float inVal = (i - inOffsetVal) * inStepsizeVal;
    float result = fmaxf(inMin, fminf(inVal, inMax));
    out(0, 0, 0, flatToVlut(i)) = result;
  }

  return QHPI_Success;
}

// TableLookup kernel implementation
static uint32_t tableLookupImpl(QHPI_RuntimeHandle *handle,
                                uint32_t num_outputs, QHPI_Tensor **outputs,
                                uint32_t num_inputs,
                                const QHPI_Tensor *const *inputs) {
  if (num_inputs != 2 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }
  const QHPI_Tensor *input = inputs[0];
  const QHPI_Tensor *table = inputs[1];
  QHPI_Tensor *output = outputs[0];

  // Get tensor layout information
  uint32_t input_layout = qhpi_tensor_layout(input);
  uint32_t output_layout = qhpi_tensor_layout(output);

  // Verify input and output dimensions match
  QHPI_Shape input_shape = qhpi_tensor_shape(input);
  QHPI_Shape output_shape = qhpi_tensor_shape(output);

  if (input_shape.rank != output_shape.rank) {
    return QHPI_ErrorFatal;
  }

  for (uint32_t i = 0; i < input_shape.rank; i++) {
    if (input_shape.dims[i] != output_shape.dims[i]) {
      return QHPI_ErrorFatal;
    }
  }

  const uint8_t *lookup_ptr = (const uint8_t *)qhpi_tensor_raw_data(table);
  if (!lookup_ptr) {
    return QHPI_ErrorFatal;
  }

  // Check if we have flat layout for potential HVX optimization
  if (input_layout == QHPI_Layout_Flat4 && output_layout == QHPI_Layout_Flat4) {
    // For flat layout, we can use optimized path
    const uint8_t *input_ptr = (const uint8_t *)qhpi_tensor_raw_data(input);
    uint8_t *output_ptr = (uint8_t *)qhpi_tensor_raw_data(output);
    if (!input_ptr || !output_ptr) {
      return QHPI_ErrorFatal;
    }

    // Calculate total elements
    uint32_t total_elements = 1;
    for (uint32_t i = 0; i < input_shape.rank; i++) {
      total_elements *= input_shape.dims[i];
    }

    // Use vectorized lookup if available, otherwise fall back to scalar
    // For now, implement scalar version but structured for future HVX
    // optimization
    for (uint32_t i = 0; i < total_elements; i++) {
      uint8_t input_val = input_ptr[i];
      output_ptr[i] = lookup_ptr[flatToVlut(input_val)];
    }
  } else {
    // Fall back to element-wise processing for other layouts
    const Flat4<uint8_t> in(input);
    const Flat4<uint8_t> lut(table);
    Flat4<uint8_t> out(output);

    uint32_t b = in.shape.dims[0], h = in.shape.dims[1], w = in.shape.dims[2],
             d = in.shape.dims[3];

    // Apply lookup table element by element
    for (uint32_t bi = 0; bi < b; bi++) {
      for (uint32_t hi = 0; hi < h; hi++) {
        for (uint32_t wi = 0; wi < w; wi++) {
          for (uint32_t di = 0; di < d; di++) {
            uint8_t input_val = in.get_raw(bi, hi, wi, di);
            uint8_t output_val = lut.get_raw(0, 0, 0, flatToVlut(input_val));
            out.get_raw(bi, hi, wi, di) = output_val;
          }
        }
      }
    }
  }

  return QHPI_Success;
}

// Tensor signatures for different kernels - Updated to use  structs
static QHPI_Tensor_Signature_v1 relu_quint8_in[] = {
    {.element_type = QHPI_QUInt8,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_quint8_out[] = {
    {.element_type = QHPI_QUInt8,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_float_in[] = {
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_float_out[] = {
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 relu_minmax_float_in[] = {
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_minmax_quint8_in[] = {
    {.element_type = QHPI_QUInt8,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_minmax_quint8_out[] = {
    {.element_type = QHPI_QUInt8,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_tablegen_in[] = {
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Int32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
    {.element_type = QHPI_Float32,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_tablegen_out[] = {
    {.element_type = QHPI_QUInt8,
     .layout = QHPI_Layout_Flat4,
     .storage = QHPI_Storage_Direct,
     .mem_placement = QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 table_lookup_input_sig[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only}};

static QHPI_Tensor_Signature_v1 table_lookup_output_sig[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only}};

// Kernel definitions - Updated to use QHPI_Kernel_v1
static QHPI_Kernel_v1 relu_quint8_kernel = {
    .function_name = THIS_PKG_NAME_STR "::reluImpl<Flat4<uint8_t>>",
    .function = reluImpl<Flat4<uint8_t>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = relu_quint8_in,
    .min_outputs = 1,
    .output_signature = relu_quint8_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu_float_kernel = {
    .function_name = THIS_PKG_NAME_STR "::reluImpl<Flat4<float>>",
    .function = reluImpl<Flat4<float>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = relu_float_in,
    .min_outputs = 1,
    .output_signature = relu_float_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu_minmax_float_kernel = {
    .function_name =
        THIS_PKG_NAME_STR "::reluMinMaxImpl<Flat4<float>, Flat4<float>>",
    .function = reluMinMaxImpl<Flat4<float>, Flat4<float>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 3,
    .input_signature = relu_minmax_float_in,
    .min_outputs = 1,
    .output_signature = relu_minmax_quint8_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu_minmax_quint8_kernel = {
    .function_name =
        THIS_PKG_NAME_STR "::reluMinMaxImpl<Flat4<uint8_t>, Flat4<float>>",
    .function = reluMinMaxImpl<Flat4<uint8_t>, Flat4<float>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 3,
    .input_signature = relu_minmax_quint8_in,
    .min_outputs = 1,
    .output_signature = relu_minmax_quint8_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu_tablegen_kernel = {
    .function_name = THIS_PKG_NAME_STR "::reluTablegenImpl",
    .function = reluTablegenImpl,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 4,
    .input_signature = relu_tablegen_in,
    .min_outputs = 1,
    .output_signature = relu_tablegen_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 table_lookup_kernel = {
    .function_name = THIS_PKG_NAME_STR "::tableLookupImpl",
    .function = tableLookupImpl,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = table_lookup_input_sig,
    .min_outputs = 1,
    .output_signature = table_lookup_output_sig,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

// Kernel arrays for multi-kernel ops
static QHPI_Kernel_v1 relu_minmax_kernels[] = {relu_minmax_float_kernel,
                                               relu_minmax_quint8_kernel};
static QHPI_Kernel_v1 relu_kernels[] = {relu_quint8_kernel, relu_float_kernel};

// OpInfo definitions - Updated to use QHPI_OpInfo_v1
static QHPI_OpInfo_v1 ops[] = {{.name = THIS_PKG_NAME_STR "::Relu",
                                .num_kernels = 2,
                                .kernels = relu_kernels,
                                .early_rewrite = relu_to_relu_minmax_quant,
                                .shape_required = relu_shape_required,
                                .shape_legalized = nullptr,
                                .build_tile = relu_build_tile,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::ReluMinMax",
                                .num_kernels = 2,
                                .kernels = relu_minmax_kernels,
                                .early_rewrite = relu_minmax_quant_passthrough,
                                .shape_required = relu_minmax_shape_required,
                                .shape_legalized = nullptr,
                                .build_tile = relu_minmax_build_tile,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::ReluTableGen",
                                .num_kernels = 1,
                                .kernels = &relu_tablegen_kernel,
                                .early_rewrite = nullptr,
                                .shape_required = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile = nullptr,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::Relu6",
                                .num_kernels = 0,
                                .kernels = nullptr,
                                .early_rewrite = relu6_to_relu_minmax,
                                .shape_required = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile = nullptr,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::Relu1",
                                .num_kernels = 0,
                                .kernels = nullptr,
                                .early_rewrite = relu1_to_relu_minmax,
                                .shape_required = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile = nullptr,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::ReluX",
                                .num_kernels = 0,
                                .kernels = nullptr,
                                .early_rewrite = relux_to_relu_minmax,
                                .shape_required = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile = nullptr,
                                .late_rewrite = nullptr},
                               {.name = THIS_PKG_NAME_STR "::TableLookup",
                                .num_kernels = 1,
                                .kernels = &table_lookup_kernel,
                                .early_rewrite = nullptr,
                                .shape_required = nullptr,
                                .shape_legalized = nullptr,
                                .build_tile = nullptr,
                                .late_rewrite = nullptr}};

// Registration function for regular ReLU operations - Updated to use
// qhpi_register_ops_v1
void register_relu_ops() {
  qhpi_register_ops_v1(sizeof(ops) / sizeof(ops[0]), ops, THIS_PKG_NAME_STR);
}
