// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#include "HTP/core/intrinsics.h"
#include "HTP/core/qhpi.h"
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <type_traits>

// HVX intrinsics
#ifdef __hexagon__
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#endif
#include "HTP/core/log.h"
#include "qhpi_example_utils.h"

#define RELU_FP16_TILE_HEIGHT 8

// op execute function declarations
template <typename T_Ttype>
static uint32_t reluImplFp16(QHPI_RuntimeHandle *, uint32_t num_outputs,
                             QHPI_Tensor **outputs, uint32_t num_inputs,
                             const QHPI_Tensor *const *inputs);

template <typename T_TtypeI, typename T_TtypeX>
static uint32_t reluXImplFp(QHPI_RuntimeHandle *, uint32_t num_outputs,
                            QHPI_Tensor **outputs, uint32_t num_inputs,
                            const QHPI_Tensor *const *inputs);

template <typename T_Ttype>
static uint32_t relu1ImplFp(QHPI_RuntimeHandle *, uint32_t num_outputs,
                            QHPI_Tensor **outputs, uint32_t num_inputs,
                            const QHPI_Tensor *const *inputs);

// Early rewrite functions (equivalent to DEF_PACKAGE_OPTIMIZATION)
static const QHPI_Op *relaxed_precision_relu_early_rewrite(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);
  QHPI_OutputDef op_output = qhpi_op_output(op, 0);

  // Check if input is Float32 and output is Float32
  if (input_output.type != QHPI_Float32 || op_output.type != QHPI_Float32) {
    return op;
  }

  // Check if relaxed precision flag is enabled
  int32_t relaxed_precision = 0;
  if (qhpi_option_int(op, "relaxed_precision_flag", &relaxed_precision) !=
          QHPI_Success ||
      !relaxed_precision) {
    return op;
  }

  // Create Cast to Float16
  QHPI_OutputDef cast_to_fp16_output = input_output;
  cast_to_fp16_output.type = QHPI_Float16;
  QHPI_OpRef cast_to_fp16 = {
      qhpi_op_create(op, "q::QNN_Cast", 1, &input, 1, &cast_to_fp16_output), 0};

  // Create Relu.fp16 operation
  QHPI_OutputDef relu_fp16_output = cast_to_fp16_output;
  QHPI_OpRef relu_fp16 = {qhpi_op_create(op, THIS_PKG_NAME_STR "::Relu.fp16", 1,
                                         &cast_to_fp16, 1, &relu_fp16_output),
                          0};

  // Create Cast back to Float32
  QHPI_OutputDef cast_to_fp32_output = op_output;
  return qhpi_op_create(op, "q::QNN_Cast", 1, &relu_fp16, 1,
                        &cast_to_fp32_output);
}

static const QHPI_Op *relaxed_precision_relux_early_rewrite(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OpRef input_x = qhpi_op_input(op, 1);
  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);
  QHPI_OutputDef input_x_output =
      qhpi_op_output(input_x.op, input_x.output_number);
  QHPI_OutputDef op_output = qhpi_op_output(op, 0);

  // Check if inputs are Float32 and output is Float32
  if (input_output.type != QHPI_Float32 ||
      input_x_output.type != QHPI_Float32 || op_output.type != QHPI_Float32) {
    return op;
  }

  // Check if relaxed precision flag is enabled
  int32_t relaxed_precision = 0;
  if (qhpi_option_int(op, "relaxed_precision_flag", &relaxed_precision) !=
          QHPI_Success ||
      !relaxed_precision) {
    return op;
  }

  // Create Cast inputs to Float16
  QHPI_OutputDef cast_input_output = input_output;
  cast_input_output.type = QHPI_Float16;
  QHPI_OpRef cast_input = {
      qhpi_op_create(op, "q::QNN_Cast", 1, &input, 1, &cast_input_output), 0};

  QHPI_OutputDef cast_x_output = input_x_output;
  cast_x_output.type = QHPI_Float16;
  QHPI_OpRef cast_x = {
      qhpi_op_create(op, "q::QNN_Cast", 1, &input_x, 1, &cast_x_output), 0};

  // Create ReluX operation
  QHPI_OpRef relux_inputs[] = {cast_input, cast_x};
  QHPI_OutputDef relux_output = cast_input_output;
  QHPI_OpRef relux = {qhpi_op_create(op, THIS_PKG_NAME_STR "::ReluX", 2,
                                     relux_inputs, 1, &relux_output),
                      0};

  // Create Cast back to Float32
  return qhpi_op_create(op, "q::QNN_Cast", 1, &relux, 1, &op_output);
}

static const QHPI_Op *relaxed_precision_relu1_early_rewrite(const QHPI_Op *op) {
  QHPI_OpRef input = qhpi_op_input(op, 0);
  QHPI_OutputDef input_output = qhpi_op_output(input.op, input.output_number);
  QHPI_OutputDef op_output = qhpi_op_output(op, 0);

  // Check if input is Float32 and output is Float32
  if (input_output.type != QHPI_Float32 || op_output.type != QHPI_Float32) {
    return op;
  }

  // Check if relaxed precision flag is enabled
  int32_t relaxed_precision = 0;
  if (qhpi_option_int(op, "relaxed_precision_flag", &relaxed_precision) !=
          QHPI_Success ||
      !relaxed_precision) {
    return op;
  }

  // Create Cast to Float16
  QHPI_OutputDef cast_to_fp16_output = input_output;
  cast_to_fp16_output.type = QHPI_Float16;
  QHPI_OpRef cast_to_fp16 = {
      qhpi_op_create(op, "q::QNN_Cast", 1, &input, 1, &cast_to_fp16_output), 0};

  // Create Relu1 operation
  QHPI_OutputDef relu1_output = cast_to_fp16_output;
  QHPI_OpRef relu1 = {qhpi_op_create(op, THIS_PKG_NAME_STR "::Relu1", 1,
                                     &cast_to_fp16, 1, &relu1_output),
                      0};

  // Create Cast back to Float32
  return qhpi_op_create(op, "q::QNN_Cast", 1, &relu1, 1, &op_output);
}

// Tiling support functions (equivalent to AUTOSPLIT)
static QHPI_Shape relu_fp16_shape_required(const QHPI_Op *op) {
  // Define tiling requirements - split on height dimension
  static QHPI_Shape required = {
      .rank = 4,
      .dims = {1, RELU_FP16_TILE_HEIGHT, 0,
               0} // Split on height with RELU_FP16_TILE_HEIGHT
  };
  return required;
}

static const QHPI_Op *relu_fp16_build_tile(const QHPI_Op *op,
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

// Similar tiling functions for other ReLU variants
static QHPI_Shape relu1_shape_required(const QHPI_Op *op) {
  static QHPI_Shape required = {.rank = 4,
                                .dims = {1, RELU_FP16_TILE_HEIGHT, 0, 0}};
  return required;
}

static const QHPI_Op *relu1_build_tile(const QHPI_Op *op,
                                       const QHPI_Shape *out_start,
                                       const QHPI_Shape *out_extent) {
  QHPI_OpRef input_ref = qhpi_op_input(op, 0);
  QHPI_Shape in_start = *out_start;
  QHPI_Shape in_extent = *out_extent;
  QHPI_OpRef input_slice = qhpi_op_slice(input_ref, &in_start, &in_extent);

  QHPI_OpRef inputs[] = {input_slice};
  QHPI_OutputDef outputs[] = {
      {.type = qhpi_op_output(op, 0).type,
       .quant_parameters = qhpi_op_output(op, 0).quant_parameters,
       .shape = *out_extent}};

  return qhpi_op_create(op, qhpi_op_name(op), 1, inputs, 1, outputs);
}

static QHPI_Shape relux_shape_required(const QHPI_Op *op) {
  static QHPI_Shape required = {.rank = 4,
                                .dims = {1, RELU_FP16_TILE_HEIGHT, 0, 0}};
  return required;
}

static const QHPI_Op *relux_build_tile(const QHPI_Op *op,
                                       const QHPI_Shape *out_start,
                                       const QHPI_Shape *out_extent) {
  QHPI_OpRef input_ref = qhpi_op_input(op, 0);
  QHPI_OpRef input_x_ref = qhpi_op_input(op, 1);

  QHPI_Shape in_start = *out_start;
  QHPI_Shape in_extent = *out_extent;
  QHPI_OpRef input_slice = qhpi_op_slice(input_ref, &in_start, &in_extent);

  // X input is typically a scalar, so no slicing needed
  QHPI_OpRef inputs[] = {input_slice, input_x_ref};
  QHPI_OutputDef outputs[] = {
      {.type = qhpi_op_output(op, 0).type,
       .quant_parameters = qhpi_op_output(op, 0).quant_parameters,
       .shape = *out_extent}};

  return qhpi_op_create(op, qhpi_op_name(op), 2, inputs, 1, outputs);
}

/* execute functions for ops */
// op 1 Relu fp
template <typename T_Ttype>
static uint32_t reluImplFp16(QHPI_RuntimeHandle *, uint32_t num_outputs,
                             QHPI_Tensor **outputs, uint32_t num_inputs,
                             const QHPI_Tensor *const *inputs) {
  T_Ttype out(outputs[0]);
  const T_Ttype in(inputs[0]);

  debuglog("relu execute... dims=(%zdx%zdx%zdx%zd)", in.shape.dims[0],
           in.shape.dims[1], in.shape.dims[2], in.shape.dims[3]);
  debuglog("in=%p out=%p", &in, &out);

  if constexpr (std::is_same_v<T_Ttype, Crouton4<Float16>>) {
    // Crouton4 tensor processing
    size_t inBlocks = in.block_table_length;
    auto inBlocktab = in.block_table;
    auto outBlocktab = out.block_table;

    // HVX Vector operations for FP16
    // vminval = 0x0 for FP16
    for (uint32_t i = 0; i < inBlocks; ++i) {
      auto inVptr = (const HVX_Vector *)(inBlocktab[i]);
      auto outVptr = (HVX_Vector *)(outBlocktab[i]);
      for (uint32_t j = 0; j < 16; ++j) {
        HVX_Vector vin = inVptr[j];
        // Q6_Vhf_vmax_VhfVhf performs max operation on FP16 vectors
        vin = Q6_Vhf_vmax_VhfVhf(vin, Q6_Vh_vsplat_R(0x0));
        outVptr[j] = vin;
      }
    }
  } else {
    // Flat4 tensor processing - element-wise operation
    const size_t bOut = out.shape.dims[0];
    const size_t hOut = out.shape.dims[1];
    const size_t wOut = out.shape.dims[2];
    const size_t dOut = out.shape.dims[3];

    for (size_t b = 0; b < bOut; b++) {
      for (size_t h = 0; h < hOut; h++) {
        for (size_t w = 0; w < wOut; w++) {
          for (size_t d = 0; d < dOut; d++) {
            Float16 inval = in.get_raw(b, h, w, d);
            Float16 outval = Float16(std::max(0.0f, float(inval)));
            out.get_raw(b, h, w, d) = outval;
          }
        }
      }
    }
  }
  return QHPI_Success;
}

// op 2 ReluX fp
template <typename T_TtypeI, typename T_TtypeX>
static uint32_t reluXImplFp(QHPI_RuntimeHandle *, uint32_t num_outputs,
                            QHPI_Tensor **outputs, uint32_t num_inputs,
                            const QHPI_Tensor *const *inputs) {
  T_TtypeI out(outputs[0]);
  const T_TtypeI in(inputs[0]);
  const T_TtypeX inX(inputs[1]);

  debuglog("relu execute... dims=(%zdx%zdx%zdx%zd)", in.shape.dims[0],
           in.shape.dims[1], in.shape.dims[2], in.shape.dims[3]);
  debuglog("in=%p out=%p", &in, &out);

  // Get the X value - assuming it's a scalar in a flat tensor
  float x = 0.0f;
  if constexpr (std::is_same_v<T_TtypeX, Flat4<float>>) {
    x = inX.get_raw(0, 0, 0, 0);
  } else {
    // For crouton tensor, extract first element
    if (inX.block_table_length > 0) {
      auto xVptr = (const HVX_Vector *)(inX.block_table[0]);
      // Extract first FP16 value and convert to float
      uint16_t fp16_val = *((uint16_t *)xVptr);
      // Simple FP16 to float conversion
      union {
        float f;
        uint32_t i;
      } u;
      uint32_t sign = (fp16_val & 0x8000) << 16;
      uint32_t exp = ((fp16_val & 0x7c00) >> 10);
      uint32_t mant = (fp16_val & 0x3ff) << 13;

      if (exp == 0) {
        if (mant == 0) {
          u.i = sign;  // True zero
        } else {
          // Denormalized number: (-1)^sign * 2^-14 * (mant/1024)
          u.i = sign | ((127 - 15 - 1) << 23) | (mant << 13);
        }
      } else if (exp == 31) {
        u.i = sign | 0x7f800000 | mant;  // Inf or NaN
      } else {
        u.i = sign | ((exp - 15 + 127) << 23) | mant;  // Normal number
      }
      x = u.f;
    }
  }

  if (!(x > 0.0f)) {
    return QHPI_ErrorFatal;
  }

  if constexpr (std::is_same_v<T_TtypeI, Crouton4<float>>) {
    // Crouton4 tensor processing
    size_t inBlocks = in.block_table_length;
    auto inBlocktab = in.block_table;
    auto outBlocktab = out.block_table;

    Float16 minval(0.0f);
    Float16 maxval(x);
    HVX_Vector vminval = Q6_Vh_vsplat_R(minval.raw());
    HVX_Vector vmaxval = Q6_Vh_vsplat_R(maxval.raw());

    for (uint32_t i = 0; i < inBlocks; ++i) {
      auto inVptr = (const HVX_Vector *)(inBlocktab[i]);
      auto outVptr = (HVX_Vector *)(outBlocktab[i]);
      for (uint32_t j = 0; j < 16; ++j) {
        HVX_Vector vin = inVptr[j];
        vin = Q6_Vhf_vmax_VhfVhf(vin, vminval);
        vin = Q6_Vhf_vmin_VhfVhf(vin, vmaxval);
        outVptr[j] = vin;
      }
    }
  } else {
    // Flat4 tensor processing - element-wise operation
    const size_t bOut = out.shape.dims[0];
    const size_t hOut = out.shape.dims[1];
    const size_t wOut = out.shape.dims[2];
    const size_t dOut = out.shape.dims[3];

    for (size_t b = 0; b < bOut; b++) {
      for (size_t h = 0; h < hOut; h++) {
        for (size_t w = 0; w < wOut; w++) {
          for (size_t d = 0; d < dOut; d++) {
            float inval = in.get_raw(b, h, w, d);
            float outval = std::max(0.0f, std::min(inval, x));
            out.get_raw(b, h, w, d) = outval;
          }
        }
      }
    }
  }
  return QHPI_Success;
}

// op 3 Relu1 fp
template <typename T_Ttype>
static uint32_t relu1ImplFp(QHPI_RuntimeHandle *, uint32_t num_outputs,
                            QHPI_Tensor **outputs, uint32_t num_inputs,
                            const QHPI_Tensor *const *inputs) {
  T_Ttype out(outputs[0]);
  const T_Ttype in(inputs[0]);

  if constexpr (std::is_same_v<T_Ttype, Crouton4<float>>) {
    // Crouton4 tensor processing
    size_t inBlocks = in.block_table_length;
    auto inBlocktab = in.block_table;
    auto outBlocktab = out.block_table;

    HVX_Vector vminval = Q6_Vh_vsplat_R(0xbc00); //-1.0 in FP16
    HVX_Vector vmaxval = Q6_Vh_vsplat_R(0x3c00); // 1.0 in FP16

    for (uint32_t i = 0; i < inBlocks; ++i) {
      auto inVptr = (const HVX_Vector *)(inBlocktab[i]);
      auto outVptr = (HVX_Vector *)(outBlocktab[i]);
      for (uint32_t j = 0; j < 16; ++j) {
        HVX_Vector vin = inVptr[j];
        vin = Q6_Vhf_vmax_VhfVhf(vin, vminval);
        vin = Q6_Vhf_vmin_VhfVhf(vin, vmaxval);
        outVptr[j] = vin;
      }
    }
  } else {
    // Flat4 tensor processing - element-wise operation
    const size_t bOut = out.shape.dims[0];
    const size_t hOut = out.shape.dims[1];
    const size_t wOut = out.shape.dims[2];
    const size_t dOut = out.shape.dims[3];

    for (size_t b = 0; b < bOut; b++) {
      for (size_t h = 0; h < hOut; h++) {
        for (size_t w = 0; w < wOut; w++) {
          for (size_t d = 0; d < dOut; d++) {
            float inval = in.get_raw(b, h, w, d);
            float outval = std::max(-1.0f, std::min(inval, 1.0f));
            out.get_raw(b, h, w, d) = outval;
          }
        }
      }
    }
  }
  return QHPI_Success;
}

// Tensor signatures for different kernels
static QHPI_Tensor_Signature_v1 relu_fp16_crouton_in[] = {
    {QHPI_Float16, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect,
     QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_fp16_crouton_out[] = {
    {QHPI_Float16, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect,
     QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_fp16_flat_in[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu_fp16_flat_out[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relux_crouton_in[] = {
    {QHPI_Float32, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect,
     QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect,
     QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 relu1_crouton_in[] = {
    {QHPI_Float32, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect,
     QHPI_MemLoc_DDR_Only},
};

// Kernel definitions
static QHPI_Kernel_v1 relu_fp16_crouton_kernel = {
    .function_name = THIS_PKG_NAME_STR "::reluImplFp16<Crouton4<Float16>>",
    .function = reluImplFp16<Crouton4<Float16>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = relu_fp16_crouton_in,
    .min_outputs = 1,
    .output_signature = relu_fp16_crouton_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu_fp16_flat_kernel = {
    .function_name = THIS_PKG_NAME_STR "::reluImplFp16<Flat4<Float16>>",
    .function = reluImplFp16<Flat4<Float16>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = relu_fp16_flat_in,
    .min_outputs = 1,
    .output_signature = relu_fp16_flat_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relux_crouton_kernel = {
    .function_name =
        THIS_PKG_NAME_STR "::reluXImplFp<Crouton4<float>, Crouton4<float>>",
    .function = reluXImplFp<Crouton4<float>, Crouton4<float>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = relux_crouton_in,
    .min_outputs = 1,
    .output_signature = relu_fp16_crouton_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

static QHPI_Kernel_v1 relu1_crouton_kernel = {
    .function_name = THIS_PKG_NAME_STR "::relu1ImplFp<Crouton4<float>>",
    .function = relu1ImplFp<Crouton4<float>>,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = relu1_crouton_in,
    .min_outputs = 1,
    .output_signature = relu_fp16_crouton_out,
    .cost_function = nullptr,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr};

// Kernel arrays for multi-kernel ops
static QHPI_Kernel_v1 relu_fp16_kernels[] = {relu_fp16_crouton_kernel,
                                             relu_fp16_flat_kernel};

// OpInfo definitions
static QHPI_OpInfo_v1 ops[] = {
    {.name = THIS_PKG_NAME_STR "::Relu",
     .num_kernels = 0,
     .kernels = nullptr,
     .early_rewrite = relaxed_precision_relu_early_rewrite,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::Relu.fp16",
     .num_kernels = 2,
     .kernels = relu_fp16_kernels,
     .early_rewrite = nullptr,
     .shape_required = relu_fp16_shape_required,
     .shape_legalized = nullptr,
     .build_tile = relu_fp16_build_tile,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::ReluX",
     .num_kernels = 1,
     .kernels = &relux_crouton_kernel,
     .early_rewrite = nullptr,
     .shape_required = relux_shape_required,
     .shape_legalized = nullptr,
     .build_tile = relux_build_tile,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::Relu1",
     .num_kernels = 1,
     .kernels = &relu1_crouton_kernel,
     .early_rewrite = nullptr,
     .shape_required = relu1_shape_required,
     .shape_legalized = nullptr,
     .build_tile = relu1_build_tile,
     .late_rewrite = nullptr}};

// Registration function for FP16 ReLU operations
void register_relu_fp16_ops() {
  qhpi_register_ops_v1(sizeof(ops) / sizeof(ops[0]), ops, THIS_PKG_NAME_STR);
}
