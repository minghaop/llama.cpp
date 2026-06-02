//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "DDPMScheduler.hpp"

// Disable warnings for fp16 library
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "fp16/fp16.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace qualla {

// Helper function to create linearly spaced values
static std::vector<float> linspace(float start, float end, int num) {
  std::vector<float> result(static_cast<size_t>(num));
  if (num == 1) {
    result[0] = start;
    return result;
  }

  float step = (end - start) / (num - 1);
  for (size_t i = 0; i < static_cast<size_t>(num); ++i) {
    result[i] = start + static_cast<float>(i) * step;
  }
  return result;
}

// Helper function to compute cumulative product
static std::vector<float> cumprod(const std::vector<float>& input) {
  std::vector<float> result(input.size());
  if (input.empty()) {
    return result;
  }

  result[0] = input[0];
  for (size_t i = 1; i < input.size(); ++i) {
    result[i] = result[i - 1] * input[i];
  }
  return result;
}

static nlohmann::json load_config(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Failed to open config file: {}", path));
  }
  nlohmann::json config;
  file >> config;
  return config;
}

DDPMScheduler::DDPMScheduler(std::shared_ptr<Env> env, const std::string& config_path)
    : DiffusionScheduler(env, config_path) {
  nlohmann::json config = load_config(config_path);

  if (config["_class_name"] != "DDPMScheduler") {
    __WARN("DDPMScheduler not found in model; applying default DDPMScheduler.");
  }

  num_train_timesteps       = config["num_train_timesteps"];
  float beta_start          = config.value("beta_start", 0.0001f);
  float beta_end            = config.value("beta_end", 0.02f);
  std::string beta_schedule = config.value("beta_schedule", "linear");

  std::vector<float> betas;

  if (beta_schedule == "linear") {
    betas = linspace(beta_start, beta_end, num_train_timesteps);
  } else if (beta_schedule == "scaled_linear") {
    std::vector<float> sqrt_betas =
        linspace(std::sqrt(beta_start), std::sqrt(beta_end), num_train_timesteps);
    betas.resize(sqrt_betas.size());
    for (size_t i = 0; i < sqrt_betas.size(); ++i) {
      betas[i] = sqrt_betas[i] * sqrt_betas[i];
    }
  } else {
    __ERROR("Unsupported beta_schedule: {}", beta_schedule);
    return;
  }

  std::vector<float> alphas(betas.size());
  for (size_t i = 0; i < betas.size(); ++i) {
    alphas[i] = 1.0f - betas[i];
  }
  alphas_cumprod_ = cumprod(alphas);
}

std::vector<float> DDPMScheduler::add_noise(const std::vector<float>& original_samples,
                                            const std::vector<float>& noise,
                                            const std::vector<int>& timesteps,
                                            const std::vector<size_t>& shape) {
  // Validate inputs
  if (original_samples.size() != noise.size()) {
    __ERROR("original_samples and noise must have the same size");
    return std::vector<float>();
  }

  // Calculate total size from shape
  size_t total_size = 1;
  for (size_t dim : shape) {
    total_size *= dim;
  }

  if (original_samples.size() != total_size) {
    __ERROR("original_samples size does not match shape");
    return std::vector<float>();
  }

  // Get batch size (first dimension of shape)
  size_t batch_size = shape.empty() ? 1 : shape[0];

  if (timesteps.size() != batch_size) {
    __ERROR("timesteps size must match batch size");
    return std::vector<float>();
  }

  // Calculate elements per batch item
  size_t elements_per_batch = total_size / batch_size;

  // Extract alpha_prod values for each timestep
  std::vector<float> alpha_prod(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    int t = timesteps[i];
    if (t < 0 || t >= static_cast<int>(alphas_cumprod_.size())) {
      __ERROR("timestep {} out of range [0, {})", t, alphas_cumprod_.size());
      return std::vector<float>();
    }
    alpha_prod[i] = alphas_cumprod_[static_cast<size_t>(t)];
  }

  // Compute sqrt values
  std::vector<float> sqrt_alpha_prod(batch_size);
  std::vector<float> sqrt_one_minus_alpha_prod(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    sqrt_alpha_prod[i]           = std::sqrt(alpha_prod[i]);
    sqrt_one_minus_alpha_prod[i] = std::sqrt(1.0f - alpha_prod[i]);
  }

  // Apply noise formula: sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
  std::vector<float> noisy_samples(total_size);
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    size_t offset              = batch_idx * elements_per_batch;
    float sqrt_alpha           = sqrt_alpha_prod[batch_idx];
    float sqrt_one_minus_alpha = sqrt_one_minus_alpha_prod[batch_idx];

    for (size_t i = 0; i < elements_per_batch; ++i) {
      size_t idx         = offset + i;
      noisy_samples[idx] = sqrt_alpha * original_samples[idx] + sqrt_one_minus_alpha * noise[idx];
    }
  }

  return noisy_samples;
}

// Helper function to convert float16 to float32 using fp16 library
static std::vector<float> float16_to_float32(const void* data, size_t num_elements) {
  std::vector<float> result(num_elements);
  const uint16_t* fp16_data = static_cast<const uint16_t*>(data);

  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = fp16_ieee_to_fp32_value(fp16_data[i]);
  }

  return result;
}

// Helper function to convert float32 to float16 using fp16 library
static void float32_to_float16(const std::vector<float>& data, void* output) {
  uint16_t* fp16_output = static_cast<uint16_t*>(output);

  for (size_t i = 0; i < data.size(); ++i) {
    fp16_output[i] = fp16_ieee_from_fp32_value(data[i]);
  }
}

Tensor DDPMScheduler::add_noise(const Tensor& original_samples,
                                const Tensor& noise,
                                const std::vector<int>& timesteps) {
  // Validate inputs
  if (original_samples.getSize() != noise.getSize()) {
    __ERROR("original_samples and noise must have the same size");
    return Tensor();
  }

  if (original_samples.getDataType() != noise.getDataType()) {
    __ERROR("original_samples and noise must have the same data type");
    return Tensor();
  }

  size_t num_elements  = original_samples.getSize();
  TensorDataType dtype = original_samples.getDataType();

  // Convert to float32 if needed
  std::vector<float> original_float32;
  std::vector<float> noise_float32;

  if (dtype == TENSOR_DATATYPE_FLOAT_POINT_16) {
    original_float32 = float16_to_float32(original_samples.getData(), num_elements);
    noise_float32    = float16_to_float32(noise.getData(), num_elements);
  } else if (dtype == TENSOR_DATATYPE_FLOAT_32) {
    const float* orig_data  = static_cast<const float*>(original_samples.getData());
    const float* noise_data = static_cast<const float*>(noise.getData());
    original_float32.assign(orig_data, orig_data + num_elements);
    noise_float32.assign(noise_data, noise_data + num_elements);
  } else {
    __ERROR("Unsupported data type for add_noise. Only float32 and float16 are supported.");
    return Tensor();
  }

  // Use dimensions to infer shape
  std::vector<uint32_t> dims = original_samples.getDimensions();
  std::vector<size_t> shape(dims.begin(), dims.end());

  // Call the vector-based add_noise
  std::vector<float> noisy_samples = add_noise(original_float32, noise_float32, timesteps, shape);

  if (noisy_samples.empty()) {
    __ERROR("Failed to add noise");
    return Tensor();
  }

  // Create output tensor
  Tensor result;
  result.setDataType(dtype);
  result.setSize(num_elements);
  result.setDimensions(dims);
  result.setQuantizationParams(original_samples.getQuantizationParams().scale,
                               original_samples.getQuantizationParams().offset);

  // Create quantization parameters
  TensorQuantizationParams qparams = original_samples.getQuantizationParams();
  if (dtype == TENSOR_DATATYPE_FLOAT_POINT_16) {
    // Convert float32 to float16 in a temporary buffer
    std::vector<uint16_t> fp16_data(num_elements);
    float32_to_float16(noisy_samples, fp16_data.data());
    result = Tensor(fp16_data.data(), dtype, qparams, num_elements, dims, false);
  } else {  // TENSOR_DATATYPE_FLOAT_32
    result = Tensor(noisy_samples.data(), dtype, qparams, num_elements, dims, false);
  }
  return result;
}

}  // namespace qualla
