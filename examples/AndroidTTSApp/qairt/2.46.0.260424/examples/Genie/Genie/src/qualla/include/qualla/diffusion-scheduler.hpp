//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DIFFUSION_SCHEDULER_HPP
#define QUALLA_DIFFUSION_SCHEDULER_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {

class DiffusionScheduler {
 public:
  DiffusionScheduler(std::shared_ptr<Env> env, const std::string& config_path);
  virtual ~DiffusionScheduler() = default;

  // Add noise to samples using the diffusion process (Tensor-based)
  // Parameters:
  //   original_samples: Tensor containing original sample values (float32 or float16)
  //   noise: Tensor containing noise values (float32 or float16)
  //   timesteps: vector of timestep indices for each batch item
  // Returns: Tensor containing noisy samples (same dtype as input)
  // Note: Internally converts float16 to float32 for computation, then converts back
  virtual Tensor add_noise(const Tensor& original_samples,
                           const Tensor& noise,
                           const std::vector<int>& timesteps);

  // Create DiffusionScheduler instance
  QUALLA_API static std::shared_ptr<DiffusionScheduler> create(
      std::shared_ptr<Env> env, const std::filesystem::path& json_path);

  int num_train_timesteps{1000};

 protected:
  std::string _type;  // scheduler type
  std::vector<float> alphas_cumprod_;
  std::shared_ptr<Env> _env;
};

}  // namespace qualla

#endif  // QUALLA_DIFFUSION_SCHEDULER_HPP
