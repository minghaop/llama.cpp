//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DDPM_SCHEDULER_HPP
#define QUALLA_DDPM_SCHEDULER_HPP

#include <string>
#include <vector>

#include "qualla/diffusion-scheduler.hpp"

namespace qualla {

/**
 * @brief DDPM (Denoising Diffusion Probabilistic Models) Scheduler
 *
 * This class implements the DDPM scheduler for diffusion models without
 * using PyTorch dependencies. It uses standard C++ containers and algorithms.
 * Inherits from DiffusionScheduler base class.
 */
class DDPMScheduler : public DiffusionScheduler {
 public:
  /**
   * @brief Constructor that initializes the scheduler from a configuration file
   *
   * @param env Shared pointer to the environment for logging
   * @param config_path Path to the JSON configuration file containing scheduler parameters
   *                    Expected keys: num_train_timesteps, beta_start, beta_end, beta_schedule
   */
  DDPMScheduler(std::shared_ptr<Env> env, const std::string& config_path);

  /**
   * @brief Add noise to samples using the diffusion process (Tensor-based override)
   *
   * This method implements the forward diffusion process by adding noise to the
   * original samples according to the DDPM noise schedule. Handles both float32
   * and float16 tensors by converting to float32 for computation and back to
   * the original dtype.
   *
   * @param original_samples Tensor containing original sample values (float32 or float16)
   * @param noise Tensor containing noise values (float32 or float16)
   * @param timesteps Vector of timestep indices for each batch item
   * @return Tensor containing noisy samples (same dtype as input)
   *
   * @note The formula used is: sqrt(alpha_prod) * original_samples + sqrt(1 - alpha_prod) * noise
   * @note Internally converts float16 to float32 for computation, then converts back
   */
  Tensor add_noise(const Tensor& original_samples,
                   const Tensor& noise,
                   const std::vector<int>& timesteps) override;

  /**
   * @brief Get the cumulative product of alphas
   * @return Vector containing the cumulative product of alphas for each timestep
   */
  const std::vector<float>& get_alphas_cumprod() const { return alphas_cumprod_; }

 private:
  /// Cumulative product of alphas (1 - beta) for each timestep
  std::vector<float> alphas_cumprod_;

  /**
   * @brief Private helper method for vector-based noise addition
   *
   * @param original_samples Flattened vector of original sample values
   * @param noise Flattened vector of noise values
   * @param timesteps Vector of timestep indices
   * @param shape Shape of the tensor
   * @return Flattened vector of noisy samples
   */
  std::vector<float> add_noise(const std::vector<float>& original_samples,
                               const std::vector<float>& noise,
                               const std::vector<int>& timesteps,
                               const std::vector<size_t>& shape);
};

}  // namespace qualla

#endif  // QUALLA_DDPM_SCHEDULER_HPP
