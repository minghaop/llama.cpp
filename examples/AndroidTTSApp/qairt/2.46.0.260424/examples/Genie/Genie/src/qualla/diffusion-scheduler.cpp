//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <fstream>

#include "diffusion-scheduler/DDPMScheduler.hpp"
#include "qualla/diffusion-scheduler.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

DiffusionScheduler::DiffusionScheduler(std::shared_ptr<Env> env, const std::string& /*config_path*/)
    : _env(env) {}

qualla::Tensor DiffusionScheduler::add_noise(const Tensor& /*original_samples*/,
                                             const Tensor& /*noise*/,
                                             const std::vector<int>& /*timesteps*/) {
  __ERROR("{}-DiffusionScheduler does not support add_noise", _type);
  return Tensor();
}

std::shared_ptr<DiffusionScheduler> DiffusionScheduler::create(
    std::shared_ptr<Env> env, const std::filesystem::path& json_path) {
  if (!std::filesystem::exists(json_path)) {
    throw std::runtime_error(fmt::format("{}: file does not exist", json_path.string()));
  }

  const std::string absolutePath = std::filesystem::absolute(json_path).string();
  static std::unordered_map<std::string, std::shared_ptr<DiffusionScheduler>> s_schedulers;

  if (!s_schedulers.contains(absolutePath) || !s_schedulers[absolutePath]) {
    s_schedulers[absolutePath] = std::make_shared<DDPMScheduler>(env, absolutePath);
  }

  return s_schedulers[absolutePath];
}

}  // namespace qualla
