//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <span>

#include "qualla/dialog.hpp"

namespace qualla {

class SpecDecDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "spec-dec";

  SpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& /*tokens*/, DialogCallback /*callback*/) override {
    return false;
  }

  virtual void reset() override;

  virtual const char* getTraceNamespace() const override { return "Dialog::SPD"; };

 private:
  size_t _draft_len;  // Number of draft tokens
  bool _parallel;     // Enable parallel processing (where possible)

  // For keeping track of the number of tokens that were accepted in each iteration.
  std::vector<int32_t> _accepted_counts;

  Sampler& _d_sampler;  // Draft sampler
  Sampler& _t_sampler;  // Target sampler

  // Token acceptor, called for each accepted token.
  // Returns true to continue, false to stop
  using Acceptor = std::function<bool(int32_t token)>;

  // Follow on processing of the tokens
  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& t_logits,
                                 Tensor& d_logits,
                                 Dialog::Callback callback);

  // Rejection sampling.
  // Returns number of accepted tokens
  size_t rejectionSampling(std::span<int32_t> tokens,
                           Tensor& target_logits,
                           std::span<float> draft_probs,
                           Acceptor accept);

  int32_t sampleFromModifiedDist(std::span<float> src0_dst, std::span<float> src1);
};

}  // namespace qualla
