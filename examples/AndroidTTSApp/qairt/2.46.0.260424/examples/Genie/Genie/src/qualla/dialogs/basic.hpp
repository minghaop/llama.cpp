//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "qualla/dialog.hpp"

namespace qualla {

class BasicDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "basic";

  BasicDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);

  virtual bool process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       qualla::DialogCallback callback) override;

  virtual bool process(std::vector<int32_t>& tokens,
                       std::vector<size_t>& tokenNumPerBatch,
                       qualla::DialogCallback callback) override;

  virtual bool process(std::vector<int32_t>& tokens,
                       std::vector<size_t>& tokenNumPerBatch,
                       Dialog::BatchCallback callback) override;

  virtual bool supportsPauseResume() override { return true; };

  void completeInit() override;

  virtual const char* getTraceNamespace() const override { return "Dialog::Basic"; };

 protected:
  virtual bool supportsLongContext() const override { return true; };

 private:
  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 Dialog::Callback callback);

  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 qualla::DialogCallback callback);

  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 std::vector<Tensor>& logits,
                                 qualla::DialogCallback callback);
};

}  // namespace qualla
