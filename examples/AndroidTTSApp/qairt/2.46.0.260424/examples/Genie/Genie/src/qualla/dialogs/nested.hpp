//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <vector>

#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/dialog.hpp"

namespace qualla {
/**
    @brief Used by NestARModelAdaptor to run the nested dialog.
 */
class NestedDialogExecutionPolicy {
 public:
  virtual ~NestedDialogExecutionPolicy() = default;
  /**
    @brief  The primary and nested engine are passed in to the policy so it has chance to
    register ModelIOAdaptor to them (if needed).
    @return true if success

   */
  virtual bool setup([[maybe_unused]] Engine& primary, [[maybe_unused]] Engine& nested) {
    return true;
  }
  /**
    @brief Called by NestARDialog, informing that a new query is about to start on the
     primary dialog.
   */
  virtual void onNewQuery([[maybe_unused]] const std::vector<int32_t>& input_tokens) {}
  /**
   @brief Called by NestARDialog, informing that a new query is about to start on the
     primary dialog.
   */
  virtual void onNewQuery([[maybe_unused]] const std::vector<uint8_t>& input_tokens) {}
  /**
   @brief Queried by NestARModelAdaptor about when to run the nested dialog
   @return true, nested dialog will run in its prepareInputs() method
   @return false, nested dialog will be run in its handleOutputs() method
   */
  virtual bool isForInputPrepration() const { return false; }
  /**
    @brief Called by NestARModelAdaptor to run the nested dialog.
    @param dialog The dialog to run
    @param info The contextual information about the current inference.
   */
  virtual bool execute([[maybe_unused]] Dialog& dialog,
                       [[maybe_unused]] const InferenceInfo& info) {
    return true;
  }
  /**
    @brief Factory method to create a concrete policy by name.
   */
  static std::shared_ptr<NestedDialogExecutionPolicy> createByName(const std::string& name);
};

class NestedARModelAdaptor : public ModelIOAdaptor {
 public:
  NestedARModelAdaptor(std::shared_ptr<NestedDialogExecutionPolicy> policy,
                       std::shared_ptr<Dialog> dialog);

  bool prepareInputs(const InferenceInfo& info, const TensorMap& inputs) override;

  void handleOutputs(const InferenceInfo& info, const TensorMap& outputs) override;

  const std::string& name() override;

 private:
  std::shared_ptr<NestedDialogExecutionPolicy> m_policy;
  std::shared_ptr<Dialog> m_dialog;
};

template <typename PrimaryDialog>
class NestedDialog : public PrimaryDialog {
 public:
  static constexpr const char* TYPE = "nested";

  NestedDialog(std::shared_ptr<Env> env,
               const std::string& name,
               const nlohmann::json& conf,
               std::unique_ptr<Dialog> nestedDialog);

  void completeInit() override;

  bool process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) override;

  bool process(std::vector<uint8_t>& embedding_vectors,
               Dialog::T2ECallback t2eCallback,
               qualla::DialogCallback callback) override;

  std::shared_ptr<Dialog> m_nestedDialog;
  std::shared_ptr<NestedDialogExecutionPolicy> m_executionPolicy;
  std::shared_ptr<NestedARModelAdaptor> m_adaptor;
};

}  // namespace qualla
