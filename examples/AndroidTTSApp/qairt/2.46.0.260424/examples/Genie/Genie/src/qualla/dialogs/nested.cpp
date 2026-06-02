//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>

#include "basic.hpp"
#include "nested.hpp"


namespace qualla {

std::shared_ptr<NestedDialogExecutionPolicy> NestedDialogExecutionPolicy::createByName(
    [[maybe_unused]] const std::string& name) {
  // TODO return the Qwen3ExecutionPolicy once it is ready.
  return std::make_shared<NestedDialogExecutionPolicy>();
}

NestedARModelAdaptor::NestedARModelAdaptor(std::shared_ptr<NestedDialogExecutionPolicy> policy,
                                           std::shared_ptr<Dialog> dialog)
    : m_policy(std::move(policy)), m_dialog(std::move(dialog)) {}

bool NestedARModelAdaptor::prepareInputs(const InferenceInfo& info,
                                         [[maybe_unused]] const TensorMap& inputs) {
  if (m_policy && m_policy->isForInputPrepration()) {
    m_policy->execute(*m_dialog, info);
  }
  return true;
}

void NestedARModelAdaptor::handleOutputs(const InferenceInfo& info,
                                         [[maybe_unused]] const TensorMap& outputs) {
  if (m_policy && !m_policy->isForInputPrepration()) {
    m_policy->execute(*m_dialog, info);
  }
}

const std::string& NestedARModelAdaptor::name() {
  static const std::string ADAPTOR_NAME = "NestedARModelAdaptor";
  return ADAPTOR_NAME;
}

template <typename PrimaryDialog>
NestedDialog<PrimaryDialog>::NestedDialog(std::shared_ptr<Env> env,
                                          const std::string& name,
                                          const nlohmann::json& conf,
                                          std::unique_ptr<Dialog> nestedDialog)
    : PrimaryDialog(env, name, conf),
      m_nestedDialog(std::move(nestedDialog)),
      m_executionPolicy(NestedDialogExecutionPolicy::createByName("Qwen3Omni")),
      m_adaptor(std::make_shared<NestedARModelAdaptor>(m_executionPolicy, m_nestedDialog)) {}

template <typename PrimaryDialog>
void NestedDialog<PrimaryDialog>::completeInit() {
  if (PrimaryDialog::m_initFinished) return;
  PrimaryDialog::completeInit();
  if (!m_executionPolicy->setup(*PrimaryDialog::_engine["primary"],
                                m_nestedDialog->engine("primary"))) {
    State::fatal("execution policy failed to register primary or secondary engine");
    return;
  }
  if (!(PrimaryDialog::_engine["primary"])->registerModelAdaptor(m_adaptor)) {
    State::fatal("primary engine failed to register adaptor");
    return;
  }
}

template <typename PrimaryDialog>
bool NestedDialog<PrimaryDialog>::process(std::vector<int32_t>& tokens,
                                          qualla::DialogCallback callback) {
  m_executionPolicy->onNewQuery(tokens);
  return PrimaryDialog::process(tokens, callback);
}

template <typename PrimaryDialog>
bool NestedDialog<PrimaryDialog>::process(std::vector<uint8_t>& embedding_vectors,
                                          Dialog::T2ECallback t2eCallback,
                                          qualla::DialogCallback callback) {
  m_executionPolicy->onNewQuery(embedding_vectors);
  return PrimaryDialog::process(embedding_vectors, t2eCallback, callback);
}

template class NestedDialog<BasicDialog>;

}  // namespace qualla
