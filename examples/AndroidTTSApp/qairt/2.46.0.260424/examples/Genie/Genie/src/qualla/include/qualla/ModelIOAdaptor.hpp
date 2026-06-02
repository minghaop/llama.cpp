//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <map>
#include <memory>
#include <set>

#include "detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {
// a Model variant is defined by the arN and clN.
using ModelVariant = std::pair<int32_t, int32_t>;
using TensorMap    = std::map<std::string, Tensor*>;

/**
  @brief Exposes all the variant and their input/output spec.
*/
class ModelIOSpec {
 public:
  virtual ~ModelIOSpec()                                                    = default;
  virtual const std::set<ModelVariant>& getVariants() const                 = 0;
  virtual const TensorMap* getInputSpec(const ModelVariant& variant) const  = 0;
  virtual const TensorMap* getOutputSpec(const ModelVariant& variant) const = 0;
};

/**
 @brief Containing information describing the current inference
 */
struct InferenceInfo {
  /**
    @brief [arN,clN] for decoder, [0,-1] otherwise.
  */
  ModelVariant variant;
  /**
    @brief how many input tokens (or embeddings) have been previously processed.
  */
  uint32_t n_past;
  /**
    @brief how many input tokens (or embeddings) have been processed by the current inference.
  */
  uint32_t n_processed;
  /**
    @brief total tokens (or embeddings) need to be processed by the current inference.
  */
  uint32_t n_inputs;
  /**
    @brief true if the current inference is the last step in the inference loop.
  */
  bool isLastInference;
};

/**
    An adaptor, once registered to an Engine, can provide and handle extra inputs and outputs of
    the model.
 */
class ModelIOAdaptor {
 public:
  virtual ~ModelIOAdaptor() = default;
  /**
      @param Env  for access logging and current path
      @param ModelIOSpec the model io spec of the model.
      @return false on validation error.

      Called once after the engine loads the model and before any inference. It will validate the
     model using the given ModelIOSpec and fail fast ( by returning false) if needed. It is expected
     that the setup() method does the most validation so that prepareInputs() and handleOutputs()
     will very unlikely to fail.

      The engine will abort if one of the registered adaptor fails to setup.
   */
  virtual bool setup(std::shared_ptr<Env>, const ModelIOSpec&) { return true; }
  /**
    @brief Called before each inference to repair the model inputs that is not handled by the engine
    core logic.
    @param InferenceInfo Contextual information about the current inference
    @param TensorMap the input tensors of the current inference
    @return false to inform the engine to skip the current inference

   */
  virtual bool prepareInputs([[maybe_unused]] const InferenceInfo& info,
                             [[maybe_unused]] const TensorMap& inputs) {
    return true;
  }
  /**
    @brief Called after each inference.
    @param InferenceInfo Contextual information about the current inference
    @param TensorMap The output tensors of the current inference

    */
  virtual void handleOutputs([[maybe_unused]] const InferenceInfo& info,
                             [[maybe_unused]] const TensorMap& outputs) {}
  /**
    @brief Reset the adaptor to the state after setup() is called and before any inference is
    happened.
  */
  virtual void reset() {}
  /**
    @brief Return the name of the adpator for tracing purpose.
   */
  virtual const std::string& name() = 0;
};
}  // namespace qualla
