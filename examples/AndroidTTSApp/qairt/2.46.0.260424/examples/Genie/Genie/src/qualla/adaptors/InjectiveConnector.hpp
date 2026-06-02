//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <string>

#include "qualla/ModelIOAdaptor.hpp"
#include "qualla/detail/tensor.hpp"

/**
 * Injective Connectors can pass outputs from one model to inputs of another in a 1-to-1 fashion
 * (i.e. an injective mapping.)
 *
 * Suppose an encoder needs to pass hidden states to a generator:
 *
 *   Producer                Consumer
 * +==========+            +===========+
 * |          |            |           |
 * |  hidden_0|----------->|hidden_0   |
 * |          |            |           |
 * |  hidden_1|----------->|hidden_1   |
 * |          |            |           |
 * |  hidden_2|----------->|hidden_2   |
 * |          |            |           |
 * +==========+            +===========+
 *
 * By registering a linked InjectiveProducer and InjectiveConsumer to their respective nodes, they
 * can share model identify their common tensor names and manage propagating tensor data during
 * execution.
 *
 * */

namespace qualla {

/**
 * InputSequenceTracker provides a view of the current input size to a model.
 *
 * When an InjectiveProducer generates outputs, the index of the sequence tracker is recorded
 * such that the InjectiveConsumer can properly time the consumption of its connected hidden states.
 *
 * A sequence tracker needs to be updated by a component that is aware of the overall input sequence
 * length as it is being prepared (e.g. the Accumulator in a Pipeline.)
 */
struct InputSequenceTracker {
  uint32_t index{0};
};

/**
 * A Tensor with a time range.
 */
struct TemporalTensor {
  qualla::Tensor tensor;
  uint32_t startIndex{0};
  uint32_t endIndex{0};
};

/**
 * Maintains a time sequence of Tensor data.
 *
 * A TensorSequence can be populated in a sparse manner with respect to time; if a time range
 * is not explicitly populated, then the tensor data for that range is treated as all zero.
 */
class TensorSequence {
 private:
  std::queue<TemporalTensor> m_segments;

  // Tracks the amount of the sequence that has been consumed.
  uint32_t m_consumptionIndex{0};

 public:
  TensorSequence() = default;

  /**
   * Resets the sequence, freeing all cached tensor data.
   */
  void reset();

  /**
   * Adds a range of tensor data to this sequence.
   *
   * Segments must be added in chronological order and cannot overlap with previously added
   * segments.
   */
  void add(TemporalTensor&& segment);

  /**
   * Populates the provided tensor's buffer with data starting at startIndex.
   *
   * Tensor data is consumed in chronological order.
   *
   * @param destination      tensor to populate
   * @param sequenceLength   number of entries to populate
   *
   * @throw genie::Exception if sequenceLength exceeds the sequence length dimension of the
   * destination tensor
   */
  void consumeNextSegment(Tensor& destination, const uint32_t sequenceLength);
};

// Maps tensor names to TensorSequences
using ConnectedTensorMap = std::unordered_map<std::string, TensorSequence>;

class InjectiveProducer : public ModelIOAdaptor {
 private:
  std::shared_ptr<InputSequenceTracker> m_sequenceTracker;
  std::shared_ptr<ConnectedTensorMap> m_tensorSequenceMap;

  const ModelIOSpec* m_ioSpec;

 public:
  InjectiveProducer(std::shared_ptr<InputSequenceTracker> sequenceTracker,
                    std::shared_ptr<ConnectedTensorMap> tensorSequenceMap)
      : m_sequenceTracker(sequenceTracker), m_tensorSequenceMap(tensorSequenceMap) {}

  bool setup(std::shared_ptr<Env>, const ModelIOSpec&) override;
  void reset() override;

  bool prepareInputs(const InferenceInfo& info, const qualla::TensorMap& inputs) override;
  void handleOutputs(const InferenceInfo& info, const qualla::TensorMap& outputs) override;
  const std::string& name() override {
    static std::string n = "InjectiveProducer";
    return n;
  }

  const ModelIOSpec& getIOSpec() { return *m_ioSpec; }
};

class InjectiveConsumer : public ModelIOAdaptor {
 private:
  std::shared_ptr<InputSequenceTracker> m_sequenceTracker;
  std::shared_ptr<ConnectedTensorMap> m_tensorSequenceMap;
  std::shared_ptr<InjectiveProducer> m_producer;

 public:
  InjectiveConsumer(std::shared_ptr<InputSequenceTracker> sequenceTracker,
                    std::shared_ptr<ConnectedTensorMap> tensorSequenceMap,
                    std::shared_ptr<InjectiveProducer> producer)
      : m_sequenceTracker(sequenceTracker),
        m_tensorSequenceMap(tensorSequenceMap),
        m_producer(producer) {}

  bool setup(std::shared_ptr<Env>, const ModelIOSpec&) override;
  void reset() override;

  bool prepareInputs(const InferenceInfo& info, const qualla::TensorMap& inputs) override;
  void handleOutputs(const InferenceInfo& info, const qualla::TensorMap& outputs) override;
  const std::string& name() override {
    static std::string n = "InjectiveConsumer";
    return n;
  }
};

}  // namespace qualla