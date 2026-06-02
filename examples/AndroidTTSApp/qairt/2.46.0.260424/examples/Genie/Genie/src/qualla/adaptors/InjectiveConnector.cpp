//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>

#include "Exception.hpp"
#include "InjectiveConnector.hpp"
#include "Quantization.hpp"

namespace qualla {

static genie::quantization::DataType toQuantDataType(TensorDataType type) {
  switch (type) {
    case TENSOR_DATATYPE_UFIXED_POINT_8:
      return genie::quantization::DataType::UFixed8;
    case TENSOR_DATATYPE_UFIXED_POINT_16:
      return genie::quantization::DataType::UFixed16;
    case TENSOR_DATATYPE_FLOAT_POINT_16:
      return genie::quantization::DataType::Float16;
    case TENSOR_DATATYPE_FLOAT_32:
      return genie::quantization::DataType::Float32;
    default:
      throw std::runtime_error("Unsupported quantization type.");
  }
}

static genie::quantization::Encoding tensorQuantParamsToEncoding(TensorQuantizationParams params,
                                                                 TensorDataType type) {
  return {params.scale, params.offset, toQuantDataType(type)};
}

static uint32_t getSequenceDimension(std::vector<uint32_t> dimensions) {
  // The first non-1 dimension is the sequence dimension to be populated (e.g. AR-N)
  const auto& sequenceDimIterator =
      std::find_if(dimensions.begin(), dimensions.end(), [](size_t dim) { return dim != 1; });
  if (sequenceDimIterator == dimensions.end()) {
    return 1;
  }
  return *sequenceDimIterator;
}

template <typename T>
static void zeroFillTensorRegionT(Tensor& tensor, size_t embeddingDim, size_t start, size_t end) {
  T zero = static_cast<T>(-tensor.getQuantizationParams().offset);
  for (uint32_t i = start; i < end; i++) {
    T* buf = static_cast<T*>(tensor.getData());
    for (uint32_t j = 0; j < embeddingDim; j++) buf[i * embeddingDim + j] = zero;
  }
}

/**
 * Helper to fill in a region of a Tensor buffer with zeroes using its quantization encodings.
 *
 * @param   tensor        the tensor to populate with zero padding
 * @param   embeddingDim  the embedding dimension of the tensor
 * @param   start         the starting index of zero padding (inclusive)
 * @param   end           the ending index of zero padding (exclusive)
 * */
static void zeroFillTensorRegion(Tensor& tensor, size_t embeddingDim, size_t start, size_t end) {
  genie::quantization::DataType dataType = toQuantDataType(tensor.getDataType());
  switch (dataType) {
    case genie::quantization::DataType::UFixed8:
      zeroFillTensorRegionT<uint8_t>(tensor, embeddingDim, start, end);
      break;
    case genie::quantization::DataType::SFixed8:
      zeroFillTensorRegionT<int8_t>(tensor, embeddingDim, start, end);
      break;
    case genie::quantization::DataType::UFixed16:
      zeroFillTensorRegionT<uint16_t>(tensor, embeddingDim, start, end);
      break;
    case genie::quantization::DataType::SFixed16:
      zeroFillTensorRegionT<int16_t>(tensor, embeddingDim, start, end);
      break;
    default:
      throw std::runtime_error("Unsupported quantization type for zero fill.");
  }
}

void TensorSequence::reset() {
  std::queue<TemporalTensor> emptyQueue;
  m_segments.swap(emptyQueue);
  m_consumptionIndex = 0;
}

void TensorSequence::add(TemporalTensor&& segment) {
  if ((!m_segments.empty()) && (segment.startIndex <= m_segments.back().endIndex)) {
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "Attempted to add overlapping or non-chronological tensor segment.");
  }
  m_segments.push(segment);
}

void TensorSequence::consumeNextSegment(Tensor& destination, const uint32_t sequenceLength) {
  const size_t totalElements = destination.getSize();
  const uint32_t sequenceDim = getSequenceDimension(destination.getDimensions());
  const size_t embeddingDim  = totalElements / static_cast<size_t>(sequenceDim);

  if (sequenceDim < sequenceLength) {
    throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                           "Requested tensor segment of " + std::to_string(sequenceLength) +
                               " entries, which exceeds the size of the destination tensor: " +
                               std::to_string(sequenceDim));
  }

  uint32_t totalConsumed = 0;
  while (totalConsumed < sequenceLength) {
    uint32_t consumedCount   = 0;
    const uint32_t remaining = (sequenceLength - totalConsumed);
    if (!m_segments.empty()) {
      // Peak at the next populated tensor segment.
      TemporalTensor& nextSegment = m_segments.front();

      if (m_consumptionIndex >= nextSegment.endIndex) {
        // Segment has been fully consumed. Delete it.
        m_segments.pop();
      } else if (m_consumptionIndex + remaining < nextSegment.startIndex) {
        // Next segment hasn't been reached. Fill the destination with zeroes.
        zeroFillTensorRegion(destination, embeddingDim, totalConsumed, sequenceLength);
        consumedCount += remaining;
      } else {
        // There is overlap between the consumed region and a produced region.
        const uint32_t overlapStart = std::max(m_consumptionIndex, nextSegment.startIndex);
        const uint32_t overlapEnd = std::min(m_consumptionIndex + remaining, nextSegment.endIndex);

        // Translate the global overlap indices to indices within the source tensor.
        const uint32_t srcOverlapStart = overlapStart - nextSegment.startIndex;
        // Translate the global overlap indices to indices within the destination tensor.
        const uint32_t dstOverlapStart = overlapStart - m_consumptionIndex;

        // Zero-pad the region prior to the overlap.
        zeroFillTensorRegion(destination, embeddingDim, 0, dstOverlapStart);

        // Copy the overlapping regions.
        const size_t srcOffset                          = srcOverlapStart * embeddingDim;
        const size_t dstOffset                          = dstOverlapStart * embeddingDim;
        const genie::quantization::Encoding srcEncoding = tensorQuantParamsToEncoding(
            nextSegment.tensor.getQuantizationParams(), nextSegment.tensor.getDataType());
        const genie::quantization::Encoding dstEncoding = tensorQuantParamsToEncoding(
            destination.getQuantizationParams(), destination.getDataType());
        genie::quantization::requantize(nextSegment.tensor.getData(),
                                        srcOffset,
                                        srcEncoding,
                                        destination.getData(),
                                        dstOffset,
                                        dstEncoding,
                                        (overlapEnd - overlapStart) * embeddingDim);

        consumedCount += (overlapEnd - m_consumptionIndex);
      }
    } else {
      // No segments remaining. Exit the loop and set the remaining tensor region to zero.
      zeroFillTensorRegion(destination, embeddingDim, totalConsumed, sequenceLength);
      consumedCount += remaining;
    }
    totalConsumed += consumedCount;
    m_consumptionIndex += consumedCount;
  }
  // Zero the remaining rows of the tensor beyond the requested sequence length.
  zeroFillTensorRegion(destination, embeddingDim, totalConsumed, sequenceDim);
}

bool InjectiveProducer::setup(std::shared_ptr<Env>, const ModelIOSpec& ioSpec) {
  m_ioSpec = &ioSpec;
  return true;
}

void InjectiveProducer::reset() {
  for (auto& [name, seq] : *m_tensorSequenceMap) {
    seq.reset();
  }
}

bool InjectiveProducer::prepareInputs(const InferenceInfo&, const qualla::TensorMap&) {
  return true;
}

void InjectiveProducer::handleOutputs(const InferenceInfo&, const qualla::TensorMap& outputs) {
  const uint32_t start = m_sequenceTracker->index;
  for (auto& [name, seq] : *m_tensorSequenceMap) {
    if (outputs.contains(name)) {
      Tensor tensorCopy = Tensor::deepCopy(*outputs.at(name));
      seq.add({tensorCopy, start, start + getSequenceDimension(outputs.at(name)->getDimensions())});
    }
  }
}

bool InjectiveConsumer::setup(std::shared_ptr<Env>, const ModelIOSpec& ioSpec) {
  const ModelIOSpec& producerSpec = m_producer->getIOSpec();

  // Choose arbitrary variants for tensor name matching.
  const ModelVariant& producerVariant = *producerSpec.getVariants().begin();
  const ModelVariant& consumerVariant = *ioSpec.getVariants().begin();

  const TensorMap& producerTensorMap = *producerSpec.getOutputSpec(producerVariant);
  const TensorMap& consumerTensorMap = *ioSpec.getInputSpec(consumerVariant);

  // Find all identical tensor names and insert an empty TensorSequence.
  for (const auto& [name, tensor] : consumerTensorMap) {
    if (m_tensorSequenceMap->contains(name)) {
      throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                             "Unsupported operation: Cannot broadcast wildcard tensors to multiple "
                             "engines in the same generator.");
    } else if (producerTensorMap.contains(name)) {
      m_tensorSequenceMap->insert({name, TensorSequence()});
    }
  }

  // Validate that the hidden state dimensions are compatible between the producer and consumer.
  // The connected tensors may differ along the first non-unitary axis. This is treated as the
  // time-dependent axis and is "broadcasted" across inference iterations.
  for (auto& [name, seq] : *m_tensorSequenceMap) {
    const std::vector<uint32_t> producerDims = producerTensorMap.at(name)->getDimensions();
    const std::vector<uint32_t> consumerDims = consumerTensorMap.at(name)->getDimensions();

    if (producerDims.size() != consumerDims.size()) {
      throw genie::Exception(GENIE_STATUS_ERROR_GENERAL,
                             "Mismatched wildcard tensor rank between consumer and producer.");
    }

    bool firstMismatch = true;
    for (size_t i = 0; i < producerDims.size(); i++) {
      if (producerDims[i] != consumerDims[i]) {
        if (firstMismatch) {
          // Allow one mismatched axis to be broadcasted.
          firstMismatch = false;
          continue;
        }
        throw genie::Exception(
            GENIE_STATUS_ERROR_GENERAL,
            "Mismatch in wildcard tensor dimension between consumer and producer.");
      }
    }
  }

  return true;
}

void InjectiveConsumer::reset() {
  for (auto& [name, seq] : *m_tensorSequenceMap) {
    seq.reset();
  }
}

bool InjectiveConsumer::prepareInputs(const InferenceInfo& info, const qualla::TensorMap& inputs) {
  for (auto& [name, seq] : *m_tensorSequenceMap) {
    if (inputs.contains(name)) {
      seq.consumeNextSegment(*inputs.at(name), info.n_inputs);
    }
  }
  // Update the sequence tracker to point to the amount of processed tokens. This ensures
  // that producer nodes are synchronized with the consumer after a response is generated.
  m_sequenceTracker->index = info.n_past + info.n_inputs;
  return true;
}

void InjectiveConsumer::handleOutputs(const InferenceInfo&, const qualla::TensorMap&) {}

}  // namespace qualla