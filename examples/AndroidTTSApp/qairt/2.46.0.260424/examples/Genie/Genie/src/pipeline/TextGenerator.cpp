//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <vector>
#include <algorithm>

#include "Exception.hpp"
#include "Pipeline.hpp"
#include "TextGenerator.hpp"

using namespace genie;

pipeline::TextGenerator::TextGenerator(nlohmann::json config,
                                       std::shared_ptr<ProfileStat> profileStat,
                                       std::shared_ptr<genie::log::Logger> logger,
                                       std::shared_ptr<qualla::Env> env)
    : Node(config) {
  m_typeGenerator = true;
  for (auto item : m_config.items()) {
    Dialog::validateDialogConfig(item.value());
    // Create dialog
    nlohmann::json dialogConfig;
    dialogConfig["dialog"] = item.value();
    if (dialogConfig["dialog"].contains("accumulator-size")) {
      m_accumulatorSize = item.value()["accumulator-size"];
    }

    m_generator = std::make_shared<genie::Dialog>(dialogConfig, profileStat, logger, nullptr, env);

    if (dialogConfig["dialog"].contains("engine")) {
      nlohmann::json engineConfig;
      if (dialogConfig["dialog"]["engine"].is_array()) {
        if (dialogConfig["dialog"]["engine"].empty()) {
          throw Exception(GENIE_STATUS_ERROR_GENERAL, "Engine array is empty");
        }
        const auto& engines = dialogConfig["dialog"]["engine"];
        // Find first engine with role == "target"
        bool found = false;
        for (const auto& eng : engines) {
          if (!eng.is_object()) {
            // Skip non-object entries to be resilient.
            continue;
          }
          if (eng.contains("role") && eng["role"].is_string() && eng["role"] == "target") {
            engineConfig = eng;
            found        = true;
            break;
          }
        }

        if (!found) {
          throw Exception(GENIE_STATUS_ERROR_GENERAL, "No engine with role 'target' found");
        }
      } else {
        engineConfig = dialogConfig["dialog"]["engine"];
      }

      if (engineConfig.contains("model") && engineConfig["model"].contains("cross-attention")) {
        m_crossAttention = engineConfig["model"]["cross-attention"];
      }

      m_type = qualla::Config::optional<std::string>(engineConfig, "type", "inference");

      if (engineConfig.contains("training-config")) {
        m_batchSize     = engineConfig["training-config"]["batch-size"];
        m_maxIterations = engineConfig["training-config"]["max-iterations"];
      }

      if (engineConfig.contains("model") && engineConfig["model"].contains("positional-encoding")) {
        nlohmann::json& posEncodingConfig = engineConfig["model"]["positional-encoding"];
        if (posEncodingConfig.contains("rope-scaling") &&
            (posEncodingConfig["rope-scaling"]["rope-type"] == "qwen2vl-mrope" ||
             posEncodingConfig["rope-scaling"]["rope-type"] == "qwen3vl-mrope")) {
          m_usingMRope = true;
        }
      }
    }
  }
}

int32_t pipeline::TextGenerator::bindPipeline(Pipeline& pipeline) {
  if (m_pipeline) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Node already bound to Pipeline");
  }
  m_pipeline = &pipeline;
  m_pipeline->setupAccumulator(m_accumulatorSize, m_generator->getEmbeddingLength());
  std::string inputDataType = "QNN_DATATYPE_FLOAT_32";
  double inputScale         = 1.0;
  int32_t inputOffset       = 0;
  size_t inputByteWidth     = 4;
  m_generator->getInputQuantParam(
      inputDataType, inputScale, inputOffset, inputByteWidth, m_crossAttention);
  // Set encoding for accumulator
  m_pipeline->m_accumulator->setEncoding(inputDataType, inputScale, inputOffset, inputByteWidth);
  return GENIE_STATUS_SUCCESS;
}

// Appends query string to existing query string
int32_t pipeline::TextGenerator::setTextInputData(GenieNode_IOName_t nodeIOName,
                                                  const char* txt,
                                                  std::shared_ptr<ProfileStat>) {
  // Helper lambda from amending data to DataLoader
  auto addToDataLoader = [&](const std::string& name) {
    std::shared_ptr<DataLoader> dataLoader = m_pipeline->getDataLoader();
    if(dataLoader) {
      // Fetch tokenizer
      GenieTokenizer_Handle_t tokenizerHandle = genie::Dialog::getTokenizerHandle(m_generator);
      if (tokenizerHandle == nullptr) {
        throw Exception(GENIE_STATUS_ERROR_GET_HANDLE_FAILED, "Failed to retrieve Tokenizer handle");
      }
      std::shared_ptr<genie::Tokenizer> tokenizer = genie::Tokenizer::get(tokenizerHandle);
      if (tokenizer == nullptr) {
        throw Exception(GENIE_STATUS_ERROR_GET_HANDLE_FAILED, "Failed to retrieve Tokenizer object");
      }

      // Tokenize the text
      const uint32_t numTokens    = tokenizer->encode(txt);
      const uint32_t numBytesI32  = numTokens * sizeof(int32_t);

      std::vector<int32_t> tokens32(numTokens);
      const int32_t* tokensData = tokens32.data();
      tokenizer->getEncodedTokenIds(&tokensData, numBytesI32);

      // Cast to long
      std::vector<int64_t> tokens64(numTokens);
      for (uint32_t j = 0; j < numTokens; j++) {
        tokens64[j] = static_cast<int64_t>(tokens32[j]);
      }
      const uint32_t numBytes = numTokens * sizeof(int64_t);

      // Fetch dimensions
      std::vector<uint32_t> dims;
      m_generator->getDimensions(qualla::LayerType::INPUT, dims);
      std::replace(dims.begin(), dims.end(), 0u, numTokens);

      dataLoader->addTensorData(name, tokens64.data(), numBytes, "QNN_DATATYPE_INT_64", 1.0, 0, dims);
    }
  };

  switch(nodeIOName) {
    case GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT:
      m_queryString += txt;
      addToDataLoader("input_ids");
      break;
    case GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT:
      addToDataLoader("targets");
      break;
    default:
      throw Exception(GENIE_STATUS_ERROR_GENERAL, "Unsupported Node IO for setTextInputData");
  }

  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                                       const void* embedding,
                                                       size_t embeddingSize,
                                                       std::shared_ptr<ProfileStat>) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setTextInputData can only be set for GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT");
  }
  m_pipeline->m_accumulator->append(reinterpret_cast<uint8_t*>(const_cast<void*>(embedding)),
                                    embeddingSize);
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::setTextOutputCallback(GenieNode_IOName_t /*nodeIOName*/,
                                                       GenieNode_TextOutput_Callback_t callback) {
  m_textOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::execute(void* userData,
                                         std::shared_ptr<ProfileStat> profileStat,
                                         nlohmann::json /*executionConfig*/) {
  try {
    bool useEmbedding   = false;
    const void* embData = nullptr;
    uint32_t embSize    = 0;

    const void* accData    = m_pipeline->m_accumulator->getData();
    const uint32_t accSize = m_pipeline->m_accumulator->getDataSize();

    if (!m_crossAttention) {
      if (accSize > 0) {
        useEmbedding = true;
        embData      = accData;
        embSize      = accSize;
      }
    } else {
      useEmbedding = false;
      if (!m_generator->setCrossAttentionHiddenStates(accData, accSize)) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL,
                        "setCrossAttentionHiddenStates fail on pipeline execute");
      }
    }

    if (useEmbedding) {
      if (m_usingMRope) {
        if (!m_generator->setVisionParam(m_pipeline->m_accumulator->getVisionParam())) {
          throw Exception(GENIE_STATUS_ERROR_GENERAL, "setVisionParam fail on pipeline execute");
        }
      }
      m_generator->embeddingQuery(embData,
                                  embSize,
                                  GenieNode_TextOutput_SentenceCode_t::GENIE_NODE_SENTENCE_COMPLETE,
                                  m_textOutputCallback,
                                  userData,
                                  profileStat);
    } else {
      m_generator->query(m_queryString.c_str(),
                         GenieNode_TextOutput_SentenceCode_t::GENIE_NODE_SENTENCE_COMPLETE,
                         m_textOutputCallback,
                         userData,
                         profileStat);
    }
    m_pipeline->m_accumulator->flush();
    m_queryString.clear();
  } catch (const ContextLimitException&) {
    m_textOutputCallback("", GENIE_NODE_SENTENCE_END, userData);
    throw;
  } catch (const Exception&) {
    m_textOutputCallback("", GENIE_NODE_SENTENCE_ABORT, userData);
    throw;
  } catch (const std::exception&) {
    m_textOutputCallback("", GENIE_NODE_SENTENCE_ABORT, userData);
    throw;
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::TextGenerator::validateTrainingDataConsistency() {
  qualla::TrainingData& trainingData = m_pipeline->getDataLoader()->getTrainingData();

  // Validate required keys
  static const std::unordered_set<std::string> requiredInputs = {
      "input_ids", "targets" };
  for (const auto& key : requiredInputs) {
    if (trainingData.find(key) == trainingData.end()) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      fmt::format("Missing required training data: {}", key));
    }
  }

  // Validate training data dimensions
  auto validateTrainingDataShape = [&](const std::string& key){
    if (trainingData[key].size() != m_maxIterations) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      fmt::format("Training data \'{}\' has an inconsistent number of iterations: "
                                  "expected: {} actual: {}",
                                  key, m_maxIterations, trainingData[key].size()));
    }
    for(size_t i = 0; i < m_maxIterations; i++) {
      if (trainingData[key][i].size() != m_batchSize) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL,
                        fmt::format("Training data \'{}\' @ iteration={} has inconsistent batch size: "
                                    "expected: {} actual: {}",
                                    key, i, m_batchSize, trainingData[key][i].size()));
      }
    }
  };

  validateTrainingDataShape("input_ids");
  validateTrainingDataShape("targets");

  // Ensure all targets tensors are smaller than their input_ids counterparts
  for (size_t i = 0; i < m_maxIterations; i++) {
    for (size_t batch = 0; batch < m_batchSize; batch++) {
      const qualla::Tensor& inputIds = trainingData["input_ids"][i][batch];
      const qualla::Tensor& targets  = trainingData["targets"][i][batch];
      if(targets.getSize() >= inputIds.getSize()) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL,
                        fmt::format("Training data @ iteration={} and batch={}: \'targets\' (size={}) "
                                    "is larger than its corresponding \'input_ids\' (size={})",
                                    i, batch, targets.getSize(), inputIds.getSize()));
      }
    }
  }
}

int32_t pipeline::TextGenerator::train() {
  if (!m_pipeline || !m_pipeline->getDataLoader()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "No DataLoader available for training");
  }

  validateTrainingDataConsistency();
  auto& trainingData = m_pipeline->getDataLoader()->getTrainingData();
  m_generator->train(trainingData);
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::save(const std::string& name) {
  return m_generator->save(name);
}

int32_t pipeline::TextGenerator::restore(const std::string& name) {
  return m_generator->restore(name);
}

void pipeline::TextGenerator::reset() { m_generator->reset(); }

int32_t pipeline::TextGenerator::setPriority(std::string engine,
                                             GeniePipeline_Priority_t priority) {
  return m_generator->setPriority(engine, static_cast<GenieDialog_Priority_t>(priority));
}

int32_t pipeline::TextGenerator::setOemkey(const std::string& oemKey) {
  return m_generator->setOemkey(oemKey);
}

int32_t pipeline::TextGenerator::applyLora(std::string loraAdapterName,
                                           std::string engine,
                                           std::shared_ptr<ProfileStat> profileStat) {
  auto status = m_generator->applyLora(loraAdapterName, engine, profileStat);
  if (m_pipeline) {  // Node configured to pipeline
    std::string inputDataType = "QNN_DATATYPE_FLOAT_32";
    double inputScale         = 1.0;
    int32_t inputOffset       = 0;
    size_t inputByteWidth     = 4;
    m_generator->getInputQuantParam(
        inputDataType, inputScale, inputOffset, inputByteWidth, m_crossAttention);
    // Set encoding for accumulator
    m_pipeline->m_accumulator->flush();
    m_pipeline->m_accumulator->setEncoding(inputDataType, inputScale, inputOffset, inputByteWidth);
  }
  return status;
}

int32_t pipeline::TextGenerator::applyLoraStrength(std::string tensorName,
                                                   std::string engine,
                                                   float alpha) {
  return m_generator->applyLoraStrength(tensorName, engine, alpha);
}

GenieEngine_Handle_t pipeline::TextGenerator::getEngineHandle(
    const std::string& engineRole, std::shared_ptr<ProfileStat> profileStat) {
  return m_generator->getEngineHandle(engineRole, profileStat);
}

int32_t pipeline::TextGenerator::bindEngine(const std::string& engineRole,
                                            std::shared_ptr<Engine> engine,
                                            std::shared_ptr<ProfileStat> profileStat) {
  return m_generator->bindEngine(engineRole, engine, profileStat);
}

GenieSampler_Handle_t pipeline::TextGenerator::getSamplerHandle() {
  return genie::Dialog::getSamplerHandle(m_generator);
}

GenieTokenizer_Handle_t pipeline::TextGenerator::getTokenizerHandle() {
  return genie::Dialog::getTokenizerHandle(m_generator);
}

void pipeline::TextGenerator::registerModelAdaptor(
    std::shared_ptr<qualla::ModelIOAdaptor> adaptor) {
  m_generator->registerModelAdaptor(adaptor);
}
