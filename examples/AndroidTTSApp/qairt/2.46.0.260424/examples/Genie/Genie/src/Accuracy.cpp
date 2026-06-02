//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <fstream>
#include <set>

#include "Accuracy.hpp"
#include "Engine.hpp"
#include "Macro.hpp"
#include "basic.hpp"

using namespace genie;

//=============================================================================
// Auxiliary Functions
//=============================================================================
bool checkFileExistsAndReadable(const std::ifstream &fileStream, const std::string) {
  bool res = fileStream.good();
  return res;
}

std::vector<float> softmax(const std::span<float> &logits) {
  std::vector<float> probs(logits.size());

  float maxLogit = *std::max_element(logits.begin(), logits.end());

  double sumExp = 0.0;
  for (size_t i = 0; i < logits.size(); i++) {
    // Subtract the maximum logit value from the current logit value for numerical stability
    const float logit     = logits[i] - maxLogit;
    const float exp_logit = expf(logit);
    sumExp += static_cast<double>(exp_logit);
  }
  sumExp = std::log(sumExp);
  for (size_t i = 0; i < probs.size(); i++) {
    probs[i] = logits[i] - maxLogit - static_cast<float>(sumExp);
  }

  return probs;
}

static void validateAccuracyEngineConfig(const nlohmann::json &config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Engine config is not an object");
  }

  // component is used in the "ENFORCE" macros
  std::string component = "engine";
  for (auto &item : config.items()) {
    if (item.key() == "convert-to-basic") {
      JSON_ENFORCE_BOOLEAN();
      if (item.value().get<bool>() != true) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid accuracy engine config: convert-to-basic needs to be true");
      }
    } else if (item.key() == "role") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown config key: " + item.key());
    }
  }
}

//=============================================================================
// Accuracy::Config Functions
//=============================================================================
qnn::util::HandleManager<Accuracy::Config> &Accuracy::Config::getManager() {
  static qnn::util::HandleManager<Accuracy::Config> s_manager;
  return s_manager;
}

Accuracy::Config::Config(const char *configStr) {
  nlohmann::json config;
  config = nlohmann::json::parse(configStr);

  std::set<std::string> mandatoryFields{"dataset", "type"};
  for (const auto &field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "accuracy";
  for (auto &item : config.items()) {
    if (item.key() == "dataset") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      if (item.value().get<std::string>() != "perplexity") {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unknown accuracy type: " + std::string(item.value()));
      }
    } else if (item.key() == "engine") {
      JSON_ENFORCE_OBJECT();
      validateAccuracyEngineConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown config key: " + item.key());
    }
  }
  m_config = config;
}

GenieAccuracyConfig_Handle_t Accuracy::Config::add(std::shared_ptr<Accuracy::Config> config) {
  return reinterpret_cast<GenieAccuracyConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<Accuracy::Config> Accuracy::Config::get(GenieAccuracyConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Accuracy::Config::remove(GenieAccuracyConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

nlohmann::json &Accuracy::Config::getJson() { return m_config; }

//=============================================================================
// Accuracy Functions
//=============================================================================
Accuracy::Accuracy(std::shared_ptr<Config> config) {
  auto jsonConfig = config->getJson();

  m_dataset = jsonConfig["dataset"];
  if (jsonConfig.contains("engine")) {
    if (jsonConfig["engine"].contains("role")) {
      m_engineRole = jsonConfig["engine"]["role"];
    }
    if (jsonConfig["engine"].contains("convert-to-basic")) {
      m_configContainsConvertField = true;
    } else {
      m_configContainsConvertField = false;
    }
  }
}

qnn::util::HandleManager<Accuracy> &Accuracy::getManager() {
  static qnn::util::HandleManager<Accuracy> s_manager;
  return s_manager;
}

GenieAccuracy_Handle_t Accuracy::add(std::shared_ptr<Accuracy> accuracyObj) {
  return reinterpret_cast<GenieAccuracy_Handle_t>(getManager().add(accuracyObj));
}

std::shared_ptr<Accuracy> Accuracy::get(GenieAccuracy_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void Accuracy::bindDialog(std::shared_ptr<Dialog> &dialog) {
  if (!dynamic_cast<qualla::BasicDialog *>(dialog->m_quallaDialog.get())) {
    if (!m_configContainsConvertField) {
      throw Exception(
          GENIE_STATUS_ERROR_JSON_VALUE,
          "Invalid accuracy engine config: convert-to-basic not set for non-basic dialog");
    }
  }
  m_dialog = dialog;
}

void Accuracy::unbindDialog() { m_dialog = nullptr; }

int32_t Accuracy::compute(const int32_t *queryTokens, uint32_t queryTokensSize) {
  auto quallaDialog   = m_dialog->m_quallaDialog.get();
  auto &quallaContext = quallaDialog->context();
  std::string role    = Engine::changeRole(m_engineRole);
  auto &quallaEngine  = quallaDialog->engine(role);
  quallaEngine.reset();

  std::vector<int32_t> tokens;
  tokens.assign(queryTokens, queryTokens + queryTokensSize);
  uint32_t ctxSize       = static_cast<uint32_t>(quallaContext.n_ctx());
  uint32_t n_vocab       = static_cast<uint32_t>(quallaContext.n_vocab());
  const uint32_t nChunks = queryTokensSize / ctxSize;

  if (nChunks == 0) {
    return -1;
  }

  double totalNLL     = 0.0;
  uint32_t totalCount = 0;
  // compute perplexity for chunks
  for (uint32_t i = 0; i < nChunks; ++i) {
    double nll     = 0.0;
    uint32_t count = 0;

    // For each chunk of size ctxSize, NLL is calculated by comparing each logit in [0, ctxSize -
    // 1] against its corresponding token, i.e. [1, ctxSize]. In total, ctxSize-1 comparisons are
    // made
    const uint32_t nCompare = ctxSize - 1;
    const uint32_t offset   = i * ctxSize;
    // Instead of processing all nCompare at once, it is further chunked into blocks of size 4096
    // This is to avoid OOM errors when accumulating logits
    const uint32_t blockSize = std::min(nCompare, 4096u);
    for (uint32_t startIdx = 0; startIdx < nCompare; startIdx += blockSize) {
      const uint32_t endIdx = std::min(startIdx + blockSize, nCompare);
      std::vector<int32_t> t(&tokens[offset + startIdx], &tokens[offset + endIdx]);
      std::vector<float> logits;
      size_t n = quallaEngine.process(t, logits, true);  // Process and return all logits

      const size_t nLogitSize = n_vocab * t.size();
      if (n != t.size() || logits.size() != nLogitSize) {
        return -1;
      }

      if (endIdx != nCompare) {
        // Trigger background KV$ updates
        quallaEngine.updateKV(endIdx);
      }

      // Based on https://huggingface.co/docs/transformers/perplexity
      // This implementation uses stride = max_length = ctx.size()
      // PPL is calculated by comparing all logits in the window to its corresponding next token
      for (size_t j = 0; j < t.size(); ++j) {
        const std::span<float> tokLogits(&logits[j * n_vocab], n_vocab);
        const size_t idx = static_cast<size_t>(tokens[offset + startIdx + j + 1]);
        const float prob = softmax(tokLogits)[idx];
        nll -= static_cast<double>(prob);
        ++count;
      }
    }
    quallaEngine.reset();
    totalNLL += nll;
    totalCount += count;
  }
  quallaDialog->reset();
  m_ppl = std::exp(totalNLL / totalCount);
  return GENIE_STATUS_SUCCESS;
}

int32_t Accuracy::computeEmbeddings(const std::vector<int32_t> &tokens) {
  auto quallaDialog   = m_dialog->m_quallaDialog.get();
  auto &quallaContext = quallaDialog->context();
  std::string role    = Engine::changeRole(m_engineRole);
  auto &quallaEngine  = quallaDialog->engine(role);
  quallaEngine.reset();

  std::vector<uint8_t> encoderOutput;
  quallaDialog->encoder()->encode(tokens, encoderOutput);

  uint32_t ctxSize         = quallaContext.n_ctx();
  uint32_t n_vocab         = quallaContext.n_vocab();
  uint32_t embeddingLength = quallaContext.n_embd();
  size_t lutSize           = quallaDialog->encoder()->getEmbeddingLutSize();  // in bytes
  float lutByteWidth = static_cast<float>(lutSize) / static_cast<float>(embeddingLength * n_vocab);
  size_t numElements = (encoderOutput.size() / lutByteWidth) / embeddingLength;
  uint32_t inputByteWidth =
      (quallaDialog->getLUTDataType() == "QNN_DATATYPE_FLOAT_32") ? 4 : quallaDialog->inputBitWidth;
  const uint32_t nChunks = numElements / ctxSize;

  {
    std::vector<uint8_t> eosEncoding, eosEmbedding;
    eosEmbedding.resize(embeddingLength * inputByteWidth);

    std::vector<int32_t> eosTokenInput(1, quallaContext.eos());
    quallaDialog->encoder()->encode(eosTokenInput, eosEncoding);
    quallaDialog->requantEmbedding(eosEncoding.data(), eosEmbedding.data(), embeddingLength);
    if (quallaDialog->getLUTDataType() == "QNN_DATATYPE_FLOAT_32") {
      if (!quallaEngine.cacheEosEmbedding(eosEncoding)) {
        return -1;
      }
    } else {
      if (!quallaEngine.cacheEosEmbedding(eosEmbedding)) {
        return -1;
      }
    }
  }

  if (nChunks == 0) {
    return -1;
  }

  std::vector<uint8_t> decoderInput;
  if (quallaDialog->getLUTDataType() == "QNN_DATATYPE_FLOAT_32") {
    decoderInput = encoderOutput;
  } else {
    decoderInput.resize(numElements * embeddingLength * inputByteWidth);
    quallaDialog->requantEmbedding(
        encoderOutput.data(), decoderInput.data(), numElements * embeddingLength);
  }

  double totalNLL     = 0.0;
  uint32_t totalCount = 0;
  // compute perplexity for chunks
  for (uint32_t i = 0; i < nChunks; ++i) {
    double nll     = 0.0;
    uint32_t count = 0;

    const uint32_t nCompare = (ctxSize - 1) * embeddingLength * inputByteWidth;
    const uint32_t offset   = i * ctxSize * embeddingLength * inputByteWidth;
    // Instead of processing all nCompare at once, it is further chunked into blocks of size 4096
    // This is to avoid OOM errors when accumulating logits
    const uint32_t blockSize = std::min(nCompare, 4096 * embeddingLength * inputByteWidth);
    for (uint32_t startIdx = 0; startIdx < nCompare; startIdx += blockSize) {
      const uint32_t endIdx = std::min(startIdx + blockSize, nCompare);
      std::vector<uint8_t> e(&decoderInput[offset + startIdx], &decoderInput[offset + endIdx]);
      std::vector<float> logits;

      const size_t n = quallaEngine.process(e, {}, logits, true);  // Process and return all logits
      const size_t nTokensExpected = e.size() / (embeddingLength * inputByteWidth);
      const size_t nLogitSize      = n_vocab * nTokensExpected;
      if (n != nTokensExpected || logits.size() != nLogitSize) {
        return -1;
      }

      if (endIdx != nCompare) {
        // Trigger background KV$ updates
        quallaEngine.updateKV(endIdx / (embeddingLength * inputByteWidth));
      }

      // Based on https://huggingface.co/docs/transformers/perplexity
      // This implementation uses stride = max_length = ctx.size()
      // PPL is calculated by comparing all logits in the window to its corresponding next token
      for (size_t j = 0; j < nTokensExpected; ++j) {
        const std::span<float> tok_logits(&logits[j * n_vocab], n_vocab);
        size_t idx =
            static_cast<size_t>(tokens[offset / (embeddingLength * inputByteWidth) +
                                       startIdx / (embeddingLength * inputByteWidth) + j + 1]);
        double prob = static_cast<double>(softmax(tok_logits)[idx]);

        nll -= prob;
        ++count;
      }
    }
    quallaEngine.reset();
    totalNLL += nll;
    totalCount += count;
  }
  quallaDialog->reset();
  m_ppl = std::exp(totalNLL / totalCount);
  return GENIE_STATUS_SUCCESS;
}

int32_t Accuracy::computeFromText() {
  std::ifstream datasetStream = std::ifstream(m_dataset);
  if (!checkFileExistsAndReadable(datasetStream, m_dataset)) {
    return EXIT_FAILURE;
  }

  std::string datasetStr{};
  std::getline(datasetStream, datasetStr, '\0');
  std::vector<int32_t> queryTokens;
  auto quallaDialog     = m_dialog->m_quallaDialog.get();
  auto &quallaTokenizer = quallaDialog->tokenizer();
  quallaTokenizer.encode(datasetStr, queryTokens);
  if (quallaDialog->encoder() && quallaDialog->encoder()->type() == "lut") {  // lut implementation
    return Accuracy::computeEmbeddings(queryTokens);
  } else {
    return Accuracy::compute(queryTokens.data(), queryTokens.size());
  }
}

uint32_t Accuracy::serialize() {
  // Serialize accuracy data into JSON and return JSON size
  m_jsonData["perplexity"] = m_ppl;
  m_data                   = m_jsonData.dump(2);
  return m_data.length() + 1;
}

void Accuracy::getJsonData(const char **jsonData) {
  memcpy(static_cast<void *>(const_cast<char *>(*jsonData)),
         static_cast<void *>(const_cast<char *>(m_data.c_str())),
         m_data.length());
  (const_cast<char *>(*jsonData))[m_data.length()] = '\0';
}

void Accuracy::remove(GenieAccuracy_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}