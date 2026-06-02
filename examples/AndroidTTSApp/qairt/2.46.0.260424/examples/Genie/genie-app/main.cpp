//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#include "GenieAccuracy.hpp"
#include "GenieCommon.hpp"
#include "GenieDialog.hpp"
#include "GenieEmbedding.hpp"
#include "GenieLog.hpp"
#include "GenieNode.hpp"
#include "GeniePipeline.hpp"
#include "GenieProfile.hpp"
#include "GenieSampler.hpp"
#include "GenieTokenizer.hpp"
#include "nlohmann/json.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cleanup();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INV_ARG(str) throw std::invalid_argument("[" + std::to_string(__LINE__) + "]: " + str);
#define RT_ERR(str)  throw std::runtime_error("[" + std::to_string(__LINE__) + "]: " + str);

#if defined(__linux__)
#define RED       "\033[91m"
#define GREY      "\033[37m"
#define YELLOW    "\033[33;1m"
#define GREEN     "\033[32m"
#define UNDERLINE "\033[4m"
#define RESET     "\033[0m"
#else
#define RED       ""
#define GREY      ""
#define YELLOW    ""
#define GREEN     ""
#define UNDERLINE ""
#define RESET     ""
#endif

#if defined(DEBUG)
#define TRACE() std::cout << GREY << "[" << __LINE__ << "] " << __FUNCTION__ << RESET << std::endl
#else
#define TRACE()
#endif

std::shared_ptr<void> g_embeddingLut{};
size_t g_embeddingLutSize{0};
std::string g_nodeOutputDirectory;
uint32_t g_nodeIOCallbackIteration = 0;

std::streamsize getFileSize(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  return file.tellg();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Callbacks
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<const void*, std::ofstream> g_queryResponseMap;
std::map<const void*, std::string> g_queryBufferMap;
std::map<const void*, std::ofstream> g_nodeResponseMap;
const void* g_primaryConsoleUserData = nullptr;

Genie_Status_t nodeQueryCallback(const char* responseStr,
                                 GenieNode_TextOutput_SentenceCode_t sentenceCode,
                                 const void* userData) {
  auto it = g_nodeResponseMap.find(userData);
  std::ostream& os =
      (it == g_nodeResponseMap.end()) ? std::cout : static_cast<std::ostream&>(it->second);
  switch (sentenceCode) {
    case GENIE_NODE_SENTENCE_COMPLETE:
      os << "[COMPLETE]: " << std::flush;
      break;
    case GENIE_NODE_SENTENCE_BEGIN:
      os << "[BEGIN]: " << std::flush;
      break;
    case GENIE_NODE_SENTENCE_CONTINUE:
      break;
    case GENIE_NODE_SENTENCE_END:
      os << "[END]" << std::flush;
      break;
    case GENIE_NODE_SENTENCE_ABORT:
      os << "[ABORT]: " << std::flush;
      break;
    default:
      os << "[UNKNOWN]: " << std::flush;
      break;
  }
  if (responseStr) {
    os << responseStr << std::flush;
  }
  return 1;
}

void queryCallback(const char* responseStr,
                   GenieDialog_SentenceCode_t sentenceCode,
                   const void* userData) {
  auto it     = g_queryResponseMap.find(userData);
  bool isFile = (it != g_queryResponseMap.end());
  bool isGlobalEnd =
      (sentenceCode == GENIE_DIALOG_SENTENCE_COMPLETE ||
       sentenceCode == GENIE_DIALOG_SENTENCE_END || sentenceCode == GENIE_DIALOG_SENTENCE_ABORT);
  if (!isFile && userData != g_primaryConsoleUserData && !isGlobalEnd) {
    if (responseStr) g_queryBufferMap[userData] += responseStr;
    return;
  }

  std::ostream& os = isFile ? (it->second) : (std::cout);

  if (!isFile) {
    if (isGlobalEnd && (!g_queryBufferMap.empty())) {
      const uintptr_t userData0 = reinterpret_cast<uintptr_t>(g_primaryConsoleUserData);
      for (size_t i = 1; i < g_queryBufferMap.size() + 1; ++i) {
        const void* userDataIdx = reinterpret_cast<const void*>(userData0 + i);
        auto itBuf              = g_queryBufferMap.find(userDataIdx);
        if (itBuf != g_queryBufferMap.end() && !itBuf->second.empty()) {
          std::cout << "\n[Batch " << i << " Output]: " << itBuf->second << std::flush;
          itBuf->second.clear();
        }
      }
    }

    switch (sentenceCode) {
      case GENIE_DIALOG_SENTENCE_COMPLETE:
        os << "[COMPLETE]: " << std::flush;
        break;
      case GENIE_DIALOG_SENTENCE_BEGIN:
        os << "[BEGIN]: " << std::flush;
        break;
      case GENIE_DIALOG_SENTENCE_CONTINUE:
        break;
      case GENIE_DIALOG_SENTENCE_END:
        os << "[END]" << std::flush << std::endl;
        break;
      case GENIE_DIALOG_SENTENCE_ABORT:
        os << "[ABORT]: " << std::flush;
        break;
      default:
        os << "[UNKNOWN]: " << std::flush;
        break;
    }
  }

  if (responseStr) {
    os << responseStr << std::flush;
  }
}

void tokenQueryCallback(const uint32_t* token,
                        uint32_t tokensLength,
                        GenieDialog_SentenceCode_t sentenceCode,
                        const void* userData) {
  auto it          = g_queryResponseMap.find(userData);
  bool isFile      = (it != g_queryResponseMap.end());
  std::ostream& os = isFile ? (it->second) : (std::cout);

  if (!isFile) {
    switch (sentenceCode) {
      case GENIE_DIALOG_SENTENCE_COMPLETE:
        os << "[COMPLETE]: " << std::flush;
        break;
      case GENIE_DIALOG_SENTENCE_BEGIN:
        os << "[BEGIN]: " << std::flush;
        break;
      case GENIE_DIALOG_SENTENCE_CONTINUE:
        break;
      case GENIE_DIALOG_SENTENCE_END:
        os << "[END]" << std::flush << std::endl;
        break;
      case GENIE_DIALOG_SENTENCE_ABORT:
        os << "[ABORT]: " << std::flush;
        break;
      default:
        os << "[UNKNOWN]: " << std::flush;
        break;
    }
  }
  if (token) {
    for (uint32_t i = 0; i < tokensLength; i++) {
      os << token[i] << " " << std::flush;
    }
  }
}

std::string g_embeddingOutputFile = "output.raw";
void embeddingCallback(const uint32_t* dimensions,
                       uint32_t rank,
                       const float* embeddingBuffer,
                       const void*) {
  // Function will save embedding vector to a file.
  // calculate the size of the embedding buffers using dimension vector
  uint64_t embeddingBufferSize = 1;
  std::cout << "RANK of DIMENSIONS : " << rank << "\n" << std::endl;
  std::cout << "EMBEDDING DIMENSIONS : [ ";
  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dimensions[i] << ((i != rank - 1) ? ", " : " ]\n");
    embeddingBufferSize = embeddingBufferSize * dimensions[i];
  }
  std::cout << std::endl;
  std::cout << "GENERATED EMBEDDING SIZE : " << embeddingBufferSize << std::endl;

  // Open a binary file for writing
  std::ofstream outFile(g_embeddingOutputFile, std::ios::binary);
  if (!outFile.good()) {
    std::cerr << "Error in opening file for writing!" << std::endl;
    return;
  }

  // Write the buffer to the file
  outFile.write(reinterpret_cast<const char*>(embeddingBuffer),
                static_cast<std::streamsize>(embeddingBufferSize * sizeof(float)));

  // Close the file
  outFile.close();

  std::cout << "Embedding vectors saved in " << g_embeddingOutputFile << std::endl;
}

int g_embeddingOutputFileIndex = 0;
void nodeEmbeddingCallback(const uint32_t* dimensions,
                           uint32_t rank,
                           const size_t embeddingBufferSize,
                           const void* embeddingBuffer,
                           const void*) {
  // Function will save embedding vector to a file.
  // calculate the size of the embedding buffers using dimension vector
  uint64_t numElements = 1;

  std::cout << "RANK of DIMENSIONS : " << rank << "\n" << std::endl;

  std::cout << "EMBEDDING DIMENSIONS : [ ";
  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dimensions[i] << ((i != rank - 1) ? ", " : " ]\n");
    numElements = numElements * dimensions[i];
  }
  std::cout << std::endl;

  std::cout << "GENERATED EMBEDDING SIZE : " << embeddingBufferSize << std::endl;

  std::string outputFile = "output_" + std::to_string(g_embeddingOutputFileIndex++) + ".raw";
  // Open a binary file for writing
  std::ofstream outFile(outputFile, std::ios::binary);
  if (!outFile.good()) {
    std::cerr << "Error in opening file for writing!" << std::endl;
    return;
  }

  // Write the buffer to the file
  outFile.write(reinterpret_cast<const char*>(embeddingBuffer),
                static_cast<std::streamsize>(embeddingBufferSize));

  // Close the file
  outFile.close();

  std::cout << "Embedding vectors saved in " << outputFile << std::endl;
}

void nodeIOCallback(const void* data,
                    const size_t dataSize,
                    const char* outputConfig,
                    const void* /*userData*/) { 
  if (!data || dataSize == 0) {
    std::cerr << "nodeIOCallback received empty data" << std::endl;
    return;
  }
 
  std::string baseFileName = "logits_" + std::to_string(g_nodeIOCallbackIteration++);

  // Save data to .raw file
  std::string dataPath = g_nodeOutputDirectory + "/" + baseFileName + ".raw";
  std::ofstream dataFile(dataPath, std::ios::binary);
  if (dataFile.is_open()) {
    dataFile.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(dataSize));
    dataFile.close();
    std::cout << "Node data saved to " << dataPath << std::endl;
  } else {
    std::cerr << "Failed to open data file: " << dataPath << std::endl;
    return;
  }

  // Save outputConfig to .json file
  if (outputConfig) {
    std::string configPath = g_nodeOutputDirectory + "/" + baseFileName + ".json";
    std::ofstream configFile(configPath);
    if (configFile.is_open()) {
      configFile << outputConfig;
      configFile.close();
      std::cout << "Node output config saved to " << configPath << std::endl;
    } else {
      std::cerr << "Failed to open config file: " << configPath << std::endl;
    }
  }
}

void tokenToEmbedCallback(int32_t token,
                          void* embedding,
                          uint32_t embeddingSize,
                          const void* /*userData*/) {
  size_t lutIndex = static_cast<size_t>(token) * embeddingSize;
  if ((lutIndex + embeddingSize) <= g_embeddingLutSize) {
    int8_t* embeddingSrc = static_cast<int8_t*>(g_embeddingLut.get()) + lutIndex;
    int8_t* embeddingDst = static_cast<int8_t*>(embedding);
    std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
  } else {
    std::cerr << "Error: T2E conversion overflow." << std::endl;
  }
}

// Function to convert string to enum
GenieNode_IOName_t stringToNodeIO(const std::string& nodeIOString) {
  static const std::map<std::string, GenieNode_IOName_t> nodeIOMap = {
      {"GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT",
       GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT},
      {"GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT},
      {"GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT",
       GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT},
      {"GENIE_NODE_TEXT_ENCODER_TEXT_INPUT",
       GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_TEXT_INPUT},
      {"GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT",
       GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT},
      {"GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT},
      {"GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT},
      {"GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN},
      {"GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS},
      {"GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK},
      {"GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK},
      {"GENIE_NODE_IMAGE_ENCODER_PRETILE_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_PRETILE_EMBEDDING_INPUT},
      {"GENIE_NODE_IMAGE_ENCODER_POSTTILE_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_POSTTILE_EMBEDDING_INPUT},
      {"GENIE_NODE_IMAGE_ENCODER_GATED_POS_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_GATED_POS_EMBEDDING_INPUT},
      {"GENIE_NODE_DIFFUSER_TEXT_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_DIFFUSER_TEXT_EMBEDDING_INPUT},
      {"GENIE_NODE_DIFFUSER_NOISE_INPUT", GenieNode_IOName_t::GENIE_NODE_DIFFUSER_NOISE_INPUT},
      {"GENIE_NODE_DIFFUSER_TIMESTEP_INPUT",
       GenieNode_IOName_t::GENIE_NODE_DIFFUSER_TIMESTEP_INPUT},
      {"GENIE_NODE_DIFFUSER_IMAGE_EMBEDDING_OUTPUT",
       GenieNode_IOName_t::GENIE_NODE_DIFFUSER_IMAGE_EMBEDDING_OUTPUT},
      {"GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT",
       GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT},
      {"GENIE_NODE_LM_EXECUTOR_EMBEDDING_INPUT",
       GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_EMBEDDING_INPUT},
      {"GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT",
       GenieNode_IOName_t::GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT},
      {"GENIE_NODE_WILDCARD", GenieNode_IOName_t::GENIE_NODE_WILDCARD}};

  try {
    return nodeIOMap.at(nodeIOString);
  } catch (const std::out_of_range&) {
    throw std::invalid_argument("Invalid Node IO value passed: " + nodeIOString);
  }
}

// Function to convert string to enum
Genie_PerformancePolicy_t stringToGeniePerformance(const std::string& performance) {
  static const std::map<std::string, Genie_PerformancePolicy_t> performanceMap = {
      {"GENIE_PERFORMANCE_BURST", Genie_PerformancePolicy_t::GENIE_PERFORMANCE_BURST},
      {"GENIE_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE",
       Genie_PerformancePolicy_t::GENIE_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE},
      {"GENIE_PERFORMANCE_HIGH_PERFORMANCE",
       Genie_PerformancePolicy_t::GENIE_PERFORMANCE_HIGH_PERFORMANCE},
      {"GENIE_PERFORMANCE_BALANCED", Genie_PerformancePolicy_t::GENIE_PERFORMANCE_BALANCED},
      {"GENIE_PERFORMANCE_LOW_BALANCED", Genie_PerformancePolicy_t::GENIE_PERFORMANCE_LOW_BALANCED},
      {"GENIE_PERFORMANCE_HIGH_POWER_SAVER",
       Genie_PerformancePolicy_t::GENIE_PERFORMANCE_HIGH_POWER_SAVER},
      {"GENIE_PERFORMANCE_LOW_POWER_SAVER",
       Genie_PerformancePolicy_t::GENIE_PERFORMANCE_LOW_POWER_SAVER},
      {"GENIE_PERFORMANCE_POWER_SAVER", Genie_PerformancePolicy_t::GENIE_PERFORMANCE_POWER_SAVER},
      {"GENIE_PERFORMANCE_EXTREME_POWER_SAVER",
       Genie_PerformancePolicy_t::GENIE_PERFORMANCE_EXTREME_POWER_SAVER}};

  try {
    return performanceMap.at(performance);
  } catch (const std::out_of_range&) {
    throw std::invalid_argument("Invalid Performance value passed: " + performance);
  }
}

// Function to convert string to enum
GenieDialog_Priority_t stringToGeniePriority(const std::string& priority) {
  static const std::map<std::string, GenieDialog_Priority_t> priorityMap = {
      {"GENIE_DIALOG_PRIORITY_LOW", GenieDialog_Priority_t::GENIE_DIALOG_PRIORITY_LOW},
      {"GENIE_DIALOG_PRIORITY_NORMAL", GenieDialog_Priority_t::GENIE_DIALOG_PRIORITY_NORMAL},
      {"GENIE_DIALOG_PRIORITY_NORMAL_HIGH",
       GenieDialog_Priority_t::GENIE_DIALOG_PRIORITY_NORMAL_HIGH},
      {"GENIE_DIALOG_PRIORITY_HIGH", GenieDialog_Priority_t::GENIE_DIALOG_PRIORITY_HIGH}};

  try {
    return priorityMap.at(priority);
  } catch (const std::out_of_range&) {
    throw std::invalid_argument("Invalid Priority value passed: " + priority);
  }
}

// Function to convert string to enum
GeniePipeline_Priority_t stringToGeniePipelinePriority(const std::string& priority) {
  static const std::map<std::string, GeniePipeline_Priority_t> priorityMap = {
      {"GENIE_PIPELINE_PRIORITY_LOW", GeniePipeline_Priority_t::GENIE_PIPELINE_PRIORITY_LOW},
      {"GENIE_PIPELINE_PRIORITY_NORMAL", GeniePipeline_Priority_t::GENIE_PIPELINE_PRIORITY_NORMAL},
      {"GENIE_PIPELINE_PRIORITY_NORMAL_HIGH",
       GeniePipeline_Priority_t::GENIE_PIPELINE_PRIORITY_NORMAL_HIGH},
      {"GENIE_PIPELINE_PRIORITY_HIGH", GeniePipeline_Priority_t::GENIE_PIPELINE_PRIORITY_HIGH}};

  try {
    return priorityMap.at(priority);
  } catch (const std::out_of_range&) {
    throw std::invalid_argument("Invalid Pipeline Priority value passed: " + priority);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Containers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Context {
  std::map<std::string, std::shared_ptr<genie::Pipeline::Config>> pipelineConfigs;
  std::map<std::string, std::shared_ptr<genie::Pipeline>> pipelines;
  std::map<std::string, std::shared_ptr<genie::Node::Config>> nodeConfigs;
  std::map<std::string, std::shared_ptr<genie::Node>> nodes;
  std::map<std::string, std::shared_ptr<genie::Dialog::Config>> dialogConfigs;
  std::map<std::string, std::shared_ptr<genie::Dialog>> dialogs;
  std::map<std::string, std::shared_ptr<genie::Embedding::Config>> embeddingConfigs;
  std::map<std::string, std::shared_ptr<genie::Embedding>> embeddings;
  std::map<std::string, std::shared_ptr<genie::Accuracy::Config>> accuracyConfigs;
  std::map<std::string, std::shared_ptr<genie::Accuracy>> accuracies;
  std::map<std::string, std::shared_ptr<genie::Dlc::Config>> dlcConfigs;
  std::map<std::string, std::shared_ptr<genie::Dlc>> dlcs;
  std::map<std::string, std::shared_ptr<genie::Profile::Config>> profileConfigs;
  std::map<std::string, std::shared_ptr<genie::Profile>> profiles;
  std::map<std::string, std::shared_ptr<genie::Log>> loggers;
  std::map<std::string, std::shared_ptr<genie::Tokenizer>> tokenizers;
  std::map<std::string, std::shared_ptr<genie::Sampler::Config>> samplerConfigs;
  std::map<std::string, std::shared_ptr<genie::Sampler>> samplers;
  std::map<std::string, std::vector<std::shared_ptr<void>>> data;
  std::vector<std::thread> activeThreads;

  void clear() {
    for (size_t i = 0; i < activeThreads.size(); i++) {
      if (activeThreads[i].joinable()) {
        activeThreads[i].join();
      }
    }
    accuracyConfigs.clear();
    accuracies.clear();
    pipelineConfigs.clear();
    pipelines.clear();
    nodes.clear();
    nodeConfigs.clear();
    dialogConfigs.clear();
    dialogs.clear();
    embeddingConfigs.clear();
    embeddings.clear();
    profiles.clear();
    loggers.clear();
    tokenizers.clear();
    samplerConfigs.clear();
    samplers.clear();
    data.clear();
  }

  void list() {
    if (dialogConfigs.size() > 0) {
      std::cout << "Dialog Configs:" << std::endl;
      for (const auto& item : dialogConfigs) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (dialogs.size() > 0) {
      std::cout << "Dialogs:" << std::endl;
      for (const auto& item : dialogs) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (samplerConfigs.size() > 0) {
      std::cout << "Sampler Configs:" << std::endl;
      for (const auto& item : samplerConfigs) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (samplers.size() > 0) {
      std::cout << "Samplers:" << std::endl;
      for (const auto& item : samplers) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (profiles.size() > 0) {
      std::cout << "Profiles:" << std::endl;
      for (const auto& item : profiles) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (loggers.size() > 0) {
      std::cout << "Loggers:" << std::endl;
      for (const auto& item : loggers) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (tokenizers.size() > 0) {
      std::cout << "Tokenizers:" << std::endl;
      for (const auto& item : tokenizers) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (accuracyConfigs.size() > 0) {
      std::cout << "Accuracy Configs:" << std::endl;
      for (const auto& item : accuracyConfigs) {
        std::cout << "  " << item.first << std::endl;
      }
    }
    if (accuracies.size() > 0) {
      std::cout << "Accuracies:" << std::endl;
      for (const auto& item : accuracies) {
        std::cout << "  " << item.first << std::endl;
      }
    }
  }
};

Context ctx;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Utilities
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> split(const std::string& str, char sep) {
  std::vector<std::string> words;
  std::string word;
  int state = 0;
  for (size_t i = 0; i < str.length(); i++) {
    switch (state) {
      case 0:
        // Normal state
        if (str[i] == sep) {
          if (word != "") {
            words.push_back(word);
            word.clear();
          }
        } else if (str[i] == '"') {
          state = 1;
        } else {
          word.push_back(str[i]);
        }
        break;
      case 1:
        // Inside quotes
        if (str[i] == '"') {
          state = 0;
        } else if (str[i] == '\\') {
          state = 2;
        } else {
          word.push_back(str[i]);
        }
        break;
      case 2:
        // Inside quotes and escaped
        state = 1;
        word.push_back(str[i]);
        break;
    }
  }
  if (word != "") {
    words.push_back(word);
  }

  if (state) {
    RT_ERR("Cannot split string, asymmetric \" found.")
  }

  return words;
}

std::string combine(const std::vector<std::string>& sections, char sep) {
  std::string str;
  for (size_t i = 0; i < sections.size(); i++) {
    if (i) {
      str += sep;
    }
    std::string quotes = (sections[i].find(' ') != std::string::npos) ? ("\"") : ("");
    str += quotes + sections[i] + quotes;
  }
  return str;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Command Classes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Command {
 public:
  virtual void validate(const std::vector<std::string>& cmd) = 0;
  virtual void process(const std::vector<std::string>& cmd)  = 0;
  virtual std::list<std::string> help()                      = 0;
  virtual ~Command() {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Pipeline Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PipelineCommand : public Command {
 public:
  static std::string key() { return "pipeline"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5 && cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "bind") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "create") || (cmd[1] == "save") || (cmd[1] == "restore") ||
               (cmd[1] == "add") || (cmd[1] == "setOemKey")) {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "setPriority") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "reset") || (cmd[1] == "free")) {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "execute") {
      if (cmd.size() != 3 && cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "connect")) {
      if (cmd.size() > 7) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.pipelineConfigs.find(configName);
        if (it != ctx.pipelineConfigs.end()) {
          RT_ERR("Pipeline config already exists: " + configName);
        }
        std::string jsonFile;
        if (cmd.size() == 5) {
          const auto& configPath = cmd[4];
          std::ifstream ifs(configPath);
          if (!ifs.is_open()) {
            INV_ARG("Could not open JSON config file: " + configPath + ".");
          }
          std::getline(ifs, jsonFile, '\0');
        }
        ctx.pipelineConfigs[configName] = std::make_shared<genie::Pipeline::Config>(jsonFile);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.pipelineConfigs.find(configName);
        if (it == ctx.pipelineConfigs.end()) {
          RT_ERR("Pipeline config does not exist: " + configName);
        }
        ctx.pipelineConfigs.erase(configName);
      } else if (cmd[2] == "bind") {
        const auto& configName = cmd[4];
        auto pipelineConfigIt  = ctx.pipelineConfigs.find(configName);
        if (pipelineConfigIt == ctx.pipelineConfigs.end()) {
          RT_ERR("Pipeline config does not exist: " + configName);
        }
        auto pipeline = pipelineConfigIt->second;
        if (cmd[3] == "profile") {
          const auto& profileName = cmd[5];
          auto profileIt          = ctx.profiles.find(profileName);
          if (profileIt == ctx.profiles.end()) {
            RT_ERR("Profile does not exist: " + profileName);
          }
          pipeline->bindProfile(profileIt->second);
        } else if (cmd[3] == "log") {
          const auto& loggerName = cmd[5];
          auto loggerIt          = ctx.loggers.find(loggerName);
          if (loggerIt == ctx.loggers.end()) {
            RT_ERR("Logger does not exist: " + loggerName);
          }
          pipeline->bindLogger(loggerIt->second);
        } else {
          RT_ERR("Unknown binding type: " + cmd[3]);
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& pipelineName = cmd[2];
      auto pipelineIt          = ctx.pipelines.find(pipelineName);
      if (pipelineIt != ctx.pipelines.end()) {
        RT_ERR("Pipeline already exists: " + pipelineName);
      }
      const auto& configName = cmd[3];
      auto configIt          = ctx.pipelineConfigs.find(configName);
      if (configIt == ctx.pipelineConfigs.end()) {
        RT_ERR("Pipeline config does not exist: " + configName);
      }
      ctx.pipelines[pipelineName] = std::make_shared<genie::Pipeline>(*(configIt->second));
    } else if (cmd[1] == "add") {
      const auto& pipelineName = cmd[2];
      auto it                  = ctx.pipelines.find(pipelineName);
      if (it == ctx.pipelines.end()) {
        RT_ERR("Pipeline does not exist: " + pipelineName);
      }
      auto pipeline        = it->second;
      const auto& nodeName = cmd[3];
      auto itr             = ctx.nodes.find(nodeName);
      if (itr == ctx.nodes.end()) {
        RT_ERR("Node does not exist: " + nodeName);
      }
      auto node = itr->second;
      pipeline->addNode(node);
    } else if (cmd[1] == "connect") {
      const auto& pipelineName = cmd[2];
      auto it                  = ctx.pipelines.find(pipelineName);
      if (it == ctx.pipelines.end()) {
        RT_ERR("Pipeline does not exist: " + pipelineName);
      }
      auto pipeline                = it->second;
      const auto& producerNodeName = cmd[3];
      auto itr                     = ctx.nodes.find(producerNodeName);
      if (itr == ctx.nodes.end()) {
        RT_ERR("Node does not exist: " + producerNodeName);
      }
      auto producerNode                 = itr->second;
      const auto& producerIO            = cmd[4];
      GenieNode_IOName_t producerIOEnum = stringToNodeIO(producerIO);
      const auto& consumerNodeName      = cmd[5];
      itr                               = ctx.nodes.find(consumerNodeName);
      if (itr == ctx.nodes.end()) {
        RT_ERR("Node does not exist: " + consumerNodeName);
      }
      auto consumerNode                 = itr->second;
      const auto& consumerIO            = cmd[6];
      GenieNode_IOName_t consumerIOEnum = stringToNodeIO(consumerIO);
      pipeline->connect(producerNode, producerIOEnum, consumerNode, consumerIOEnum);
    } else {
      const auto& pipelineName = cmd[2];
      auto it                  = ctx.pipelines.find(pipelineName);
      if (it == ctx.pipelines.end()) {
        RT_ERR("Pipeline does not exist: " + pipelineName);
      }
      auto pipeline = it->second;
      if (cmd[1] == "free") {
        ctx.pipelines.erase(pipelineName);
      } else if (cmd[1] == "execute") {
        void* userData = nullptr;

        if (cmd.size() == 4) {
          const std::string& outputPath = cmd[3];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }

          static std::atomic<intptr_t> nodeResponseKey(1);
          userData = reinterpret_cast<void*>(nodeResponseKey++);

          g_nodeResponseMap[userData] = std::move(ofs);
        }

        pipeline->execute(userData);

        if (userData != nullptr) {
          g_nodeResponseMap.erase(userData);
        }
      } else if (cmd[1] == "reset") {
        pipeline->reset();
      } else if (cmd[1] == "save") {
        pipeline->save(cmd[3]);
      } else if (cmd[1] == "restore") {
        pipeline->restore(cmd[3]);
      } else if (cmd[1] == "setOemKey") {
        pipeline->setOemKey(cmd[3]);
      } else if (cmd[1] == "setPriority") {
        const auto& engineRole                    = cmd[3];
        const auto& priority                      = cmd[4];
        GeniePipeline_Priority_t priorityEnum     = stringToGeniePipelinePriority(priority);
        pipeline->setPriority(engineRole, priorityEnum);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("pipeline config create       CONFIG_NAME FILE.json");
    strs.push_back("pipeline config bind profile CONFIG_NAME PROFILE_NAME");
    strs.push_back("pipeline config bind log     CONFIG_NAME LOG_NAME");
    strs.push_back("pipeline config free         CONFIG_NAME");
    strs.push_back("pipeline create              PIPELINE_NAME CONFIG_NAME");
    strs.push_back("pipeline add                 PIPELINE_NAME NODE_NAME");
    strs.push_back(
        "pipeline connect             PIPELINE_NAME PRODUCER_NODE_NAME PRODUCER_NODE_IO "
        "CONSUMER_NODE_NAME CONSUMER_NODE_IO");
    strs.push_back("pipeline execute             PIPELINE_NAME [OUTPUT_FILE]");
    strs.push_back("pipeline reset               PIPELINE_NAME");
    strs.push_back("pipeline save                PIPELINE_NAME PATH");
    strs.push_back("pipeline restore             PIPELINE_NAME PATH");
    strs.push_back("pipeline free                PIPELINE_NAME");
    strs.push_back("pipeline setOemKey           PIPELINE_NAME OEM_KEY");
    strs.push_back("pipeline setPriority         PIPELINE_NAME ENGINE_ROLE PRIORITY");
    return strs;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Node Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class NodeCommand : public Command {
 public:
  static std::string key() { return "node"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "createFromDlc") {
        if (!(cmd.size() == 6 || cmd.size() == 7)) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "bind") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "create")) {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "set")) {
      if ((cmd[2] == "text")) {
        if (!(cmd.size() == 6 || cmd.size() == 5)) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "textFile")) {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "image")) {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "embedding")) {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "textCallback")) {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "embeddingCallback")) {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if ((cmd[2] == "data")) {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "free")) {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "reset")) {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "applyLora") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "train") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "saveLora") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "setLoraStrength") {
      // cmd.size() =  4 + 2*i
      if (cmd.size() < 4 || cmd.size() % 2 == 1) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "execute") {
      if (cmd.size() != 3 && cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "getSampler") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "get") {
      if (cmd[2] == "data") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.nodeConfigs.find(configName);
        if (it != ctx.nodeConfigs.end()) {
          RT_ERR("Node config already exists: " + configName);
        }
        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');
        ctx.nodeConfigs[configName] = std::make_shared<genie::Node::Config>(jsonFile);
      } else if (cmd[2] == "createFromDlc") {
        const auto& configName = cmd[3];
        auto it                = ctx.nodeConfigs.find(configName);

        if (it != ctx.nodeConfigs.end()) {
          RT_ERR("Node config already exists: " + configName);
        }

        const auto& dlcName = cmd[4];
        auto dlcIt          = ctx.dlcs.find(dlcName);
        if (dlcIt == ctx.dlcs.end()) {
          RT_ERR("Dlc does not exist: " + dlcName);
        }
        const auto& useCaseName = cmd[5];
        std::string genieConfigPath;
        std::string configStr;
        if (cmd.size() == 7) {
          genieConfigPath = cmd[6];
          std::ifstream ifs(genieConfigPath);
          if (!ifs.is_open()) {
            INV_ARG("Could not open JSON config file: " + genieConfigPath + ".");
          }
          std::getline(ifs, configStr, '\0');
        }
        ctx.nodeConfigs[configName] =
            std::make_shared<genie::Node::Config>(dlcIt->second, useCaseName, configStr);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.nodeConfigs.find(configName);
        if (it == ctx.nodeConfigs.end()) {
          RT_ERR("Node config does not exist: " + configName);
        }
        ctx.nodeConfigs.erase(configName);
      } else if (cmd[2] == "bind") {
        const auto& configName = cmd[4];
        auto nodeConfigIt      = ctx.nodeConfigs.find(configName);
        if (nodeConfigIt == ctx.nodeConfigs.end()) {
          RT_ERR("Node config does not exist: " + configName);
        }
        auto node = nodeConfigIt->second;
        if (cmd[3] == "profile") {
          const auto& profileName = cmd[5];
          auto profileIt          = ctx.profiles.find(profileName);
          if (profileIt == ctx.profiles.end()) {
            RT_ERR("Profile does not exist: " + profileName);
          }
          node->bindProfile(profileIt->second);
        } else if (cmd[3] == "log") {
          const auto& loggerName = cmd[5];
          auto loggerIt          = ctx.loggers.find(loggerName);
          if (loggerIt == ctx.loggers.end()) {
            RT_ERR("Logger does not exist: " + loggerName);
          }
          node->bindLogger(loggerIt->second);
        } else {
          RT_ERR("Unknown binding type: " + cmd[3]);
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& nodeName = cmd[2];
      auto nodeIt          = ctx.nodes.find(nodeName);
      if (nodeIt != ctx.nodes.end()) {
        RT_ERR("Node already exists: " + nodeName);
      }

      const auto& configName = cmd[3];
      auto configIt          = ctx.nodeConfigs.find(configName);
      if (configIt == ctx.nodeConfigs.end()) {
        RT_ERR("Node config does not exist: " + configName);
      }
      ctx.nodes[nodeName] = std::make_shared<genie::Node>(*(configIt->second));
    } else if (cmd[1] == "set") {
      if (cmd[2] == "text") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        auto& textPrompt          = (cmd.size() == 6) ? cmd.at(5) : "";
        ctx.nodes[nodeName]->setData(nodeIO, textPrompt);
      } else if (cmd[2] == "textFile") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        const auto& filePath      = cmd[5];
        std::ifstream file(filePath);
        if (file.is_open() && file.good()) {
          std::ostringstream buffer;
          buffer << file.rdbuf();
          const auto& textPrompt = buffer.str();
          file.close();
          ctx.nodes[nodeName]->setData(nodeIO, textPrompt);
        } else {
          RT_ERR("Text file does not exists: " + filePath);
        }
      } else if (cmd[2] == "image") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        const auto& imagePath     = cmd[5];
        std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
        if (!file) {
          RT_ERR("Image file does not exist: " + imagePath);
        }
        uint32_t fileSize                 = file.tellg();
        std::shared_ptr<void> imageBuffer = std::shared_ptr<void>(new int8_t[fileSize]);
        std::ifstream embeddingStream(imagePath, std::ifstream::binary);
        embeddingStream.read(static_cast<char*>(imageBuffer.get()), fileSize);
        ctx.nodes[nodeName]->setData(nodeIO, imageBuffer.get(), fileSize);
      } else if (cmd[2] == "embedding") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        const auto& embeddingPath = cmd[5];
        std::ifstream file(embeddingPath, std::ios::binary | std::ios::ate);
        if (!file) {
          RT_ERR("Embedding file does not exist: " + embeddingPath);
        }
        uint32_t fileSize                     = file.tellg();
        std::shared_ptr<void> embeddingBuffer = std::shared_ptr<void>(new int8_t[fileSize]);
        std::ifstream embeddingStream(embeddingPath, std::ifstream::binary);
        embeddingStream.read(static_cast<char*>(embeddingBuffer.get()), fileSize);
        ctx.nodes[nodeName]->setData(nodeIO, embeddingBuffer.get(), fileSize);
      } else if (cmd[2] == "textCallback") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        ctx.nodes[nodeName]->setTextCallback(nodeIO, nodeQueryCallback);
      } else if (cmd[2] == "embeddingCallback") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }
        const auto& nodeIOEnum    = cmd[4];
        GenieNode_IOName_t nodeIO = stringToNodeIO(nodeIOEnum);
        ctx.nodes[nodeName]->setEmbeddingCallback(nodeIO, nodeEmbeddingCallback);
      } else if (cmd[2] == "data") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }

        nlohmann::json ioConfigJson = nlohmann::json::parse(ifs);
        if (ioConfigJson.contains("Inputs")) {
          for (auto it = ioConfigJson["Inputs"].begin(); it != ioConfigJson["Inputs"].end(); ++it) {
            std::string ioNameStr     = it.key();
            GenieNode_IOName_t nodeIO = stringToNodeIO(ioNameStr);

            auto& inputVal       = it.value();
            std::string filePath = inputVal["file-path"];

            std::ifstream file(filePath, std::ios::binary | std::ios::ate);
            if (!file) {
              RT_ERR("Data file does not exist: " + filePath);
            }
            uint32_t fileSize = file.tellg();
            // might need to save the buffer as global memory to keep it alive for the whole proces
            // or till explicit destructor is triggered due to persistent buffers.
            std::shared_ptr<void> buffer = std::shared_ptr<void>(new int8_t[fileSize]);
            file.seekg(0, std::ios::beg);
            file.read(static_cast<char*>(buffer.get()), fileSize);
            file.close();

            std::string ioConfigStr = inputVal["io-config"].dump();
            // Add data pointer to map, to keep it alive for execute
            ctx.data[nodeName].push_back(buffer);
            ctx.nodes[nodeName]->setData(nodeIO, buffer.get(), fileSize, ioConfigStr.c_str());
          }
        }
      }
    } else if (cmd[1] == "train") {
      const auto& nodeName = cmd[2];
      ctx.nodes[nodeName]->train();
    } else if (cmd[1] == "saveLora") {
      const auto& nodeName = cmd[2];
      auto engineName      = cmd[3];
      auto adapterName     = cmd[4];
      ctx.nodes[nodeName]->saveLora(engineName, adapterName);
    } else if (cmd[1] == "applyLora") {
      const auto& nodeName = cmd[2];
      auto engineName      = cmd[3];
      auto adapterName     = cmd[4];
      ctx.nodes[nodeName]->applyLora(engineName, adapterName);
    } else if (cmd[1] == "setLoraStrength") {
      const auto& nodeName = cmd[2];
      auto engineName      = cmd[3];
      for (size_t idx = 5; idx < cmd.size(); idx += 2) {
        ctx.nodes[nodeName]->setLoraStrength(engineName, cmd[idx - 1], std::stof(cmd[idx]));
      }
    } else if (cmd[1] == "execute") {
      const auto& nodeName = cmd[2];
      auto nodeIt          = ctx.nodes.find(nodeName);
      if (nodeIt == ctx.nodes.end()) {
        RT_ERR("Node does not exists: " + nodeName);
      }
      std::string executionConfig{};
      if (cmd.size() == 4) {
        const auto& configPath = cmd[3];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        std::getline(ifs, executionConfig, '\0');
      }
      ctx.nodes[nodeName]->execute(executionConfig);
      ctx.data[nodeName].clear();
    } else if (cmd[1] == "get") {
      if (cmd[2] == "data") {
        const auto& nodeName = cmd[3];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exists: " + nodeName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        nlohmann::json outputConfigJson = nlohmann::json::parse(ifs);

        g_nodeOutputDirectory = cmd[5];
        for (auto it = outputConfigJson["Outputs"].begin(); it != outputConfigJson["Outputs"].end();
             ++it) {
          std::string ioNameStr     = it.key();
          GenieNode_IOName_t nodeIO = stringToNodeIO(ioNameStr);

          auto inputVal = it.value();
          std::string ioConfigStr{};
          if (inputVal.contains("io-config")) {
            ioConfigStr = inputVal["io-config"].dump();
          }

          ctx.nodes[nodeName]->getData(nodeIO, ioConfigStr, nodeIOCallback);
        }
      }
    } else if (cmd[1] == "getSampler") {
      const auto& nodeName = cmd[2];
      auto nodeIt          = ctx.nodes.find(nodeName);
      if (nodeIt == ctx.nodes.end()) {
        RT_ERR("Node does not exists: " + nodeName);
      }
      const auto& samplerName = cmd[3];
      ctx.samplers[samplerName] =
          std::make_shared<genie::Sampler>(ctx.nodes[nodeName]->getSampler());
    } else if (cmd[1] == "reset") {
      const auto& nodeName = cmd[2];
      auto nodeIt          = ctx.nodes.find(nodeName);
      if (nodeIt == ctx.nodes.end()) {
        RT_ERR("Node does not exists: " + nodeName);
      }
      ctx.nodes[nodeName]->reset();
    } else {
      const auto& nodeName = cmd[2];
      auto it              = ctx.nodes.find(nodeName);
      if (it == ctx.nodes.end()) {
        RT_ERR("Node does not exist: " + nodeName);
      }
      if (cmd[1] == "free") {
        ctx.nodes.erase(nodeName);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("node config create         CONFIG_NAME FILE.json");
    strs.push_back(
        "node config createFromDlc    CONFIG_NAME DLC_NAME USE_CASE_NAME "
        "[STANDALONE_GENIE_CONFIG]");
    strs.push_back("node config free           CONFIG_NAME");
    strs.push_back("node config bind profile   CONFIG_NAME PROFILE_NAME");
    strs.push_back("node config bind log       CONFIG_NAME LOG_NAME");
    strs.push_back("node create                NODE_NAME CONFIG_NAME");
    strs.push_back("node set text              NODE_NAME NODE_IO TEXT_PROMPT");
    strs.push_back("node set textFile          NODE_NAME NODE_IO TEXT_PROMPT_FILE_PATH");
    strs.push_back("node set image             NODE_NAME NODE_IO IMAGE_PATH");
    strs.push_back("node set embedding         NODE_NAME NODE_IO EMBEDDING_PATH");
    strs.push_back("node set textCallback      NODE_NAME NODE_IO");
    strs.push_back("node set embeddingCallback NODE_NAME");
    strs.push_back("node set data              NODE_NAME INPUT_CONFIG.json");
    strs.push_back("node execute               NODE_NAME EXECUTION_CONFIG.json");
    strs.push_back("node get data              NODE_NAME OUTPUT_CONFIG.json OUTPUT_DIR_NAME");
    strs.push_back("node getSampler            NODE_NAME SAMPLER_NAME");
    strs.push_back("node applyLora             NODE_NAME ENGINE_NAME ADAPTER_NAME");
    strs.push_back("node setLoraStrength       NODE_NAME ENGINE_NAME TENSOR_NAME_1 ALPHA_1 ...");
    strs.push_back("node reset                 NODE_NAME");
    strs.push_back("node free                  NODE_NAME");
    return strs;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dialog Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DialogCommand : public Command {
 public:
  static std::string key() { return "dialog"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "createFromDlc") {
        if (!(cmd.size() == 6 || cmd.size() == 7)) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "bind") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "create") || (cmd[1] == "save") || (cmd[1] == "restore") ||
               (cmd[1] == "setPerformance") || (cmd[1] == "setMaxNumTokens") ||
               (cmd[1] == "getValue") || (cmd[1] == "setStopSequence") ||
               (cmd[1] == "setOemKey")) {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "setPriority") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "reset") || (cmd[1] == "free")) {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "query" || cmd[1] == "rewindQuery") {
      if ((cmd.size() != 4) && (cmd.size() != 5)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "abort" || cmd[1] == "pause") {
      if ((cmd.size() != 3)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "resume") {
      if ((cmd.size() != 3) && (cmd.size() != 4)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "embeddingQuery") {
      if (cmd.size() != 5 && cmd.size() != 6) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "embeddingTokenQuery") {
      if (cmd.size() != 5 && cmd.size() != 6) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "tokenQuery") {
      if (cmd.size() != 4 && cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "applyLora") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "setLoraStrength") {
      // cmd.size() =  4 + 2*i
      if (cmd.size() < 4 || cmd.size() % 2 == 1) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "releaseLoraMemory") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "getTokenizer") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "batchQuery") {
      if (cmd.size() < 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "getSampler") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.dialogConfigs.find(configName);
        if (it != ctx.dialogConfigs.end()) {
          RT_ERR("Dialog config already exists: " + configName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }

        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');

        ctx.dialogConfigs[configName] = std::make_shared<genie::Dialog::Config>(jsonFile);
      } else if (cmd[2] == "createFromDlc") {
        const auto& configName = cmd[3];
        auto it                = ctx.dialogConfigs.find(configName);

        if (it != ctx.dialogConfigs.end()) {
          RT_ERR("Dialog config already exists: " + configName);
        }

        const auto& dlcName = cmd[4];
        auto dlcIt          = ctx.dlcs.find(dlcName);
        if (dlcIt == ctx.dlcs.end()) {
          RT_ERR("Dlc does not exist: " + dlcName);
        }
        const auto& useCaseName = cmd[5];
        std::string genieConfigPath;
        std::string configStr;
        if (cmd.size() == 7) {
          genieConfigPath = cmd[6];
          std::ifstream ifs(genieConfigPath);
          if (!ifs.is_open()) {
            INV_ARG("Could not open JSON config file: " + genieConfigPath + ".");
          }
          std::getline(ifs, configStr, '\0');
        }
        ctx.dialogConfigs[configName] =
            std::make_shared<genie::Dialog::Config>(dlcIt->second, useCaseName, configStr);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.dialogConfigs.find(configName);
        if (it == ctx.dialogConfigs.end()) {
          RT_ERR("Dialog config does not exist: " + configName);
        }
        ctx.dialogConfigs.erase(configName);
      } else if (cmd[2] == "bind") {
        const auto& configName = cmd[4];
        auto dialogConfigIt    = ctx.dialogConfigs.find(configName);
        if (dialogConfigIt == ctx.dialogConfigs.end()) {
          RT_ERR("Dialog config does not exist: " + configName);
        }
        auto dialog = dialogConfigIt->second;
        if (cmd[3] == "profile") {
          const auto& profileName = cmd[5];
          auto profileIt          = ctx.profiles.find(profileName);
          if (profileIt == ctx.profiles.end()) {
            RT_ERR("Profile does not exist: " + profileName);
          }
          dialog->bindProfile(profileIt->second);
        } else if (cmd[3] == "log") {
          const auto& loggerName = cmd[5];
          auto loggerIt          = ctx.loggers.find(loggerName);
          if (loggerIt == ctx.loggers.end()) {
            RT_ERR("Logger does not exist: " + loggerName);
          }
          dialog->bindLogger(loggerIt->second);
        } else {
          RT_ERR("Unknown binding type: " + cmd[3]);
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& dialogName = cmd[2];
      auto dialogIt          = ctx.dialogs.find(dialogName);
      if (dialogIt != ctx.dialogs.end()) {
        RT_ERR("Dialog already exists: " + dialogName);
      }

      const auto& configName = cmd[3];
      auto configIt          = ctx.dialogConfigs.find(configName);
      if (configIt == ctx.dialogConfigs.end()) {
        RT_ERR("Dialog config does not exist: " + configName);
      }
      ctx.dialogs[dialogName] = std::make_shared<genie::Dialog>(*(configIt->second));
    } else if (cmd[1] == "free") {
      const auto& dialogName = cmd[2];
      auto it                = ctx.dialogs.find(dialogName);
      if (it == ctx.dialogs.end()) {
        RT_ERR("Dialog does not exist: " + dialogName);
      }
      ctx.dialogs.erase(dialogName);
    } else {
      const auto& dialogName = cmd[2];
      auto it                = ctx.dialogs.find(dialogName);
      if (it == ctx.dialogs.end()) {
        RT_ERR("Dialog does not exist: " + dialogName);
      }
      auto dialog = it->second;
      if (cmd[1] == "query" || cmd[1] == "rewindQuery") {
        GenieDialog_SentenceCode_t sentenceCode =
            (cmd[1] == "rewindQuery") ? GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_REWIND
                                      : GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE;
        std::string prompt;
        std::ifstream ifs(cmd[3]);
        if (ifs.is_open()) {
          std::getline(ifs, prompt, '\0');
        } else {
          prompt = cmd[3];
        }
        const void* userData = nullptr;
        if (cmd.size() == 5) {
          std::string outputPath = cmd[4];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }
          static std::atomic<intptr_t> queryResponseKey(1);
          userData                     = reinterpret_cast<const void*>(queryResponseKey++);
          g_queryResponseMap[userData] = std::move(ofs);
          g_primaryConsoleUserData     = userData;
        }
        std::cout << "[PROMPT]: " << prompt << std::endl;
        dialog->query(prompt, sentenceCode, queryCallback, userData);
        g_queryResponseMap.erase(userData);
      } else if (cmd[1] == "abort") {
        GenieDialog_Action_t dialogAction = GENIE_DIALOG_ACTION_ABORT;
        dialog->signal(dialogAction);
      } else if (cmd[1] == "pause") {
        GenieDialog_Action_t dialogAction = GENIE_DIALOG_ACTION_PAUSE;
        dialog->signal(dialogAction);
      } else if (cmd[1] == "resume") {
        const void* userData = nullptr;
        if (cmd.size() == 4) {
          std::string outputPath = cmd[3];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }
          static std::atomic<intptr_t> queryResponseKey(1);
          userData                     = reinterpret_cast<const void*>(queryResponseKey++);
          g_queryResponseMap[userData] = std::move(ofs);
        }
        dialog->query(
            "", GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_RESUME, queryCallback, userData);
      } else if (cmd[1] == "embeddingQuery") {
        std::string embeddingFilename = cmd[3];
        uint32_t embeddingBufferSize  = getFileSize(embeddingFilename);
        std::shared_ptr<void> embeddingBuffer =
            std::shared_ptr<void>(new int8_t[embeddingBufferSize]);
        std::ifstream embeddingStream(embeddingFilename, std::ifstream::binary);
        if (!embeddingStream.good()) {
          RT_ERR("Embedding file doesn't exists or is in bad shape.");
        }
        embeddingStream.read(static_cast<char*>(embeddingBuffer.get()), embeddingBufferSize);

        std::string embeddingTableFilename = cmd[4];
        uint32_t fileSize                  = getFileSize(embeddingTableFilename);
        g_embeddingLut                     = std::shared_ptr<void>(new int8_t[fileSize]);
        g_embeddingLutSize                 = fileSize;
        std::ifstream embeddingTable(embeddingTableFilename, std::ifstream::binary);
        if (!embeddingTable.good()) {
          RT_ERR("embeddingTable File doesn't exists or is in bad shape.");
        }

        embeddingTable.read(static_cast<char*>(g_embeddingLut.get()), fileSize);
        GenieDialog_TokenToEmbeddingCallback_t t2eCallback{nullptr};
        t2eCallback          = tokenToEmbedCallback;
        const void* userData = nullptr;
        bool hasOutputFile   = (cmd.size() == 6);

        if (hasOutputFile) {
          const auto& outputPath = cmd[5];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }
          static std::atomic<intptr_t> queryResponseKey(1);
          userData                     = reinterpret_cast<const void*>(queryResponseKey++);
          g_queryResponseMap[userData] = std::move(ofs);
        }
        if (embeddingBufferSize != 0) {
          dialog->embeddingQuery(
              embeddingBuffer.get(), embeddingBufferSize, t2eCallback, queryCallback, userData);
          if (hasOutputFile) {
            g_queryResponseMap.erase(userData);
          }
        } else {
          RT_ERR("Invalid embedding input.")
        }
      } else if (cmd[1] == "embeddingTokenQuery") {
        std::string embeddingFilename = cmd[3];
        uint32_t embeddingBufferSize  = getFileSize(embeddingFilename);
        std::shared_ptr<void> embeddingBuffer =
            std::shared_ptr<void>(new int8_t[embeddingBufferSize]);
        std::ifstream embeddingStream(embeddingFilename, std::ifstream::binary);
        if (!embeddingStream.good()) {
          RT_ERR("Embedding file doesn't exists or is in bad shape.");
        }
        embeddingStream.read(static_cast<char*>(embeddingBuffer.get()), embeddingBufferSize);

        std::string embeddingTableFilename = cmd[4];
        uint32_t fileSize                  = getFileSize(embeddingTableFilename);
        g_embeddingLut                     = std::shared_ptr<void>(new int8_t[fileSize]);
        g_embeddingLutSize                 = fileSize;
        std::ifstream embeddingTable(embeddingTableFilename, std::ifstream::binary);
        if (!embeddingTable.good()) {
          RT_ERR("embeddingTable File doesn't exists or is in bad shape.");
        }

        embeddingTable.read(static_cast<char*>(g_embeddingLut.get()), fileSize);
        GenieDialog_TokenToEmbeddingCallback_t t2eCallback{nullptr};
        t2eCallback          = tokenToEmbedCallback;
        const void* userData = nullptr;
        bool hasOutputFile   = (cmd.size() == 6);

        if (hasOutputFile) {
          const auto& outputPath = cmd[5];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }
          static std::atomic<intptr_t> queryResponseKey(1);
          userData                     = reinterpret_cast<const void*>(queryResponseKey++);
          g_queryResponseMap[userData] = std::move(ofs);
        }

        if (embeddingBufferSize != 0) {
          dialog->embeddingTokenQuery(embeddingBuffer.get(),
                                      embeddingBufferSize,
                                      t2eCallback,
                                      tokenQueryCallback,
                                      userData);
          if (hasOutputFile) {
            g_queryResponseMap.erase(userData);
          }
        } else {
          RT_ERR("Invalid embedding input.")
        }
      } else if (cmd[1] == "tokenQuery") {
        std::string tokenFileName = cmd[3];
        std::ifstream tokenStream(tokenFileName, std::ifstream::binary);
        if (!tokenStream.good()) {
          RT_ERR("Token file doesn't exists or is in bad shape.");
        }
        std::vector<uint32_t> tokens;
        std::string prompt{};
        while (std::getline(tokenStream, prompt)) {
          std::istringstream iss(prompt);
          uint32_t token;
          while (iss >> token) {
            tokens.push_back(token);
          }
        }
        uint32_t totalTokens = static_cast<uint32_t>(tokens.size());
        const void* userData = nullptr;

        if (cmd.size() == 5) {
          const auto& outputPath = cmd[4];
          std::ofstream ofs(outputPath);
          if (!ofs.is_open()) {
            RT_ERR("Could not open output file: " + outputPath);
          }
          static std::atomic<intptr_t> queryResponseKey(1);
          userData                     = reinterpret_cast<const void*>(queryResponseKey++);
          g_queryResponseMap[userData] = std::move(ofs);
        }

        if (totalTokens != 0) {
          dialog->tokenQuery(static_cast<uint32_t*>(tokens.data()),
                             totalTokens,
                             GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                             tokenQueryCallback,
                             userData);

          if (cmd.size() == 5) {
            g_queryResponseMap.erase(userData);
          }

        } else {
          RT_ERR("Invalid token input.")
        }
      } else if (cmd[1] == "batchQuery") {
        std::vector<std::string> promptStorage;
        std::vector<const char*> promptPtrs;
        std::string outputDir;

        size_t endIdx = cmd.size();
        // Check if last argument is a directory to use as OUTPUT_DIR
        if (cmd.size() > 3) {
          const std::string& lastArg = cmd.back();
          bool isDir                 = false;
#ifdef _WIN32
          struct _stat info;
          if (_stat(lastArg.c_str(), &info) == 0) {
            if (info.st_mode & _S_IFDIR) {
              isDir = true;
            }
#else
          struct stat info;
          if (stat(lastArg.c_str(), &info) == 0) {
            if (info.st_mode & S_IFDIR) {
              isDir = true;
            }
#endif
          } else {
            // If stat failed, it doesn't exist.
            // Check if it is NOT a file that can be opened (as a prompt).
            std::ifstream ifs(lastArg);
            if (!ifs.is_open()) {
              // Not a file we can read.
              // Heuristic: If it contains path separators, assume it's an output directory path.
              if (lastArg.find('/') != std::string::npos ||
                  lastArg.find('\\') != std::string::npos) {
                isDir = true;
// Try to create the directory
#ifdef _WIN32
                if (_mkdir(lastArg.c_str()) != 0 && errno != EEXIST) {
                  std::cerr << "[Genie] Warning: Failed to create output directory: " << lastArg
                            << std::endl;
                }
#else
                if (mkdir(lastArg.c_str(), 0777) != 0 && errno != EEXIST) {
                  std::cerr << "[Genie] Warning: Failed to create output directory: " << lastArg
                            << std::endl;
                }
#endif
              }
            }
          }

          if (isDir) {
            outputDir = lastArg;
            endIdx--;
            std::cout << "[Genie] Output directory set to: " << outputDir << std::endl;
          }
        }

        for (size_t i = 3; i < endIdx; ++i) {
          std::string currentPrompt;
          std::ifstream ifs(cmd[i]);
          if (ifs.is_open()) {
            std::getline(ifs, currentPrompt, '\0');
          } else {
            std::cout << "[Genie] Warning: " << cmd[i] << " is not a file, using as raw string."
                      << std::endl;
            currentPrompt = cmd[i];
          }

          promptStorage.push_back(std::move(currentPrompt));
          const std::string& lastPrompt = promptStorage.back();
          promptPtrs.push_back(lastPrompt.c_str());
          std::cout << "[BATCH PROMPT]: " << lastPrompt.substr(0, 50)
                    << (lastPrompt.length() > 50 ? "..." : "") << std::endl;
        }

        std::vector<const void*> userDataVec;
        for (size_t i = 0; i < promptPtrs.size(); ++i) {
          static std::atomic<intptr_t> queryResponseKey(1);
          const void* userData = reinterpret_cast<const void*>(queryResponseKey++);
          userDataVec.push_back(userData);

          if (!outputDir.empty()) {
            std::string filename = outputDir + "/batch_" + std::to_string(i) + "_output.txt";
            std::ofstream ofs(filename);
            if (!ofs.is_open()) {
              RT_ERR("Could not open output file: " + filename);
            }
            g_queryResponseMap[userData] = std::move(ofs);
          }
        }

        if (outputDir.empty() && !userDataVec.empty()) {
          g_primaryConsoleUserData = userDataVec[0];
        }

        std::cout << "[Genie] Sending " << promptPtrs.size() << " prompts to engine..."
                  << std::endl;
        dialog->batchQuery(promptPtrs,
                           GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                           queryCallback,
                           userDataVec.data());

        for (const auto& userData : userDataVec) {
          g_queryResponseMap.erase(userData);
          g_queryBufferMap.erase(userData);
        }
      } else if (cmd[1] == "applyLora") {
        auto engineName  = cmd[3];
        auto adapterName = cmd[4];
        dialog->applyLora(engineName, adapterName);
      } else if (cmd[1] == "setLoraStrength") {
        auto engineName = cmd[3];
        for (size_t idx = 5; idx < cmd.size(); idx += 2) {
          dialog->setLoraStrength(engineName, cmd[idx - 1], std::stof(cmd[idx]));
        }
      } else if (cmd[1] == "releaseLoraMemory") {
        auto engineName  = cmd[3];
        auto adapterName = cmd[4];
        dialog->releaseLoraMemory(engineName, adapterName);
      } else if (cmd[1] == "reset") {
        dialog->reset();
      } else if (cmd[1] == "save") {
        dialog->save(cmd[3]);
      } else if (cmd[1] == "restore") {
        dialog->restore(cmd[3]);
      } else if (cmd[1] == "setPerformance") {
        const auto& performance                   = cmd[3];
        Genie_PerformancePolicy_t performanceEnum = stringToGeniePerformance(performance);
        dialog->setPerformancePolicy(performanceEnum);
      } else if (cmd[1] == "getTokenizer") {
        const auto& tokenizerName     = cmd[3];
        ctx.tokenizers[tokenizerName] = std::make_shared<genie::Tokenizer>(dialog->getTokenizer());
      } else if (cmd[1] == "getSampler") {
        const auto& samplerName   = cmd[3];
        ctx.samplers[samplerName] = std::make_shared<genie::Sampler>(dialog->getSampler());
      } else if (cmd[1] == "setStopSequence") {
        std::string stopSequences;
        std::ifstream file(cmd[3]);
        if (file.is_open() && file.good()) {
          std::ostringstream buffer;
          buffer << file.rdbuf();
          stopSequences = buffer.str();
          file.close();
        } else {
          stopSequences = cmd[3];
        }
        dialog->setStopSequence(stopSequences);
      } else if (cmd[1] == "setMaxNumTokens") {
        const uint32_t maxNumTokens = static_cast<uint32_t>(std::stoi(cmd[3]));
        dialog->setMaxNumTokens(maxNumTokens);
      } else if (cmd[1] == "setOemKey") {
        dialog->setOemKey(cmd[3]);
      } else if (cmd[1] == "setPriority") {
        const auto& engineRole                = cmd[3];
        const auto& priority                  = cmd[4];
        GenieDialog_Priority_t priorityEnum   = stringToGeniePriority(priority);
        dialog->setPriority(engineRole, priorityEnum);
      } else if (cmd[1] == "getValue") {
        const auto& keyStr = cmd[3];
        GenieDialog_Param_t key;

        // Map string key to enum
        if (keyStr == "GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY") {
          key = GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY;
        } else if (keyStr == "GENIE_DIALOG_PARAM_APPLIED_LORA_ADAPTER") {
          key = GENIE_DIALOG_PARAM_APPLIED_LORA_ADAPTER;
        } else {
          std::cerr << "ERROR: Unknown parameter key: " << keyStr << std::endl;
          return;
        }

        const Genie_AllocCallback_t callback = [](size_t size, const char** data) {
          *data = new char[size];
        };

        try {
          auto result = dialog->getValue(key, callback);

          std::cout << "Parameter: ";
          switch (key) {
            case GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY:
              std::cout << "Context Occupancy: ";
              break;
            case GENIE_DIALOG_PARAM_APPLIED_LORA_ADAPTER:
              std::cout << "Applied LoRA Adapter: ";
              break;
            default:
              std::cout << "unknown: ";
              break;
          }

          switch (result.first) {
            case GENIE_DATATYPE_INT_32:
              std::cout << result.second.int32Value << " (int32)" << std::endl;
              break;
            case GENIE_DATATYPE_UINT_32:
              std::cout << result.second.uint32Value << " (uint32)" << std::endl;
              break;
            case GENIE_DATATYPE_UINT_64:
              std::cout << result.second.uint64Value << " (uint64)" << std::endl;
              break;
            case GENIE_DATATYPE_FLOAT_32:
              std::cout << result.second.floatValue << " (float)" << std::endl;
              break;
            case GENIE_DATATYPE_STRING:
              if (result.second.stringValue) {
                std::string str(result.second.stringValue);
                std::cout << str << " (string)" << std::endl;
                delete[] const_cast<char*>(result.second.stringValue);
              } else {
                std::cout << "(null string)" << std::endl;
              }
              break;
            default:
              std::cout << "unknown data type" << std::endl;
              break;
          }
        } catch (const genie::Exception& e) {
          std::cerr << "Error: " << e.what() << std::endl;
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("dialog config create       CONFIG_NAME FILE.json");
    strs.push_back(
        "dialog config createFromDlc    CONFIG_NAME DLC_NAME USE_CASE_NAME "
        "[STANDALONE_GENIE_CONFIG]");
    strs.push_back("dialog config bind profile CONFIG_NAME PROFILE_NAME");
    strs.push_back("dialog config bind log     CONFIG_NAME LOG_NAME");
    strs.push_back("dialog config free         CONFIG_NAME");
    strs.push_back("dialog create              DIALOG_NAME CONFIG_NAME");
    strs.push_back(
        "dialog query               DIALOG_NAME [\"PROMPT_STR\"/PROMPT_FILE] [OUTPUT_FILE]");
    strs.push_back(
        "dialog rewindQuery         DIALOG_NAME [\"PROMPT_STR\"/PROMPT_FILE] [OUTPUT_FILE]");
    strs.push_back(
        "dialog batchQuery     DIALOG_NAME [\"PROMPT_STR\"/PROMPT_FILE] "
        "[\"PROMPT_STR\"/PROMPT_FILE]... [OUTPUT_DIR]");
    strs.push_back(
        "dialog embeddingQuery      DIALOG_NAME EMBEDDINGS_PATH EMBEDDINGS_TABLE_PATH "
        "[OUTPUT_FILE]");
    strs.push_back(
        "dialog embeddingTokenQuery DIALOG_NAME EMBEDDINGS_PATH EMBEDDINGS_TABLE_PATH "
        "[OUTPUT_FILE]");
    strs.push_back("dialog tokenQuery          DIALOG_NAME TOKENS_PATH [OUTPUT_FILE]");
    strs.push_back("dialog applyLora           DIALOG_NAME ENGINE_NAME ADAPTER_NAME");
    strs.push_back("dialog setLoraStrength     DIALOG_NAME ENGINE_NAME TENSOR_NAME_1 ALPHA_1 ...");
    strs.push_back("dialog reset               DIALOG_NAME");
    strs.push_back("dialog save                DIALOG_NAME PATH");
    strs.push_back("dialog restore             DIALOG_NAME PATH");
    strs.push_back("dialog setPerformance      DIALOG_NAME PERFORMANCE_MODE");
    strs.push_back("dialog free                DIALOG_NAME");
    strs.push_back("dialog getValue            DIALOG_NAME GENIE_DIALOG_PARAM");
    strs.push_back("dialog getTokenizer        DIALOG_NAME TOKENIZER_NAME");
    strs.push_back("dialog getSampler          DIALOG_NAME SAMPLER_NAME");
    strs.push_back(
        "dialog setStopSequence     DIALOG_NAME "
        "[\"STOP_SEQUENCE_JSON_STR\"/STOP_SEQUENCE_JSON_FILE]");
    strs.push_back("dialog setMaxNumTokens     DIALOG_NAME MAX_NUM_TOKENS");
    strs.push_back("dialog setOemKey           DIALOG_NAME OEM_KEY");
    strs.push_back("dialog setPriority         DIALOG_NAME ENGINE_ROLE PRIORITY");
    return strs;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Embedding Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class EmbeddingCommand : public Command {
 public:
  static std::string key() { return "embedding"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "createFromDlc") {
        if (!(cmd.size() == 6 || cmd.size() == 7)) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "bind") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if ((cmd[1] == "create") || (cmd[1] == "setPerformance")) {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "generate") {
      if (cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.embeddingConfigs.find(configName);
        if (it != ctx.embeddingConfigs.end()) {
          RT_ERR("Embedding config already exists: " + configName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');
        ctx.embeddingConfigs[configName] = std::make_shared<genie::Embedding::Config>(jsonFile);
      } else if (cmd[2] == "createFromDlc") {
        const auto& configName = cmd[3];
        auto it                = ctx.embeddingConfigs.find(configName);

        if (it != ctx.embeddingConfigs.end()) {
          RT_ERR("Embedding config already exists: " + configName);
        }

        const auto& dlcName = cmd[4];
        auto dlcIt          = ctx.dlcs.find(dlcName);
        if (dlcIt == ctx.dlcs.end()) {
          RT_ERR("Dlc does not exist: " + dlcName);
        }
        const auto& useCaseName = cmd[5];
        std::string genieConfigPath;
        std::string configStr;
        if (cmd.size() == 7) {
          genieConfigPath = cmd[6];
          std::ifstream ifs(genieConfigPath);
          if (!ifs.is_open()) {
            INV_ARG("Could not open JSON config file: " + genieConfigPath + ".");
          }
          std::getline(ifs, configStr, '\0');
        }
        ctx.embeddingConfigs[configName] =
            std::make_shared<genie::Embedding::Config>(dlcIt->second, useCaseName, configStr);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.embeddingConfigs.find(configName);
        if (it == ctx.embeddingConfigs.end()) {
          RT_ERR("Embedding config does not exist: " + configName);
        }
        ctx.embeddingConfigs.erase(configName);
      } else if (cmd[2] == "bind") {
        const auto& configName = cmd[4];
        auto embeddingConfigIt = ctx.embeddingConfigs.find(configName);
        if (embeddingConfigIt == ctx.embeddingConfigs.end()) {
          RT_ERR("Embedding config does not exist: " + configName);
        }
        auto embedding = embeddingConfigIt->second;
        if (cmd[3] == "profile") {
          const auto& profileName = cmd[5];
          auto profileIt          = ctx.profiles.find(profileName);
          if (profileIt == ctx.profiles.end()) {
            RT_ERR("Profile does not exist: " + profileName);
          }
          embedding->bindProfile(profileIt->second);
        } else if (cmd[3] == "log") {
          const auto& loggerName = cmd[5];
          auto loggerIt          = ctx.loggers.find(loggerName);
          if (loggerIt == ctx.loggers.end()) {
            RT_ERR("Logger does not exist: " + loggerName);
          }
          embedding->bindLogger(loggerIt->second);
        } else {
          RT_ERR("Unknown binding type: " + cmd[3]);
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& embeddingName = cmd[2];
      auto embeddingIt          = ctx.embeddings.find(embeddingName);
      if (embeddingIt != ctx.embeddings.end()) {
        RT_ERR("Embedding already exists: " + embeddingName);
      }

      const auto& configName = cmd[3];
      auto configIt          = ctx.embeddingConfigs.find(configName);
      if (configIt == ctx.embeddingConfigs.end()) {
        RT_ERR("Embedding config does not exist: " + configName);
      }
      ctx.embeddings[embeddingName] = std::make_shared<genie::Embedding>(*(configIt->second));
    } else if (cmd[1] == "free") {
      const auto& embeddingName = cmd[2];
      auto it                   = ctx.embeddings.find(embeddingName);
      if (it == ctx.embeddings.end()) {
        RT_ERR("Embedding does not exist: " + embeddingName);
      }
      ctx.embeddings.erase(embeddingName);
    } else {
      const auto& embeddingName = cmd[2];
      auto it                   = ctx.embeddings.find(embeddingName);
      if (it == ctx.embeddings.end()) {
        RT_ERR("Embedding does not exist: " + embeddingName);
      }
      auto embedding = it->second;
      if (cmd[1] == "generate") {
        std::string prompt;
        std::ifstream ifs(cmd[3]);
        if (ifs.is_open()) {
          std::getline(ifs, prompt, '\0');
        } else {
          prompt = cmd[3];
        }

        g_embeddingOutputFile = cmd[4];
        std::ofstream ofs(g_embeddingOutputFile);
        if (!ofs.is_open()) {
          RT_ERR("Could not open output file: " + g_embeddingOutputFile);
        }

        std::cout << "[PROMPT]: " << prompt << std::endl;
        embedding->generate(prompt, embeddingCallback, nullptr);
      } else if (cmd[1] == "setPerformance") {
        const auto& performance                   = cmd[3];
        Genie_PerformancePolicy_t performanceEnum = stringToGeniePerformance(performance);
        embedding->setPerformancePolicy(performanceEnum);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("embedding config create       CONFIG_NAME FILE.json");
    strs.push_back(
        "embedding config createFromDlc    CONFIG_NAME DLC_NAME USE_CASE_NAME "
        "[STANDALONE_GENIE_CONFIG]");
    strs.push_back("embedding config bind profile CONFIG_NAME PROFILE_NAME");
    strs.push_back("embedding config bind log     CONFIG_NAME LOG_NAME");
    strs.push_back("embedding config free         CONFIG_NAME");
    strs.push_back("embedding create              EMBEDDING_NAME CONFIG_NAME");
    strs.push_back("embedding generate  EMBEDDING_NAME [\"PROMPT_STR\"/PROMPT_FILE] [OUTPUT_FILE]");
    strs.push_back("embedding setPerformance      EMBEDDING_NAME PERFORMANCE_MODE");
    strs.push_back("embedding free                EMBEDDING_NAME");
    return strs;
  }
};

class TokenizerCommand : public Command {
 public:
  static std::string key() { return "tokenizer"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "encode") {
      if (cmd.size() != 4 && cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "encodeFromFile") {
      if (cmd.size() != 4 && cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "decode") {
      if (cmd.size() < 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
      size_t numTokens = static_cast<size_t>(std::stoi(cmd[3]));
      if (cmd.size() < 4 + numTokens || cmd.size() > 5 + numTokens) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "decodeFromFile") {
      if (cmd.size() != 4 && cmd.size() != 5) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "encode") {
      const auto& tokenizerName = cmd[2];
      auto it                   = ctx.tokenizers.find(tokenizerName);
      if (it == ctx.tokenizers.end()) {
        RT_ERR("Tokenizer does not exist: " + tokenizerName);
      }
      std::string inputStr = cmd[3];
      const int32_t* tokenIds;
      uint32_t numTokenIds = 0;
      it->second->encode(inputStr, &tokenIds, &numTokenIds);

      std::cout << "[Input Text]: " << inputStr << std::endl;
      std::cout << "[Encoded Tokens Size]: " << numTokenIds << std::endl;
      bool hasOutputFile  = (cmd.size() == 5);
      std::string outPath = hasOutputFile ? cmd[4] : "";
      std::ofstream ofs;

      if (hasOutputFile) {
        ofs.open(outPath);
        if (!ofs.is_open() || !ofs.good()) {
          RT_ERR("Could not open output file: " + outPath);
        }
      }

      std::ostream& os = hasOutputFile ? ofs : std::cout;

      if (!hasOutputFile) {
        os << "[Encoded Tokens]:";
      }

      for (size_t idx = 0; idx < numTokenIds; idx++) {
        os << ((idx > 0) ? " " : "") << tokenIds[idx];
      }
      os << std::endl;

      if (hasOutputFile) {
        std::cout << "[Encoded Tokens] saved in: " << outPath << std::endl;
      }
    } else if (cmd[1] == "encodeFromFile") {
      const auto& tokenizerName = cmd[2];
      auto it                   = ctx.tokenizers.find(tokenizerName);
      if (it == ctx.tokenizers.end()) {
        RT_ERR("Tokenizer does not exist: " + tokenizerName);
      }

      const auto& filePath = cmd[3];
      std::ifstream ifs(filePath);
      if (!ifs.is_open() || !ifs.good()) {
        RT_ERR("Could not open input text file: " + filePath);
      }

      std::string inputStr;
      std::getline(ifs, inputStr, '\0');

      const int32_t* tokenIds;
      uint32_t numTokenIds = 0;

      it->second->encode(inputStr, &tokenIds, &numTokenIds);
      std::cout << "[Input File]: " << filePath << std::endl;
      std::cout << "[Input Text]: " << inputStr << std::endl;
      std::cout << "[Encoded Tokens Size]: " << numTokenIds << std::endl;

      bool hasOutputFile  = (cmd.size() == 5);
      std::string outPath = hasOutputFile ? cmd[4] : "";
      std::ofstream ofs;

      if (hasOutputFile) {
        ofs.open(outPath);
        if (!ofs.is_open() || !ofs.good()) {
          RT_ERR("Could not open output file: " + outPath);
        }
      }

      std::ostream& os = hasOutputFile ? ofs : std::cout;

      if (!hasOutputFile) {
        os << "[Encoded Tokens]:";
      }

      for (size_t idx = 0; idx < numTokenIds; idx++) {
        os << ((idx > 0) ? " " : "") << tokenIds[idx];
      }
      os << std::endl;

      if (hasOutputFile) {
        std::cout << "[Encoded Tokens] saved in: " << outPath << std::endl;
      }

    } else if (cmd[1] == "decode") {
      const auto& tokenizerName = cmd[2];
      auto it                   = ctx.tokenizers.find(tokenizerName);
      if (it == ctx.tokenizers.end()) {
        RT_ERR("Tokenizer does not exist: " + tokenizerName);
      }

      uint32_t numTokens = static_cast<uint32_t>(std::stoi(cmd[3]));
      bool hasOutputFile = (cmd.size() > 4 + numTokens);

      size_t tokensStart = 4;
      size_t tokensEnd   = hasOutputFile ? (cmd.size() - 1) : cmd.size();

      std::vector<int32_t> tokenVec;
      tokenVec.reserve(numTokens);
      for (size_t idx = tokensStart; idx < tokensEnd; ++idx) {
        tokenVec.push_back(std::stoi(cmd[idx]));
      }

      const auto outStr =
          it->second->decode(tokenVec.data(), static_cast<uint32_t>(tokenVec.size()));
      std::cout << "[Input Tokens]:";
      for (const auto& tokenIt : tokenVec) {
        std::cout << " " << tokenIt;
      }
      std::cout << std::endl;

      std::string outPath = hasOutputFile ? cmd.back() : "";
      std::ofstream ofs;

      if (hasOutputFile) {
        ofs.open(outPath);
        if (!ofs.is_open() || !ofs.good()) {
          RT_ERR("Could not open output file: " + outPath);
        }
      }

      std::ostream& os = hasOutputFile ? ofs : std::cout;

      if (!hasOutputFile) {
        os << "[Decoded Text]: ";
      }
      os << outStr << std::endl;

      if (hasOutputFile) {
        std::cout << "[Decoded Text] saved in: " << outPath << std::endl;
      }
    } else if (cmd[1] == "decodeFromFile") {
      const auto& tokenizerName = cmd[2];
      auto it                   = ctx.tokenizers.find(tokenizerName);
      if (it == ctx.tokenizers.end()) {
        RT_ERR("Tokenizer does not exist: " + tokenizerName);
      }

      const auto& filePath = cmd[3];
      std::ifstream ifs(filePath);
      if (!ifs.is_open() || !ifs.good()) {
        RT_ERR("Could not open token file: " + filePath);
      }

      std::vector<int32_t> tokenVec;
      std::string line;
      while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        int32_t token;
        while (iss >> token) {
          tokenVec.push_back(token);
        }
      }

      if (tokenVec.empty()) {
        RT_ERR("Token file is empty or contains no valid tokens: " + filePath);
      }

      const auto outStr =
          it->second->decode(tokenVec.data(), static_cast<uint32_t>(tokenVec.size()));

      std::cout << "[Input File]: " << filePath << std::endl;
      std::cout << "[Input Tokens]:";
      for (const auto& tokenIt : tokenVec) {
        std::cout << " " << tokenIt;
      }
      std::cout << std::endl;
      bool hasOutputFile  = (cmd.size() == 5);
      std::string outPath = hasOutputFile ? cmd[4] : "";
      std::ofstream ofs;

      if (hasOutputFile) {
        ofs.open(outPath);
        if (!ofs.is_open() || !ofs.good()) {
          RT_ERR("Could not open output file: " + outPath);
        }
      }

      std::ostream& os = hasOutputFile ? ofs : std::cout;

      if (!hasOutputFile) {
        os << "[Decoded Text]: ";
      }
      os << outStr << std::endl;

      if (hasOutputFile) {
        std::cout << "[Decoded Text] saved in: " << outPath << std::endl;
      }
    }

    else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("tokenizer encode         TOKENIZER_NAME INPUT_STRING [OUTPUT_FILE]");
    strs.push_back("tokenizer encodeFromFile TOKENIZER_NAME INPUT_FILE [OUTPUT_FILE]");
    strs.push_back(
        "tokenizer decode         TOKENIZER_NAME TOKEN_SIZE TOKEN_ID_1 ... [OUTPUT_FILE]");
    strs.push_back("tokenizer decodeFromFile TOKENIZER_NAME TOKENS_FILE [OUTPUT_FILE]");
    return strs;
  }
};

class AccuracyCommand : public Command {
 public:
  static std::string key() { return "accuracy"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      if ((cmd.size() != 6)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "compute") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.accuracyConfigs.find(configName);
        if (it != ctx.accuracyConfigs.end()) {
          RT_ERR("Accuracy config already exists: " + configName);
        }
        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');
        ctx.accuracyConfigs[configName] = std::make_shared<genie::Accuracy::Config>(jsonFile);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.accuracyConfigs.find(configName);
        if (it == ctx.accuracyConfigs.end()) {
          RT_ERR("Accuracy config does not exist: " + configName);
        }
        ctx.accuracyConfigs.erase(configName);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& accuracyName = cmd[2];
      auto it                  = ctx.accuracies.find(accuracyName);
      if (it != ctx.accuracies.end()) {
        RT_ERR("Accuracy object already exists: " + accuracyName);
      }
      const auto& configName = cmd[3];
      auto configIt          = ctx.accuracyConfigs.find(configName);
      if (configIt == ctx.accuracyConfigs.end()) {
        RT_ERR("Accuracy config does not exist: " + configName);
      }
      std::shared_ptr<genie::Accuracy::Config> accuracyConfig = configIt->second;
      if (cmd[4] == "dialog") {
        const auto& dialogName = cmd[5];
        auto dialogIt          = ctx.dialogs.find(dialogName);
        if (dialogIt == ctx.dialogs.end()) {
          RT_ERR("Dialog does not exist: " + dialogName);
        }
        std::shared_ptr<genie::Dialog> dialogObj = dialogIt->second;
        ctx.accuracies[accuracyName] =
            std::make_shared<genie::Accuracy>(*accuracyConfig, dialogObj);
      } else if (cmd[4] == "node") {
        const auto& nodeName = cmd[5];
        auto nodeIt          = ctx.nodes.find(nodeName);
        if (nodeIt == ctx.nodes.end()) {
          RT_ERR("Node does not exist: " + nodeName);
        }
        std::shared_ptr<genie::Node> nodeObj = nodeIt->second;
        ctx.accuracies[accuracyName] = std::make_shared<genie::Accuracy>(*accuracyConfig, nodeObj);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      const auto& accuracyName = cmd[2];
      auto it                  = ctx.accuracies.find(accuracyName);
      if (it == ctx.accuracies.end()) {
        RT_ERR("Accuracy does not exist: " + accuracyName);
      }
      ctx.accuracies.erase(accuracyName);
    } else if (cmd[1] == "compute") {
      const auto& accuracyName = cmd[2];
      auto it                  = ctx.accuracies.find(accuracyName);
      if (it == ctx.accuracies.end()) {
        RT_ERR("Accuracy does not exist: " + accuracyName);
      }
      const auto& accuracyPath = cmd[3];
      std::ofstream outFile(accuracyPath);
      if (!outFile.good()) {
        RT_ERR("Cannot create accuracy output file with name:" + accuracyPath);
      }
      outFile << it->second->getJsonData();
      outFile.close();
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("accuracy config create CONFIG_NAME FILE.json");
    strs.push_back("accuracy config free   CONFIG_NAME");
    strs.push_back("accuracy create        ACCURACY_NAME CONFIG_NAME [dialog/node] DIALOG_NAME");
    strs.push_back("accuracy compute       ACCURACY_NAME OUTPUT_FILE.json");
    strs.push_back("accuracy free          ACCURACY_NAME");
    return strs;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Sampler Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SamplerCommand : public Command {
 public:
  static std::string key() { return "sampler"; }

  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "setParam") {
        if (cmd.size() != 6) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "applyConfig") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "sampleData") {
      if (cmd.size() != 5 && cmd.size() != 6) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  // Simple callback to collect sampled tokens
  static void samplerCallback(const uint32_t numTokens,
                              const int32_t* tokens,
                              const void* userData) {
    auto* out = static_cast<std::vector<int32_t>*>(const_cast<void*>(userData));
    if (!out || !tokens) {
      return;
    }
    out->assign(tokens, tokens + numTokens);
  }

  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.samplerConfigs.find(configName);
        if (it != ctx.samplerConfigs.end()) {
          RT_ERR("Sampler config already exists: " + configName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }

        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');

        ctx.samplerConfigs[configName] = std::make_shared<genie::Sampler::Config>(jsonFile);
      } else if (cmd[2] == "setParam") {
        const auto& configName = cmd[3];
        auto it                = ctx.samplerConfigs.find(configName);
        if (it == ctx.samplerConfigs.end()) {
          RT_ERR("Sampler config does not exist: " + configName);
        }
        const auto& key   = cmd[4];
        const auto& value = cmd[5];
        it->second->setParam(key, value);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.samplerConfigs.find(configName);
        if (it == ctx.samplerConfigs.end()) {
          RT_ERR("Sampler config does not exist: " + configName);
        }
        ctx.samplerConfigs.erase(configName);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& samplerName = cmd[2];
      auto samplerIt          = ctx.samplers.find(samplerName);
      if (samplerIt != ctx.samplers.end()) {
        RT_ERR("Sampler already exists: " + samplerName);
      }

      const auto& configName = cmd[3];
      auto configIt          = ctx.samplerConfigs.find(configName);
      if (configIt == ctx.samplerConfigs.end()) {
        RT_ERR("Sampler config does not exist: " + configName);
      }
      ctx.samplers[samplerName] = std::make_shared<genie::Sampler>(*(configIt->second));
    } else if (cmd[1] == "applyConfig") {
      const auto& samplerName = cmd[2];
      auto it                 = ctx.samplers.find(samplerName);
      if (it == ctx.samplers.end()) {
        RT_ERR("Sampler does not exist: " + samplerName);
      }
      const auto& configName = cmd[3];
      auto configIt          = ctx.samplerConfigs.find(configName);
      if (configIt == ctx.samplerConfigs.end()) {
        RT_ERR("Sampler config does not exist: " + configName);
      }
      it->second->applyConfig(*(configIt->second));
    } else if (cmd[1] == "free") {
      const auto& samplerName = cmd[2];
      auto it                 = ctx.samplers.find(samplerName);
      if (it == ctx.samplers.end()) {
        RT_ERR("Sampler does not exist: " + samplerName);
      }
      ctx.samplers.erase(samplerName);
    } else if (cmd[1] == "sampleData") {
      const auto& samplerName = cmd[2];
      auto it                 = ctx.samplers.find(samplerName);
      if (it == ctx.samplers.end()) {
        RT_ERR("Sampler does not exist: " + samplerName);
      }
      auto sampler = it->second;

      const std::string& dataPath       = cmd[3];
      const std::string& dataConfigPath = cmd[4];

      const bool hasOutputFile = (cmd.size() == 6);
      std::string outPath      = hasOutputFile ? cmd[5] : "";
      std::ofstream ofs;
      if (hasOutputFile) {
        ofs.open(outPath);
        if (!ofs.is_open()) {
          RT_ERR("Could not open output file: " + outPath);
        }
      }
      std::ostream& os = hasOutputFile ? ofs : std::cout;

      // Read data config JSON (as raw string for GenieSampler_sampleData)
      std::ifstream dcIfs(dataConfigPath);
      if (!dcIfs.is_open()) {
        RT_ERR("Could not open data config: " + dataConfigPath);
      }
      std::string dataConfigStr;
      std::getline(dcIfs, dataConfigStr, '\0');

      // Read raw data
      std::ifstream dataIfs(dataPath, std::ios::binary | std::ios::ate);
      if (!dataIfs.is_open()) {
        RT_ERR("Could not open data file: " + dataPath);
      }
      std::streamsize rawSize = dataIfs.tellg();
      dataIfs.seekg(0, std::ios::beg);
      std::vector<char> buffer(static_cast<size_t>(rawSize));
      if (!dataIfs.read(buffer.data(), rawSize)) {
        RT_ERR("Failed to read data file: " + dataPath);
      }

      try {
        // Sample using C++ API
        std::vector<int32_t> sampledTokens;
        sampler->sampleData(buffer.data(),
                            static_cast<size_t>(rawSize),
                            dataConfigStr,
                            &SamplerCommand::samplerCallback,
                            &sampledTokens);

        if (!hasOutputFile) {
          os << "[Sample Results]:";
        }
        for (size_t i = 0; i < sampledTokens.size(); ++i) {
          os << " " << sampledTokens[i];
        }
        os << std::endl;

        if (hasOutputFile) {
          std::cout << "[Sample Results] saved in: " << outPath << std::endl;
        }
      } catch (const genie::Exception& e) {
        RT_ERR("Sampler operation failed: " + std::string(e.what()));
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("sampler config create   CONFIG_NAME FILE.json");
    strs.push_back("sampler config setParam CONFIG_NAME KEY VALUE");
    strs.push_back("sampler config free     CONFIG_NAME");
    strs.push_back("sampler create          SAMPLER_NAME CONFIG_NAME");
    strs.push_back("sampler applyConfig     SAMPLER_NAME CONFIG_NAME");
    strs.push_back("sampler sampleData      SAMPLER_NAME DATA.raw DATA_CONFIG.json [OUTPUT_FILE]");
    strs.push_back("sampler free            SAMPLER_NAME");
    return strs;
  }
};

class DlcCommand : public Command {
 public:
  static std::string key() { return "dlc"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "getUseCases") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.dlcConfigs.find(configName);
        if (it != ctx.dlcConfigs.end()) {
          RT_ERR("Dlc config already exists: " + configName);
        }

        const auto& dlcPath        = cmd[4];
        ctx.dlcConfigs[configName] = std::make_shared<genie::Dlc::Config>(dlcPath);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.dlcConfigs.find(configName);
        if (it == ctx.dlcConfigs.end()) {
          RT_ERR("Dlc config does not exist: " + configName);
        }
        ctx.dlcConfigs.erase(configName);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& dlcName = cmd[2];
      auto it             = ctx.dlcs.find(dlcName);
      if (it != ctx.dlcs.end()) {
        RT_ERR("Dlc already exists: " + dlcName);
      }

      const auto& configName = cmd[3];
      auto configIt          = ctx.dlcConfigs.find(configName);
      if (configIt == ctx.dlcConfigs.end()) {
        RT_ERR("Dlc config does not exist: " + configName);
      }
      ctx.dlcs[dlcName] = std::make_shared<genie::Dlc>(*(configIt->second));
    } else if (cmd[1] == "free") {
      const auto& dlcName = cmd[2];
      auto it             = ctx.dlcs.find(dlcName);
      if (it == ctx.dlcs.end()) {
        RT_ERR("Dlc does not exist: " + dlcName);
      }
      ctx.dlcs.erase(dlcName);
    } else if (cmd[1] == "getUseCases") {
      const auto& dlcName = cmd[2];
      auto it             = ctx.dlcs.find(dlcName);
      if (it == ctx.dlcs.end()) {
        RT_ERR("Dlc does not exist: " + dlcName);
      }

      const auto& useCasesPath = cmd[3];
      std::ofstream outFile(useCasesPath);
      if (!outFile.good()) {
        RT_ERR("Cannot create useCases output file with name:" + useCasesPath);
      }
      outFile << it->second->getUseCases();
      outFile.close();
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("dlc config create CONFIG_NAME FILE.dlc");
    strs.push_back("dlc config free   CONFIG_NAME");
    strs.push_back("dlc create        DLC_NAME CONFIG_NAME");
    strs.push_back("dlc getUseCases   DLC_NAME OUTPUT_FILE.json");
    strs.push_back("dlc free          DLC_NAME");
    return strs;
  }
};

class ProfileCommand : public Command {
 public:
  static std::string key() { return "profile"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        if (cmd.size() != 5) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else if (cmd[2] == "free") {
        if (cmd.size() != 4) {
          INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
        }
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      if ((cmd.size() != 3) && (cmd.size() != 4)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "save") {
      if (cmd.size() != 4) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "config") {
      if (cmd[2] == "create") {
        const auto& configName = cmd[3];
        auto it                = ctx.profileConfigs.find(configName);
        if (it != ctx.profileConfigs.end()) {
          RT_ERR("Profile config already exists: " + configName);
        }

        const auto& configPath = cmd[4];
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
          INV_ARG("Could not open JSON config file: " + configPath + ".");
        }
        std::string jsonFile;
        std::getline(ifs, jsonFile, '\0');
        ctx.profileConfigs[configName] = std::make_shared<genie::Profile::Config>(jsonFile);
      } else if (cmd[2] == "free") {
        const auto& configName = cmd[3];
        auto it                = ctx.profileConfigs.find(configName);
        if (it == ctx.profileConfigs.end()) {
          RT_ERR("Profile config does not exist: " + configName);
        }
        ctx.profileConfigs.erase(configName);
      } else {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "create") {
      const auto& profileName = cmd[2];
      auto it                 = ctx.profiles.find(profileName);
      if (it != ctx.profiles.end()) {
        RT_ERR("Profile already exists: " + profileName);
      }

      std::shared_ptr<genie::Profile::Config> profileConfig{new genie::Profile::Config()};
      if (cmd.size() == 4) {
        const auto& configName = cmd[3];
        auto configIt          = ctx.profileConfigs.find(configName);
        if (configIt == ctx.profileConfigs.end()) {
          RT_ERR("Profile config does not exist: " + configName);
        }
        profileConfig = configIt->second;
      }
      ctx.profiles[profileName] = std::make_shared<genie::Profile>(*profileConfig);
    } else if (cmd[1] == "free") {
      const auto& profileName = cmd[2];
      auto it                 = ctx.profiles.find(profileName);
      if (it == ctx.profiles.end()) {
        RT_ERR("Profile does not exist: " + profileName);
      }
      ctx.profiles.erase(profileName);
    } else if (cmd[1] == "save") {
      const auto& profileName = cmd[2];
      auto it                 = ctx.profiles.find(profileName);
      if (it == ctx.profiles.end()) {
        RT_ERR("Profile does not exist: " + profileName);
      }
      const auto jsonStr      = it->second->getJsonData();
      const auto& profilePath = cmd[3];
      std::ofstream outFile(profilePath);
      if (!outFile.good()) {
        RT_ERR("Cannot create profile output file with name:" + profilePath);
      }
      outFile << it->second->getJsonData();
      outFile.close();
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("profile config create CONFIG_NAME FILE.json");
    strs.push_back("profile config free   CONFIG_NAME");
    strs.push_back("profile create        PROFILE_NAME [CONFIG_NAME]");
    strs.push_back("profile save          PROFILE_NAME [OUTPUT_FILE.json]");
    strs.push_back("profile free          PROFILE_NAME");
    return strs;
  }
};

class LogCommand : public Command {
 public:
  static std::string key() { return "log"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "create") {
      if ((cmd.size() != 4) && (cmd.size() != 5)) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd[1] == "free") {
      if (cmd.size() != 3) {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd[1] == "create") {
      const auto& logLevelStr = cmd[3];
      GenieLog_Level_t level;
      if (logLevelStr == "error") {
        level = GENIE_LOG_LEVEL_ERROR;
      } else if (logLevelStr == "warning") {
        level = GENIE_LOG_LEVEL_WARN;
      } else if (logLevelStr == "info") {
        level = GENIE_LOG_LEVEL_INFO;
      } else if (logLevelStr == "verbose") {
        level = GENIE_LOG_LEVEL_VERBOSE;
      } else {
        RT_ERR("Invalid log level: " + logLevelStr);
      }

      const auto& logName = cmd[2];
      auto it             = ctx.loggers.find(logName);
      if (it != ctx.loggers.end()) {
        RT_ERR("Logger already exists: " + logName);
      }

      GenieLog_Callback_t cb = nullptr;
      if (cmd.size() == 5) {
        cb = callback;
      }

      auto logger = std::make_shared<genie::Log>(level, cb);

      if (cmd.size() == 5) {
        std::ofstream ofs(cmd[4]);
        if (!ofs.is_open()) {
          INV_ARG("Could not open log output file: " + cmd[4] + ".");
        }
        m_streams[(*logger)()] = std::move(ofs);
      }

      ctx.loggers[logName] = logger;
    } else if (cmd[1] == "free") {
      const auto& logName = cmd[2];
      auto it             = ctx.loggers.find(logName);
      if (it == ctx.loggers.end()) {
        RT_ERR("Logger does not exist: " + logName);
      }
      if (m_streams.count((*it->second)()) > 0) {
        m_streams.erase((*it->second)());
      }
      ctx.loggers.erase(logName);
    } else {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  static void cleanup() { m_streams.clear(); }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("log create LOG_NAME LEVEL [OUTPUT_FILE.txt]");
    strs.push_back("log free   LOG_NAME");
    return strs;
  }

 private:
  static void callback(const GenieLog_Handle_t handle,
                       const char* fmt,
                       GenieLog_Level_t level,
                       uint64_t timestamp,
                       va_list args) {
    char buf[1024];
    const char* levelStr = "";
    switch (level) {
      case GENIE_LOG_LEVEL_ERROR:
        levelStr = "ERROR";
        break;
      case GENIE_LOG_LEVEL_WARN:
        levelStr = "WARNING";
        break;
      case GENIE_LOG_LEVEL_INFO:
        levelStr = "INFO";
        break;
      case GENIE_LOG_LEVEL_VERBOSE:
        levelStr = "VERBOSE";
        break;
    }

    double ms  = static_cast<double>(timestamp) / 1000000.0;
    int offset = snprintf(buf, sizeof(buf), "%8.1fms [%s]: ", ms, levelStr);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    vsnprintf(buf + offset, sizeof(buf) - static_cast<size_t>(offset), fmt, args);
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) || defined(__clang__)
    m_streams[handle] << buf << "\n";
  }

  static std::map<const GenieLog_Handle_t, std::ofstream> m_streams;
};

std::map<const GenieLog_Handle_t, std::ofstream> LogCommand::m_streams;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Command Variables
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<std::string, std::unique_ptr<Command>> commands;

// Aliases map: maps a single-token alias to its canonical replacement (space-separated).
// Add entries here to support legacy command names without modifying command logic.
static const std::map<std::string, std::string> g_commandAliases{
    {"endasync", "async end"},
};
std::list<std::vector<std::string>> commandHistory;

void processCommand(const std::vector<std::string>& cmd, bool scriptMode = true);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Basic Commands
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ListCommand : public Command {
 public:
  static std::string key() { return "ls"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 1) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& /*cmd*/) override { ctx.list(); }
  std::list<std::string> help() override { return {key()}; }
};

class VersionCommand : public Command {
 public:
  static std::string key() { return "version"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 1) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& /*cmd*/) override {
#ifdef _WIN32
    std::cout << "genie-app pid: " << _getpid() << std::endl;
#else
    std::cout << "genie-app pid: " << getpid() << std::endl;
#endif
    std::cout << "libGenie.so = " << Genie_getApiMajorVersion() << "." << Genie_getApiMinorVersion()
              << "." << Genie_getApiPatchVersion() << std::endl;
  }
  std::list<std::string> help() override { return {key()}; }
};

class HistoryCommand : public Command {
 public:
  static std::string key() { return "history"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() == 3) {
      if (cmd[1] != "save") {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd.size() == 2) {
      if (cmd[1] != "clear") {
        INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
      }
    } else if (cmd.size() != 1) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    if (cmd.size() == 1) {
      writeHistory(std::cout);
    } else if (cmd.size() == 2) {
      if (cmd[1] == "clear") {
        commandHistory.clear();
      }
    } else if (cmd.size() == 3) {
      auto& historyFile = cmd[2];
      std::ofstream ofs(historyFile);
      if (!ofs.is_open()) {
        INV_ARG("Could not open history file: " + std::string(historyFile) + ".");
      }
      writeHistory(ofs);
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("history");
    strs.push_back("history clear");
    strs.push_back("history save FILE");
    return strs;
  }

 private:
  void writeHistory(std::ostream& os) {
    for (const auto& command : commandHistory) {
      os << combine(command, ' ') << "\n";
    }
  }
};

class AsyncCommand : public Command {
 private:
  static std::atomic<bool> asyncMode;
  static std::vector<std::vector<std::string>> asyncCmds;
  static std::mutex asyncMutex;

  static void setAsync(bool async) { asyncMode.store(async); }

  static void processAsync(std::vector<std::vector<std::string>> cmds, int sleepTimeInMs = 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeInMs));
    for (auto cmd : cmds) {
      try {
        processCommand(cmd, true);
      } catch (const std::exception& e) {
        std::cerr << "Error in async command: " << e.what() << std::endl;
      }
    }
  }

 public:
  static std::string key() { return "async"; }

  void validate(const std::vector<std::string>& cmd) override {
    if ((cmd.size() > 2) || (cmd.size() == 2 && cmd[1] != "end" && cmd[1] != "wait")) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }

  static void addJob(const std::vector<std::string>& cmd) {
    if (!asyncMode.load()) {
      INV_ARG("Not in async mode.");
    }
    if (cmd.size() > 0 && (cmd[0] == "loop" || cmd[0] == "endloop")) {
      INV_ARG("Can't execute loop in async mode");
    }
    std::lock_guard<std::mutex> lock(asyncMutex);
    asyncCmds.push_back(cmd);
  }

  void process(const std::vector<std::string>& cmd) override {
    (void)cmd;  // Suppress unused parameter warning
    if (cmd.size() == 1) {
      if (asyncMode) {
        INV_ARG("Ill-formatted async command syntax. Nested asyncs are not supported.");
      }
      setAsync(true);
    } else if (cmd[1] == "end") {
      if (!asyncMode) {
        INV_ARG("Not in async mode.");
      }
      std::vector<std::vector<std::string>> cmdsCopy;
      {
        std::lock_guard<std::mutex> lock(asyncMutex);
        cmdsCopy = std::move(asyncCmds);
        asyncCmds.clear();
      }
      ctx.activeThreads.emplace_back(processAsync, std::move(cmdsCopy), 0);
      setAsync(false);
    } else if (cmd[1] == "wait") {
      if (asyncMode) {
        INV_ARG("Can't do wait job in async mode: " + combine(cmd, ' ') + ".");
      }
      for (size_t i = 0; i < ctx.activeThreads.size(); i++) {
        if (ctx.activeThreads[i].joinable()) {
          ctx.activeThreads[i].join();
        }
      }
      ctx.activeThreads.clear();
    }
  }
  std::list<std::string> help() override { return {key() + " Process command in async mode"}; }
  static bool isAsync() { return asyncMode.load(); }
};
std::atomic<bool> AsyncCommand::asyncMode(false);
std::vector<std::vector<std::string>> AsyncCommand::asyncCmds;
std::mutex AsyncCommand::asyncMutex;

class ScriptCommand : public Command {
 public:
  struct Loop {
    bool enabled       = false;
    uint32_t start_idx = 0;  // inclusive
    uint32_t end_idx   = 0;  // exclusive
    uint32_t increment = 1;
    std::vector<std::vector<std::string>> cmds;
  };

  static std::string key() { return "script"; }

  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 2) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    std::string path = cmd[1];
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      INV_ARG("Could not open script file: " + std::string(path) + ".");
    }
    Loop loop;
    std::string line;
    while (std::getline(ifs, line)) {
      auto _cmd = split(line, ' ');

      if ((_cmd.size() == 0 || _cmd[0] != "endloop") && loop.enabled == true) {
        if (_cmd.size() == 0) {
          loop.cmds.push_back(_cmd);
        } else if (_cmd[0] == "loop") {
          INV_ARG("Ill-formatted loop command syntax. Nested loops are not supported.");
        } else if (_cmd[0][0] == '#') {
          loop.cmds.push_back({line});
        } else {
          loop.cmds.push_back(_cmd);
        }
      } else if (_cmd.size() == 0) {
        std::cout << std::endl;
      } else if (_cmd[0] == "loop") {
        loop.enabled = true;

        int32_t start_idx = 0;
        int32_t end_idx   = 0;
        int32_t increment = 1;
        if (_cmd.size() == 2) {
          end_idx = static_cast<int32_t>(std::stoi(_cmd[1]));
        } else if (_cmd.size() == 3) {
          start_idx = static_cast<int32_t>(std::stoi(_cmd[1]));
          end_idx   = static_cast<int32_t>(std::stoi(_cmd[2]));
        } else if (_cmd.size() == 4) {
          start_idx = static_cast<int32_t>(std::stoi(_cmd[1]));
          end_idx   = static_cast<int32_t>(std::stoi(_cmd[2]));
          increment = static_cast<int32_t>(std::stoi(_cmd[3]));
        } else {
          INV_ARG(
              "Ill-formatted loop command syntax. Loop command must be formatted "
              "in one of the following ways:"
              " (1) loop <end>"
              " (2) loop <start> <end>"
              " (3) loop <start> <end> <increment>");
        }

        // Loop boundary validation
        if (start_idx < 0 || end_idx < 0 || increment <= 0) {
          INV_ARG(
              "Bad loop argument. Expecting non-negative <start> and <end> arguments "
              "and strictly-positive <increment> argument");
        }

        loop.start_idx = static_cast<uint32_t>(start_idx);
        loop.end_idx   = static_cast<uint32_t>(end_idx);
        loop.increment = static_cast<uint32_t>(increment);
      } else if (_cmd[0][0] == '#') {
        std::cout << line << std::endl;
      } else {
        if (_cmd[0] == "endloop") {
          if (loop.enabled == false) {
            INV_ARG("Ill-formatted loop command syntax. Dangling endloop command.");
          } else {
            const size_t maxItrs =
                std::ceil(static_cast<float>(loop.end_idx - loop.start_idx) / loop.increment);
            for (size_t i = loop.start_idx, itr = 1; i < loop.end_idx; i += loop.increment, ++itr) {
              std::cout << "Starting Loop Iteration: " << itr << "/" << maxItrs << std::endl;
              for (size_t j = 0; j < loop.cmds.size(); ++j) {
                if (loop.cmds[j].size() == 0) {
                  std::cout << std::endl;
                } else if (loop.cmds[j][0][0] == '#') {
                  std::cout << loop.cmds[j][0] << std::endl;
                } else if (loop.cmds[j][0] != "async" && AsyncCommand::isAsync()) {
                  AsyncCommand::addJob(loop.cmds[j]);
                } else {
                  // Substitute "{index}" in commands with value of i
                  std::vector<std::string> replacedCmd = loop.cmds[j];
                  for (size_t t = 0; t < replacedCmd.size(); ++t) {
                    std::string& token        = replacedCmd[t];
                    const std::string pattern = "{index}";
                    size_t pos                = 0;
                    while ((pos = token.find(pattern, pos)) != std::string::npos) {
                      const std::string indexStr = std::to_string(i);
                      token.replace(pos, pattern.size(), indexStr);
                      pos += indexStr.size();
                    }
                  }
                  try {
                    processCommand(replacedCmd, true);
                  } catch (const std::exception& e) {
                    std::cerr << RED << e.what() << RESET << std::endl;
                  }
                }
              }
            }
            loop = Loop();
          }
        } else if (_cmd[0] != "async" && AsyncCommand::isAsync()) {
          AsyncCommand::addJob(_cmd);
        } else {
          try {
            processCommand(_cmd, true);
          } catch (const std::exception& e) {
            std::cerr << RED << e.what() << RESET << std::endl;
          }
        }
      }
    }
    if (loop.enabled == true || !loop.cmds.empty()) {
      loop = Loop();
      ctx.clear();
      INV_ARG("Ill-formatted loop command syntax. Loop is not enclosed via endloop.");
    }
  }

  std::list<std::string> help() override {
    std::list<std::string> strs;
    strs.push_back("script   FILE");
    strs.push_back("loop     [END INDEX] or [START INDEX] [END INDEX] or ");
    strs.push_back("         [START INDEX] [END INDEX] [INCREMENT]");
    strs.push_back("endloop");
    return strs;
  }
};

class SleepCommand : public Command {
 public:
  static std::string key() { return "sleep"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 2) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>& cmd) override {
    sleepTimeInMs = std::stoi(cmd[1]);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeInMs));
  }
  std::list<std::string> help() override { return {key() + " [DURATION in ms]"}; }

 private:
  int sleepTimeInMs = 0;
};

class ExitCommand : public Command {
 public:
  static std::string key() { return "exit"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 1) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>&) override {
    cleanup();
    std::exit(0);
  }
  std::list<std::string> help() override { return {key()}; }
};

class HelpCommand : public Command {
 public:
  static std::string key() { return "help"; }
  void validate(const std::vector<std::string>& cmd) override {
    if (cmd.size() != 1) {
      INV_ARG("Ill-formatted command: " + combine(cmd, ' ') + ".");
    }
  }
  void process(const std::vector<std::string>&) override {
    for (const auto& command : commands) {
      for (const auto& str : command.second->help()) {
        std::cout << str << std::endl;
      }
    }
  }
  std::list<std::string> help() override { return {key()}; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Command Infrastructure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void processCommand(const std::vector<std::string>& cmd, bool scriptMode) {
  // Resolve aliases: split the alias string into tokens and prepend to any trailing args.
  auto aliasIt = g_commandAliases.find(cmd[0]);
  if (aliasIt != g_commandAliases.end()) {
    std::vector<std::string> resolved = split(aliasIt->second, ' ');
    resolved.insert(resolved.end(), cmd.begin() + 1, cmd.end());
    return processCommand(resolved, scriptMode);
  }

  commandHistory.push_back(cmd);
  if (scriptMode) {
    std::cout << "> " << combine(cmd, ' ') << std::endl;
  }

  auto it = commands.find(cmd[0]);
  if (it == commands.end()) {
    INV_ARG("Unknown command: " + cmd[0] + ".");
  }
  it->second->validate(cmd);
  it->second->process(cmd);
}

void validateCommand(const std::vector<std::string>& cmd) {
  auto it = commands.find(cmd[0]);
  if (it == commands.end()) {
    INV_ARG("Unknown command: " + cmd[0] + ".");
  }
  it->second->validate(cmd);
}

void registerCommands() {
  commands[HelpCommand::key()]      = std::unique_ptr<HelpCommand>(new HelpCommand());
  commands[ScriptCommand::key()]    = std::unique_ptr<ScriptCommand>(new ScriptCommand());
  commands[SleepCommand::key()]     = std::unique_ptr<SleepCommand>(new SleepCommand());
  commands[ExitCommand::key()]      = std::unique_ptr<ExitCommand>(new ExitCommand());
  commands[HistoryCommand::key()]   = std::unique_ptr<HistoryCommand>(new HistoryCommand());
  commands[VersionCommand::key()]   = std::unique_ptr<VersionCommand>(new VersionCommand());
  commands[ListCommand::key()]      = std::unique_ptr<ListCommand>(new ListCommand());
  commands[PipelineCommand::key()]  = std::unique_ptr<PipelineCommand>(new PipelineCommand());
  commands[NodeCommand::key()]      = std::unique_ptr<NodeCommand>(new NodeCommand());
  commands[DialogCommand::key()]    = std::unique_ptr<DialogCommand>(new DialogCommand());
  commands[EmbeddingCommand::key()] = std::unique_ptr<EmbeddingCommand>(new EmbeddingCommand());
  commands[ProfileCommand::key()]   = std::unique_ptr<ProfileCommand>(new ProfileCommand());
  commands[LogCommand::key()]       = std::unique_ptr<LogCommand>(new LogCommand());
  commands[TokenizerCommand::key()] = std::unique_ptr<TokenizerCommand>(new TokenizerCommand());
  commands[AccuracyCommand::key()]  = std::unique_ptr<AccuracyCommand>(new AccuracyCommand());
  commands[SamplerCommand::key()]   = std::unique_ptr<SamplerCommand>(new SamplerCommand());
  commands[DlcCommand::key()]       = std::unique_ptr<DlcCommand>(new DlcCommand());
  commands[AsyncCommand::key()]     = std::unique_ptr<AsyncCommand>(new AsyncCommand());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Application Command Line Interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::list<std::string> startupScripts;

void printUsage() {
  std::cout << "Usage:\n"
            << "-----\n\n"
            << "genie-app"
            << " [options]\n"
            << std::endl;
  std::cout << "Options:\n"
            << "--------\n\n";

  int width = 30;

  std::cout << std::left << std::setw(width) << "  -h, --help";
  std::cout << "Show this help message and exit.\n\n";

  std::cout << std::setw(width) << "  -s FILE or --script FILE";
  std::cout << "Script file.\n\n";

  std::cout << "Script commands:\n"
            << "----------------\n\n";
  try {
    processCommand({"help"}, false);
  } catch (const std::exception& e) {
    std::cerr << RED << e.what() << RESET << std::endl;
  }

  std::cout << std::endl;
}

void parseCommandLineInput(int argc, char** argv) {
  // Process help commands first
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "-h" || (arg == "--help")) {
      printUsage();
      exit(0);
    }
  }

  // Process the remainder of commands
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if ((arg == "-s") || (arg == "--script")) {
      if (i + 1 == argc) {
        INV_ARG("Missing script file.");
      }
      i++;
      startupScripts.push_back(argv[i]);
    } else {
      INV_ARG("Invalid option: " + arg + ".");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main Function
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cleanup() {
  ctx.clear();
  g_queryResponseMap.clear();
  g_nodeResponseMap.clear();
  LogCommand::cleanup();
  startupScripts.clear();
  commands.clear();
}

int main(int argc, char** argv) {
  registerCommands();
  bool errorOccurred = false;

  try {
    parseCommandLineInput(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << RED << e.what() << RESET << "\n" << std::endl;
    printUsage();
    return EXIT_FAILURE;
  }

  try {
    for (const auto& script : startupScripts) {
      try {
        processCommand({"script", script}, true);
      } catch (const std::exception& e) {
        std::cerr << RED << e.what() << RESET << std::endl;
        errorOccurred = true;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << RED << e.what() << RESET << std::endl;
    errorOccurred = true;
  }

  cleanup();

  return errorOccurred ? EXIT_FAILURE : EXIT_SUCCESS;
}
