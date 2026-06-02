//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <set>
#include <sstream>

#include "Exception.hpp"
#include "ImageEncoder.hpp"
#include "LmExecutor.hpp"
#include "Macro.hpp"
#include "Node.hpp"
#include "Pipeline.hpp"
#include "TextEncoder.hpp"
#include "TextGenerator.hpp"
#include "Unet.hpp"

using namespace genie;

const pipeline::ExecutionStrategy& pipeline::Node::getExecutionStrategyFromString(
    const std::string& strategy) {
  static const std::unordered_map<std::string, pipeline::ExecutionStrategy>
      s_strategyStringToStrategyEnum{{"implicit", GENIE_EXECUTION_STRATEGY_IMPLICIT},
                                     {"explicit", GENIE_EXECUTION_STRATEGY_EXPLICIT}};

  auto it = s_strategyStringToStrategyEnum.find(strategy);
  if (it == s_strategyStringToStrategyEnum.end()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unrecognized strategy type: " + strategy);
  }

  return it->second;
}

const qualla::LayerType& pipeline::Node::getLayerTypeFromNodeIO(const GenieNode_IOName_t& ioName) {
  static const std::unordered_map<GenieNode_IOName_t, qualla::LayerType> s_ioNameToLayerType{
      {GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT, qualla::LayerType::INPUT},
      {GENIE_NODE_LM_EXECUTOR_EMBEDDING_INPUT, qualla::LayerType::INPUT},
      {GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT, qualla::LayerType::OUTPUT}};

  auto it = s_ioNameToLayerType.find(ioName);
  if (it == s_ioNameToLayerType.end()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Failed to get Layer Name");
  }

  return it->second;
}
qnn::util::HandleManager<pipeline::Node::Config>& pipeline::Node::Config::getManager() {
  static qnn::util::HandleManager<pipeline::Node::Config> s_manager;
  return s_manager;
}

GenieNodeConfig_Handle_t pipeline::Node::Config::add(std::shared_ptr<Config> config) {
  return reinterpret_cast<GenieNodeConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<pipeline::Node::Config> pipeline::Node::Config::get(
    GenieNodeConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void pipeline::Node::Config::remove(GenieNodeConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void pipeline::Node::Config::createConfigFromDlc(const char* useCaseName, const char* configStr) {
  std::shared_ptr<const char[]> metadataStr;
  uint64_t metadataBufferSize{};
  nlohmann::json metadataJson;
  m_dlc->getGenieMetadataBuffer(metadataStr, &metadataBufferSize);
  util::getJsonFromStr(metadataStr.get(), metadataJson);
  bool useCaseFound = false;
  for (auto& item : metadataJson.items()) {
    if (item.key() == "use-cases") {
      if (item.value().type() != nlohmann::json::value_t::array) {
        throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "use-cases must be an array");
      }
      for (const auto& useCase : item.value()) {
        if (!useCase.is_object()) continue;
        if (!useCase.contains("name")) {
          throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                 "name field must be present in a genie config use-case.");
        }
        if (useCaseName == nullptr ||
            strcmp(useCase["name"].get<std::string>().c_str(), useCaseName) == 0) {
          useCaseFound = true;
          if (!useCase.contains("config")) {
            throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                   "Missing or invalid config field in use-case");
          } else {
            if (!useCase["config"].contains("name")) {
              throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                     "Missing name field in config of use-case");
            }
          }
          std::string configName = useCase["config"]["name"].get<std::string>();
          if (configName.empty()) {
            throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                                   "Must provide Genie config name for the use case");
          }
          std::string genieConfigRecordName = "genai.artifact." + configName;
          nlohmann::json genieConfigJson;
          util::getJsonFromDlc(m_dlc, genieConfigRecordName, genieConfigJson);
          m_config = genieConfigJson;

          // If user passes a standalone genie config, overwrite the config in DLC with it
          if (configStr != nullptr) {
            util::overwriteConfig(m_config, configStr);
          }
          break;
        }
      }
    }
  }

  if (!useCaseFound) {
    throw genie::Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                           "Use case not found: " + std::string(useCaseName));
  }
}

pipeline::Node::Config::Config(const char* configStr) {
  m_config = nlohmann::json::parse(configStr);
}

pipeline::Node::Config::Config(std::shared_ptr<Dlc> dlc,
                               const char* useCaseName,
                               const char* configStr) {
  m_dlc = dlc;
  createConfigFromDlc(useCaseName, configStr);
}

void pipeline::Node::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void pipeline::Node::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Node::Config::getLogger() {
  return m_logger;
}

void pipeline::Node::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void pipeline::Node::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Node::Config::getProfiler() {
  return m_profiler;
}

std::shared_ptr<Dlc>& pipeline::Node::Config::getDlc() { return m_dlc; }

//=============================================================================
// Node functions
//=============================================================================
std::atomic<std::uint32_t> pipeline::Node::s_nameCounter{0u};

qnn::util::HandleManager<pipeline::Node>& pipeline::Node::getManager() {
  static qnn::util::HandleManager<pipeline::Node> s_manager;
  return s_manager;
}

std::shared_ptr<pipeline::Node> pipeline::Node::createNode(
    std::shared_ptr<Config> configObj,
    std::shared_ptr<ProfileStat> profileStat,
    std::shared_ptr<genie::log::Logger> logger) {
  auto config = configObj->getJson();
  auto env =
      qualla::Env::create(configObj->getDlc() != nullptr ? configObj->getDlc()->getResourceManager()
                                                         : std::make_shared<ResourceManager>(),
                          nlohmann::json{});
  // component is used in the "ENFORCE" macros
  std::string component = "node";
  // Translate pipeline-config to Dialog/ Embedding config
  for (auto item : config.items()) {
    if (item.key() == "lut-encoder") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextEncoder>(item, profileStat, logger, env);
    } else if (item.key() == "text-encoder") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextEncoder>(item, profileStat, logger, env);
    } else if (item.key() == "text-generator") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextGenerator>(item, profileStat, logger, env);
    } else if (item.key() == "image-encoder") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::ImageEncoder>(item, profileStat, logger, env);
    } else if (item.key() == "diffuser") {
      return std::make_shared<pipeline::Unet>(item, profileStat, logger, env);
    } else if (item.key() == "lm-executor") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::LmExecutor>(item, profileStat, logger, env);
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown config key: " + item.key());
    }
  }

  return nullptr;
}

pipeline::Node::Node(nlohmann::json config) {
  m_config            = config;
  m_executionStrategy = ExecutionStrategy::GENIE_EXECUTION_STRATEGY_IMPLICIT;
  m_name              = "node" + std::to_string(s_nameCounter.fetch_add(1u));
}

GenieNode_Handle_t pipeline::Node::add(std::shared_ptr<pipeline::Node> node) {
  return reinterpret_cast<GenieNode_Handle_t>(getManager().add(node));
}

std::shared_ptr<pipeline::Node> pipeline::Node::get(GenieNode_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void pipeline::Node::remove(GenieNode_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

int32_t pipeline::Node::bindPipeline(Pipeline& pipeline) {
  if (m_pipeline) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Node already bound to Pipeline");
  }
  m_pipeline = &pipeline;
  return GENIE_STATUS_SUCCESS;
}

std::string pipeline::Node::getName() const { return m_name; }

int32_t pipeline::Node::execute(void* /*userData*/,
                                std::shared_ptr<ProfileStat> /*profileStat*/,
                                nlohmann::json /*executionConfig*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "execute not supported on Node");
}

int32_t pipeline::Node::train() {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "train not supported in node");
}

int32_t pipeline::Node::saveLora(std::string /*loraAdapterName*/,
                                 std::string /*engine*/,
                                 std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "saveLora not supported in node");
}

int32_t pipeline::Node::save(const std::string& /*name*/) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::restore(const std::string& /*name*/) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::setPriority(const std::string /*engine*/,
                                    const GeniePipeline_Priority_t /*priority*/) {
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Node::setOemkey(const std::string& /*oemKey*/) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::applyLora(std::string /*loraAdapterName*/,
                                  std::string /*engine*/,
                                  std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "applyLora not supported in node");
}

int32_t pipeline::Node::applyLoraStrength(std::string /*tensorName*/,
                                          std::string /*engine*/,
                                          float /*alpha*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "applyLoraStrength not supported in node");
}

GenieEngine_Handle_t pipeline::Node::getEngineHandle(const std::string& /*engineRole*/,
                                                     std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getEngineHandle not supported by Node");
  return nullptr;
}

int32_t pipeline::Node::bindEngine(const std::string& /*engineRole*/,
                                   std::shared_ptr<Engine> /*engine*/,
                                   std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "bindEngine not supported by Node");
  return GENIE_STATUS_ERROR_GENERAL;
}

size_t pipeline::Node::getMaxIterations() { return m_maxIterations; }

size_t pipeline::Node::getBatchSize() { return m_batchSize; }

GenieSampler_Handle_t pipeline::Node::getSamplerHandle() {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getSamplerHandle not supported by Node");
  return nullptr;
}

GenieTokenizer_Handle_t pipeline::Node::getTokenizerHandle() {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getTokenizerHandle not supported by Node");
  return nullptr;
}

// Input Modality setters
int32_t pipeline::Node::setTextInputData(GenieNode_IOName_t /*nodeIOName*/,
                                         const char* /*txt*/,
                                         std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setTextInputData not supported in node");
}

int32_t pipeline::Node::setEmbeddingInputData(GenieNode_IOName_t /*nodeIOName*/,
                                              const void* /*embedding*/,
                                              size_t /*embeddingSize*/,
                                              std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setEmbeddingInputData not supported in node");
}

int32_t pipeline::Node::setImageInputData(GenieNode_IOName_t /*nodeIOName*/,
                                          const void* /*imageData*/,
                                          size_t /*imageSize*/,
                                          std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData not supported in node");
}

int32_t pipeline::Node::setInputBufferData(GenieNode_IOName_t /*nodeIOName*/,
                                           const void* /*embedding*/,
                                           size_t /*embeddingSize*/,
                                           nlohmann::json /*dataConfig*/,
                                           std::shared_ptr<ProfileStat> /*profileStat*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setInputBufferData not supported in node");
}

// Output Modality setters
int32_t pipeline::Node::setTextOutputCallback(GenieNode_IOName_t /*nodeIOName*/,
                                              GenieNode_TextOutput_Callback_t /*callback*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setTextOutputCallback not supported in node");
}

int32_t pipeline::Node::setEmbeddingOutputCallback(
    GenieNode_IOName_t /*nodeIOName*/, GenieNode_EmbeddingOutputCallback_t /*callback*/) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setEmbeddingOutputCallback not supported in node");
}

// get Output buffer data
int32_t pipeline::Node::getOutputBufferData(GenieNode_IOName_t /*nodeIOName*/,
                                            nlohmann::json& /*ioConfig*/,
                                            GenieNode_IOCallback_t /*callback*/,
                                            const void* /*userData*/,
                                            std::shared_ptr<ProfileStat>) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getOutputBufferData not supported in node");
}

void pipeline::Node::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
  }
}

void pipeline::Node::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Node::getLogger() {
  return m_logger;
}

void pipeline::Node::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void pipeline::Node::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Node::getProfiler() { return m_profiler; }

pipeline::Node::~Node() {}
