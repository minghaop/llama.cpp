//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>

#include "Exception.hpp"
#include "InjectiveConnector.hpp"
#include "Pipeline.hpp"

using namespace genie;

//=============================================================================
// Pipeline::Config functions
//=============================================================================
qnn::util::HandleManager<pipeline::Pipeline::Config>& pipeline::Pipeline::Config::getManager() {
  static qnn::util::HandleManager<pipeline::Pipeline::Config> s_manager;
  return s_manager;
}

GeniePipelineConfig_Handle_t pipeline::Pipeline::Config::add(
    std::shared_ptr<pipeline::Pipeline::Config> config) {
  return reinterpret_cast<GeniePipelineConfig_Handle_t>(getManager().add(config));
}

std::shared_ptr<pipeline::Pipeline::Config> pipeline::Pipeline::Config::get(
    GeniePipelineConfig_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void pipeline::Pipeline::Config::remove(GeniePipelineConfig_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

pipeline::Pipeline::Config::Config(const char* configStr) {
  if (configStr != nullptr && (configStr[0] != '\0')) {
    m_config = nlohmann::json::parse(configStr);
  }
}

nlohmann::json& pipeline::Pipeline::Config::getJson() { return m_config; }

void pipeline::Pipeline::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void pipeline::Pipeline::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Pipeline::Config::getLogger() {
  return m_logger;
}

void pipeline::Pipeline::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void pipeline::Pipeline::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Pipeline::Config::getProfiler() {
  return m_profiler;
}

//=============================================================================
// Pipeline functions
//=============================================================================
std::atomic<std::uint32_t> pipeline::Pipeline::s_nameCounter{0u};

qnn::util::HandleManager<pipeline::Pipeline>& pipeline::Pipeline::getManager() {
  static qnn::util::HandleManager<pipeline::Pipeline> s_manager;
  return s_manager;
}

pipeline::Pipeline::Pipeline(std::shared_ptr<Config> /*config*/,
                             std::shared_ptr<ProfileStat> /*profileStat*/,
                             std::shared_ptr<genie::log::Logger> logger) {
  auto env = qualla::Env::create(std::make_shared<ResourceManager>(), nlohmann::json{});
  if (logger) env->bindLogger(logger);
  nlohmann::json quallaConfig;
  m_name = "pipeline" + std::to_string(s_nameCounter.fetch_add(1u));
}

void pipeline::Pipeline::setupAccumulator(size_t accumulatorSize, size_t embeddingDimension) {
  // Create Accumulator for each input to generator
  m_accumulator = std::make_shared<Accumulator>(accumulatorSize, embeddingDimension);
}

void pipeline::Pipeline::setupDataLoader() {
  // Create DataLoader for each input to generator
  m_dataLoader = std::make_shared<DataLoader>();
}
GeniePipeline_Handle_t pipeline::Pipeline::add(std::shared_ptr<pipeline::Pipeline> pipeline) {
  return reinterpret_cast<GeniePipeline_Handle_t>(getManager().add(pipeline));
}

std::shared_ptr<pipeline::Pipeline> pipeline::Pipeline::get(GeniePipeline_Handle_t handle) {
  return getManager().get(reinterpret_cast<qnn::util::Handle_t>(handle));
}

void pipeline::Pipeline::remove(GeniePipeline_Handle_t handle) {
  getManager().remove(reinterpret_cast<qnn::util::Handle_t>(handle));
}

int32_t pipeline::Pipeline::addNode(std::shared_ptr<Node> node) {
  m_nodes.insert(node);
  node->bindPipeline(*this);
  if (node->getType() == "train" && !m_dataLoader) {
    setupDataLoader();
    m_dataLoader->setMaxIterations(node->getMaxIterations());
    m_dataLoader->setBatchSize(node->getBatchSize());
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::pipelineExecute(void* userData,
                                            std::shared_ptr<ProfileStat> profileStat) {
  for (auto node : m_nodes) {
    node->execute(userData, profileStat);
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::Pipeline::connectMatchingTensors(std::shared_ptr<Node> producer,
                                                std::shared_ptr<Node> consumer) {
  if (!producer->isConnected()) {
    // Force wildcard connections to occur after a normal I/O connection to avoid a "partially
    // connected" state.
    throw genie::Exception(
        GENIE_STATUS_ERROR_INVALID_ARGUMENT,
        "Make a primary connection between nodes before using GENIE_NODE_WILDCARD.");
  } else if (producer->hasWildcard() || consumer->hasWildcard()) {
    throw genie::Exception(
        GENIE_STATUS_ERROR_INVALID_ARGUMENT,
        "Cannot make multiple GENIE_NODE_WILDCARD connections to the same node.");
  }

  std::shared_ptr<qualla::ConnectedTensorMap> connectedTensorMap =
      std::make_shared<qualla::ConnectedTensorMap>();
  std::shared_ptr<qualla::InjectiveProducer> producerConnector =
      std::make_shared<qualla::InjectiveProducer>(m_accumulator->getSequenceTracker(),
                                                  connectedTensorMap);
  std::shared_ptr<qualla::InjectiveConsumer> consumerConnector =
      std::make_shared<qualla::InjectiveConsumer>(
          m_accumulator->getSequenceTracker(), connectedTensorMap, producerConnector);
  // Register the connectors
  producer->registerModelAdaptor(producerConnector);
  consumer->registerModelAdaptor(consumerConnector);

  producer->markWildcard();
  consumer->markWildcard();
}

int32_t pipeline::Pipeline::save(const std::string& name) { /* TODO */
  for (auto node : m_nodes) {
    return node->save(name);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::restore(const std::string& name) { /* TODO */
  for (auto node : m_nodes) {
    return node->restore(name);
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::Pipeline::reset() {
  for (auto node : m_nodes) {
    node->reset();
  }
  if (m_accumulator){
    m_accumulator->reset();
  }
}

int32_t pipeline::Pipeline::setPriority(const std::string engine,
                                        const GeniePipeline_Priority_t priority) {
  for (auto node : m_nodes) {
    return node->setPriority(engine, priority);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::setOemkey(const std::string& oemKey) { /* TODO */
  for (auto node : m_nodes) {
    return node->setOemkey(oemKey);
  }
  return GENIE_STATUS_SUCCESS;
}

pipeline::Pipeline::~Pipeline() {}

std::string pipeline::Pipeline::getName() const { return m_name; }

void pipeline::Pipeline::bindLogger(
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
  }
}

void pipeline::Pipeline::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Pipeline::getLogger() {
  return m_logger;
}

void pipeline::Pipeline::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void pipeline::Pipeline::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Pipeline::getProfiler() {
  return m_profiler;
}
