//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <set>

#include "Accumulator.hpp"
#include "DataLoader.hpp"
#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline {
  // pipeline Utilities
 public:
  class Config {
   public:
    static GeniePipelineConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GeniePipelineConfig_Handle_t handle);
    static void remove(GeniePipelineConfig_Handle_t handle);
    Config(const char* configStr);
     nlohmann::json& getJson();

    void bindProfiler(std::shared_ptr<Profiler> profiler);
    void unbindProfiler();
    std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
    void bindLogger(std::shared_ptr<genie::log::Logger> log);
    void unbindLogger();
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

   private:
    static qnn::util::HandleManager<Config>& getManager();

    nlohmann::json m_config = nlohmann::json();
    std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
    std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  };

  static GeniePipeline_Handle_t add(std::shared_ptr<Pipeline> pipeline);
  static std::shared_ptr<Pipeline> get(GeniePipeline_Handle_t handle);
  static void remove(GeniePipeline_Handle_t handle);

  Pipeline(std::shared_ptr<Config> config,
           std::shared_ptr<ProfileStat> profileStat,
           std::shared_ptr<genie::log::Logger> logger = nullptr);
  Pipeline(const Pipeline&)            = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&)                 = delete;
  Pipeline& operator=(Pipeline&&)      = delete;
  ~Pipeline();

  void setupAccumulator(size_t accumulatorSize = 0, size_t embeddingDimension = 0);

  int32_t addNode(std::shared_ptr<Node> node);

  int32_t pipelineExecute(void* userData, std::shared_ptr<ProfileStat> profileStat);

  /**
   * Connects any outputs from the producer with identical tensor names to the inputs of the
   * consumer.
   */
  void connectMatchingTensors(std::shared_ptr<Node> producer, std::shared_ptr<Node> consumer);

  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();

  std::string getName() const;

  int32_t setPriority(std::string engine, const GeniePipeline_Priority_t priority);
  int32_t setOemkey(const std::string& oemKey);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();

  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

  // DataLoader accessor
  std::shared_ptr<DataLoader> getDataLoader() { return m_dataLoader; }
  void setupDataLoader();

  std::shared_ptr<Accumulator> m_accumulator{nullptr};  // Accumulators are unique to Genie pipelines
  std::shared_ptr<DataLoader> m_dataLoader{nullptr};    // DataLoader for training data collection

 private:
  static qnn::util::HandleManager<Pipeline>& getManager();

  std::set<std::shared_ptr<pipeline::Node>> m_nodes;
  std::unordered_map<std::string, std::shared_ptr<Node>> m_pipelineNodeMap;
  std::unordered_map<std::string, std::set<std::string>> m_connections;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  std::string m_name;
};

}  // namespace pipeline

}  // namespace genie
