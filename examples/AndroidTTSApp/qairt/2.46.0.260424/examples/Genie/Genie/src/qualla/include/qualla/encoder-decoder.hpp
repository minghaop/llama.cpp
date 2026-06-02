//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sentence.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"

namespace qualla {

class EncoderDecoder : public State {
 public:
  QUALLA_API EncoderDecoder(std::shared_ptr<Env> env,
                            const std::string& name,
                            const nlohmann::json& conf);
  QUALLA_API virtual ~EncoderDecoder();

  // EncoderDecoder registration
  using Creator =
      std::function<EncoderDecoder*(std::shared_ptr<Env>, const std::string&, const nlohmann::json&)>;
  QUALLA_API static void __register(const std::string& type, Creator func);

  // Create EncoderDecoder instance
  QUALLA_API static std::unique_ptr<EncoderDecoder> create(std::shared_ptr<Env> env,
                                                           const std::string& name,
                                                           const nlohmann::json& conf = {});
  QUALLA_API static std::unique_ptr<EncoderDecoder> create(std::shared_ptr<Env> env,
                                                           const std::string& name,
                                                           std::istream& json_stream);
  QUALLA_API static std::unique_ptr<EncoderDecoder> create(std::shared_ptr<Env> env,
                                                           const std::string& name,
                                                           const std::filesystem::path& json_path);

  QUALLA_API virtual bool saveLoraAdapter(std::string lora_adapter_name,
                                          std::string engine_role = "primary");

  QUALLA_API virtual std::vector<float> train(TrainingData& training_data);

  // Get input names
  QUALLA_API virtual void input_names(std::unordered_set<std::string>& inputTensorNames);

  // Get tensor dimensions (generic for both input and output)
  QUALLA_API virtual void get_dimensions(std::vector<std::uint32_t>& dimensions,
                                         LayerType layerType);

  // Get tensor quantization parameters (generic for both input and output)
  QUALLA_API virtual void get_tensorQuantParam(
      std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth, LayerType layerType);

  QUALLA_API std::string type() { return _type; }

  QUALLA_API std::shared_ptr<Env> getEnv() { return _env; };

  // Embedding KPIs
  struct KPIs {
    Kpi init;  // init (model load, mem allocs, etc) stats
    Kpi lora;  // lora stats

    KPIs() {}
    // void reset();  // reset to initial state

    QUALLA_API std::string dump(
        std::string_view sep = " ") const;  // dump KPIs as a formated string
  };

  // Get latest KPIs.
  // Updates TPS, etc as needed.
  QUALLA_API virtual KPIs& kpis();
  Engine& engine() { return *_engine; }

 protected:
  const std::string _type;
  KPIs _kpis;
  std::shared_ptr<Env> _env;
  std::shared_ptr<Engine> _engine;  // engines
  qualla::PerformanceProfile m_perfProfile;
  qualla::PerformanceProfile m_defaultPerfProfile;
};

}  // namespace qualla
