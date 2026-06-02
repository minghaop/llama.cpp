//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "gpu-model.hpp"
#include "qualla/engine.hpp"

namespace qualla {

class GpuEngine : public Engine {
 private:
  QnnGpuModel::Params _params;
  std::unique_ptr<QnnGpuModel> _model;

 public:
  static constexpr const char* TYPE = "qnn-gpu";

  GpuEngine(Context& ctx, const nlohmann::json& json);
  ~GpuEngine();

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual bool updateKV(size_t n_past) override;

  virtual bool setKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;

  virtual bool save(const std::string& name) override;

  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;

  virtual void reset() override;

  virtual bool load() override;

  virtual bool unload() override;

  virtual const char* getTraceNamespace() const override { return "QnnGpu"; }
};

}  // namespace qualla
