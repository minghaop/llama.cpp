//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <QnnApi.hpp>
#include <memory>
#include <qualla/detail/tensor.hpp>
#include <stdexcept>
#define _USE_MATH_DEFINES  // Used for M_PI

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <span>
#include <sstream>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif  // defined(__GNUC__) || defined(__clang__)
#include "fp16/fp16.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) || defined(__clang__)

#include "ResourceManager.hpp"
#include "Trace.hpp"
#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"
#include "nsp-base-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "smart-mask.hpp"

namespace fs = std::filesystem;

namespace qualla {

inline TensorDataType mapQnnDataTypeToQualla(const QnnUtils::DataType& qnnDataType) {
  switch (static_cast<Qnn_DataType_t>(qnnDataType)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      return TENSOR_DATATYPE_UFIXED_POINT_8;
    case QNN_DATATYPE_UFIXED_POINT_16:
      return TENSOR_DATATYPE_UFIXED_POINT_16;
    case QNN_DATATYPE_FLOAT_16:
      return TENSOR_DATATYPE_FLOAT_POINT_16;
    case QNN_DATATYPE_FLOAT_32:
      return TENSOR_DATATYPE_FLOAT_32;
    default:
      return TENSOR_DATATYPE_UNKNOWN;
  }
}

// NspModelIOSpec method implementations
void NspModelIOSpec::addModelVariant(ModelVariant variant,
                                     const QnnUtils::TensorMap& qnn_input_spec,
                                     const QnnUtils::TensorMap& qnn_output_spec,
                                     IOTensor& ioTensorManager) {
  m_variants.insert(variant);
  std::unique_ptr<TensorMap>& input_spec = m_input_specs[variant];
  if (!input_spec) {
    input_spec = std::make_unique<TensorMap>();
  }
  for (const auto& [name, qnn_tensor] : qnn_input_spec) {
    auto itr = input_spec->find(name);
    if (itr == input_spec->end()) {
      std::unique_ptr<Tensor> tensor = createTensorFromQnnTensor(qnn_tensor, ioTensorManager);
      input_spec->insert({name, tensor.get()});
      m_tensors.push_back(std::move(tensor));
    }
  }

  std::unique_ptr<TensorMap>& output_spec = m_output_specs[variant];
  if (!output_spec) {
    output_spec = std::make_unique<TensorMap>();
  }
  for (const auto& [name, qnn_tensor] : qnn_output_spec) {
    auto itr = output_spec->find(name);
    if (itr == output_spec->end()) {
      std::unique_ptr<Tensor> tensor = createTensorFromQnnTensor(qnn_tensor, ioTensorManager);
      output_spec->insert({name, tensor.get()});
      m_tensors.push_back(std::move(tensor));
    }
  }
}
const std::set<ModelVariant>& NspModelIOSpec::getVariants() const { return m_variants; }

const TensorMap* NspModelIOSpec::getInputSpec(const ModelVariant& variant) const {
  auto it = m_input_specs.find(variant);
  if (it != m_input_specs.end()) {
    return it->second.get();
  }
  return nullptr;
}

const TensorMap* NspModelIOSpec::getOutputSpec(const ModelVariant& variant) const {
  auto it = m_output_specs.find(variant);
  if (it != m_output_specs.end()) {
    return it->second.get();
  }
  return nullptr;
}

std::unique_ptr<Tensor> NspModelIOSpec::createTensorFromQnnTensor(
    const QnnUtils::Tensor& qnn_tensor, IOTensor& ioTensorManager) {
  void* tensorData              = ioTensorManager.getBuffer(qnn_tensor.tensor);
  size_t numElements            = qnn_tensor.dims.getNumElements();
  TensorDataType quallaDataType = mapQnnDataTypeToQualla(qnn_tensor.dtype);
  TensorQuantizationParams quantizationParams{1.0, 0};
  if (!qnn_tensor.quantParam.empty()) {
    const auto& qp     = qnn_tensor.quantParam[0];
    quantizationParams = {qp.scale, qp.offset};
    if (qnn_tensor.quantParam.size() > 1) {
      throw std::runtime_error("only QNN_QUANTIZATION_ENCODING_SCALE_OFFSET is supported!");
    }
  }

  return std::make_unique<Tensor>(tensorData,
                                  quallaDataType,
                                  quantizationParams,
                                  numElements,
                                  qnn_tensor.dims.getVector(),
                                  true);
}

QnnNspBaseModel::QnnNspBaseModel(std::shared_ptr<Env> env, const Params& params)
    : State(env->getTraceLogger()), model_basedir(params.model_basedir), _env(env) {
  GENIE_TRACE();
  // Initialize QnnAPI
  m_qnnApi = std::unique_ptr<QnnApi>(new QnnApi(env->getTraceLogger(), env->getResourceManager()));
  // Debug flags
  _debug_path          = params.debug_path;
  _debug_specs         = params.debug_specs;
  _debug_tensors       = params.debug_tensors;
  _debug_outputs       = params.debug_outputs;
  _debug_qnn           = params.debug_qnn;
  lora_conf_type       = params.lora_conf_type;
  _backend_lib         = params.backend_lib;
  _backend_ext_conf    = params.backend_ext_conf;
  m_use_async_Init     = params.use_async_Init;
  m_lazyInitialization = params.shared_Engine;

  if (lora_conf_type != LoraConfigType::LORA_DISABLE) {
    lora_config = params.lora_config;
  }

  if (!m_lazyInitialization) {
    // Initialize QNN IO Tensor
    m_ioTensor = std::shared_ptr<IOTensor>(
        new IOTensor(m_sharedBuffer ? BufferType::SHARED_BUFFER : BufferType::DEFAULT,
                     m_sharedBuffer ? m_qnnApi->getQnnInterfaceVer() : nullptr));
    m_qnnApi->setIOTensor(m_ioTensor);
  }
}

QnnNspBaseModel::~QnnNspBaseModel() {
  if (m_ioTensor) {
    m_ioTensor->deRegisterAll();
  }
}

bool QnnNspBaseModel::isSupportedActivation(Qnn_DataType_t type) const {
  static const std::unordered_set<Qnn_DataType_t> s_supportedActivations{
      QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_UFIXED_POINT_16,
      QNN_DATATYPE_INT_32,
      QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_16};

  return s_supportedActivations.contains(type);
}

bool QnnNspBaseModel::float32ToFloat16(uint8_t* out, float* in, size_t numElements) {
  if (!numElements) return false;
  uint16_t* temp = reinterpret_cast<uint16_t*>(out);
  for (size_t i = 0; i < numElements; i++) {
    temp[i] = fp16_ieee_from_fp32_value(in[i]);
  }
  return true;
}

bool QnnNspBaseModel::setOemKey(const std::string& oemKey) {
  if (m_qnnApi != nullptr) {
    return m_qnnApi->setOemKey(oemKey);
  }
  return false;
}

bool QnnNspBaseModel::releaseLoraAdapter(const std::string &lora_adapter_name) {
  if (lora_conf_type != LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    __ERROR("qnn-htp: Lora config is not enabled for adapters");
    return false;
  }
  auto adapter = lora_config->getAdapter(lora_adapter_name);
  if (!adapter) {
    __ERROR("qnn-htp: Could not find lora adapter {} to release", lora_adapter_name);
    return false;
  }

  if ((graph_switching && lazy_lora == "lazy") || weight_shared_lora) {
    m_qnnApi->m_adapterCache.clear();
    m_qnnApi->resetAdapterGraphHandleState();
  }

  // Release adapter binary sections and track memory
  for (const auto& binSection : adapter->m_binList) {
    if (!binSection.empty()) {
      m_qnnApi->clearAdapterBuffer(binSection);
    }
  }

  if (!adapter->m_groupName.empty()) {
    for (const auto& quantBinSection : adapter->m_quantBinList) {
      if (!quantBinSection.empty()) {
        m_qnnApi->clearAdapterBuffer(quantBinSection);
      }
    }
  }

  if (lora_config->getAppliedAdapterName() == lora_adapter_name) {
    lora_config->updateAppliedAdapterName("");
    if (lora_conf_type == LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE && _lora_enabled) {
      if (!flushLoraWeightsBuffers()) {
        __ERROR("qnn-htp: Failed to flush LoRA weights buffers");
        return false;
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::setExecutionPriority(const uint32_t executionPriority) {
  if (m_qnnApi != nullptr) {
    return m_qnnApi->setExecutionPriority(static_cast<Qnn_Priority_t>(executionPriority));
  }
  return false;
}

bool QnnNspBaseModel::flushLoraWeightsBuffers(void) {
  if (!_lora_enabled) {
    __ERROR("qnn-htp: Model does not support LoRA weights.");
    return false;
  }

  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.find("lora") !=
          std::string::npos) {  // find lora weights tensors and flush them out
        if (getBuffer(tspec) == nullptr) return false;
        size_t numElements = tspec.dims.getNumElements();
        auto offset        = tspec.quantParam[0].offset;
        // Since values needs to be quantized so zero is going to get translated.
        switch (tspec.dtype) {
          case QNN_DATATYPE_UFIXED_POINT_8:
            std::fill_n(reinterpret_cast<uint8_t*>(getBuffer(tspec)),
                        numElements,
                        static_cast<uint8_t>(-offset));
            break;
          case QNN_DATATYPE_UFIXED_POINT_16:
            std::fill_n(reinterpret_cast<uint16_t*>(getBuffer(tspec)),
                        numElements,
                        static_cast<uint16_t>(-offset));
            break;
          case QNN_DATATYPE_FLOAT_16: {
            uint16_t* buffer = reinterpret_cast<uint16_t*>(getBuffer(tspec));
            for (size_t i = 0; i < numElements; i++) {
              buffer[i] = fp16_ieee_from_fp32_value(static_cast<float>(-offset));
            }
            break;
          }
          default:
            __ERROR("Unsupported {} datatype for {} tensor", tspec.dtype.str(), tname);
            return false;
        }
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::applyLoraWeights(const std::string& lora_weights_name) {
  if (!_lora_enabled) {
    __ERROR("qnn-htp: Model does not support LoRA weights.");
    return false;
  }
  if (lora_conf_type != LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    __ERROR("qnn-htp: LoRA config is not enable for input weights");
    return false;
  }
  auto curAdapter = lora_config->getAdapter(lora_weights_name);
  if (!curAdapter) {
    __ERROR("qnn-htp: Could not find lora weights config to apply ");
    return false;
  }

  if (curAdapter->m_weightPath.empty()) {
    __ERROR("qnn-htp: LoRA weights dir is empty for {}", lora_weights_name);
    return false;
  }

  for (auto& alphaName : curAdapter->m_alphaTensorList) {
    if (!applyLoraStrength(alphaName, lora_config->getCachedAlphaVal(alphaName))) {
      __ERROR("qnn-htp: Could not apply Alpha tensor ");
      return false;
    }
  }

  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.find("lora") != std::string::npos && tname != lora_config->getAlphaTensorName()) {
        if (getBuffer(tspec) == nullptr) return false;
        // lora tensor file names should be in same format as tensor names present in graph
        std::string lora_weights_file =
            (model_basedir / fs::path(curAdapter->m_weightPath) / fs::path(tname + ".raw"))
                .string();

        size_t numElements = tspec.dims.getNumElements();
        auto scale         = tspec.quantParam[0].scale;
        auto offset        = tspec.quantParam[0].offset;

        size_t size = sizeof(float);
        std::vector<float> lora_weights_f32;  // Temporary variable to load fp32 values
        lora_weights_f32.resize(numElements);

        FILE* fp = fopen(lora_weights_file.c_str(), "rb");
        if (fp == NULL) {
          __ERROR("NSPModel: Error opening file: {}", lora_weights_file);
          return false;
        }

        size_t count = fread(lora_weights_f32.data(), size, numElements, fp);
        fclose(fp);

        if (count != numElements) {
          __ERROR("NSPModel: Could not load {} - expected file size {}",
                  lora_weights_file,
                  numElements * size);
          return false;
        }

        // Quantize the values
        switch (tspec.dtype) {
          case QNN_DATATYPE_UFIXED_POINT_8:
            QnnUtils::quantizeTensorPtr(lora_weights_f32.data(),
                                        reinterpret_cast<uint8_t*>(getBuffer(tspec)),
                                        offset,
                                        scale,
                                        numElements);
            break;
          case QNN_DATATYPE_UFIXED_POINT_16:
            QnnUtils::quantizeTensorPtr(lora_weights_f32.data(),
                                        reinterpret_cast<uint16_t*>(getBuffer(tspec)),
                                        offset,
                                        scale,
                                        numElements);
            break;
          case QNN_DATATYPE_FLOAT_16:
            float32ToFloat16(
                reinterpret_cast<uint8_t*>(getBuffer(tspec)), lora_weights_f32.data(), numElements);
            break;
          default:
            __ERROR("Unsupported {} datatype for {} tensor", tspec.dtype.str(), tname);
            return false;
        }
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::applyBinarySections(std::vector<std::string>& binsection_list) {
  if (weight_shared_lora && binsection_list.size() == 0) {
    __ERROR(
        "qnn-htp: weight_shared_lora model requires an adapter to be applied before execution.");
    return false;
  }

  if ((graph_switching && lazy_lora == "lazy") || weight_shared_lora) {
    m_qnnApi->m_adapterCache.clear();
  }

  // apply binary section for lora config
  for (size_t i = 0; i < binsection_list.size(); i++) {
    if (binsection_list.at(i).empty()) continue;
    __DEBUG("qnn-htp: applyBinarySections adapters {}", binsection_list.at(i));
    if (!m_qnnApi->applyBinarySection(
            i, binsection_list.at(i), m_use_mmap, graph_switching, lazy_lora, weight_shared_lora)) {
      __ERROR("qnn-htp: Error in applyBinarySections {}", i);
      return false;
    }
  }
  return true;
}

bool QnnNspBaseModel::applyLoraStrength(const std::string& alpha_name, const float alpha_val) {
  if (lora_config->getAlphaTensorName().empty() || alpha_name.empty()) return true;

  if (!lora_config->hasAlpha(alpha_name)) {
    __ERROR("qnn-htp: Could not find lora alpha tensor to apply");
    return false;
  }

  lora_config->updateCacheAlphaVal(alpha_name, alpha_val);
  auto curAdapter = lora_config->getAppliedAdapter();
  if (curAdapter) {
    for (size_t idx = 0; idx < curAdapter->m_alphaTensorVal.size(); idx++) {
      curAdapter->m_alphaTensorVal[idx] = lora_config->getCachedAlphaVal(alpha_name);
    }
  } else {
    // Alpha tensor gets set (below) when adapter is applied.
    return true;
  }

  for (auto& variant : m_variant_list) {
    if (!variant.input_specs.contains(lora_config->getAlphaTensorName())) continue;

    auto& tspec          = variant.input_specs.at(lora_config->getAlphaTensorName());
    auto [scale, offset] = tspec.quantParam[0];

    switch (tspec.dtype) {
      case QNN_DATATYPE_UFIXED_POINT_8:
        QnnUtils::quantizeTensorPtr(curAdapter->m_alphaTensorVal.data(),
                                    reinterpret_cast<uint8_t*>(getBuffer(tspec)),
                                    offset,
                                    scale,
                                    curAdapter->m_alphaTensorVal.size());
        break;
      case QNN_DATATYPE_UFIXED_POINT_16:
        QnnUtils::quantizeTensorPtr(curAdapter->m_alphaTensorVal.data(),
                                    reinterpret_cast<uint16_t*>(getBuffer(tspec)),
                                    offset,
                                    scale,
                                    curAdapter->m_alphaTensorVal.size());
        break;
      case QNN_DATATYPE_FLOAT_16:
        float32ToFloat16(reinterpret_cast<uint8_t*>(getBuffer(tspec)),
                         const_cast<float*>(curAdapter->m_alphaTensorVal.data()),
                         curAdapter->m_alphaTensorVal.size());
        break;
      default:
        __ERROR("Unsupported alpha tensor dtype {}", tspec.dtype.str());
        return false;
    }
    __DEBUG("qnn-htp: applyAlphaTensor alpha = {}", alpha_val);
    return true;  // Each lora bin section should have only one alpha tensor
  }
  return false;
}

bool QnnNspBaseModel::applyLoraAdapter(const std::string& lora_adapter_name) {
  if (lora_conf_type != LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    __ERROR("qnn-htp: Lora config is not enable for adapters");
    return false;
  }

  auto curAdapter = lora_config->getAdapter(lora_adapter_name);
  if (!curAdapter) {
    __ERROR("qnn-htp: Could not find lora adapters config to apply ");
    return false;
  }

  // apply default strengths
  for (auto& alphaName : curAdapter->m_alphaTensorList) {
    if (!applyLoraStrength(alphaName, lora_config->getCachedAlphaVal(alphaName))) {
      __ERROR("qnn-htp: Could not apply Alpha tensor ");
      return false;
    }
  }

  auto lastAdapter = lora_config->getAppliedAdapter();

  std::string lastGroupName = "";
  if (lastAdapter) {
    lastGroupName = lastAdapter->m_groupName;
  }

  if (!curAdapter->m_groupName.empty() && curAdapter->m_groupName != lastGroupName) {
    if (!applyBinarySections(curAdapter->m_quantBinList)) {
      __ERROR("qnn-htp: Could not apply quant binary Sections ");
      return false;
    }
  }

  if (!applyBinarySections(curAdapter->m_binList)) {
    __ERROR("qnn-htp: Could not apply binary Sections ");
    return false;
  }

  for (auto& variant : m_variant_list) {
    variant.refreshTensorQuantParams();
  }

  lora_config->updateAppliedAdapterName(
      lora_adapter_name);  // always update applied adapter and same will be used for group adapter

  return true;
}

bool QnnNspBaseModel::setPerfProfile(qualla::PerformanceProfile& perfProfile) {
  return m_qnnApi->setPerfProfile(perfProfile);
}

bool QnnNspBaseModel::getPerfProfile(qualla::PerformanceProfile& perfProfile) {
  perfProfile = m_qnnApi->getPerfProfile();
  return true;
}

// Dumps out the specified tensor to _debug_path numbered according to m_inference_count
bool QnnNspBaseModel::debugOutputs(const InferenceStep& step, const std::string& tensor_name) {
  GraphVariant* graph_variant = m_nsp_graphs.back()(step.variant, step.ctx_size);
  QnnUtils::Tensor* tensor    = graph_variant->getOutput(tensor_name);
  if (tensor == nullptr) {
    __DEBUG("qnn-htp: Couldn't find tensor {} in graph {}", tensor_name, graph_variant->graph_name);
    return false;
  }

  uint32_t output_bitwidth =
      static_cast<uint32_t>(tensor->dtype.bw());  // Detect 8-bit vs 16-bit logits
  uint32_t rank      = tensor->tensor->v1.rank;
  size_t numElements = 1;
  for (size_t i = 0; i < rank; i++) {
    numElements *= tensor->tensor->v1.dimensions[i];
  }
  size_t output_size = output_bitwidth * numElements;
  std::string fname  = fmt::format("{}/{}/{:03d}", _debug_path, tensor_name, m_inference_count);
  if (!QnnUtils::writeRawData(getBuffer(tensor), output_size, fname)) {
    __DEBUG("qnn-htp: Failed to save {}. Error when writing to {}", tensor_name, fname);
    return false;
  }

  return true;
}

void QnnNspBaseModel::dumpTensorSpecs() {
  GENIE_TRACE();
  if (!fs::exists(_debug_path) && !fs::create_directories(_debug_path)) {
    __ERROR("Could not create directory for debug - {}", _debug_path);
    return;
  }

  const uint32_t n_graphs                     = m_qnnApi->getGraphsCount();
  qnn_wrapper_api::GraphInfo_t**& graphs_info = m_qnnApi->getGraphsInfo();

  for (size_t graph_idx = 0; graph_idx < n_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];

    // Create output spec file and open it
    std::string filename = fmt::format("{}/spec.{}.json", _debug_path, graph_info->graphName);

    FILE* specFile = fopen(filename.c_str(), "w");
    if (specFile == NULL) throw std::runtime_error("Error opening file : " + filename);

    fprintf(specFile, "{\n\t\"graph_name\" : \"%s\",\n", graph_info->graphName);
    for (bool io : {true, false}) {
      uint32_t n_tensors   = io ? graph_info->numInputTensors : graph_info->numOutputTensors;
      Qnn_Tensor_t* tensor = io ? graph_info->inputTensors : graph_info->outputTensors;

      fprintf(specFile, (io) ? "\t\"inputs\" : [\n" : "\t\"outputs\" : [\n");
      while (n_tensors-- > 0) {
        QnnUtils::Tensor tensorW(tensor);

        std::string scales, offsets;
        QnnUtils::getQuantParamString(tensorW.quantParam, scales, offsets);

        fprintf(specFile,
                "\t\t{ \"name\": \"%s\", \"dims\": [%d, %d, %d, %d], \"bitwidth\": %d, "
                "\"dtype\": \"%s\", \"dataFormat\": %u, \"scale\": [%s], \"offset\": [%s] },\n",
                tensorW.name.c_str(),
                tensorW.dims.batch,
                tensorW.dims.height,
                tensorW.dims.width,
                tensorW.dims.channel,
                tensorW.dims.bitwidth,
                tensorW.dtype.str(),
                QNN_TENSOR_GET_DATA_FORMAT(tensor),
                scales.c_str(),
                offsets.c_str());

        tensor++;
      }
      fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma
      fprintf(specFile, "\n\t],\n");
    }
    fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma
    fprintf(specFile, "\n}");
    fclose(specFile);
  }
}

bool QnnNspBaseModel::finalizeLora(std::shared_ptr<EngineState>& engineState) {
  auto newLoraConfig = engineState->getLoraConfig();
  if (!newLoraConfig || newLoraConfig->getLoraConfigType() == LoraConfigType::LORA_DISABLE)
    return true;
  LoraEventType loraEvent = newLoraConfig->getEventType();
  if (lora_config &&
      newLoraConfig->getAppliedAdapterName() == lora_config->getAppliedAdapterName()) {
    loraEvent = LoraEventType::NO_EVENT;
  }
  lora_config    = newLoraConfig;
  lora_conf_type = lora_config->getLoraConfigType();
  if (lora_config->getAppliedAdapterName().empty()) loraEvent = LoraEventType::NO_EVENT;
  if (loraEvent == LoraEventType::APPLY_EVENT) {
    if (lora_conf_type == LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
      if (!applyLoraAdapter(lora_config->getAppliedAdapterName())) {
        return false;
      }
    } else if (lora_conf_type == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
      if (!applyLoraWeights(lora_config->getAppliedAdapterName())) {
        return false;
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::finalizeState(std::shared_ptr<EngineState>& engineState) {
  // Update logging env

  // Update RoPE if needed

  // Update Debug Config if needed

  // Update IO  if needed

  IOEVENT event = engineState->isInitialize() ? engineState->getIOBuffer()->m_event
                                              : IOEVENT::ALLOCATE_REGISTER_EVENT;

  if (event == IOEVENT::NO_EVENT) {
    return true;
  }

  if (m_ioTensor) {
    m_ioTensor->deRegisterAll();
  }

  if (event == IOEVENT::ALLOCATE_REGISTER_EVENT || event == IOEVENT::ALLOCATE_EVENT) {
    // ReInitialize IO Tensor
    m_ioTensor.reset();
    m_ioTensor = std::shared_ptr<IOTensor>(
        new IOTensor(m_sharedBuffer ? BufferType::SHARED_BUFFER : BufferType::DEFAULT,
                     m_sharedBuffer ? m_qnnApi->getQnnInterfaceVer() : nullptr));
  } else if (event == IOEVENT::REGISTER_EVENT) {
    m_ioTensor = std::dynamic_pointer_cast<IOTensor>(engineState->getIOBuffer());
    if (!m_ioTensor->initializeRegistrar()) {
      QNN_ERROR("Failed to register the IO buffers.");
      return false;
    }
  }

  // Always update IO Tensor
  m_qnnApi->setIOTensor(m_ioTensor);

  if (event == IOEVENT::ALLOCATE_REGISTER_EVENT || event == IOEVENT::ALLOCATE_EVENT) {
    // allocate
    // note::qnnApi will do the allocation.
    if (true != m_qnnApi->allocateAll()) {
      __ERROR("Failed to Allocate buffers");
      return false;
    }
  }
  if (event == IOEVENT::REGISTER_EVENT || event == IOEVENT::ALLOCATE_REGISTER_EVENT) {
    // only register
    // note::qnnApi will do the registration.
    if (true != m_qnnApi->registerAll()) {
      __ERROR("Failed to Register the buffers with IO Tensors");
      return false;
    }
  }

  // Update LoRA if needed
  if (!finalizeLora(engineState)) {
    __ERROR("EngineState: finalize Lora state failed");
    return false;
  }

  return true;
}

bool QnnNspBaseModel::registerModelIOAdaptor(std::shared_ptr<ModelIOAdaptor> adaptor) {
  GENIE_TRACE();

  if (!adaptor) {
    __ERROR("nsp-base-model: Cannot register null ModelIOAdaptor");
    return false;
  }

  __DEBUG("nsp-base-model: Registering ModelIOAdaptor: {}", adaptor->name());

  // Create ModelIOSpec from the loaded model variants
  auto& modelIOSpec = IOSpec();

  // Call setup on the adaptor
  if (!adaptor->setup(_env, modelIOSpec)) {
    __ERROR("nsp-base-model: ModelIOAdaptor '{}' setup failed", adaptor->name());
    return false;
  }

  // Store the adaptor
  m_adaptors.push_back(adaptor);

  __DEBUG("nsp-base-model: Successfully registered ModelIOAdaptor: {}", adaptor->name());
  return true;
}

NspModelIOSpec& QnnNspBaseModel::IOSpec() {
  if (!m_ModelIOSpec) {
    m_ModelIOSpec = std::make_unique<NspModelIOSpec>();
    for (const auto& variant : m_variant_list) {
      m_ModelIOSpec->addModelVariant({variant.n_tokens, variant.ctx_size},
                                     variant.input_specs,
                                     variant.output_specs,
                                     *m_ioTensor);
    }
  }
  return *m_ModelIOSpec.get();
}

void QnnNspBaseModel::callAdaptorsReset() {
  for (auto& adaptor : m_adaptors) {
    __DEBUG("nsp-base-model: Calling reset on adaptor: {}", adaptor->name());
    adaptor->reset();
  }
}

bool QnnNspBaseModel::prepareAdaptorInputs(int32_t variant,
                                           int32_t ctx_size,
                                           uint32_t n_past,
                                           uint32_t n_processed,
                                           uint32_t n_inputs,
                                           bool isLastInference,
                                           bool& shouldSkipInference) {
  GENIE_TRACE();

  // Early return if no adaptors registered
  if (m_adaptors.empty()) {
    shouldSkipInference = false;
    return true;
  }

  // Build InferenceInfo from the parameters
  InferenceInfo info;
  info.variant         = {variant, ctx_size};
  info.n_past          = n_past;
  info.n_processed     = n_processed;
  info.n_inputs        = n_inputs;
  info.isLastInference = isLastInference;

  const TensorMap* input_spec = IOSpec().getInputSpec(info.variant);

  if (!input_spec) {
    __ERROR("nsp-base-model: Failed to get input spec for AR-{} CL-{}", variant, ctx_size);
    return false;
  }

  // Call each adaptor's prepareInputs
  for (auto& adaptor : m_adaptors) {
    if (!adaptor->prepareInputs(info, *input_spec)) {
      __DEBUG("nsp-base-model: Adaptor '{}' requested to skip inference", adaptor->name());
      shouldSkipInference = true;
    }
  }
  return true;
}

bool QnnNspBaseModel::handleAdaptorOutputs(int32_t variant,
                                           int32_t ctx_size,
                                           uint32_t n_past,
                                           uint32_t n_processed,
                                           uint32_t n_inputs,
                                           bool isLastInference) {
  GENIE_TRACE();

  // Early return if no adaptors registered
  if (m_adaptors.empty()) {
    return true;
  }

  // Build InferenceInfo from the parameters
  InferenceInfo info;
  info.variant         = {variant, ctx_size};
  info.n_past          = n_past;
  info.n_processed     = n_processed;
  info.n_inputs        = n_inputs;
  info.isLastInference = isLastInference;

  // Build output TensorMap from the current variant
  const TensorMap* output_spec = IOSpec().getOutputSpec(info.variant);

  if (!output_spec) {
    __ERROR("nsp-base-model: Failed to get output spec for AR-{} CL-{}", variant, ctx_size);
    return false;
  }

  // Call each adaptor's handleOutputs
  for (auto& adaptor : m_adaptors) {
    adaptor->handleOutputs(info, *output_spec);
  }
  return true;
}

void QnnNspBaseModel::resetState() {
  if (m_qnnApi) {
    m_qnnApi->resetAdapterGraphHandleState();
  }
}

}  // namespace qualla
