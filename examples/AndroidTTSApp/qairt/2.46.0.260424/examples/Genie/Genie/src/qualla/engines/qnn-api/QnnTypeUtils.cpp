//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include "QnnTypeUtils.hpp"

#define STR(x) #x

namespace aiswutility {

float getDataTypeSize(const Qnn_DataType_t dataType) {
  switch (dataType) {
    // All 2 bit values
    case QNN_DATATYPE_INT_2:
    case QNN_DATATYPE_UINT_2:
    case QNN_DATATYPE_SFIXED_POINT_2:
    case QNN_DATATYPE_UFIXED_POINT_2:
      return 0.25f;
    // All 4 bit values
    case QNN_DATATYPE_INT_4:
    case QNN_DATATYPE_UINT_4:
    case QNN_DATATYPE_FLOAT_4:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_4:
      return 0.5f;
    // All 8 bit values
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_FLOAT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return 1.0f;
    // All 16 bit values
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2.0f;
    // All 32 bit values
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4.0f;
    // All 64 bit values
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8.0f;
    case QNN_DATATYPE_UNDEFINED:
    default:
      return 0.0f;
  }
}

uint32_t getDataTypeContainerSize(const Qnn_DataType_t dataType) {
  switch (dataType) {
    // All 2 and 4 bit values have a container size of 1 byte
    case QNN_DATATYPE_INT_2:
    case QNN_DATATYPE_INT_4:
    case QNN_DATATYPE_UINT_2:
    case QNN_DATATYPE_UINT_4:
    case QNN_DATATYPE_FLOAT_4:
    case QNN_DATATYPE_SFIXED_POINT_2:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_2:
    case QNN_DATATYPE_UFIXED_POINT_4:
      return 1u;
    default:
      return static_cast<uint32_t>(getDataTypeSize(dataType));
  }
}

size_t getDataTypeBitWidth(const Qnn_DataType_t dataType) {
  return (8u * getDataTypeSize(dataType));
}

const char* dataTypeToString(const Qnn_DataType_t dataType) {
  const char* valData;
  switch (dataType) {
    case QNN_DATATYPE_INT_2:
      valData = STR(QNN_DATATYPE_INT_2);
      break;
    case QNN_DATATYPE_INT_4:
      valData = STR(QNN_DATATYPE_INT_4);
      break;
    case QNN_DATATYPE_INT_8:
      valData = STR(QNN_DATATYPE_INT_8);
      break;
    case QNN_DATATYPE_INT_16:
      valData = STR(QNN_DATATYPE_INT_16);
      break;
    case QNN_DATATYPE_INT_32:
      valData = STR(QNN_DATATYPE_INT_32);
      break;
    case QNN_DATATYPE_INT_64:
      valData = STR(QNN_DATATYPE_INT_64);
      break;
    case QNN_DATATYPE_UINT_2:
      valData = STR(QNN_DATATYPE_UINT_2);
      break;
    case QNN_DATATYPE_UINT_4:
      valData = STR(QNN_DATATYPE_UINT_4);
      break;
    case QNN_DATATYPE_UINT_8:
      valData = STR(QNN_DATATYPE_UINT_8);
      break;
    case QNN_DATATYPE_UINT_16:
      valData = STR(QNN_DATATYPE_UINT_16);
      break;
    case QNN_DATATYPE_UINT_32:
      valData = STR(QNN_DATATYPE_UINT_32);
      break;
    case QNN_DATATYPE_UINT_64:
      valData = STR(QNN_DATATYPE_UINT_64);
      break;
    case QNN_DATATYPE_FLOAT_4:
      valData = STR(QNN_DATATYPE_FLOAT_4);
      break;
    case QNN_DATATYPE_FLOAT_8:
      valData = STR(QNN_DATATYPE_FLOAT_8);
      break;
    case QNN_DATATYPE_FLOAT_16:
      valData = STR(QNN_DATATYPE_FLOAT_16);
      break;
    case QNN_DATATYPE_BFLOAT_16:
      valData = STR(QNN_DATATYPE_BFLOAT_16);
      break;
    case QNN_DATATYPE_FLOAT_32:
      valData = STR(QNN_DATATYPE_FLOAT_32);
      break;
    case QNN_DATATYPE_FLOAT_64:
      valData = STR(QNN_DATATYPE_FLOAT_64);
      break;
    case QNN_DATATYPE_SFIXED_POINT_2:
      valData = STR(QNN_DATATYPE_SFIXED_POINT_2);
      break;
    case QNN_DATATYPE_SFIXED_POINT_4:
      valData = STR(QNN_DATATYPE_SFIXED_POINT_4);
      break;
    case QNN_DATATYPE_SFIXED_POINT_8:
      valData = STR(QNN_DATATYPE_SFIXED_POINT_8);
      break;
    case QNN_DATATYPE_SFIXED_POINT_16:
      valData = STR(QNN_DATATYPE_SFIXED_POINT_16);
      break;
    case QNN_DATATYPE_SFIXED_POINT_32:
      valData = STR(QNN_DATATYPE_SFIXED_POINT_32);
      break;
    case QNN_DATATYPE_UFIXED_POINT_2:
      valData = STR(QNN_DATATYPE_UFIXED_POINT_2);
      break;
    case QNN_DATATYPE_UFIXED_POINT_4:
      valData = STR(QNN_DATATYPE_UFIXED_POINT_4);
      break;
    case QNN_DATATYPE_UFIXED_POINT_8:
      valData = STR(QNN_DATATYPE_UFIXED_POINT_8);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      valData = STR(QNN_DATATYPE_UFIXED_POINT_16);
      break;
    case QNN_DATATYPE_UFIXED_POINT_32:
      valData = STR(QNN_DATATYPE_UFIXED_POINT_32);
      break;
    case QNN_DATATYPE_BOOL_8:
      valData = STR(QNN_DATATYPE_BOOL_8);
      break;
    case QNN_DATATYPE_UNDEFINED:
      valData = STR(QNN_DATATYPE_UNDEFINED);
      break;
    case QNN_DATATYPE_STRING:
      valData = STR(QNN_DATATYPE_STRING);
      break;
    default:
      valData = STR(UNKNOWN);
      break;
  }
  return valData;
}

Qnn_DataType_t dataTypefromString(const std::string& dataType) {
  if (dataType == "int2" || dataType == "QNN_DATATYPE_INT_2") {
    return QNN_DATATYPE_INT_2;
  } else if (dataType == "int4" || dataType == "QNN_DATATYPE_INT_4") {
    return QNN_DATATYPE_INT_4;
  } else if (dataType == "int8" || dataType == "QNN_DATATYPE_INT_8") {
    return QNN_DATATYPE_INT_8;
  } else if (dataType == "int16" || dataType == "QNN_DATATYPE_INT_16") {
    return QNN_DATATYPE_INT_16;
  } else if (dataType == "int32" || dataType == "int" || dataType == "QNN_DATATYPE_INT_32") {
    return QNN_DATATYPE_INT_32;
  } else if (dataType == "int64" || dataType == "QNN_DATATYPE_INT_64") {
    return QNN_DATATYPE_INT_64;
  } else if (dataType == "uint2" || dataType == "QNN_DATATYPE_UINT_2") {
    return QNN_DATATYPE_UINT_2;
  } else if (dataType == "uint4" || dataType == "QNN_DATATYPE_UINT_4") {
    return QNN_DATATYPE_UINT_4;
  } else if (dataType == "uint8" || dataType == "QNN_DATATYPE_UINT_8") {
    return QNN_DATATYPE_UINT_8;
  } else if (dataType == "uint16" || dataType == "QNN_DATATYPE_UINT_16") {
    return QNN_DATATYPE_UINT_16;
  } else if (dataType == "uint32" || dataType == "uint" || dataType == "QNN_DATATYPE_UINT_32") {
    return QNN_DATATYPE_UINT_32;
  } else if (dataType == "uint64" || dataType == "QNN_DATATYPE_UINT_64") {
    return QNN_DATATYPE_UINT_64;
  } else if (dataType == "float4" || dataType == "float4e2m1" || dataType == "QNN_DATATYPE_FLOAT_4") {
    return QNN_DATATYPE_FLOAT_4;
  } else if (dataType == "float8" || dataType == "QNN_DATATYPE_FLOAT_8") {
    return QNN_DATATYPE_FLOAT_8;
  } else if (dataType == "float16" || dataType == "QNN_DATATYPE_FLOAT_16") {
    return QNN_DATATYPE_FLOAT_16;
  } else if (dataType == "bfloat16" || dataType == "bfloat" || dataType == "QNN_DATATYPE_BFLOAT_16") {
    return QNN_DATATYPE_BFLOAT_16;
  } else if (dataType == "float32" || dataType == "float" || dataType == "QNN_DATATYPE_FLOAT_32") {
    return QNN_DATATYPE_FLOAT_32;
  } else if (dataType == "float64" || dataType == "double" || dataType == "QNN_DATATYPE_FLOAT_64") {
    return QNN_DATATYPE_FLOAT_64;
  } else if (dataType == "sfixed2" || dataType == "QNN_DATATYPE_SFIXED_POINT_2") {
    return QNN_DATATYPE_SFIXED_POINT_2;
  } else if (dataType == "sfixed4" || dataType == "QNN_DATATYPE_SFIXED_POINT_4") {
    return QNN_DATATYPE_SFIXED_POINT_4;
  } else if (dataType == "sfixed8" || dataType == "QNN_DATATYPE_SFIXED_POINT_8" ) {
    return QNN_DATATYPE_SFIXED_POINT_8;
  } else if (dataType == "sfixed16" || dataType == "QNN_DATATYPE_SFIXED_POINT_16") {
    return QNN_DATATYPE_SFIXED_POINT_16;
  } else if (dataType == "sfixed32" || dataType == "QNN_DATATYPE_SFIXED_POINT_32") {
    return QNN_DATATYPE_SFIXED_POINT_32;
  } else if (dataType == "ufixed2" || dataType == "QNN_DATATYPE_UFIXED_POINT_2") {
    return QNN_DATATYPE_UFIXED_POINT_2;
  } else if (dataType == "ufixed4" || dataType == "QNN_DATATYPE_UFIXED_POINT_4") {
    return QNN_DATATYPE_UFIXED_POINT_4;
  } else if (dataType == "ufixed8" || dataType == "QNN_DATATYPE_UFIXED_POINT_8") {
    return QNN_DATATYPE_UFIXED_POINT_8;
  } else if (dataType == "ufixed16" || dataType == "QNN_DATATYPE_UFIXED_POINT_16") {
    return QNN_DATATYPE_UFIXED_POINT_16;
  } else if (dataType == "ufixed32" || dataType == "QNN_DATATYPE_UFIXED_POINT_32") {
    return QNN_DATATYPE_UFIXED_POINT_32;
  } else if (dataType == "bool" || dataType == "QNN_DATATYPE_BOOL_8") {
    return QNN_DATATYPE_BOOL_8;
  } else if (dataType == "string" || dataType == "QNN_DATATYPE_STRING") {
    return QNN_DATATYPE_STRING;
  } else {
    return QNN_DATATYPE_UNDEFINED;
  }
}

bool isSignedIntDataType(const Qnn_DataType_t dataType) {
  bool val;
  switch (dataType) {
    case QNN_DATATYPE_INT_2:
    case QNN_DATATYPE_INT_4:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_INT_64:
      val = true;
      break;
    default:
      val = false;
      break;
  }
  return val;
}

bool isUnsignedIntDataType(const Qnn_DataType_t dataType) {
  bool val;
  switch (dataType) {
    case QNN_DATATYPE_UINT_2:
    case QNN_DATATYPE_UINT_4:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UINT_64:
      val = true;
      break;
    default:
      val = false;
      break;
  }
  return val;
}

bool isIntDataType(const Qnn_DataType_t dataType) {
  return isSignedIntDataType(dataType) || isUnsignedIntDataType(dataType);
}

bool isFloatDataType(const Qnn_DataType_t dataType) {
  bool val;
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_FLOAT_64:
      val = true;
      break;
    default:
      val = false;
      break;
  }
  return val;
}

bool isSignedQuantizedDataType(const Qnn_DataType_t dataType) {
  bool val;
  switch (dataType) {
    case QNN_DATATYPE_SFIXED_POINT_2:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_32:
      val = true;
      break;
    default:
      val = false;
      break;
  }
  return val;
}

bool isUnsignedQuantizedDataType(const Qnn_DataType_t dataType) {
  bool val;
  switch (dataType) {
    case QNN_DATATYPE_UFIXED_POINT_2:
    case QNN_DATATYPE_UFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_32:
      val = true;
      break;
    default:
      val = false;
      break;
  }
  return val;
}

bool isQuantizedDataType(const Qnn_DataType_t dataType) {
  return isSignedQuantizedDataType(dataType) || isUnsignedQuantizedDataType(dataType);
}

const char* capabilityToString(const QnnProperty_Key_t key) {
  const char* ret = nullptr;
  switch (key) {
    case QNN_PROPERTY_GROUP_CORE:
      ret = STR(QNN_PROPERTY_GROUP_CORE);
      break;
    case QNN_PROPERTY_GROUP_BACKEND:
      ret = STR(QNN_PROPERTY_GROUP_BACKEND);
      break;
    case QNN_PROPERTY_BACKEND_SUPPORT_OP_PACKAGE:
      ret = STR(QNN_PROPERTY_BACKEND_SUPPORT_OP_PACKAGE);
      break;
    case QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS:
      ret = STR(QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS);
      break;
    case QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION:
      ret = STR(QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION);
      break;
    case QNN_PROPERTY_GROUP_CONTEXT:
      ret = STR(QNN_PROPERTY_GROUP_CONTEXT);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CACHING:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CACHING);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_ENABLE_GRAPHS:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_ENABLE_GRAPHS);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_MEMORY_LIMIT_HINT:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_MEMORY_LIMIT_HINT);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_PERSISTENT_BINARY:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_PERSISTENT_BINARY);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_BINARY_COMPATIBILITY_TYPE:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_BINARY_COMPATIBILITY_TYPE);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_VALIDATE_BINARY:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_VALIDATE_BINARY);
      break;
    case QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_WITH_SIGNALS:
      ret = STR(QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_WITH_SIGNALS);
      break;
    case QNN_PROPERTY_GROUP_GRAPH:
      ret = STR(QNN_PROPERTY_GROUP_GRAPH);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_CONFIG:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_CONFIG);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_SIGNALS:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_SIGNALS);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_NULL_INPUTS:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_NULL_INPUTS);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_PRIORITY_CONTROL:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_PRIORITY_CONTROL);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_CONTINUOUS_PROFILING:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_CONTINUOUS_PROFILING);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_PER_API_PROFILING:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_PER_API_PROFILING);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_SUBGRAPH:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_SUBGRAPH);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_PROFILING_STATE:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_PROFILING_STATE);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_SET_PROFILING_NUM_EXECUTIONS:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_SET_PROFILING_NUM_EXECUTIONS);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_BIND_MEM_HANDLES:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_BIND_MEM_HANDLES);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_POPULATE_CLIENT_BUFS:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_POPULATE_CLIENT_BUFS);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_DESERIALIZED_GRAPH:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_DESERIALIZED_GRAPH);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_CUSTOM_PROPERTY:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_CUSTOM_PROPERTY);
      break;
    case QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION:
      ret = STR(QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION);
      break;
    case QNN_PROPERTY_GROUP_OP_PACKAGE:
      ret = STR(QNN_PROPERTY_GROUP_OP_PACKAGE);
      break;
    case QNN_PROPERTY_OP_PACKAGE_SUPPORTS_VALIDATION:
      ret = STR(QNN_PROPERTY_OP_PACKAGE_SUPPORTS_VALIDATION);
      break;
    case QNN_PROPERTY_OP_PACKAGE_SUPPORTS_OP_IMPLS:
      ret = STR(QNN_PROPERTY_OP_PACKAGE_SUPPORTS_OP_IMPLS);
      break;
    case QNN_PROPERTY_OP_PACKAGE_SUPPORTS_DUPLICATE_NAMES:
      ret = STR(QNN_PROPERTY_OP_PACKAGE_SUPPORTS_DUPLICATE_NAMES);
      break;
    case QNN_PROPERTY_GROUP_TENSOR:
      ret = STR(QNN_PROPERTY_GROUP_TENSOR);
      break;
    case QNN_PROPERTY_TENSOR_SUPPORT_MEMHANDLE_TYPE:
      ret = STR(QNN_PROPERTY_TENSOR_SUPPORT_MEMHANDLE_TYPE);
      break;
    case QNN_PROPERTY_TENSOR_SUPPORT_CONTEXT_TENSORS:
      ret = STR(QNN_PROPERTY_TENSOR_SUPPORT_CONTEXT_TENSORS);
      break;
    case QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS:
      ret = STR(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS);
      break;
    case QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY:
      ret = STR(QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY);
      break;
    case QNN_PROPERTY_GROUP_ERROR:
      ret = STR(QNN_PROPERTY_GROUP_ERROR);
      break;
    case QNN_PROPERTY_ERROR_GET_VERBOSE_MESSAGE:
      ret = STR(QNN_PROPERTY_ERROR_GET_VERBOSE_MESSAGE);
      break;
    case QNN_PROPERTY_GROUP_MEMORY:
      ret = STR(QNN_PROPERTY_GROUP_MEMORY);
      break;
    case QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_ION:
      ret = STR(QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_ION);
      break;
    case QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_CUSTOM:
      ret = STR(QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_CUSTOM);
      break;
    case QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_DMA_BUF:
      ret = STR(QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_DMA_BUF);
      break;
    case QNN_PROPERTY_GROUP_SIGNAL:
      ret = STR(QNN_PROPERTY_GROUP_SIGNAL);
      break;
    case QNN_PROPERTY_SIGNAL_SUPPORT_ABORT:
      ret = STR(QNN_PROPERTY_SIGNAL_SUPPORT_ABORT);
      break;
    case QNN_PROPERTY_SIGNAL_SUPPORT_TIMEOUT:
      ret = STR(QNN_PROPERTY_SIGNAL_SUPPORT_TIMEOUT);
      break;
    case QNN_PROPERTY_GROUP_LOG:
      ret = STR(QNN_PROPERTY_GROUP_LOG);
      break;
    case QNN_PROPERTY_LOG_SUPPORTS_DEFAULT_STREAM:
      ret = STR(QNN_PROPERTY_LOG_SUPPORTS_DEFAULT_STREAM);
      break;
    case QNN_PROPERTY_GROUP_PROFILE:
      ret = STR(QNN_PROPERTY_GROUP_PROFILE);
      break;
    case QNN_PROPERTY_PROFILE_SUPPORT_CUSTOM_CONFIG:
      ret = STR(QNN_PROPERTY_PROFILE_SUPPORT_CUSTOM_CONFIG);
      break;
    case QNN_PROPERTY_PROFILE_SUPPORT_MAX_EVENTS_CONFIG:
      ret = STR(QNN_PROPERTY_PROFILE_SUPPORT_MAX_EVENTS_CONFIG);
      break;
    case QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT:
      ret = STR(QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT);
      break;
    case QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG:
      ret = STR(QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG);
      break;
    case QNN_PROPERTY_GROUP_DEVICE:
      ret = STR(QNN_PROPERTY_GROUP_DEVICE);
      break;
    case QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE:
      ret = STR(QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE);
      break;
    case QNN_PROPERTY_GROUP_CUSTOM:
      ret = STR(QNN_PROPERTY_GROUP_CUSTOM);
      break;
    default:
      ret = "UNKNOWN";
      break;
  }
  return ret;
}

const char* priorityToString(Qnn_Priority_t priority) {
  const char* ret = nullptr;
  switch (priority) {
    case QNN_PRIORITY_LOW:
      ret = "QNN_PRIORITY_LOW";
      break;
    case QNN_PRIORITY_NORMAL_LOW:
      ret = "QNN_PRIORITY_NORMAL_LOW";
      break;
    case QNN_PRIORITY_NORMAL:
      // QNN_PRIORITY_DEFAULT deliberately omitted because it creates a duplicate case
      ret = "QNN_PRIORITY_NORMAL";
      break;
    case QNN_PRIORITY_NORMAL_HIGH:
      ret = "QNN_PRIORITY_NORMAL_HIGH";
      break;
    case QNN_PRIORITY_HIGH:
      ret = "QNN_PRIORITY_HIGH";
      break;
    case QNN_PRIORITY_HIGH_PLUS:
      ret = "QNN_PRIORITY_HIGH_PLUS";
      break;
    case QNN_PRIORITY_CRITICAL:
      ret = "QNN_PRIORITY_CRITICAL";
      break;
    case QNN_PRIORITY_CRITICAL_PLUS:
      ret = "QNN_PRIORITY_CRITICAL_PLUS";
      break;
    case QNN_PRIORITY_UNDEFINED:
      ret = "QNN_PRIORITY_UNDEFINED";
      break;
    default:
      ret = "UNKNOWN";
      break;
  }
  return ret;
}

Qnn_Priority_t priorityFromString(const std::string& priorityStr) {
  Qnn_Priority_t ret = QNN_PRIORITY_DEFAULT;
  if (priorityStr == "QNN_PRIORITY_LOW") {
    ret = QNN_PRIORITY_LOW;
  }
  else if (priorityStr == "QNN_PRIORITY_NORMAL_LOW") {
    ret = QNN_PRIORITY_NORMAL_LOW;
  }
  else if (priorityStr == "QNN_PRIORITY_NORMAL") {
    ret = QNN_PRIORITY_NORMAL;
  }
  else if (priorityStr == "QNN_PRIORITY_NORMAL_HIGH") {
    ret = QNN_PRIORITY_NORMAL_HIGH;
  }
  else if (priorityStr == "QNN_PRIORITY_HIGH") {
    ret = QNN_PRIORITY_HIGH;
  }
  else if (priorityStr == "QNN_PRIORITY_HIGH_PLUS") {
    ret = QNN_PRIORITY_HIGH_PLUS;
  }
  else if (priorityStr == "QNN_PRIORITY_CRITICAL") {
    ret = QNN_PRIORITY_CRITICAL;
  }
  else if (priorityStr == "QNN_PRIORITY_CRITICAL_PLUS") {
    ret = QNN_PRIORITY_CRITICAL_PLUS;
  }
  else if(priorityStr == "QNN_PRIORITY_DEFAULT") {
    ret = QNN_PRIORITY_DEFAULT;
  }
  else if (priorityStr == "QNN_PRIORITY_UNDEFINED") {
    ret = QNN_PRIORITY_UNDEFINED;
  }
  return ret;
}

// Qnn_Tensor
const char* tensorVersionToString(const Qnn_TensorVersion_t& version) {
  const char* tensorVersionString = nullptr;
  switch (version) {
    case QNN_TENSOR_VERSION_1:
      tensorVersionString = STR(QNN_TENSOR_VERSION_1);
      break;
    case QNN_TENSOR_VERSION_2:
      tensorVersionString = STR(QNN_TENSOR_VERSION_2);
      break;
    case QNN_TENSOR_VERSION_UNDEFINED:
      tensorVersionString = STR(QNN_TENSOR_VERSION_UNDEFINED);
      break;
    default:
      tensorVersionString = STR(UNKNOWN);
      break;
  }
  return tensorVersionString;
}

Qnn_TensorVersion_t tensorVersionFromString(const char* versionString) {
  Qnn_TensorVersion_t tensorVersion = QNN_TENSOR_VERSION_UNDEFINED;
  if (strcmp(versionString, "QNN_TENSOR_VERSION_1") == 0) {
    tensorVersion = QNN_TENSOR_VERSION_1;
  } else if (strcmp(versionString, "QNN_TENSOR_VERSION_2") == 0) {
    tensorVersion = QNN_TENSOR_VERSION_2;
  }
  return tensorVersion;
}

const char* tensorTypeToString(const Qnn_TensorType_t& type) {
  const char* tensorTypeString = nullptr;
  switch (type) {
    case QNN_TENSOR_TYPE_APP_WRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_APP_WRITE);
      break;
    case QNN_TENSOR_TYPE_APP_READ:
      tensorTypeString = STR(QNN_TENSOR_TYPE_APP_READ);
      break;
    case QNN_TENSOR_TYPE_APP_READWRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_APP_READWRITE);
      break;
    case QNN_TENSOR_TYPE_NATIVE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_NATIVE);
      break;
    case QNN_TENSOR_TYPE_STATIC:
      tensorTypeString = STR(QNN_TENSOR_TYPE_STATIC);
      break;
    case QNN_TENSOR_TYPE_NULL:
      tensorTypeString = STR(QNN_TENSOR_TYPE_NULL);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_STATIC:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UPDATEABLE_STATIC);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_NATIVE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UPDATEABLE_NATIVE);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_APP_READ:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UPDATEABLE_APP_READ);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE);
      break;
    case QNN_TENSOR_TYPE_UNDEFINED:
      tensorTypeString = STR(QNN_TENSOR_TYPE_UNDEFINED);
      break;
    case QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE);
      break;
    case QNN_TENSOR_TYPE_OPTIONAL_APP_READ:
      tensorTypeString = STR(QNN_TENSOR_TYPE_OPTIONAL_APP_READ);
      break;
    case QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE:
      tensorTypeString = STR(QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE);
      break;
    default:
      tensorTypeString = STR(UNKNOWN);
      break;
  }
  return tensorTypeString;
}

Qnn_TensorType_t tensorTypeFromString(const char* typeString) {
  Qnn_TensorType_t tensorType = QNN_TENSOR_TYPE_UNDEFINED;
  if (strcmp(typeString, "QNN_TENSOR_TYPE_APP_WRITE") == 0) {
    tensorType = QNN_TENSOR_TYPE_APP_WRITE;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_APP_READ") == 0) {
    tensorType = QNN_TENSOR_TYPE_APP_READ;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_APP_READWRITE") == 0) {
    tensorType = QNN_TENSOR_TYPE_APP_READWRITE;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_NATIVE") == 0) {
    tensorType = QNN_TENSOR_TYPE_NATIVE;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_STATIC") == 0) {
    tensorType = QNN_TENSOR_TYPE_STATIC;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_NULL") == 0) {
    tensorType = QNN_TENSOR_TYPE_NULL;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_UPDATEABLE_STATIC") == 0) {
    tensorType = QNN_TENSOR_TYPE_UPDATEABLE_STATIC;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_UPDATEABLE_NATIVE") == 0) {
    tensorType = QNN_TENSOR_TYPE_UPDATEABLE_NATIVE;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE") == 0) {
    tensorType = QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_UPDATEABLE_APP_READ") == 0) {
    tensorType = QNN_TENSOR_TYPE_UPDATEABLE_APP_READ;
  } else if (strcmp(typeString, "QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE") == 0) {
    tensorType = QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE;
  }
  return tensorType;
}

const char* tensorMemTypeToString(const Qnn_TensorMemType_t& memType) {
  const char* tensorMemTypeString = nullptr;
  switch (memType) {
    case QNN_TENSORMEMTYPE_RAW:
      tensorMemTypeString = STR(QNN_TENSORMEMTYPE_RAW);
      break;
    case QNN_TENSORMEMTYPE_MEMHANDLE:
      tensorMemTypeString = STR(QNN_TENSORMEMTYPE_MEMHANDLE);
      break;
    case QNN_TENSORMEMTYPE_UNDEFINED:
      tensorMemTypeString = STR(QNN_TENSORMEMTYPE_UNDEFINED);
      break;
    default:
      tensorMemTypeString = STR(UNKNOWN);
      break;
  }
  return tensorMemTypeString;
}

Qnn_TensorMemType_t tensorMemTypeFromString(const char* tensorMemTypeString) {
  Qnn_TensorMemType_t tensorMemType = QNN_TENSORMEMTYPE_UNDEFINED;
  if (strcmp(tensorMemTypeString, "QNN_TENSORMEMTYPE_RAW") == 0) {
    tensorMemType = QNN_TENSORMEMTYPE_RAW;
  } else if (strcmp(tensorMemTypeString, "QNN_TENSORMEMTYPE_MEMHANDLE") == 0) {
    tensorMemType = QNN_TENSORMEMTYPE_MEMHANDLE;
  } else if (strcmp(tensorMemTypeString, "QNN_TENSORMEMTYPE_RETRIEVE_RAW") == 0) {
    tensorMemType = QNN_TENSORMEMTYPE_RETRIEVE_RAW;
  } else {
    tensorMemType = QNN_TENSORMEMTYPE_UNDEFINED;
  }
  return tensorMemType;
}

const char* tensorDataFormatToString(const Qnn_TensorDataFormat_t& tensorDataFormat) {
  const char* tensorDataFormatString = nullptr;
  switch (tensorDataFormat) {
    case QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
      break;
    case QNN_TENSOR_DATA_FORMAT_SPARSE:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_SPARSE);
      break;
    case QNN_TENSOR_DATA_FORMAT_MX:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_MX);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV12);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV);
      break;
    case QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV124R);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y);
      break;
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV:
      tensorDataFormatString = STR(QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV);
      break;
    default:
      tensorDataFormatString = STR(UNKNOWN);
      break;
  }
  return tensorDataFormatString;
}

Qnn_TensorDataFormat_t tensorDataFormatFromString(const char* tensorDataFormatString) {
  Qnn_TensorDataFormat_t tensorDataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
  if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_SPARSE") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_SPARSE;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_MX") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_MX;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV12") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV12;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV124R") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV124R;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y;
  } else if (strcmp(tensorDataFormatString, "QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV") == 0) {
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV;
  } else {
    // tensor format default to dense
    tensorDataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
  }
  return tensorDataFormat;
}

const char* sparseLayoutTypeToString(const Qnn_SparseLayoutType_t& sparseLayoutType) {
  const char* sparseLayoutTypeString = nullptr;
  switch (sparseLayoutType) {
    case QNN_SPARSE_LAYOUT_HYBRID_COO:
      sparseLayoutTypeString = STR(QNN_SPARSE_LAYOUT_HYBRID_COO);
      break;
    case QNN_SPARSE_LAYOUT_UNDEFINED:
      sparseLayoutTypeString = STR(QNN_SPARSE_LAYOUT_UNDEFINED);
      break;
    default:
      sparseLayoutTypeString = STR(UNKNOWN);
      break;
  }
  return sparseLayoutTypeString;
}

const char* platformInfoVersionToString(const QnnDevice_PlatformInfoVersion_t& version) {
  const char* platformInfoVersionString = nullptr;
  switch (version) {
    case QNN_DEVICE_PLATFORM_INFO_VERSION_1:
      platformInfoVersionString = STR(QNN_DEVICE_PLATFORM_INFO_VERSION_1);
      break;
    default:
      platformInfoVersionString = STR(UNKNOWN);
      break;
  }
  return platformInfoVersionString;
}

const char* hardwareDeviceInfoVersionToString(const QnnDevice_HardwareDeviceInfoVersion_t& version) {
  const char* hardwareDeviceInfoVersionString = nullptr;
  switch (version) {
    case QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1:
      hardwareDeviceInfoVersionString = STR(QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1);
      break;
    default:
      hardwareDeviceInfoVersionString = STR(UNKOWN);
      break;
  }
  return hardwareDeviceInfoVersionString;
}

const char* coreInfoVersionToString(const QnnDevice_CoreInfoVersion_t& version){
  const char* coreInfoVersionString = nullptr;
  switch (version) {
    case QNN_DEVICE_CORE_INFO_VERSION_1:
      coreInfoVersionString = STR(QNN_DEVICE_CORE_INFO_VERSION_1);
      break;
    default:
      coreInfoVersionString = STR(UNKNOWN);
      break;
  }
  return coreInfoVersionString;
}

const char* binaryInfoVersionToString(const QnnSystemContext_BinaryInfoVersion_t& version) {
  const char* binaryInfoVersionString = nullptr;
  switch (version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      binaryInfoVersionString = STR(QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1);
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      binaryInfoVersionString = STR(QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2);
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      binaryInfoVersionString = STR(QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3);
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED:
      binaryInfoVersionString = STR(QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED);
      break;
    default:
      binaryInfoVersionString = STR(UNKNOWN);
      break;
  }
  return binaryInfoVersionString;
}

const char* graphInfoVersionToString(const QnnSystemContext_GraphInfoVersion_t& version) {
  const char* graphInfoVersionString = nullptr;
  switch (version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
      graphInfoVersionString = STR(QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1);
      break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
      graphInfoVersionString = STR(QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2);
      break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
      graphInfoVersionString = STR(QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3);
      break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED:
      graphInfoVersionString = STR(QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED);
      break;
    default:
      graphInfoVersionString = STR(UNKNOWN);
      break;
  }
  return graphInfoVersionString;
}

QnnSystemContext_GraphInfoVersion_t graphInfoVersionFromString(const char* versionString) {
  QnnSystemContext_GraphInfoVersion_t graphInfoVersion = QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED;
  if (strcmp(versionString, "QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1") == 0) {
    graphInfoVersion = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  } else if (strcmp(versionString, "QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2") == 0) {
    graphInfoVersion = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2;
  } else if (strcmp(versionString, "QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3") == 0) {
    graphInfoVersion = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3;
  } else {
    graphInfoVersion = QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED;
  }
  return graphInfoVersion;
}

const char* quantizeDefinitionToString(const Qnn_Definition_t& quantizeDefinition) {
  const char* quantizeDefinitionToString = nullptr;
  switch (quantizeDefinition) {
    case QNN_DEFINITION_IMPL_GENERATED:
      quantizeDefinitionToString = STR(QNN_DEFINITION_IMPL_GENERATED);
      break;
    case QNN_DEFINITION_DEFINED:
      quantizeDefinitionToString = STR(QNN_DEFINITION_DEFINED);
      break;
    case QNN_DEFINITION_UNDEFINED:
      quantizeDefinitionToString = STR(QNN_DEFINITION_UNDEFINED);
      break;
    default:
      quantizeDefinitionToString = STR(UNKNOWN);
      break;
  }
  return quantizeDefinitionToString;
}

Qnn_Definition_t quantizeDefinitionFromString(const char* quantizeDefinitionString) {
  Qnn_Definition_t quantizeDefinition = QNN_DEFINITION_UNDEFINED;
  if (strcmp(quantizeDefinitionString, "QNN_DEFINITION_IMPL_GENERATED") == 0) {
    quantizeDefinition = QNN_DEFINITION_IMPL_GENERATED;
  } else if (strcmp(quantizeDefinitionString, "QNN_DEFINITION_DEFINED") == 0) {
    quantizeDefinition = QNN_DEFINITION_DEFINED;
  } else {
    quantizeDefinition = QNN_DEFINITION_UNDEFINED;
  }
  return quantizeDefinition;
}

const char* quantizeEncodingToString(const Qnn_QuantizationEncoding_t& quantizeEncoding) {
  const char* quantizeEncodingToString = nullptr;
  switch (quantizeEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
      break;
    case QNN_QUANTIZATION_ENCODING_BLOCK:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BLOCK);
      break;
    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION);
      break;
    case QNN_QUANTIZATION_ENCODING_VECTOR:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_VECTOR);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED);
      break;
    case QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
      break;
    case QNN_QUANTIZATION_ENCODING_MICROSCALING:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_MICROSCALING);
      break;
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      quantizeEncodingToString = STR(QNN_QUANTIZATION_ENCODING_UNDEFINED);
      break;
    default:
      quantizeEncodingToString = STR(UNKNOWN);
      break;
  }
  return quantizeEncodingToString;
}

Qnn_QuantizationEncoding_t quantizeEncodingFromString(const char* quantizeEncodingString) {
  Qnn_QuantizationEncoding_t quantizeEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET") ==
             0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BLOCK") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BLOCK;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_VECTOR") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_VECTOR;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
  } else if (strcmp(quantizeEncodingString, "QNN_QUANTIZATION_ENCODING_MICROSCALING") == 0) {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_MICROSCALING;
  } else {
    quantizeEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  }
  return quantizeEncoding;
}

const char* quantizeEncodingMappingToString(const Qnn_QuantizationEncodingMapping_t& quantizeEncodingMapping) {
  const char* quantizeEncodingMappingToString = nullptr;
  switch (quantizeEncodingMapping) {
    case QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC:
      quantizeEncodingMappingToString = STR(QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC);
      break;
    case QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE:
      quantizeEncodingMappingToString = STR(QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE);
      break;
    case QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO:
      quantizeEncodingMappingToString = STR(QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO);
      break;
    case QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED:
      quantizeEncodingMappingToString = STR(QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED);
      break;
    default:
      quantizeEncodingMappingToString = STR(UNKNOWN);
      break;
  }
  return quantizeEncodingMappingToString;
}

Qnn_QuantizationEncodingMapping_t quantizeEncodingMappingFromString(const char* quantizeEncodingMappingString) {
  Qnn_QuantizationEncodingMapping_t quantizeEncodingMapping = QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED;
  if (strcmp(quantizeEncodingMappingString, "QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC") == 0) {
    quantizeEncodingMapping = QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC;
  } else if (strcmp(quantizeEncodingMappingString, "QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE") == 0) {
    quantizeEncodingMapping = QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE;
  } else if (strcmp(quantizeEncodingMappingString, "QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO") == 0) {
    quantizeEncodingMapping = QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO;
  } else {
    quantizeEncodingMapping = QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED;
  }
  return quantizeEncodingMapping;
}

const char* floatEncodingToString(const Qnn_FloatEncoding_t& floatEncoding) {
  const char* floatEncodingString = nullptr;
  switch (floatEncoding) {
    case QNN_FLOAT_ENCODING_MXFP8_E5M2:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_MXFP8_E5M2);
      break;
    case QNN_FLOAT_ENCODING_MXFP8_E4M3:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_MXFP8_E4M3);
      break;
    case QNN_FLOAT_ENCODING_MXFP6_E3M2:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_MXFP6_E3M2);
      break;
    case QNN_FLOAT_ENCODING_MXFP6_E2M3:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_MXFP6_E2M3);
      break;
    case QNN_FLOAT_ENCODING_MXFP4_E2M1:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_MXFP4_E2M1);
      break;
    case QNN_FLOAT_ENCODING_UNDEFINED:
      floatEncodingString = STR(QNN_FLOAT_ENCODING_UNDEFINED);
      break;
    default:
      floatEncodingString = STR(UNKNOWN);
      break;
  }
  return floatEncodingString;
}

Qnn_FloatEncoding_t floatEncodingFromString(const char* floatEncodingString) {
  if (strcmp(floatEncodingString, "QNN_FLOAT_ENCODING_MXFP8_E5M2") == 0) {
    return QNN_FLOAT_ENCODING_MXFP8_E5M2;
  } else if (strcmp(floatEncodingString, "QNN_FLOAT_ENCODING_MXFP8_E4M3") == 0) {
    return QNN_FLOAT_ENCODING_MXFP8_E4M3;
  } else if (strcmp(floatEncodingString, "QNN_FLOAT_ENCODING_MXFP6_E3M2") == 0) {
    return QNN_FLOAT_ENCODING_MXFP6_E3M2;
  } else if (strcmp(floatEncodingString, "QNN_FLOAT_ENCODING_MXFP6_E2M3") == 0) {
    return QNN_FLOAT_ENCODING_MXFP6_E2M3;
  } else if (strcmp(floatEncodingString, "QNN_FLOAT_ENCODING_MXFP4_E2M1") == 0) {
    return QNN_FLOAT_ENCODING_MXFP4_E2M1;
  }
  return QNN_FLOAT_ENCODING_UNDEFINED;
}

}  // namespace aiswutility
