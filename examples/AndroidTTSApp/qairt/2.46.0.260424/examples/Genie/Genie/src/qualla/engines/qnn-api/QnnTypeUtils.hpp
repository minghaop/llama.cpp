//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>
#include <type_traits>
#include <cstring>

#include "QnnProperty.h"
#include "QnnTypes.h"
#include "System/QnnSystemContext.h"

namespace aiswutility {

template <typename T>
Qnn_DataType_t QnnDataType() {
  // Signed integers
  if (std::is_same<T, int8_t>::value) return QNN_DATATYPE_INT_8;
  if (std::is_same<T, int16_t>::value) return QNN_DATATYPE_INT_16;
  if (std::is_same<T, int32_t>::value) return QNN_DATATYPE_INT_32;
  if (std::is_same<T, int64_t>::value) return QNN_DATATYPE_INT_64;
  // Unsigned integers
  if (std::is_same<T, uint8_t>::value) return QNN_DATATYPE_UINT_8;
  if (std::is_same<T, uint16_t>::value) return QNN_DATATYPE_UINT_16;
  if (std::is_same<T, uint32_t>::value) return QNN_DATATYPE_UINT_32;
  if (std::is_same<T, uint64_t>::value) return QNN_DATATYPE_UINT_64;
  // Floating point
  if (std::is_same<T, float>::value) return QNN_DATATYPE_FLOAT_32;
  if (std::is_same<T, double>::value) return QNN_DATATYPE_FLOAT_64;
  // Bool
  if (std::is_same<T, bool>::value) return QNN_DATATYPE_BOOL_8;
  // String
  if (std::is_same<std::string, T>::value) return QNN_DATATYPE_STRING;

  return QNN_DATATYPE_UNDEFINED;
}

float getDataTypeSize(Qnn_DataType_t dataType);
uint32_t getDataTypeContainerSize(Qnn_DataType_t dataType);
size_t getDataTypeBitWidth(Qnn_DataType_t dataType);
const char* dataTypeToString(Qnn_DataType_t dataType);
Qnn_DataType_t dataTypefromString(const std::string& dataType);
bool isSignedIntDataType(Qnn_DataType_t dataType);
bool isUnsignedIntDataType(Qnn_DataType_t dataType);
bool isIntDataType(Qnn_DataType_t dataType);
bool isFloatDataType(Qnn_DataType_t dataType);
bool isSignedQuantizedDataType(Qnn_DataType_t dataType);
bool isUnsignedQuantizedDataType(Qnn_DataType_t dataType);
bool isQuantizedDataType(Qnn_DataType_t dataType);

const char* capabilityToString(const QnnProperty_Key_t key);

const char* priorityToString(Qnn_Priority_t priority);
Qnn_Priority_t priorityFromString(const std::string& priorityStr);

const char* tensorVersionToString(const Qnn_TensorVersion_t& version);
Qnn_TensorVersion_t tensorVersionFromString(const char* versionString);
const char* tensorTypeToString(const Qnn_TensorType_t& type);
Qnn_TensorType_t tensorTypeFromString(const char* typeString);
const char* tensorMemTypeToString(const Qnn_TensorMemType_t& memType);
Qnn_TensorMemType_t tensorMemTypeFromString(const char* tensorMemTypeString);
const char* tensorDataFormatToString(const Qnn_TensorDataFormat_t& type);
Qnn_TensorDataFormat_t tensorDataFormatFromString(const char* tensorDataFormatString);
const char* sparseLayoutTypeToString(const Qnn_SparseLayoutType_t& sparseLayoutType);

const char* platformInfoVersionToString(const QnnDevice_PlatformInfoVersion_t &version);
const char* hardwareDeviceInfoVersionToString(const QnnDevice_HardwareDeviceInfoVersion_t& version);
const char* coreInfoVersionToString(const QnnDevice_CoreInfoVersion_t& version);

const char* binaryInfoVersionToString(const QnnSystemContext_BinaryInfoVersion_t& version);
const char* graphInfoVersionToString(const QnnSystemContext_GraphInfoVersion_t& version);
QnnSystemContext_GraphInfoVersion_t graphInfoVersionFromString(const char* versionString);

const char* quantizeDefinitionToString(const Qnn_Definition_t& quantizeDefinition);
Qnn_Definition_t quantizeDefinitionFromString(const char* quantizeDefinitionString);
const char* quantizeEncodingToString(const Qnn_QuantizationEncoding_t& quantizeEncoding);
Qnn_QuantizationEncoding_t quantizeEncodingFromString(const char* quantizeEncodingString);
const char* quantizeEncodingMappingToString(const Qnn_QuantizationEncodingMapping_t& quantizeEncodingMapping);
Qnn_QuantizationEncodingMapping_t quantizeEncodingMappingFromString(const char* quantizeEncodingMappingString);
const char* floatEncodingToString(const Qnn_FloatEncoding_t& floatEncoding);
Qnn_FloatEncoding_t floatEncodingFromString(const char* floatEncodingString);

}  // namespace aiswutility
