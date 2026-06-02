// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpDef.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/*
 * Relevant information on writing HTP op packages can be found in
 * "Op Writing Guidelines" section in QNN SDK docs/general/backend.html
 */

// op package info
static constexpr auto sg_examplePackageName =
    THIS_PKG_NAME_STR; // package name passed in as compile flag THIS_PKG_NAME
                       // in this example: "examples_OpPackage"
static constexpr auto sg_opNameRelu = "Relu";
static constexpr auto sg_opNameMaxPool2d = "PoolMax2d";
static constexpr auto sg_opNameSoftmax = "Softmax";
static constexpr auto sg_opNameSoftmaxCrouton = "Softmax_Crouton";
static std::array<const char *, 4> sg_opNames{
    {sg_opNameRelu, sg_opNameMaxPool2d, sg_opNameSoftmax,
     sg_opNameSoftmaxCrouton}};

static Qnn_ApiVersion_t sg_exampleSdkApiVersion = QNN_HTP_API_VERSION_INIT;
// Version of the set of operations implemented in the op package
// User can define own custom opset version
static Qnn_Version_t sg_exampleOpsetVersion = {
    QNN_OPSET_VERSION_MAJOR, QNN_OPSET_VERSION_MINOR, QNN_OPSET_VERSION_PATCH};
static QnnOpPackage_Info_t sg_examplePackageInfo = {sg_examplePackageName,
                                                    sg_opNames.data(),
                                                    nullptr,
                                                    sg_opNames.size(),
                                                    nullptr,
                                                    0,
                                                    QNN_SDK_BUILD_ID,
                                                    &sg_exampleSdkApiVersion,
                                                    nullptr,
                                                    &sg_exampleOpsetVersion,
                                                    {0}};

// global data
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra =
    nullptr; // global infrastructure not in use for now
static bool sg_packageInitialized =
    false; // tracks whether this package has been initialized
/*
 * user provided logging call back function
 * currently only supported on linux x86-64 and nonrpc versions
 * typedef void (*QnnLog_Callback_t)(const char* fmt,
 *                                   QnnLog_Level_t level,
 *                                   uint64_t timestamp,
 *                                   va_list args);
 * usage: if(sg_logInitialized && level <= sg_maxLogLevel)
 *            sg_logCallback(fmt, level, timestamp, args);
 *
 * for cross rpc versions, skel side user provided logging call back function
 * can be defined as part of op packages. maximal log level sg_maxLogLevel
 * can be set by Qnn_ErrorHandle_t exampleLogSetLevel(QnnLog_Level_t
 * maxLogLevel)
 */
/*
 * for alternative logging method provided by HTP core, please refer to log.h
 */
static QnnLog_Callback_t sg_logCallback =
    nullptr; // user provided call back function pointer for logging
static QnnLog_Level_t sg_maxLogLevel =
    (QnnLog_Level_t)0; // maximal log level used in user provided logging
static bool sg_logInitialized =
    false; // tracks whether user provided logging method has been initialized

// Forward declaration for Relu/ReluFP16 registration
extern void register_relu_ops();
extern void register_relu_fp16_ops();
extern void register_maxpool_ops();
extern void register_softmax_ops();

Qnn_ErrorHandle_t
exampleInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized)
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  sg_globalInfra = infrastructure;
  sg_packageInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t exampleGetInfo(const QnnOpPackage_Info_t **info) {
  if (!sg_packageInitialized)
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!info)
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

  *info = &sg_examplePackageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t exampleLogInitialize(QnnLog_Callback_t callback,
                                       QnnLog_Level_t maxLogLevel) {
  if (!callback)
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR)
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_logCallback = callback;
  sg_maxLogLevel = maxLogLevel;
  sg_logInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t exampleLogSetLevel(QnnLog_Level_t maxLogLevel) {
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR)
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_maxLogLevel = maxLogLevel;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t exampleLogTerminate() {
  sg_logCallback = nullptr;
  sg_maxLogLevel = (QnnLog_Level_t)0;
  sg_logInitialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t exampleValidateOpConfig(Qnn_OpConfig_t opConfig) {
  if (std::string(sg_examplePackageName) != opConfig.v1.packageName) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  if (std::string(opConfig.v1.typeName) == sg_opNameRelu) {
    if (opConfig.v1.numOfParams > 0 || opConfig.v1.numOfInputs != 1 ||
        opConfig.v1.numOfOutputs != 1)
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  } else if (std::string(opConfig.v1.typeName) == sg_opNameSoftmax) {
    if (opConfig.v1.numOfParams > 2 || opConfig.v1.numOfInputs != 1 ||
        opConfig.v1.numOfOutputs != 1)
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  } else if (std::string(opConfig.v1.typeName) == sg_opNameSoftmaxCrouton) {
    if (opConfig.v1.numOfParams > 1 || opConfig.v1.numOfInputs != 1 ||
        opConfig.v1.numOfOutputs != 1)
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  } else if (std::string(opConfig.v1.typeName) == sg_opNameMaxPool2d) {
    if (opConfig.v1.numOfParams != 3 || opConfig.v1.numOfInputs != 1 ||
        opConfig.v1.numOfOutputs != 1)
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  } else {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  return QNN_SUCCESS;
}

/* The following functions in this comment are not required by HTP backend,
 * thus no implementations needed.
 *
 * (version 1.4.0) Qnn_ErrorHandle_t exampleCreateOpImpl
 * (QnnOpPackage_GraphInfrastructure_t graphInfrastructure, QnnOpPackage_Node_t
 * node, QnnOpPackage_OpImpl_t* opImpl) (version 1.4.0) Qnn_ErrorHandle_t
 * exampleFreeOpImpl (QnnOpPackage_OpImpl_t opImpl)
 */

Qnn_ErrorHandle_t
exampleCreateOpImpl(QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                    QnnOpPackage_Node_t node, QnnOpPackage_OpImpl_t *opImpl) {
  (void)graphInfrastructure;
  (void)node;
  (void)opImpl;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t exampleFreeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
  (void)opImpl;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t exampleTerminate() {
  if (!sg_packageInitialized)
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

  sg_globalInfra = nullptr;
  sg_packageInitialized = false;
  return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

/* latest version */
Qnn_ErrorHandle_t
exampleInterfaceProvider(QnnOpPackage_Interface_t *interface) {
  if (!interface)
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = exampleInit;
  interface->v1_4.terminate = exampleTerminate;
  interface->v1_4.getInfo = exampleGetInfo;
  interface->v1_4.validateOpConfig = exampleValidateOpConfig;
  interface->v1_4.createOpImpl = exampleCreateOpImpl;
  interface->v1_4.freeOpImpl = exampleFreeOpImpl;
  interface->v1_4.logInitialize = exampleLogInitialize;
  interface->v1_4.logSetLevel = exampleLogSetLevel;
  interface->v1_4.logTerminate = exampleLogTerminate;
  return QNN_SUCCESS;
}

extern "C" const char *qhpi_init() {
  // Register both regular ReLU and FP16 ReLU operations
  register_relu_ops();
  register_relu_fp16_ops();
  register_maxpool_ops();
  register_softmax_ops();
  return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
