<%doc> define all relevant variables</%doc>
<%doc>
//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================
</%doc>
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import is_valid_cpp_identifier, log_info %>
<% is_valid_cpp_identifier(package_info.name) %>

//==============================================================================
// Auto Generated Code for ${package_info.name} - QHPI Interface
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include "HTP/core/qhpi.h"
#include <array>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

%for operator in package_info.operators:
// Forward declaration for ${operator.type_name} registration
extern void register_${operator.type_name.lower()}_ops();
%endfor

// External declarations for operator infos from individual operator files
%for operator in package_info.operators:
extern QHPI_OpInfo_v1 ${operator.type_name.lower()}OpInfo;
%endfor

// op package info
const char* const sg_packageName = "${package_info.name}";  // package name passed in as compile flag

static std::array<const char*, ${len(package_info.operators)}> sg_opNames{${_format_list_to_cpp_brace([operator.type_name for operator in package_info.operators])}};

static Qnn_ApiVersion_t sg_sdkApiVersion  = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

// global data
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra =
nullptr;  // global infrastructure not in use for now
static bool sg_packageInitialized = false;

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
 * can be set by Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(QnnLog_Level_t maxLogLevel)
 */
/*
 * for alternative logging method provided by HTP core, please refer to log.h
 */
static QnnLog_Callback_t sg_logCallback =
    nullptr;  // user provided call back function pointer for logging
static QnnLog_Level_t sg_maxLogLevel =
    (QnnLog_Level_t)0;  // maximal log level used in user provided logging
static bool sg_logInitialized =
    false;  // tracks whether user provided logging method has been initialized

/* op package API's */

Qnn_ErrorHandle_t ${package_info.name}Init(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;

    /*
     * QHPI packages don't use traditional DEF_OP registration macros
     * Plugin registration is handled through  qhpi_register_ops_vxx in the source files
     */

    sg_globalInfra        = infrastructure;
    sg_packageInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}GetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

    sg_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
    sg_packageInfo.packageName    = sg_packageName;
    sg_packageInfo.operationNames = sg_opNames.data();
    sg_packageInfo.numOperations  = sg_opNames.size();
    sg_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
    sg_packageInfo.sdkApiVersion  = &sg_sdkApiVersion;

    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
    if (sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback    = callback;
    sg_maxLogLevel    = maxLogLevel;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(QnnLog_Level_t maxLogLevel) {
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_maxLogLevel = maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogTerminate() {
    if (!sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_logCallback    = nullptr;
    sg_maxLogLevel    = (QnnLog_Level_t)0;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}ValidateOpConfig (Qnn_OpConfig_t opConfig){
    if (std::string(sg_packageName) != opConfig.v1.packageName) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    %if len(package_info.operators) > 0:
    /* auto-generated validation code below
     * Check if op config type matches any registered ops
     * If a match is found, check number of inputs, outputs and params
     */
    if (std::string(opConfig.v1.typeName) == "${package_info.operators[0].type_name}"){
        if (opConfig.v1.numOfParams != ${len(package_info.operators[0].param)} || opConfig.v1.numOfInputs != ${len(package_info.operators[0].input)} || opConfig.v1.numOfOutputs != ${len(package_info.operators[0].output)}){
          return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
        }
    }
    % for operator in package_info.operators[1:]:
    else if (std::string(opConfig.v1.typeName) == "${operator.type_name}"){
        if (opConfig.v1.numOfParams != ${len(operator.param)} || opConfig.v1.numOfInputs != ${len(operator.input)} || opConfig.v1.numOfOutputs != ${len(operator.output)}){
          return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
        }
    }
    %endfor
    else{
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    /*
    * additional validation code here
    * */

    %else:
    /*
    * add code here
    * */
    %endif
    return QNN_SUCCESS;
}

/* The following three functions in this comment are not called by HTP backend for now,
 * no auto-generated implementations are created. Users should see example for full function signatures.
 * (version 1.3.0) Qnn_ErrorHandle_t ${package_info.name}CreateKernels (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_Kernel_t** kernels, uint32_t*
 * numKernels)
 * (version 1.3.0) Qnn_ErrorHandle_t ${package_info.name}FreeKernels (QnnOpPackage_Kernel_t* kernels)
 *
 * (version 1.4.0) Qnn_ErrorHandle_t ${package_info.name}CreateOpImpl (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_OpImpl_t* opImpl)
 *(version 1.4.0) Qnn_ErrorHandle_t ${package_info.name}FreeOpImpl (QnnOpPackage_OpImpl_t opImpl)
 */

Qnn_ErrorHandle_t ${package_info.name}Terminate() {
if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

sg_globalInfra        = nullptr;
sg_packageInitialized = false;
return QNN_SUCCESS;
}


<% log_info("Note: Interface provider function will be named: {}".format(package_info.name + "InterfaceProvider")) %>
/* latest version */
Qnn_ErrorHandle_t ${package_info.name}InterfaceProvider(QnnOpPackage_Interface_t* interface) {
  if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion      = {1, 4, 0};
  interface->v1_4.init             = ${package_info.name}Init;
  interface->v1_4.terminate        = ${package_info.name}Terminate;
  interface->v1_4.getInfo          = ${package_info.name}GetInfo;
  interface->v1_4.validateOpConfig = ${package_info.name}ValidateOpConfig;
  interface->v1_4.createOpImpl     = nullptr;
  interface->v1_4.freeOpImpl       = nullptr;
  interface->v1_4.logInitialize    = ${package_info.name}LogInitialize;
  interface->v1_4.logSetLevel      = ${package_info.name}LogSetLevel;
  interface->v1_4.logTerminate     = ${package_info.name}LogTerminate;
  return QNN_SUCCESS;
}

// Implementation of qhpi_init function
const char* qhpi_init() {
%for operator in package_info.operators:
    register_${operator.type_name.lower()}_ops();
%endfor
    return sg_packageName;
}
#ifdef __cplusplus
}
#endif
<%def name="_format_list_to_cpp_brace(list_object)" filter="trim">
<% string_list_type = str(list_object) %>
${string_list_type.replace('[', '{').replace(']', '}').replace('\'', '\"')}
</%def>
