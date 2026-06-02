//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "GPU/QnnGpuBackend.h"
#include "GPU/QnnGpuGraph.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnSampleAppUtils.hpp"
#include "QnnTensor.h"
#include "QnnTypes.h"

// OpenCL Headers
#include <CL/cl.h>

#include "CL/cl_ext_qcom.h"

namespace qnn {
namespace tools {
namespace sample_app {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

// Async notification structure
typedef struct QnnGraphAsyncNotifyParam {
  bool m_status                 = false;
  size_t *m_infCount            = nullptr;
  std::mutex *m_mutex           = nullptr;
  std::condition_variable *m_cv = nullptr;
} QnnGraphAsyncNotifyParam;

class QnnGpuFencingSampleApp {
 public:
  QnnGpuFencingSampleApp(std::string backendPath, int iterations = 3);

  ~QnnGpuFencingSampleApp();

  // @brief Print a message to STDERR then return a nonzero
  //  exit status.
  int32_t reportError(const std::string &err);

  StatusCode initialize();

  StatusCode runFencingExample();

  StatusCode cleanup();

 private:
  // QNN and OpenCL library loading
  StatusCode loadQnnLibrary();
  StatusCode loadOpenCLLibrary();

  // QNN operations
  StatusCode initializeQnn();
  StatusCode createGraph();
  StatusCode serializeAndDeserializeContext();
  StatusCode setupFenceConfiguration();
  StatusCode executeAsyncLoop();

  // OpenCL operations
  StatusCode initializeOpenCL();

  // Cleanup operations
  StatusCode cleanupQnn();
  StatusCode cleanupOpenCL();

  // Async notification callback
  static void notifyFn(void *notifyParam, Qnn_NotifyStatus_t notifyStatus);

  // Configuration
  std::string m_backendPath;
  int m_iterations;

  // QNN handles and interface
  void *m_qnnLib                       = nullptr;
  const QnnInterface_t *m_qnnInterface = nullptr;
  Qnn_LogHandle_t m_logHandle          = nullptr;
  Qnn_BackendHandle_t m_backendHandle  = nullptr;
  Qnn_ContextHandle_t m_context        = nullptr;
  Qnn_GraphHandle_t m_graph            = nullptr;
  std::string m_graphName              = "FenceGraph";

  // OpenCL handles and interface
  void *m_openclLib         = nullptr;
  cl_platform_id m_platform = nullptr;
  cl_device_id m_device     = nullptr;
  cl_context m_clContext    = nullptr;
  cl_command_queue m_queue  = nullptr;

  // OpenCL function pointers
  typedef cl_int (*clGetPlatformIDs_fn)(cl_uint, cl_platform_id *, cl_uint *);
  typedef cl_int (*clGetDeviceIDs_fn)(
      cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  typedef cl_context (*clCreateContext_fn)(
      const cl_context_properties *,
      cl_uint,
      const cl_device_id *,
      void(CL_CALLBACK *)(const char *, const void *, size_t, void *),
      void *,
      cl_int *);
  typedef cl_command_queue (*clCreateCommandQueueWithProperties_fn)(cl_context,
                                                                    cl_device_id,
                                                                    const cl_queue_properties *,
                                                                    cl_int *);
  typedef cl_int (*clFlush_fn)(cl_command_queue);
  typedef cl_int (*clReleaseCommandQueue_fn)(cl_command_queue);
  typedef cl_int (*clReleaseContext_fn)(cl_context);

  clGetPlatformIDs_fn clGetPlatformIDs_ptr                                     = nullptr;
  clGetDeviceIDs_fn clGetDeviceIDs_ptr                                         = nullptr;
  clCreateContext_fn clCreateContext_ptr                                       = nullptr;
  clCreateCommandQueueWithProperties_fn clCreateCommandQueueWithProperties_ptr = nullptr;
  typedef cl_semaphore_khr (*clCreateSemaphoreWithPropertiesKHR_fn)(
      cl_context, const cl_semaphore_properties_khr *, cl_int *);
  typedef cl_int (*clEnqueueSignalSemaphoresKHR_fn)(cl_command_queue,
                                                    cl_uint,
                                                    const cl_semaphore_khr *,
                                                    const cl_semaphore_payload_khr *,
                                                    cl_uint,
                                                    const cl_event *,
                                                    cl_event *);
  typedef cl_int (*clEnqueueWaitSemaphoresKHR_fn)(cl_command_queue,
                                                  cl_uint,
                                                  const cl_semaphore_khr *,
                                                  const cl_semaphore_payload_khr *,
                                                  cl_uint,
                                                  const cl_event *,
                                                  cl_event *);
  typedef cl_int (*clGetSemaphoreHandleForTypeKHR_fn)(cl_semaphore_khr,
                                                      cl_device_id,
                                                      cl_external_semaphore_handle_type_khr,
                                                      size_t,
                                                      void *,
                                                      size_t *);
  typedef cl_int (*clReleaseSemaphoreKHR_fn)(cl_semaphore_khr);

  clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR_ptr = nullptr;
  clEnqueueSignalSemaphoresKHR_fn clEnqueueSignalSemaphoresKHR_ptr             = nullptr;
  clEnqueueWaitSemaphoresKHR_fn clEnqueueWaitSemaphoresKHR_ptr                 = nullptr;
  clFlush_fn clFlush_ptr                                                       = nullptr;
  clGetSemaphoreHandleForTypeKHR_fn clGetSemaphoreHandleForTypeKHR_ptr         = nullptr;
  clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR_ptr                           = nullptr;
  clReleaseCommandQueue_fn clReleaseCommandQueue_ptr                           = nullptr;
  clReleaseContext_fn clReleaseContext_ptr                                     = nullptr;

  // Fence configuration
  int m_outputFence         = -2;
  uint32_t m_numInputFences = 2;
  std::vector<FenceHandle> m_fenceHandleArr;

  // Tensor data
  std::vector<float> m_inputData;
  std::vector<float> m_outputData;
  std::vector<uint32_t> m_graphDims;
  Qnn_Tensor_t m_input;
  Qnn_Tensor_t m_output;

  // Semaphore management
  std::vector<cl_semaphore_khr> m_semaphores;
  std::vector<void *> m_semaphoreHandles;
};

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn
