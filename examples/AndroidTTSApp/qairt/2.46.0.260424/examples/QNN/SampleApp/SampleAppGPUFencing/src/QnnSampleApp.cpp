//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <poll.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "Logger.hpp"
#include "QnnSampleApp.hpp"

#define CHECK_QNN(status, message)                             \
  do {                                                         \
    if (status != QNN_SUCCESS) {                               \
      QNN_ERROR("QNN Error: %s (status=%d)", message, status); \
      return StatusCode::FAILURE;                              \
    }                                                          \
  } while (0)

#define CHECK_CL(status, message)                                 \
  do {                                                            \
    if (status != CL_SUCCESS) {                                   \
      QNN_ERROR("OpenCL Error: %s (status=%d)", message, status); \
      return StatusCode::FAILURE;                                 \
    }                                                             \
  } while (0)

namespace qnn {
namespace tools {
namespace sample_app {

QnnGpuFencingSampleApp::QnnGpuFencingSampleApp(std::string backendPath, int iterations)
    : m_backendPath(backendPath), m_iterations(iterations) {
  // Initialize tensor data
  uint32_t numElements = 16;
  m_inputData.resize(numElements, 1.0f);
  m_outputData.resize(numElements, 0.0f);
}

QnnGpuFencingSampleApp::~QnnGpuFencingSampleApp() { cleanup(); }

int32_t QnnGpuFencingSampleApp::reportError(const std::string& err) {
  QNN_ERROR("%s", err.c_str());
  return EXIT_FAILURE;
}

StatusCode QnnGpuFencingSampleApp::initialize() {
  QNN_INFO("=================================================");
  QNN_INFO("QNN GPU Fencing Sample App");
  QNN_INFO("=================================================");
  QNN_INFO("This sample demonstrates QNN GPU fence usage");
  QNN_INFO("with OpenCL semaphore integration.");
  QNN_INFO("=================================================");

  // Load libraries
  if (StatusCode::SUCCESS != loadQnnLibrary()) {
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != loadOpenCLLibrary()) {
    return StatusCode::FAILURE;
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::runFencingExample() {
  // Initialize QNN
  if (StatusCode::SUCCESS != initializeQnn()) {
    return StatusCode::FAILURE;
  }

  // Initialize OpenCL
  if (StatusCode::SUCCESS != initializeOpenCL()) {
    return StatusCode::FAILURE;
  }

  // Create graph
  if (StatusCode::SUCCESS != createGraph()) {
    return StatusCode::FAILURE;
  }

  // Serialize and deserialize context
  if (StatusCode::SUCCESS != serializeAndDeserializeContext()) {
    return StatusCode::FAILURE;
  }

  // Setup fence configuration
  if (StatusCode::SUCCESS != setupFenceConfiguration()) {
    return StatusCode::FAILURE;
  }

  // Execute async loop
  if (StatusCode::SUCCESS != executeAsyncLoop()) {
    return StatusCode::FAILURE;
  }

  QNN_INFO("=================================================");
  QNN_INFO("SUCCESS: GPU Fencing example completed successfully!");
  QNN_INFO("=================================================");

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::cleanup() {
  cleanupQnn();
  cleanupOpenCL();

  if (m_openclLib) {
    dlclose(m_openclLib);
    m_openclLib = nullptr;
  }

  if (m_qnnLib) {
    dlclose(m_qnnLib);
    m_qnnLib = nullptr;
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::loadQnnLibrary() {
  QNN_INFO("Loading QNN library...");

  m_qnnLib = dlopen(m_backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!m_qnnLib) {
    QNN_ERROR("Failed to load %s: %s", m_backendPath.c_str(), dlerror());
    return StatusCode::FAILURE;
  }

  typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(const QnnInterface_t***, uint32_t*);
  QnnInterface_getProviders_t getProviders =
      (QnnInterface_getProviders_t)dlsym(m_qnnLib, "QnnInterface_getProviders");
  if (!getProviders) {
    QNN_ERROR("Failed to get QnnInterface_getProviders");
    return StatusCode::FAILURE;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders            = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    QNN_ERROR("Failed to get QNN providers");
    return StatusCode::FAILURE;
  }

  m_qnnInterface = providers[0];
  QNN_INFO("QNN library loaded successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::loadOpenCLLibrary() {
  QNN_INFO("Loading OpenCL library...");

  // First try loading without a path, which will use LD_LIBRARY_PATH
  m_openclLib = dlopen("libOpenCL.so", RTLD_LOCAL | RTLD_NOW);

  // If that fails, try hardcoded paths
  if (!m_openclLib) {
    const std::string openclSearchPaths[] = {"/vendor/lib64/libOpenCL.so",
                                             "/system/lib64/libOpenCL.so",
                                             "/system/vendor/lib64/libOpenCL.so"};

    for (const auto& path : openclSearchPaths) {
      m_openclLib = dlopen(path.c_str(), RTLD_LOCAL | RTLD_NOW);
      if (m_openclLib) {
        break;
      }
    }
  }

  if (!m_openclLib) {
    QNN_ERROR("Failed to load libOpenCL.so: %s", dlerror());
    return StatusCode::FAILURE;
  }

  clGetPlatformIDs_ptr = (clGetPlatformIDs_fn)dlsym(m_openclLib, "clGetPlatformIDs");
  clGetDeviceIDs_ptr   = (clGetDeviceIDs_fn)dlsym(m_openclLib, "clGetDeviceIDs");
  clCreateContext_ptr  = (clCreateContext_fn)dlsym(m_openclLib, "clCreateContext");
  clCreateCommandQueueWithProperties_ptr = (clCreateCommandQueueWithProperties_fn)dlsym(
      m_openclLib, "clCreateCommandQueueWithProperties");
  clCreateSemaphoreWithPropertiesKHR_ptr = (clCreateSemaphoreWithPropertiesKHR_fn)dlsym(
      m_openclLib, "clCreateSemaphoreWithPropertiesKHR");
  clEnqueueSignalSemaphoresKHR_ptr =
      (clEnqueueSignalSemaphoresKHR_fn)dlsym(m_openclLib, "clEnqueueSignalSemaphoresKHR");
  clEnqueueWaitSemaphoresKHR_ptr =
      (clEnqueueWaitSemaphoresKHR_fn)dlsym(m_openclLib, "clEnqueueWaitSemaphoresKHR");
  clFlush_ptr = (clFlush_fn)dlsym(m_openclLib, "clFlush");
  clGetSemaphoreHandleForTypeKHR_ptr =
      (clGetSemaphoreHandleForTypeKHR_fn)dlsym(m_openclLib, "clGetSemaphoreHandleForTypeKHR");
  clReleaseSemaphoreKHR_ptr = (clReleaseSemaphoreKHR_fn)dlsym(m_openclLib, "clReleaseSemaphoreKHR");
  clReleaseCommandQueue_ptr = (clReleaseCommandQueue_fn)dlsym(m_openclLib, "clReleaseCommandQueue");
  clReleaseContext_ptr      = (clReleaseContext_fn)dlsym(m_openclLib, "clReleaseContext");

  if (!clGetPlatformIDs_ptr || !clGetDeviceIDs_ptr || !clCreateContext_ptr ||
      !clCreateCommandQueueWithProperties_ptr || !clCreateSemaphoreWithPropertiesKHR_ptr ||
      !clEnqueueSignalSemaphoresKHR_ptr || !clEnqueueWaitSemaphoresKHR_ptr || !clFlush_ptr ||
      !clGetSemaphoreHandleForTypeKHR_ptr || !clReleaseSemaphoreKHR_ptr ||
      !clReleaseCommandQueue_ptr || !clReleaseContext_ptr) {
    QNN_ERROR("Failed to load required OpenCL functions");
    return StatusCode::FAILURE;
  }

  QNN_INFO("OpenCL library loaded successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::initializeQnn() {
  QNN_INFO("Initializing QNN GPU Backend...");

  Qnn_ErrorHandle_t status =
      m_qnnInterface->QNN_INTERFACE_VER_NAME.logCreate(nullptr, QNN_LOG_LEVEL_WARN, &m_logHandle);
  CHECK_QNN(status, "Failed to create log handle");

  status =
      m_qnnInterface->QNN_INTERFACE_VER_NAME.backendCreate(m_logHandle, nullptr, &m_backendHandle);
  CHECK_QNN(status, "Failed to create backend");

  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.contextCreate(
      m_backendHandle, nullptr, nullptr, &m_context);
  CHECK_QNN(status, "Failed to create context");

  QNN_INFO("QNN initialized successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::initializeOpenCL() {
  QNN_INFO("Initializing OpenCL...");

  if (!clGetPlatformIDs_ptr) {
    QNN_ERROR("clGetPlatformIDs function pointer is null");
    return StatusCode::FAILURE;
  }
  cl_int err = clGetPlatformIDs_ptr(1, &m_platform, nullptr);
  CHECK_CL(err, "Failed to get platform IDs");

  if (!clGetDeviceIDs_ptr) {
    QNN_ERROR("clGetDeviceIDs function pointer is null");
    return StatusCode::FAILURE;
  }
  err = clGetDeviceIDs_ptr(m_platform, CL_DEVICE_TYPE_GPU, 1, &m_device, nullptr);
  CHECK_CL(err, "Failed to get device IDs");

  if (!clCreateContext_ptr) {
    QNN_ERROR("clCreateContext function pointer is null");
    return StatusCode::FAILURE;
  }
  m_clContext = clCreateContext_ptr(nullptr, 1, &m_device, nullptr, nullptr, &err);
  CHECK_CL(err, "Failed to create OpenCL context");

  if (!clCreateCommandQueueWithProperties_ptr) {
    QNN_ERROR("clCreateCommandQueueWithProperties function pointer is null");
    return StatusCode::FAILURE;
  }
  m_queue = clCreateCommandQueueWithProperties_ptr(m_clContext, m_device, nullptr, &err);
  CHECK_CL(err, "Failed to create command queue");

  QNN_INFO("OpenCL initialized successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::createGraph() {
  QNN_INFO("Creating graph...");

  Qnn_ErrorHandle_t status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphCreate(
      m_context, m_graphName.c_str(), nullptr, &m_graph);
  CHECK_QNN(status, "Failed to create graph");

  m_graphDims = {1, 4, 4, 1};  // NHWC example

  // Create graph I/O tensors (APP_*)
  m_input.version       = QNN_TENSOR_VERSION_1;
  m_input.v1.id         = 0;
  m_input.v1.name       = "input";
  m_input.v1.type       = QNN_TENSOR_TYPE_APP_WRITE;  // graph input
  m_input.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  m_input.v1.dataType   = QNN_DATATYPE_FLOAT_32;
  m_input.v1.rank       = m_graphDims.size();
  m_input.v1.dimensions = m_graphDims.data();
  m_input.v1.memType    = QNN_TENSORMEMTYPE_RAW;
  m_input.v1.clientBuf  = {nullptr, 0};

  m_output.version       = QNN_TENSOR_VERSION_1;
  m_output.v1.id         = 1;
  m_output.v1.name       = "output";
  m_output.v1.type       = QNN_TENSOR_TYPE_APP_READ;  // graph output
  m_output.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  m_output.v1.dataType   = QNN_DATATYPE_FLOAT_32;
  m_output.v1.rank       = m_graphDims.size();
  m_output.v1.dimensions = m_graphDims.data();
  m_output.v1.memType    = QNN_TENSORMEMTYPE_RAW;
  m_output.v1.clientBuf  = {nullptr, 0};

  // Create the graph tensors
  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(m_graph, &m_input);
  CHECK_QNN(status, "Failed to create input tensor");
  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(m_graph, &m_output);
  CHECK_QNN(status, "Failed to create output tensor");

  // Create tensor references for the operation (following the test pattern)
  const uint32_t inputId  = m_input.v1.id;
  const uint32_t outputId = m_output.v1.id;
  const char* inputName   = m_input.v1.name;
  const char* outputName  = m_output.v1.name;

  // Create minimal tensor references with only ID for connectivity
  Qnn_Tensor_t inputRef;
  memset(&inputRef, 0, sizeof(inputRef));
  inputRef.version = QNN_TENSOR_VERSION_1;
  inputRef.v1.id   = inputId;

  Qnn_Tensor_t outputRef;
  memset(&outputRef, 0, sizeof(outputRef));
  outputRef.version = QNN_TENSOR_VERSION_1;
  outputRef.v1.id   = outputId;

  // Configure a ReLU node
  Qnn_OpConfig_t opConfig;
  opConfig.version          = QNN_OPCONFIG_VERSION_1;
  opConfig.v1.name          = "ReluNode";
  opConfig.v1.packageName   = "qti.aisw";
  opConfig.v1.typeName      = "Relu";
  opConfig.v1.numOfParams   = 0;
  opConfig.v1.params        = nullptr;
  opConfig.v1.numOfInputs   = 1;
  opConfig.v1.inputTensors  = &inputRef;
  opConfig.v1.numOfOutputs  = 1;
  opConfig.v1.outputTensors = &outputRef;

  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphAddNode(m_graph, opConfig);
  CHECK_QNN(status, "Failed to add ReLU node");

  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize(m_graph, nullptr, nullptr);
  CHECK_QNN(status, "Failed to finalize graph");

  // Restore m_input and m_output for later use in executeAsyncLoop
  m_input.v1.name  = inputName;
  m_output.v1.name = outputName;
  QNN_INFO("Graph created successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::serializeAndDeserializeContext() {
  QNN_INFO("Serializing context...");

  Qnn_ContextBinarySize_t binarySize = 0;
  Qnn_ErrorHandle_t status =
      m_qnnInterface->QNN_INTERFACE_VER_NAME.contextGetBinarySize(m_context, &binarySize);
  CHECK_QNN(status, "Failed to get context binary size");

  std::vector<uint8_t> binaryBuffer(binarySize);
  Qnn_ContextBinarySize_t writtenSize = 0;
  status                              = m_qnnInterface->QNN_INTERFACE_VER_NAME.contextGetBinary(
      m_context, binaryBuffer.data(), binarySize, &writtenSize);
  CHECK_QNN(status, "Failed to get context binary");

  QNN_INFO("Context serialized successfully (size: %zu bytes)", writtenSize);

  // Create new context from binary
  QNN_INFO("Deserializing context...");
  Qnn_ContextHandle_t deserializedContext = nullptr;
  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(m_backendHandle,
                                                                          nullptr,
                                                                          nullptr,
                                                                          binaryBuffer.data(),
                                                                          binarySize,
                                                                          &deserializedContext,
                                                                          nullptr);
  CHECK_QNN(status, "Failed to create context from binary");

  // Retrieve the graph from deserialized context
  Qnn_GraphHandle_t retrievedGraph;
  status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve(
      deserializedContext, m_graphName.c_str(), &retrievedGraph);
  CHECK_QNN(status, "Failed to retrieve graph from deserialized context");

  QNN_INFO("Context deserialized and graph retrieved successfully");

  // Free the original context before replacing it
  if (m_context) {
    m_qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(m_context, nullptr);
  }

  // Use the retrieved graph for the rest of the example
  m_graph   = retrievedGraph;
  m_context = deserializedContext;

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::setupFenceConfiguration() {
  QNN_INFO("Setting up fence configuration...");

  // Initialize the fence handle array with sufficient size for all iterations
  // Each iteration i uses (i+1) fences, so max iteration uses m_iterations fences
  m_fenceHandleArr.resize(m_iterations);

  // Create fence config V1
  QnnGpuGraph_FenceConfigV1_t fenceConfigV1;
  fenceConfigV1.numInputFences    = &m_numInputFences;
  fenceConfigV1.inputFenceHandles = m_fenceHandleArr.data();
  fenceConfigV1.enableOutputFence = true;
  fenceConfigV1.outputFence       = &m_outputFence;

  // Create fence config with version
  QnnGpuGraph_FenceConfig_t fenceConfig;
  fenceConfig.version       = QNN_GPU_GRAPH_FENCE_CONFIG_VERSION_V1;
  fenceConfig.fenceConfigV1 = fenceConfigV1;

  // Create array of V2 configs
  QnnGpuGraph_CustomConfigV2_t customConfigV2;
  customConfigV2.handshake   = QNN_GPU_GRAPH_V2_HANDSHAKE_FLAG;
  customConfigV2.option      = QNN_GPU_GRAPH_CONFIG_OPTION_FENCE_CONFIG;
  customConfigV2.fenceConfig = fenceConfig;

  QnnGraph_Config_t graphConfig;
  graphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphConfig.customConfig = reinterpret_cast<void*>(&customConfigV2);

  const QnnGraph_Config_t* configs[] = {&graphConfig, nullptr};

  Qnn_ErrorHandle_t status =
      m_qnnInterface->QNN_INTERFACE_VER_NAME.graphSetConfig(m_graph, configs);
  CHECK_QNN(status, "Failed to set graph config with fences");

  QNN_INFO("Fence configuration set successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::executeAsyncLoop() {
  QNN_INFO("Executing async loop with dynamic fence updates...");

  // Set up tensors for execution
  m_input.v1.clientBuf.data     = m_inputData.data();
  m_input.v1.clientBuf.dataSize = m_inputData.size() * sizeof(float);

  m_output.v1.clientBuf.data     = m_outputData.data();
  m_output.v1.clientBuf.dataSize = m_outputData.size() * sizeof(float);

  cl_semaphore_properties_khr export_props[] = {
      static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
      static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_BINARY_KHR),
      static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
      static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_HANDLE_SYNC_FD_KHR),
      static_cast<cl_semaphore_properties_khr>(0),
      static_cast<cl_semaphore_properties_khr>(0)};

  for (int i = 0; i < m_iterations; ++i) {
    QNN_INFO("--- Iteration %d ---", i + 1);

    // Track handles created in this iteration for cleanup on error
    std::vector<void*> iterationHandles;
    std::vector<cl_semaphore_khr> iterationSemaphores;

    // Update numInputFences (client holds the value)
    m_numInputFences = i + 1;  // 1, 2, 3 fences per iteration
    QNN_INFO("Using %d input fences", m_numInputFences);

    // Create new semaphores for this iteration
    cl_int err;
    for (int j = 0; j < m_numInputFences; ++j) {
      if (!clCreateSemaphoreWithPropertiesKHR_ptr) {
        QNN_ERROR("clCreateSemaphoreWithPropertiesKHR function pointer is null");
        return StatusCode::FAILURE;
      }
      cl_semaphore_khr newSemaphore =
          clCreateSemaphoreWithPropertiesKHR_ptr(m_clContext, export_props, &err);
      if (err != CL_SUCCESS) {
        // Cleanup handles created in this iteration
        for (auto handle : iterationHandles) {
          close((int)(intptr_t)handle);
        }
        for (auto sem : iterationSemaphores) {
          clReleaseSemaphoreKHR_ptr(sem);
        }
        // Remove iteration semaphores from global tracking
        m_semaphores.erase(m_semaphores.end() - iterationSemaphores.size(), m_semaphores.end());
        m_semaphoreHandles.erase(m_semaphoreHandles.end() - iterationHandles.size(),
                                 m_semaphoreHandles.end());
        // Close output fence from previous iteration if exists
        if (m_outputFence >= 0) {
          close(m_outputFence);
          m_outputFence = -2;
        }
        QNN_ERROR("OpenCL Error: Failed to create semaphore (status=%d)", err);
        return StatusCode::FAILURE;
      }
      iterationSemaphores.push_back(newSemaphore);
      m_semaphores.push_back(newSemaphore);

      // Signal semaphore
      if (!clEnqueueSignalSemaphoresKHR_ptr) {
        QNN_ERROR("clEnqueueSignalSemaphoresKHR function pointer is null");
        return StatusCode::FAILURE;
      }
      err =
          clEnqueueSignalSemaphoresKHR_ptr(m_queue, 1, &newSemaphore, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        // Cleanup handles created in this iteration
        for (auto handle : iterationHandles) {
          close((int)(intptr_t)handle);
        }
        for (auto sem : iterationSemaphores) {
          clReleaseSemaphoreKHR_ptr(sem);
        }
        // Remove iteration semaphores from global tracking
        m_semaphores.erase(m_semaphores.end() - iterationSemaphores.size(), m_semaphores.end());
        m_semaphoreHandles.erase(m_semaphoreHandles.end() - iterationHandles.size(),
                                 m_semaphoreHandles.end());
        // Close output fence from previous iteration if exists
        if (m_outputFence >= 0) {
          close(m_outputFence);
          m_outputFence = -2;
        }
        QNN_ERROR("OpenCL Error: Failed to signal semaphore (status=%d)", err);
        return StatusCode::FAILURE;
      }
      if (!clFlush_ptr) {
        QNN_ERROR("clFlush function pointer is null");
        return StatusCode::FAILURE;
      }
      err = clFlush_ptr(m_queue);
      if (err != CL_SUCCESS) {
        // Cleanup handles created in this iteration
        for (auto handle : iterationHandles) {
          close((int)(intptr_t)handle);
        }
        for (auto sem : iterationSemaphores) {
          clReleaseSemaphoreKHR_ptr(sem);
        }
        // Remove iteration semaphores from global tracking
        m_semaphores.erase(m_semaphores.end() - iterationSemaphores.size(), m_semaphores.end());
        m_semaphoreHandles.erase(m_semaphoreHandles.end() - iterationHandles.size(),
                                 m_semaphoreHandles.end());
        // Close output fence from previous iteration if exists
        if (m_outputFence >= 0) {
          close(m_outputFence);
          m_outputFence = -2;
        }
        QNN_ERROR("OpenCL Error: Failed to flush queue (status=%d)", err);
        return StatusCode::FAILURE;
      }

      // Get FD handle and update array (client holds the array)
      if (!clGetSemaphoreHandleForTypeKHR_ptr) {
        QNN_ERROR("clGetSemaphoreHandleForTypeKHR function pointer is null");
        return StatusCode::FAILURE;
      }
      void* newHandle;
      err = clGetSemaphoreHandleForTypeKHR_ptr(newSemaphore,
                                               m_device,
                                               CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                                               sizeof(void*),
                                               &newHandle,
                                               nullptr);
      if (err != CL_SUCCESS) {
        // Cleanup handles created in this iteration
        for (auto handle : iterationHandles) {
          close((int)(intptr_t)handle);
        }
        for (auto sem : iterationSemaphores) {
          clReleaseSemaphoreKHR_ptr(sem);
        }
        // Remove iteration semaphores from global tracking
        m_semaphores.erase(m_semaphores.end() - iterationSemaphores.size(), m_semaphores.end());
        m_semaphoreHandles.erase(m_semaphoreHandles.end() - iterationHandles.size(),
                                 m_semaphoreHandles.end());
        // Close output fence from previous iteration if exists
        if (m_outputFence >= 0) {
          close(m_outputFence);
          m_outputFence = -2;
        }
        QNN_ERROR("OpenCL Error: Failed to get semaphore handle (status=%d)", err);
        return StatusCode::FAILURE;
      }
      iterationHandles.push_back(newHandle);
      m_semaphoreHandles.push_back(newHandle);
      m_fenceHandleArr[j] = (FenceHandle)newHandle;
    }

    // Execute graph asynchronously - the backend should read the updated values from the pointers
    size_t numInferences = 1;
    std::mutex notifyMutex;
    std::condition_variable notifyCv;
    QnnGraphAsyncNotifyParam notifyParam;
    notifyParam.m_infCount = &numInferences;
    notifyParam.m_mutex    = &notifyMutex;
    notifyParam.m_cv       = &notifyCv;

    QNN_INFO("Executing graph asynchronously...");
    Qnn_ErrorHandle_t status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphExecuteAsync(
        m_graph, &m_input, 1, &m_output, 1, nullptr, nullptr, notifyFn, &notifyParam);
    CHECK_QNN(status, "Failed to execute graph asynchronously");

    // Wait for completion
    std::unique_lock<std::mutex> lock(notifyMutex);
    while (numInferences != 0) {
      notifyCv.wait(lock);
    }

    if (!notifyParam.m_status) {
      QNN_ERROR("Async execution failed in iteration %d", i + 1);
      // Cleanup handles from this iteration
      for (auto handle : iterationHandles) {
        close((int)(intptr_t)handle);
      }
      return StatusCode::FAILURE;
    }

    QNN_INFO("Async execution completed successfully");

    // Verify output fence is signaled
    struct pollfd outfd {
      m_outputFence, POLLIN, 0
    };
    int ret = poll(&outfd, 1, 1000);
    if (ret == -1) {
      QNN_ERROR("poll() failed with error in iteration %d: %s", i + 1, strerror(errno));
      return StatusCode::FAILURE;
    } else if (ret == 0) {
      QNN_ERROR("Output fence timed out in iteration %d!", i + 1);
      return StatusCode::FAILURE;
    } else if (ret == 1 && (outfd.revents & POLLIN)) {
      QNN_INFO("Output fence signaled successfully!");
    } else {
      QNN_ERROR("Output fence not signaled in iteration %d (revents: 0x%x)!", i + 1, outfd.revents);
      return StatusCode::FAILURE;
    }

    // Close output fence for this iteration
    close(m_outputFence);
    m_outputFence = -2;  // Reset for next iteration
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::cleanupQnn() {
  if (m_qnnInterface && m_context) {
    m_qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(m_context, nullptr);
    m_context = nullptr;
  }

  if (m_qnnInterface && m_backendHandle) {
    m_qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(m_backendHandle);
    m_backendHandle = nullptr;
  }

  if (m_qnnInterface && m_logHandle) {
    m_qnnInterface->QNN_INTERFACE_VER_NAME.logFree(m_logHandle);
    m_logHandle = nullptr;
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnGpuFencingSampleApp::cleanupOpenCL() {
  // Cleanup semaphores
  QNN_INFO("Cleaning up semaphores...");
  for (auto sem : m_semaphores) {
    if (clReleaseSemaphoreKHR_ptr) {
      clReleaseSemaphoreKHR_ptr(sem);
    }
  }
  m_semaphores.clear();

  for (auto handle : m_semaphoreHandles) {
    close((int)(intptr_t)handle);
  }
  m_semaphoreHandles.clear();

  if (m_queue && clReleaseCommandQueue_ptr) {
    clReleaseCommandQueue_ptr(m_queue);
    m_queue = nullptr;
  }

  if (m_clContext && clReleaseContext_ptr) {
    clReleaseContext_ptr(m_clContext);
    m_clContext = nullptr;
  }

  return StatusCode::SUCCESS;
}

void QnnGpuFencingSampleApp::notifyFn(void* notifyParam, Qnn_NotifyStatus_t notifyStatus) {
  QnnGraphAsyncNotifyParam* param = (QnnGraphAsyncNotifyParam*)notifyParam;
  const std::lock_guard<std::mutex> lock(*param->m_mutex);

  // Set status based on error code
  param->m_status = (notifyStatus.error == QNN_SUCCESS);
  (*param->m_infCount)--;
  if (*param->m_infCount == 0) {
    param->m_cv->notify_all();
  }
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn
