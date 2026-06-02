//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <memory>
#include <string>

#include "Logger.hpp"
#include "PAL/GetOpt.hpp"
#include "QnnSampleApp.hpp"

namespace qnn {
namespace tools {
namespace sample_app {

void showHelp() {
  std::cout
      << "\nDESCRIPTION:\n"
      << "------------\n"
      << "Sample application demonstrating QNN GPU fence usage with OpenCL semaphore\n"
      << "integration, context serialization, and asynchronous execution.\n"
      << "\n\n"
      << "REQUIRED ARGUMENTS:\n"
      << "-------------------\n"
      << "  --backend           <FILE>      Path to QNN GPU backend (libQnnGpu.so).\n"
      << "\n\n"

      << "OPTIONAL ARGUMENTS:\n"
      << "-------------------\n"

      << "  --iterations        <VAL>       Number of iterations to run (default: 3, max: 1000).\n"
      << "                                  Each iteration uses a different number of input "
         "fences.\n"
      << "\n"
#ifdef QNN_ENABLE_DEBUG
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\", \"verbose\" and "
         "\"debug\"."
         "\n"
#else
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\" and \"verbose\"."
         "\n"
#endif
      << "\n"
      << "  --version                       Print the QNN SDK version.\n"
      << "\n"
      << "  --help                          Show this help message.\n"
      << std::endl;
}

void showHelpAndExit(std::string&& error) {
  std::cerr << "ERROR: " << error << "\n";
  std::cerr << "Please check help below:\n";
  showHelp();
  std::exit(EXIT_FAILURE);
}

std::unique_ptr<sample_app::QnnGpuFencingSampleApp> processCommandLine(int argc, char** argv) {
  enum OPTIONS {
    OPT_HELP       = 0,
    OPT_BACKEND    = 1,
    OPT_LOG_LEVEL  = 2,
    OPT_VERSION    = 3,
    OPT_ITERATIONS = 4,
  };

  // Create the command line options
  static struct pal::Option s_longOptions[] = {
      {"help", pal::no_argument, NULL, OPT_HELP},
      {"backend", pal::required_argument, NULL, OPT_BACKEND},
      {"log_level", pal::required_argument, NULL, OPT_LOG_LEVEL},
      {"iterations", pal::required_argument, NULL, OPT_ITERATIONS},
      {"version", pal::no_argument, NULL, OPT_VERSION},
      {NULL, 0, NULL, 0}};

  // Command line parsing loop
  int longIndex = 0;
  int opt       = 0;
  std::string backEndPath;
  QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
  int iterations = 3;

  while ((opt = pal::getOptLongOnly(argc, argv, "", s_longOptions, &longIndex)) != -1) {
    switch (opt) {
      case OPT_HELP:
        showHelp();
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_VERSION:
        std::cout << "QNN GPU Fencing Sample App v1.1\n";
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_BACKEND:
        backEndPath = pal::g_optArg;
        break;

      case OPT_ITERATIONS: {
        char* endptr = nullptr;
        long val     = std::strtol(pal::g_optArg, &endptr, 10);

        // Check for conversion errors
        if (endptr == pal::g_optArg || *endptr != '\0') {
          showHelpAndExit("Invalid iterations value. Must be a valid integer.");
        }

        // Check for overflow and range
        if (val <= 0 || val > 1000) {
          showHelpAndExit("Invalid iterations value. Must be between 1 and 1000.");
        }

        iterations = static_cast<int>(val);
      } break;

      case OPT_LOG_LEVEL:
        logLevel = sample_app::parseLogLevel(pal::g_optArg);
        if (logLevel != QNN_LOG_LEVEL_MAX) {
          if (!log::setLogLevel(logLevel)) {
            showHelpAndExit("Unable to set log level.");
          }
        }
        break;

      default:
        std::cerr << "ERROR: Invalid argument passed: " << argv[pal::g_optInd - 1]
                  << "\nPlease check the Arguments section in the description below.\n";
        showHelp();
        std::exit(EXIT_FAILURE);
    }
  }

  if (backEndPath.empty()) {
    showHelpAndExit("Missing option: --backend\n");
  }

  QNN_INFO("Backend: %s", backEndPath.c_str());

  std::unique_ptr<sample_app::QnnGpuFencingSampleApp> app(
      new sample_app::QnnGpuFencingSampleApp(backEndPath, iterations));
  return app;
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn

int main(int argc, char** argv) {
  using namespace qnn::tools;
  int32_t status = EXIT_SUCCESS;

  if (!qnn::log::initializeLogging()) {
    std::cerr << "ERROR: Unable to initialize logging!\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<sample_app::QnnGpuFencingSampleApp> app =
      sample_app::processCommandLine(argc, argv);

  if (nullptr == app) {
    return EXIT_FAILURE;
  }

  QNN_INFO("qnn-gpu-fencing-sample-app build version: v1.1");

  if (sample_app::StatusCode::SUCCESS != app->initialize()) {
    status = app->reportError("Initialization failure");
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->runFencingExample()) {
    status = app->reportError("GPU Fencing Example execution failure");
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->cleanup()) {
    status = app->reportError("Cleanup failure");
  }

  return status;
}
