//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "GenieCommon.h"
#include "GenieEmbedding.h"
#include "GenieLog.h"
#include "GenieProfile.h"

std::string g_config        = "";
std::string g_prompt        = "";
std::string g_outputFile    = "output.raw";
std::string g_outputDimFile = "embeddingInfo.json";
std::string g_profilePath   = "";
std::string g_logLevel      = "";

std::unordered_set<std::string> g_commandLineArguments;
std::unordered_map<std::string, std::pair<bool, bool>> g_options;

bool isSet(const std::string& name) {
  auto sought = g_options.find(name);
  return (sought != g_options.end()) && (sought->second).first;
}

bool isRequired(const std::string& name) {
  auto sought = g_options.find(name);
  return (sought != g_options.end()) && (sought->second).second;
}

void addOption(const std::string& name, bool set, bool isRequired) {
  g_options.emplace(name, std::make_pair(set, isRequired));
}

void printUsage(const char* program) {
  std::cout << "Usage:\n" << program << " [options]\n" << std::endl;
  std::cout << "Options:" << std::endl;

  int width = 88;

  std::cout << std::left << std::setw(width) << "  -h, --help";
  std::cout << "Show this help message and exit.\n" << std::endl;

  std::cout << std::setw(width) << "  -c CONFIG_FILE or --config CONFIG_FILE";
  std::cout << "Embedding JSON configuration file.\n" << std::endl;

  std::cout << std::setw(width) << "  -p PROMPT or --prompt PROMPT";
  std::cout << "Prompt to query. Mutually exclusive with --prompt_file.\n" << std::endl;

  std::cout << std::setw(width) << "  --prompt_file PATH";
  std::cout << "Prompt to query provided as a file. Mutually exclusive with --prompt.\n"
            << std::endl;

  std::cout << std::setw(width) << "  --output_file PATH";
  std::cout << "Output file path to save embedding result. Default file is output.raw."
            << std::endl;

  std::cout << std::setw(width) << "";
  std::cout
      << "Output file saves the float buffer returned by GenieEmbedding_GenerateCallback_t Fn,"
      << std::endl;

  std::cout << std::setw(width) << "";
  std::cout << "User must consult the rank and dimensions, for the shape of the output.\n"
            << std::endl;

  std::cout << std::setw(width) << "  --log logLevel";
  std::cout << "Enables logging. LogLevel must be one of error, warn, info, or verbose.\n"
            << std::endl;

  std::cout << std::setw(width) << "  --profile FILE_NAME";
  std::cout << "Enables profiling. FILE_NAME is mandatory parameter and provides name of output "
               "file with profiling data.\n"
            << std::endl;

  std::cout << std::setw(width) << " --pid";
  std::cout << "Displays genie-t2e-run process id." << std::endl;

  std::cout << std::setw(width) << " --save_dim PATH";
  std::cout << "Output file path to save embedding Dimension. Default file is embeddingInfo.json."
            << std::endl;
}

bool parseCommandLineInput(int argc, char** argv) {
  bool invalidParam = false;
  std::string arg;
  if (argc == 1) {
    printUsage(argv[0]);
    std::exit(EXIT_SUCCESS);
  }
  for (int i = 1; i < argc; i++) {
    arg = argv[i];
    g_commandLineArguments.insert(arg);
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "-c" || arg == "--config") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      const std::string configPath(argv[i]);
      std::getline(std::ifstream(configPath), g_config, '\0');
      addOption("--config", true, false);
    } else if (arg == "-p" || arg == "--prompt") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      g_prompt = argv[i];
      addOption("--prompt", true, false);
    } else if (arg == "--prompt_file") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      const std::string promptFilePath(argv[i]);
      std::getline(std::ifstream(promptFilePath), g_prompt, '\0');
      addOption("--prompt_file", true, false);
    } else if (arg == "--output_file") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      g_outputFile = std::string(argv[i]);
      addOption("--output_file", true, false);
    } else if (arg == "--save_dim") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      g_outputDimFile = std::string(argv[i]);
      addOption("--save_dim", true, false);
    } else if (arg == "--profile") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      const std::string profilePath(argv[i]);
      const std::ifstream profileStream = std::ifstream(profilePath);
      if (profileStream.good()) {
        std::cerr << "ERROR: Invalid --profile argument. Output file " << argv[i]
                  << " already exists.\n";
        return false;
      }
      g_profilePath = std::string(argv[i]);
      addOption("--profile", true, false);
    } else if (arg == "--log") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      std::vector<std::string> validLogLevel = {"error", "warn", "info", "verbose"};
      std::string inputLogLevel(argv[i]);
      if (std::find(validLogLevel.begin(), validLogLevel.end(), inputLogLevel) ==
          validLogLevel.end()) {
        std::cerr << "ERROR: Invalid --log argument. Argument " << argv[i]
                  << " is not one of error, warn, info, or verbose.\n";
        return false;
      }
      g_logLevel = inputLogLevel;
      addOption("--log", true, false);
    } else if (arg == "--pid") {
      addOption("--pid", true, false);
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      return false;
    }
  }
  if (invalidParam) {
    std::cerr << "ERROR: Invalid parameter for argument: " << arg << std::endl;
    printUsage(argv[0]);
    return false;
  }

  if (!isSet("--prompt") && !isSet("--prompt_file")) {
    std::cerr << "ERROR:: Please provide prompt using --prompt or --prompt_file." << std::endl;
    return false;
  } else if (isSet("--prompt") && isSet("--prompt_file")) {
    std::cerr << "ERROR:: Please provide only one of --prompt or --prompt_file." << std::endl;
    return false;
  }

  return true;
}

// Callback
void embeddingCallback(const uint32_t* dimensions,
                       const uint32_t rank,
                       const float* embeddingBuffer,
                       const void*) {
  // Function will save embedding vector to a file.
  // calculate the size of the embedding buffers using dimension vector
  uint64_t embeddingBufferSize = 1;
  // Create and open a file
  std::ofstream outJsonFile(g_outputDimFile);
  if (outJsonFile.is_open()) {
    outJsonFile << "{\n";
    outJsonFile << " \"Rank\" : " << rank << ",\n";
    outJsonFile << " \"Dimensions\" : [";
  }

  std::cout << "RANK of DIMENSIONS : " << rank << "\n" << std::endl;
  std::cout << "EMBEDDING DIMENSIONS : [ ";

  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dimensions[i] << ((i != rank - 1) ? ", " : " ]\n");
    if (outJsonFile.is_open()) {
      if (i != 0) {
        outJsonFile << ",";
      }
      outJsonFile << dimensions[i];
    }
    embeddingBufferSize = embeddingBufferSize * dimensions[i];
  }

  if (outJsonFile.is_open()) {
    outJsonFile << "],\n";
    outJsonFile << " \"Size : " << embeddingBufferSize << "\n";
    outJsonFile << "}";
    outJsonFile.close();
  }
  std::cout << std::endl;
  std::cout << "GENERATED EMBEDDING SIZE : " << embeddingBufferSize << std::endl;

  // Open a binary file for writing
  std::ofstream outFile(g_outputFile, std::ios::binary);
  if (!outFile.good()) {
    std::cerr << "Error in opening file for writing!" << std::endl;
    return;
  }

  // Write the buffer to the file
  outFile.write(reinterpret_cast<const char*>(embeddingBuffer),
                static_cast<std::streamsize>(embeddingBufferSize * sizeof(float)));

  // Close the file
  outFile.close();

  std::cout << "Embedding vectors saved in " << g_outputFile << std::endl;
  std::cout << "Embedding Dimension saved in " << g_outputDimFile << std::endl;
}

class Log {
 public:
  Log(GenieLog_Callback_t callback, const std::string& logLevel) {
    const int32_t status = GenieLog_create(nullptr, callback, convertLogLevel(logLevel), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the Log handle.");
    }
  }

  GenieLog_Handle_t getLogHandle() { return m_handle; }

  GenieLog_Level_t convertLogLevel(const std::string& logLevel) {
    GenieLog_Level_t logLevelGenie;
    if (logLevel == "error") {
      logLevelGenie = GENIE_LOG_LEVEL_ERROR;
    } else if (logLevel == "warn") {
      logLevelGenie = GENIE_LOG_LEVEL_WARN;
    } else if (logLevel == "info") {
      logLevelGenie = GENIE_LOG_LEVEL_INFO;
    } else {
      logLevelGenie = GENIE_LOG_LEVEL_VERBOSE;
    }
    return logLevelGenie;
  }

  ~Log() {
    const int32_t status = GenieLog_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the Log handle." << std::endl;
    }
  }

 private:
  GenieLog_Handle_t m_handle = NULL;
};

class Profile {
 public:
  Profile() {
    const int32_t status = GenieProfile_create(nullptr, &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the profile handle.");
    }
  }

  GenieProfile_Handle_t getProfileHandle() { return m_handle; }

  void getJsonData() {
    const char* jsonData = nullptr;
    const Genie_AllocCallback_t callback([](size_t size, const char** data) {
      *data = reinterpret_cast<char*>(malloc(size));
      if (*data == nullptr) {
        throw std::runtime_error("Cannot allocate memory for JSON data");
      }
    });

    const int32_t status = GenieProfile_getJsonData(m_handle, callback, &jsonData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to get the profile data");
    }

    std::ofstream outFile;
    outFile.open(g_profilePath);
    if (!outFile.good()) {
      throw std::runtime_error("Cannot create profile output file with name:" + g_profilePath);
    }
    outFile << jsonData;
    outFile.close();
    free(const_cast<char*>(jsonData));
  }

  ~Profile() {
    const int32_t status = GenieProfile_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the profile handle." << std::endl;
    }
  }

 private:
  GenieProfile_Handle_t m_handle = NULL;
};

class Embedding {
 public:
  class Config {
   public:
    Config(const std::string& embeddingConfig,
           std::shared_ptr<Profile> profile,
           std::shared_ptr<Log> log) {
      int32_t status =
          GenieEmbeddingConfig_createFromJson(embeddingConfig.c_str(), &m_configHandle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_configHandle)) {
        throw std::runtime_error("Failed to create the embedding config.");
      }

      if (profile) {
        m_profileHandle = profile->getProfileHandle();
        status          = GenieEmbeddingConfig_bindProfiler(m_configHandle, m_profileHandle);
        if (GENIE_STATUS_SUCCESS != status) {
          throw std::runtime_error("Failed to bind the profile handle with the embedding config.");
        }
      }

      if (log) {
        m_logHandle = log->getLogHandle();
        status      = GenieEmbeddingConfig_bindLogger(m_configHandle, m_logHandle);
        if (GENIE_STATUS_SUCCESS != status) {
          throw std::runtime_error("Failed to bind the log handle with the embedding config.");
        }
      }
    }

    ~Config() {
      int32_t status = GenieEmbeddingConfig_free(m_configHandle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the embedding config." << std::endl;
      }
      m_configHandle = NULL;
    }

    GenieEmbeddingConfig_Handle_t operator()() const { return m_configHandle; }

    GenieEmbeddingConfig_Handle_t getHandle() { return m_configHandle; }

   private:
    GenieEmbeddingConfig_Handle_t m_configHandle = NULL;
    GenieProfile_Handle_t m_profileHandle        = NULL;
    GenieLog_Handle_t m_logHandle                = NULL;
  };

  Embedding(Config&& embeddingConfig) {
    const int32_t status = GenieEmbedding_create(embeddingConfig.getHandle(), &m_embeddingHandle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_embeddingHandle)) {
      throw std::runtime_error("Failed to create the embedding.");
    }
  }

  ~Embedding() {
    int32_t status = GenieEmbedding_free(m_embeddingHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the dialog." << std::endl;
    }
  }

  void generate(const std::string& prompt) {
    int32_t status =
        GenieEmbedding_generate(m_embeddingHandle, prompt.c_str(), embeddingCallback, nullptr);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to generate embedding.");
    }
  }

 private:
  GenieEmbedding_Handle_t m_embeddingHandle = NULL;
};

int main(int argc, char** argv) {
  if (!parseCommandLineInput(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (isSet("--pid")) {
#ifdef _WIN32
    std::cout << "genie-t2t-run pid: " << _getpid() << std::endl;
#else
    std::cout << "genie-t2t-run pid: " << getpid() << std::endl;
#endif
  }
  std::cout << "Using libGenie.so version " << Genie_getApiMajorVersion() << "."
            << Genie_getApiMinorVersion() << "." << Genie_getApiPatchVersion() << "\n"
            << std::endl;

  std::shared_ptr<Profile> profiler(nullptr);
  std::shared_ptr<Log> logger(nullptr);
  try {
    if (isSet("--profile")) profiler = std::make_shared<Profile>();
    if (isSet("--log")) {
      logger = std::make_shared<Log>(nullptr, g_logLevel);
    }
    Embedding embedding(Embedding::Config(g_config, profiler, logger));

    std::cout << "[PROMPT]: " << g_prompt.c_str() << std::endl;
    std::cout << std::endl;
    embedding.generate(g_prompt);
    std::cout << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (profiler) profiler->getJsonData();

  return EXIT_SUCCESS;
}
