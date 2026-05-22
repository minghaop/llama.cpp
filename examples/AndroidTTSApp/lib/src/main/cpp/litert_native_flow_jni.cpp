#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <EGL/egl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party_litert_c/litert_common.h"
#include "third_party_litert_c/litert_compiled_model.h"
#include "third_party_litert_c/litert_environment.h"
#include "third_party_litert_c/litert_model.h"
#include "third_party_litert_c/litert_opaque_options.h"
#include "third_party_litert_c/litert_options.h"
#include "third_party_litert_c/litert_tensor_buffer.h"
#include "third_party_litert_c/litert_tensor_buffer_requirements.h"
#include "third_party_litert_c/litert_tensor_buffer_types.h"
#include "third_party_litert_c/options/litert_gpu_options.h"

namespace {

constexpr const char *kTag = "LiteRtNativeFlowJNI";
constexpr bool kForceCpuRuntime = false;
constexpr bool kRequireGpuRuntime = false;

constexpr const char *kInputFiles[] = {
    "0_token.bin",
    "1_token_len.bin",
    "2_prompt_token.bin",
    "3_prompt_token_len.bin",
    "4_prompt_feat.bin",
    "5_prompt_feat_len.bin",
    "6_embedding.bin",
    "7_streaming.bin",
    "8_finalize.bin",
};

template <typename T>
using Fn = T;

void noOpPayloadDeleter(void *) {}
void freePayloadDeleter(void *p) { std::free(p); }

std::string escapeTomlString(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string buildGpuOptionsToml(
        LiteRtGpuBackend backend,
        LiteRtDelegatePrecision precision,
        int numSteps,
        const std::string &serializationDir,
        const std::string &modelCacheKey) {
    std::ostringstream oss;
    oss << "backend = " << static_cast<int>(backend) << "\n";
    oss << "precision = " << static_cast<int>(precision) << "\n";
    oss << "num_steps_of_command_buffer_preparations = " << numSteps << "\n";
    oss << "hint_fully_delegated_to_single_delegate = false\n";
    oss << "hint_waiting_for_completion = false\n";
    oss << "kernel_batch_size = 1\n";
    oss << "buffer_storage_type = "
        << static_cast<int>(kLiteRtDelegateBufferStorageTypeTexture2D) << "\n";
    oss << "prefer_texture_weights = true\n";
    oss << "num_threads_to_upload = 1\n";
    oss << "num_threads_to_compile = 1\n";
    oss << "serialize_program_cache = true\n";
    oss << "cache_only_compiled_programs = true\n";
    if (!serializationDir.empty()) {
        oss << "serialization_dir = \"" << escapeTomlString(serializationDir) << "\"\n";
    }
    if (!modelCacheKey.empty()) {
        oss << "model_cache_key = \"" << escapeTomlString(modelCacheKey) << "\"\n";
    }
    return oss.str();
}

void logSymbolOrigin(const char *symbolName, const void *fnPtr) {
    if (fnPtr == nullptr) return;
    Dl_info info {};
    if (dladdr(fnPtr, &info) != 0 && info.dli_fname != nullptr) {
        __android_log_print(
                ANDROID_LOG_INFO,
                kTag,
                "symbol %s resolved from %s",
                symbolName,
                info.dli_fname);
    }
}

struct LiteRtApi {
    void *handle = nullptr;
    void *gpuHandle = nullptr;
    bool coreReady = false;
    bool gpuOptionReady = false;
    bool gpuOpaqueViaLegacyApi = false;

    Fn<LiteRtStatus (*)(int, const LiteRtEnvOption *, LiteRtEnvironment *)> CreateEnvironment = nullptr;
    Fn<void (*)(LiteRtEnvironment)> DestroyEnvironment = nullptr;

    Fn<LiteRtStatus (*)(LiteRtOptions *)> CreateOptions = nullptr;
    Fn<void (*)(LiteRtOptions)> DestroyOptions = nullptr;
    Fn<LiteRtStatus (*)(LiteRtOptions, LiteRtHwAcceleratorSet)> SetOptionsHardwareAccelerators = nullptr;
    Fn<LiteRtStatus (*)(LiteRtOptions, LiteRtOpaqueOptions)> AddOpaqueOptions = nullptr;

    Fn<LiteRtStatus (*)(const char *, void *, void (*)(void *), LiteRtOpaqueOptions *)> CreateOpaqueOptions = nullptr;

    Fn<LiteRtStatus (*)(LrtGpuOptions **)> CreateGpuOptions = nullptr;
    Fn<void (*)(LrtGpuOptions *)> DestroyGpuOptions = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, LiteRtGpuBackend)> SetGpuBackend = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, LiteRtDelegatePrecision)> SetGpuPrecision = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, int)> SetGpuCmdPrepSteps = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, bool)> SetGpuHintFullyDelegated = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, bool)> SetGpuHintWaitCompletion = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, int)> SetGpuKernelBatchSize = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, LiteRtDelegateBufferStorageType)> SetGpuBufferStorageType = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, bool)> SetGpuPreferTextureWeights = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, int)> SetGpuNumThreadsToUpload = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, int)> SetGpuNumThreadsToCompile = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, bool)> SetGpuSerializeProgramCache = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, bool)> SetGpuCacheCompiledProgramsOnly = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, const char *)> SetGpuSerializationDir = nullptr;
    Fn<LiteRtStatus (*)(LrtGpuOptions *, const char *)> SetGpuModelCacheKey = nullptr;
    Fn<LiteRtStatus (*)(const LrtGpuOptions *, const char **, void **, void (**)(void *))> GetOpaqueGpuOptionsData = nullptr;
    Fn<const char *(*)()> GetGpuOptionsPayloadIdentifier = nullptr;

    Fn<LiteRtStatus (*)(const char *, LiteRtModel *)> CreateModelFromFile = nullptr;
    Fn<void (*)(LiteRtModel)> DestroyModel = nullptr;
    Fn<LiteRtStatus (*)(LiteRtModel, LiteRtParamIndex *)> GetNumModelSignatures = nullptr;
    Fn<LiteRtStatus (*)(LiteRtModel, LiteRtParamIndex, LiteRtSignature *)> GetModelSignature = nullptr;
    Fn<LiteRtStatus (*)(LiteRtSignature, LiteRtParamIndex *)> GetNumSignatureInputs = nullptr;
    Fn<LiteRtStatus (*)(LiteRtSignature, LiteRtParamIndex *)> GetNumSignatureOutputs = nullptr;
    Fn<LiteRtStatus (*)(LiteRtSignature, LiteRtParamIndex, const char **)> GetSignatureInputName = nullptr;
    Fn<LiteRtStatus (*)(LiteRtSignature, LiteRtParamIndex, LiteRtTensor *)> GetSignatureInputTensorByIndex = nullptr;
    Fn<LiteRtStatus (*)(LiteRtSignature, LiteRtParamIndex, LiteRtTensor *)> GetSignatureOutputTensorByIndex = nullptr;
    Fn<LiteRtStatus (*)(LiteRtTensor, LiteRtRankedTensorType *)> GetRankedTensorType = nullptr;

    Fn<LiteRtStatus (*)(LiteRtEnvironment, LiteRtModel, LiteRtOptions, LiteRtCompiledModel *)> CreateCompiledModel = nullptr;
    Fn<void (*)(LiteRtCompiledModel)> DestroyCompiledModel = nullptr;
    Fn<LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex, LiteRtTensorBufferRequirements *)> GetCompiledModelInputBufferRequirements = nullptr;
    Fn<LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex, LiteRtTensorBufferRequirements *)> GetCompiledModelOutputBufferRequirements = nullptr;

    Fn<LiteRtStatus (*)(LiteRtEnvironment, const LiteRtRankedTensorType *, LiteRtTensorBufferRequirements, LiteRtTensorBuffer *)> CreateManagedTensorBufferFromRequirements = nullptr;
    Fn<LiteRtStatus (*)(const LiteRtRankedTensorType *, void *, size_t, LiteRtHostMemoryDeallocator, LiteRtTensorBuffer *)> CreateTensorBufferFromHostMemory = nullptr;
    Fn<void (*)(LiteRtTensorBuffer)> DestroyTensorBuffer = nullptr;
    Fn<LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, size_t, LiteRtTensorBuffer *, size_t, LiteRtTensorBuffer *)> RunCompiledModel = nullptr;
    Fn<LiteRtStatus (*)(LiteRtTensorBuffer, void **, LiteRtTensorBufferLockMode)> LockTensorBuffer = nullptr;
    Fn<LiteRtStatus (*)(LiteRtTensorBuffer)> UnlockTensorBuffer = nullptr;
    Fn<LiteRtStatus (*)(LiteRtTensorBuffer, size_t *)> GetTensorBufferSize = nullptr;

    bool load() {
        handle = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            __android_log_print(ANDROID_LOG_ERROR, kTag, "dlopen libLiteRt.so failed: %s", dlerror());
            return false;
        }
        if (!kForceCpuRuntime) {
            const char* gpuLibCandidates[] = {
                    "libLiteRtGpuAccelerator.so",
                    "libLiteRtOpenClAccelerator.so",
                    "libLiteRtClGlAccelerator.so",
            };
            for (const char* lib : gpuLibCandidates) {
                gpuHandle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
                if (gpuHandle) {
                    __android_log_print(ANDROID_LOG_INFO, kTag, "dlopen %s success", lib);
                    break;
                } else {
                    const char *err = dlerror();
                    __android_log_print(
                            ANDROID_LOG_WARN,
                            kTag,
                            "dlopen %s failed: %s",
                            lib,
                            err ? err : "(null)");
                }
            }
            if (!gpuHandle) {
                __android_log_print(ANDROID_LOG_WARN, kTag, "dlopen GPU accelerator libs failed: %s", dlerror());
            }
        }
#define LOAD_SYM(member, name)                                                                 \
        do {                                                                                   \
            member = reinterpret_cast<decltype(member)>(dlsym(handle, name));                 \
            if (!(member)) {                                                                   \
                __android_log_print(ANDROID_LOG_ERROR, kTag, "missing symbol %s", name);      \
                return false;                                                                  \
            }                                                                                  \
        } while (0)
#define LOAD_SYM_OPTIONAL(member, name)                                                        \
        do {                                                                                   \
            member = reinterpret_cast<decltype(member)>(dlsym(handle, name));                 \
            if (!(member)) {                                                                   \
                __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing %s", name); \
            }                                                                                  \
        } while (0)
        auto loadSymAny = [&](std::initializer_list<const char *> names) -> void * {
            for (const char *name : names) {
                if (name == nullptr) continue;
                if (handle) {
                    if (void *p = dlsym(handle, name)) return p;
                }
                if (gpuHandle) {
                    if (void *p = dlsym(gpuHandle, name)) return p;
                }
                if (void *p = dlsym(RTLD_DEFAULT, name)) return p;
            }
            return nullptr;
        };

        LOAD_SYM(CreateEnvironment, "LiteRtCreateEnvironment");
        LOAD_SYM(DestroyEnvironment, "LiteRtDestroyEnvironment");
        LOAD_SYM(CreateOptions, "LiteRtCreateOptions");
        LOAD_SYM(DestroyOptions, "LiteRtDestroyOptions");
        LOAD_SYM(SetOptionsHardwareAccelerators, "LiteRtSetOptionsHardwareAccelerators");
        LOAD_SYM(AddOpaqueOptions, "LiteRtAddOpaqueOptions");
        LOAD_SYM(CreateOpaqueOptions, "LiteRtCreateOpaqueOptions");

        CreateGpuOptions = reinterpret_cast<decltype(CreateGpuOptions)>(
                loadSymAny({"LrtCreateGpuOptions", "LiteRtCreateGpuOptions"}));
        DestroyGpuOptions = reinterpret_cast<decltype(DestroyGpuOptions)>(
                loadSymAny({"LrtDestroyGpuOptions", "LiteRtDestroyGpuOptions"}));
        SetGpuBackend = reinterpret_cast<decltype(SetGpuBackend)>(
                loadSymAny({"LrtSetGpuOptionsGpuBackend", "LiteRtSetGpuOptionsGpuBackend"}));
        SetGpuPrecision = reinterpret_cast<decltype(SetGpuPrecision)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsPrecision",
                            "LiteRtSetGpuAcceleratorCompilationOptionsPrecision"}));
        SetGpuCmdPrepSteps = reinterpret_cast<decltype(SetGpuCmdPrepSteps)>(
                loadSymAny({"LrtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations",
                            "LiteRtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations"}));
        SetGpuHintFullyDelegated = reinterpret_cast<decltype(SetGpuHintFullyDelegated)>(
                loadSymAny({"LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate",
                            "LiteRtSetGpuOptionsHintFullyDelegatedToSingleDelegate"}));
        SetGpuHintWaitCompletion = reinterpret_cast<decltype(SetGpuHintWaitCompletion)>(
                loadSymAny({"LrtSetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion",
                            "LiteRtSetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion"}));
        SetGpuKernelBatchSize = reinterpret_cast<decltype(SetGpuKernelBatchSize)>(
                loadSymAny({"LrtSetGpuAcceleratorRuntimeOptionsKernelBatchSize",
                            "LiteRtSetGpuAcceleratorRuntimeOptionsKernelBatchSize"}));
        SetGpuBufferStorageType = reinterpret_cast<decltype(SetGpuBufferStorageType)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType",
                            "LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType"}));
        SetGpuPreferTextureWeights = reinterpret_cast<decltype(SetGpuPreferTextureWeights)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights",
                            "LiteRtSetGpuAcceleratorCompilationOptionsPreferTextureWeights"}));
        SetGpuNumThreadsToUpload = reinterpret_cast<decltype(SetGpuNumThreadsToUpload)>(
                loadSymAny({"LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload",
                            "LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload"}));
        SetGpuNumThreadsToCompile = reinterpret_cast<decltype(SetGpuNumThreadsToCompile)>(
                loadSymAny({"LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile",
                            "LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile"}));
        SetGpuSerializeProgramCache = reinterpret_cast<decltype(SetGpuSerializeProgramCache)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache",
                            "LiteRtSetGpuAcceleratorCompilationOptionsSerializeProgramCache"}));
        SetGpuCacheCompiledProgramsOnly = reinterpret_cast<decltype(SetGpuCacheCompiledProgramsOnly)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly",
                            "LiteRtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly"}));
        SetGpuSerializationDir = reinterpret_cast<decltype(SetGpuSerializationDir)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsSerializationDir",
                            "LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir"}));
        SetGpuModelCacheKey = reinterpret_cast<decltype(SetGpuModelCacheKey)>(
                loadSymAny({"LrtSetGpuAcceleratorCompilationOptionsModelCacheKey",
                            "LiteRtSetGpuAcceleratorCompilationOptionsModelCacheKey"}));
        GetOpaqueGpuOptionsData = reinterpret_cast<decltype(GetOpaqueGpuOptionsData)>(
                loadSymAny({"LrtGetOpaqueGpuOptionsData", "LiteRtGetOpaqueGpuOptionsData"}));
        GetGpuOptionsPayloadIdentifier = reinterpret_cast<decltype(GetGpuOptionsPayloadIdentifier)>(
                loadSymAny({"LiteRtGetGpuOptionsPayloadIdentifier", "LrtGetGpuOptionsIdentifier"}));
        logSymbolOrigin("CreateGpuOptions", reinterpret_cast<void *>(CreateGpuOptions));
        logSymbolOrigin("DestroyGpuOptions", reinterpret_cast<void *>(DestroyGpuOptions));
        logSymbolOrigin("SetGpuBackend", reinterpret_cast<void *>(SetGpuBackend));
        logSymbolOrigin("SetGpuPrecision", reinterpret_cast<void *>(SetGpuPrecision));
        logSymbolOrigin("SetGpuCmdPrepSteps", reinterpret_cast<void *>(SetGpuCmdPrepSteps));
        logSymbolOrigin("SetGpuHintFullyDelegated", reinterpret_cast<void *>(SetGpuHintFullyDelegated));
        logSymbolOrigin("SetGpuHintWaitCompletion", reinterpret_cast<void *>(SetGpuHintWaitCompletion));
        logSymbolOrigin("SetGpuKernelBatchSize", reinterpret_cast<void *>(SetGpuKernelBatchSize));
        logSymbolOrigin("SetGpuBufferStorageType", reinterpret_cast<void *>(SetGpuBufferStorageType));
        logSymbolOrigin("SetGpuPreferTextureWeights", reinterpret_cast<void *>(SetGpuPreferTextureWeights));
        logSymbolOrigin("SetGpuNumThreadsToUpload", reinterpret_cast<void *>(SetGpuNumThreadsToUpload));
        logSymbolOrigin("SetGpuNumThreadsToCompile", reinterpret_cast<void *>(SetGpuNumThreadsToCompile));
        logSymbolOrigin("SetGpuSerializeProgramCache", reinterpret_cast<void *>(SetGpuSerializeProgramCache));
        logSymbolOrigin("SetGpuCacheCompiledProgramsOnly", reinterpret_cast<void *>(SetGpuCacheCompiledProgramsOnly));
        logSymbolOrigin("SetGpuSerializationDir", reinterpret_cast<void *>(SetGpuSerializationDir));
        logSymbolOrigin("SetGpuModelCacheKey", reinterpret_cast<void *>(SetGpuModelCacheKey));
        logSymbolOrigin("GetOpaqueGpuOptionsData", reinterpret_cast<void *>(GetOpaqueGpuOptionsData));
        logSymbolOrigin("GetGpuOptionsPayloadIdentifier", reinterpret_cast<void *>(GetGpuOptionsPayloadIdentifier));
        gpuOpaqueViaLegacyApi = GetOpaqueGpuOptionsData != nullptr;
        gpuOptionReady = CreateGpuOptions && SetGpuBackend &&
                SetGpuPrecision && SetGpuCmdPrepSteps &&
                (GetOpaqueGpuOptionsData || GetGpuOptionsPayloadIdentifier);
        if (!CreateGpuOptions) __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing CreateGpuOptions");
        if (!DestroyGpuOptions) __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing DestroyGpuOptions (will skip explicit free)");
        if (!SetGpuBackend) __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing SetGpuBackend");
        if (!SetGpuPrecision) __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing SetGpuPrecision");
        if (!SetGpuCmdPrepSteps) __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing SetGpuCmdPrepSteps");
        if (!GetOpaqueGpuOptionsData && !GetGpuOptionsPayloadIdentifier) {
            __android_log_print(ANDROID_LOG_WARN, kTag, "optional symbol missing GPU opaque options APIs");
        }

        LOAD_SYM(CreateModelFromFile, "LiteRtCreateModelFromFile");
        LOAD_SYM(DestroyModel, "LiteRtDestroyModel");
        LOAD_SYM(GetNumModelSignatures, "LiteRtGetNumModelSignatures");
        LOAD_SYM(GetModelSignature, "LiteRtGetModelSignature");
        LOAD_SYM(GetNumSignatureInputs, "LiteRtGetNumSignatureInputs");
        LOAD_SYM(GetNumSignatureOutputs, "LiteRtGetNumSignatureOutputs");
        LOAD_SYM(GetSignatureInputName, "LiteRtGetSignatureInputName");
        LOAD_SYM(GetSignatureInputTensorByIndex, "LiteRtGetSignatureInputTensorByIndex");
        LOAD_SYM(GetSignatureOutputTensorByIndex, "LiteRtGetSignatureOutputTensorByIndex");
        LOAD_SYM(GetRankedTensorType, "LiteRtGetRankedTensorType");

        LOAD_SYM(CreateCompiledModel, "LiteRtCreateCompiledModel");
        LOAD_SYM(DestroyCompiledModel, "LiteRtDestroyCompiledModel");
        LOAD_SYM(GetCompiledModelInputBufferRequirements, "LiteRtGetCompiledModelInputBufferRequirements");
        LOAD_SYM(GetCompiledModelOutputBufferRequirements, "LiteRtGetCompiledModelOutputBufferRequirements");
        LOAD_SYM(CreateManagedTensorBufferFromRequirements, "LiteRtCreateManagedTensorBufferFromRequirements");
        LOAD_SYM(CreateTensorBufferFromHostMemory, "LiteRtCreateTensorBufferFromHostMemory");
        LOAD_SYM(DestroyTensorBuffer, "LiteRtDestroyTensorBuffer");
        LOAD_SYM(RunCompiledModel, "LiteRtRunCompiledModel");
        LOAD_SYM(LockTensorBuffer, "LiteRtLockTensorBuffer");
        LOAD_SYM(UnlockTensorBuffer, "LiteRtUnlockTensorBuffer");
        LOAD_SYM(GetTensorBufferSize, "LiteRtGetTensorBufferSize");
#undef LOAD_SYM
#undef LOAD_SYM_OPTIONAL
        coreReady = true;
        return true;
    }
};

LiteRtApi &api() {
    static LiteRtApi g_api;
    static bool loaded = g_api.load();
    (void)loaded;
    return g_api;
}

[[noreturn]] void throwRuntime(JNIEnv *env, const std::string &msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exClass, msg.c_str());
    throw std::runtime_error(msg);
}

std::string statusMsg(const char *name, LiteRtStatus st) {
    return std::string(name) + " failed, status=" + std::to_string(static_cast<int>(st));
}

void checkStatus(JNIEnv *env, const char *name, LiteRtStatus st) {
    if (st != kLiteRtStatusOk) {
        throwRuntime(env, statusMsg(name, st));
    }
}

bool fileExists(const std::string &path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

size_t fileSize(const std::string &path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return 0;
    return static_cast<size_t>(st.st_size);
}

std::vector<uint8_t> readAll(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return {};
    }
    ifs.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(size);
    if (size > 0) {
        ifs.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(size));
    }
    return out;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string dirNameOf(const std::string &path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

std::string baseNameOf(const std::string &path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

bool ensureDir(const std::string &path) {
    if (path.empty()) return false;
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir(path.c_str(), 0700) == 0) return true;
    if (errno == EEXIST) return true;
    return false;
}

std::string pickFileForInput(const std::string &inputName, int index) {
    const std::string n = lower(inputName);
    if (n.find("prompt_feat_len") != std::string::npos) return kInputFiles[5];
    if (n.find("prompt_token_len") != std::string::npos) return kInputFiles[3];
    if (n.find("prompt_token") != std::string::npos) return kInputFiles[2];
    if (n.find("token_len") != std::string::npos) return kInputFiles[1];
    if (n == "token" || (n.find("token") != std::string::npos && n.find("prompt") == std::string::npos)) return kInputFiles[0];
    if (n.find("prompt_feat") != std::string::npos) return kInputFiles[4];
    if (n.find("embedding") != std::string::npos) return kInputFiles[6];
    if (n.find("stream") != std::string::npos) return kInputFiles[7];
    if (n.find("final") != std::string::npos) return kInputFiles[8];
    if (index >= 0 && index < static_cast<int>(sizeof(kInputFiles) / sizeof(kInputFiles[0]))) return kInputFiles[index];
    return "";
}

struct AlignedBuffer {
    void *ptr = nullptr;
    size_t size = 0;
    ~AlignedBuffer() {
        if (ptr) free(ptr);
    }
};

struct EglScope {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    bool active = false;

    bool init() {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY) return false;
        EGLint major = 0;
        EGLint minor = 0;
        if (!eglInitialize(display, &major, &minor)) return false;

        const EGLint cfgAttrs[] = {
                EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_NONE
        };
        EGLConfig cfg = nullptr;
        EGLint numCfg = 0;
        if (!eglChooseConfig(display, cfgAttrs, &cfg, 1, &numCfg) || numCfg <= 0) return false;

        const EGLint surfAttrs[] = {
                EGL_WIDTH, 1,
                EGL_HEIGHT, 1,
                EGL_NONE
        };
        surface = eglCreatePbufferSurface(display, cfg, surfAttrs);
        if (surface == EGL_NO_SURFACE) return false;

        const EGLint ctxAttrs[] = {
                EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL_NONE
        };
        context = eglCreateContext(display, cfg, EGL_NO_CONTEXT, ctxAttrs);
        if (context == EGL_NO_CONTEXT) return false;

        if (!eglMakeCurrent(display, surface, surface, context)) return false;
        active = true;
        return true;
    }

    void reset() {
        if (display != EGL_NO_DISPLAY) {
            eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        if (context != EGL_NO_CONTEXT) {
            eglDestroyContext(display, context);
            context = EGL_NO_CONTEXT;
        }
        if (surface != EGL_NO_SURFACE) {
            eglDestroySurface(display, surface);
            surface = EGL_NO_SURFACE;
        }
        if (display != EGL_NO_DISPLAY) {
            eglTerminate(display);
            display = EGL_NO_DISPLAY;
        }
        active = false;
    }

    ~EglScope() {
        reset();
    }
};

struct FlowRunner {
    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiled = nullptr;
    LiteRtParamIndex signatureIndex = 0;
    LiteRtSignature signature = nullptr;
    std::string runtimeMode;

    std::vector<LiteRtRankedTensorType> inputTypes;
    std::vector<LiteRtRankedTensorType> outputTypes;
    std::vector<std::string> inputNames;
    std::vector<LiteRtTensorBuffer> inputBuffers;
    std::vector<LiteRtTensorBuffer> outputBuffers;
    std::vector<std::unique_ptr<AlignedBuffer>> inputBacking;
    std::string preparedInputDir;
    EglScope eglScope;

    ~FlowRunner() {
        for (auto &tb : inputBuffers) {
            if (tb) api().DestroyTensorBuffer(tb);
        }
        inputBuffers.clear();
        inputBacking.clear();
        for (auto &tb : outputBuffers) {
            if (tb) api().DestroyTensorBuffer(tb);
        }
        outputBuffers.clear();
        if (compiled) api().DestroyCompiledModel(compiled);
        if (model) api().DestroyModel(model);
        if (env) api().DestroyEnvironment(env);
    }
};

bool containsInt64Token(const std::string &name) {
    const auto n = lower(name);
    return n == "token" || n.find("prompt_token") != std::string::npos;
}

std::vector<uint8_t> maybeConvertByType(
        const std::vector<uint8_t> &raw,
        LiteRtElementType elementType,
        const std::string &name) {
    if (elementType == kLiteRtElementTypeInt32 && containsInt64Token(name) && raw.size() % sizeof(int64_t) == 0) {
        const auto *src = reinterpret_cast<const int64_t *>(raw.data());
        const size_t count = raw.size() / sizeof(int64_t);
        std::vector<uint8_t> out(count * sizeof(int32_t));
        auto *dst = reinterpret_cast<int32_t *>(out.data());
        for (size_t i = 0; i < count; ++i) {
            const int64_t v = src[i];
            if (v < INT32_MIN) dst[i] = INT32_MIN;
            else if (v > INT32_MAX) dst[i] = INT32_MAX;
            else dst[i] = static_cast<int32_t>(v);
        }
        return out;
    }
    return raw;
}

LiteRtStatus createGpuOpaqueOptions(
        LiteRtGpuBackend backend,
        LiteRtDelegatePrecision precision,
        int numSteps,
        const std::string &serializationDir,
        const std::string &modelCacheKey,
        LiteRtOpaqueOptions *opaqueOut) {
    if (!api().gpuOptionReady) {
        return kLiteRtStatusErrorUnsupported;
    }
    LrtGpuOptions *gpu = nullptr;
    auto st = api().CreateGpuOptions(&gpu);
    if (st != kLiteRtStatusOk) return st;
    if (api().DestroyGpuOptions == nullptr) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "DestroyGpuOptions symbol missing; continue with no-op payload deleter");
    }
    st = api().SetGpuBackend(gpu, backend);
    if (st != kLiteRtStatusOk) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "SetGpuBackend failed backend=%d status=%d; continue with default backend",
                static_cast<int>(backend),
                static_cast<int>(st));
    }
    st = api().SetGpuPrecision(gpu, precision);
    if (st != kLiteRtStatusOk) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "SetGpuPrecision failed precision=%d status=%d; continue with default precision",
                static_cast<int>(precision),
                static_cast<int>(st));
    }
    st = api().SetGpuCmdPrepSteps(gpu, numSteps);
    if (st != kLiteRtStatusOk) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "SetGpuCmdPrepSteps failed steps=%d status=%d; continue with default cmd-prep",
                numSteps,
                static_cast<int>(st));
    }
    if (api().SetGpuHintFullyDelegated) {
        st = api().SetGpuHintFullyDelegated(gpu, false);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuHintFullyDelegated(false) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuHintWaitCompletion) {
        st = api().SetGpuHintWaitCompletion(gpu, false);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuHintWaitCompletion(false) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuKernelBatchSize) {
        st = api().SetGpuKernelBatchSize(gpu, 1);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuKernelBatchSize(1) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuBufferStorageType) {
        st = api().SetGpuBufferStorageType(gpu, kLiteRtDelegateBufferStorageTypeTexture2D);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuBufferStorageType(Texture2D) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuPreferTextureWeights) {
        st = api().SetGpuPreferTextureWeights(gpu, true);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuPreferTextureWeights(true) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuNumThreadsToUpload) {
        st = api().SetGpuNumThreadsToUpload(gpu, 1);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuNumThreadsToUpload(1) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuNumThreadsToCompile) {
        st = api().SetGpuNumThreadsToCompile(gpu, 1);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuNumThreadsToCompile(1) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuSerializeProgramCache) {
        st = api().SetGpuSerializeProgramCache(gpu, true);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuSerializeProgramCache(true) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuCacheCompiledProgramsOnly) {
        st = api().SetGpuCacheCompiledProgramsOnly(gpu, true);
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuCacheCompiledProgramsOnly(true) status=%d",
                static_cast<int>(st));
    }
    if (api().SetGpuSerializationDir && !serializationDir.empty()) {
        st = api().SetGpuSerializationDir(gpu, serializationDir.c_str());
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuSerializationDir(%s) status=%d",
                serializationDir.c_str(),
                static_cast<int>(st));
    }
    if (api().SetGpuModelCacheKey && !modelCacheKey.empty()) {
        st = api().SetGpuModelCacheKey(gpu, modelCacheKey.c_str());
        __android_log_print(
                st == kLiteRtStatusOk ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                kTag,
                "SetGpuModelCacheKey(%s) status=%d",
                modelCacheKey.c_str(),
                static_cast<int>(st));
    }

    if (api().GetOpaqueGpuOptionsData == nullptr && api().GetGpuOptionsPayloadIdentifier == nullptr) {
        if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
        return kLiteRtStatusErrorUnsupported;
    }

    const char *identifier = nullptr;
    void *payload = nullptr;
    void (*payloadDeleter)(void *) = nullptr;
    if (api().GetOpaqueGpuOptionsData != nullptr) {
        st = api().GetOpaqueGpuOptionsData(gpu, &identifier, &payload, &payloadDeleter);
        if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
        if (st != kLiteRtStatusOk) return st;
        return api().CreateOpaqueOptions(identifier, payload, payloadDeleter, opaqueOut);
    }

    // Newer LiteRT runtime may expose only payload identifier. In this mode,
    // payload must be TOML C-string (same contract as LrtGetOpaqueGpuOptionsData),
    // not a raw LrtGpuOptions* pointer.
    identifier = api().GetGpuOptionsPayloadIdentifier();
    if (identifier == nullptr || identifier[0] == '\0') {
        if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
        return kLiteRtStatusErrorInvalidArgument;
    }
    if (std::strcmp(identifier, "gpu_options") != 0) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "unsupported GPU payload identifier=%s (expect gpu_options); fallback to default GPU compile",
                identifier);
        if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
        return kLiteRtStatusErrorUnsupported;
    }
    const std::string toml = buildGpuOptionsToml(
            backend,
            precision,
            numSteps,
            serializationDir,
            modelCacheKey);
    auto *payloadBytes = static_cast<char *>(std::malloc(toml.size() + 1));
    if (payloadBytes == nullptr) {
        if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
        return kLiteRtStatusErrorMemoryAllocationFailure;
    }
    std::memcpy(payloadBytes, toml.c_str(), toml.size() + 1);
    payload = payloadBytes;
    payloadDeleter = freePayloadDeleter;
    if (api().DestroyGpuOptions) api().DestroyGpuOptions(gpu);
    st = api().CreateOpaqueOptions(identifier, payload, payloadDeleter, opaqueOut);
    __android_log_print(
            ANDROID_LOG_INFO,
            kTag,
            "CreateOpaqueOptions identifier=%s status=%d payload=%s",
            identifier,
            static_cast<int>(st),
            payloadBytes);
    return st;
}

LiteRtCompiledModel tryCreateCompiled(
        LiteRtEnvironment env,
        LiteRtModel model,
        const std::string &serializationDir,
        const std::string &modelCacheKey,
        const std::string &mode,
        const std::vector<std::pair<LiteRtGpuBackend, LiteRtDelegatePrecision>> &gpuModes) {
    LiteRtOptions options = nullptr;
    if (api().CreateOptions(&options) != kLiteRtStatusOk) return nullptr;
    auto cleanup = [&]() {
        if (options) api().DestroyOptions(options);
        options = nullptr;
    };

    if (mode == "CPU") {
        if (api().SetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu) != kLiteRtStatusOk) {
            cleanup();
            return nullptr;
        }
        LiteRtCompiledModel cm = nullptr;
        const auto st = api().CreateCompiledModel(env, model, options, &cm);
        __android_log_print(ANDROID_LOG_INFO, kTag, "CreateCompiledModel mode=CPU status=%d", static_cast<int>(st));
        if (st == kLiteRtStatusOk) {
            cleanup();
            return cm;
        }
        cleanup();
        return nullptr;
    }

    if (api().SetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorGpu) != kLiteRtStatusOk) {
        cleanup();
        return nullptr;
    }
    if (!api().gpuOptionReady) {
        LiteRtCompiledModel cm = nullptr;
        const auto st = api().CreateCompiledModel(env, model, options, &cm);
        __android_log_print(ANDROID_LOG_INFO, kTag, "CreateCompiledModel mode=GPU(default) status=%d", static_cast<int>(st));
        if (st == kLiteRtStatusOk) {
            cleanup();
            return cm;
        }
        cleanup();
        return nullptr;
    }

    bool triedDefaultGpuNoOpaque = false;
    for (const auto &cfg : gpuModes) {
        LiteRtOpaqueOptions opaque = nullptr;
        auto st = createGpuOpaqueOptions(
                cfg.first,
                cfg.second,
                /*numSteps=*/0,
                serializationDir,
                modelCacheKey,
                &opaque);
        if (st != kLiteRtStatusOk) {
            __android_log_print(
                    ANDROID_LOG_WARN,
                    kTag,
                    "createGpuOpaqueOptions backend=%d precision=%d status=%d",
                    static_cast<int>(cfg.first),
                    static_cast<int>(cfg.second),
                    static_cast<int>(st));
            if (!triedDefaultGpuNoOpaque) {
                triedDefaultGpuNoOpaque = true;
                LiteRtCompiledModel cm = nullptr;
                const auto stDefault = api().CreateCompiledModel(env, model, options, &cm);
                __android_log_print(
                        ANDROID_LOG_INFO,
                        kTag,
                        "CreateCompiledModel mode=GPU(default-after-opaque-fail) status=%d",
                        static_cast<int>(stDefault));
                if (stDefault == kLiteRtStatusOk && cm != nullptr) {
                    cleanup();
                    return cm;
                }
            }
            continue;
        }
        st = api().AddOpaqueOptions(options, opaque);
        if (st != kLiteRtStatusOk) {
            __android_log_print(
                    ANDROID_LOG_WARN,
                    kTag,
                    "AddOpaqueOptions backend=%d precision=%d status=%d",
                    static_cast<int>(cfg.first),
                    static_cast<int>(cfg.second),
                    static_cast<int>(st));
            continue;
        }
        LiteRtCompiledModel cm = nullptr;
        st = api().CreateCompiledModel(env, model, options, &cm);
        __android_log_print(
                ANDROID_LOG_INFO,
                kTag,
                "CreateCompiledModel mode=GPU backend=%d precision=%d status=%d",
                static_cast<int>(cfg.first),
                static_cast<int>(cfg.second),
                static_cast<int>(st));
        if (st == kLiteRtStatusOk && cm != nullptr) {
            cleanup();
            return cm;
        }
        // Re-create options to clear previous opaque chain before trying next.
        cleanup();
        if (api().CreateOptions(&options) != kLiteRtStatusOk) return nullptr;
        if (api().SetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorGpu) != kLiteRtStatusOk) {
            cleanup();
            return nullptr;
        }
    }
    cleanup();
    return nullptr;
}

std::unique_ptr<FlowRunner> createRunner(JNIEnv *env, const std::string &modelPath) {
    auto &lrt = api();
    if (!lrt.handle || !lrt.coreReady) {
        throwRuntime(env, "LiteRT runtime library not loaded");
    }

    auto runner = std::make_unique<FlowRunner>();
    if (!kForceCpuRuntime && !runner->eglScope.init()) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "EGL context init failed, continue for OpenCL path");
    }
    checkStatus(env, "LiteRtCreateEnvironment", lrt.CreateEnvironment(0, nullptr, &runner->env));
    checkStatus(env, "LiteRtCreateModelFromFile", lrt.CreateModelFromFile(modelPath.c_str(), &runner->model));

    const std::string modelDir = dirNameOf(modelPath);
    const std::string serializationDir = modelDir + "/litert_gpu_cache";
    if (!ensureDir(serializationDir)) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kTag,
                "failed to create serialization dir: %s",
                serializationDir.c_str());
    }
    const std::string modelCacheKey =
            "flow_opengl_fp32_" + baseNameOf(modelPath) + "_" + std::to_string(fileSize(modelPath));

    (void)fileSize(modelPath);
    LiteRtCompiledModel cm = nullptr;
    std::string runtime;
    if (kForceCpuRuntime) {
        cm = tryCreateCompiled(runner->env, runner->model, serializationDir, modelCacheKey, "CPU", {});
        if (cm) runtime = "CPU";
    } else {
        cm = tryCreateCompiled(
                runner->env,
                runner->model,
                serializationDir,
                modelCacheKey,
                "GPU",
                {
                        {kLiteRtGpuBackendOpenGl, kLiteRtDelegatePrecisionFp32},
                });
        if (cm) runtime = lrt.gpuOptionReady ? "GPU(OPENGL)" : "GPU(DEFAULT)";
    }

    // No implicit GPU fallback here: avoid unexpected OpenGL path with missing EGL context.
    if (!cm && !kRequireGpuRuntime) {
        cm = tryCreateCompiled(runner->env, runner->model, serializationDir, modelCacheKey, "CPU", {});
        runtime = "CPU";
    }
    if (!cm) {
        if (kRequireGpuRuntime) {
            throwRuntime(env, "LiteRT GPU is required, but GPU compiled model creation failed");
        } else {
            throwRuntime(env, "LiteRT C++ API failed to create compiled model");
        }
    }
    runner->compiled = cm;
    runner->runtimeMode = runtime;

    LiteRtParamIndex numSignatures = 0;
    checkStatus(env, "LiteRtGetNumModelSignatures", lrt.GetNumModelSignatures(runner->model, &numSignatures));
    if (numSignatures <= 0) {
        throwRuntime(env, "LiteRT model has no signatures");
    }
    runner->signatureIndex = 0;
    checkStatus(env, "LiteRtGetModelSignature", lrt.GetModelSignature(runner->model, runner->signatureIndex, &runner->signature));

    LiteRtParamIndex numInputs = 0;
    LiteRtParamIndex numOutputs = 0;
    checkStatus(env, "LiteRtGetNumSignatureInputs", lrt.GetNumSignatureInputs(runner->signature, &numInputs));
    checkStatus(env, "LiteRtGetNumSignatureOutputs", lrt.GetNumSignatureOutputs(runner->signature, &numOutputs));
    if (numInputs <= 0 || numOutputs <= 0) {
        throwRuntime(env, "LiteRT signature has invalid IO count");
    }

    runner->inputTypes.resize(numInputs);
    runner->outputTypes.resize(numOutputs);
    runner->inputNames.resize(numInputs);
    runner->outputBuffers.resize(numOutputs, nullptr);

    for (LiteRtParamIndex i = 0; i < numInputs; ++i) {
        LiteRtTensor tensor = nullptr;
        checkStatus(env, "LiteRtGetSignatureInputTensorByIndex", lrt.GetSignatureInputTensorByIndex(runner->signature, i, &tensor));
        checkStatus(env, "LiteRtGetRankedTensorType(input)", lrt.GetRankedTensorType(tensor, &runner->inputTypes[i]));
        const char *name = nullptr;
        if (lrt.GetSignatureInputName(runner->signature, i, &name) == kLiteRtStatusOk && name != nullptr) {
            runner->inputNames[i] = name;
        } else {
            runner->inputNames[i] = "";
        }
    }

    for (LiteRtParamIndex i = 0; i < numOutputs; ++i) {
        LiteRtTensor tensor = nullptr;
        checkStatus(env, "LiteRtGetSignatureOutputTensorByIndex", lrt.GetSignatureOutputTensorByIndex(runner->signature, i, &tensor));
        checkStatus(env, "LiteRtGetRankedTensorType(output)", lrt.GetRankedTensorType(tensor, &runner->outputTypes[i]));
        LiteRtTensorBufferRequirements req = nullptr;
        checkStatus(env, "LiteRtGetCompiledModelOutputBufferRequirements",
                    lrt.GetCompiledModelOutputBufferRequirements(runner->compiled, runner->signatureIndex, i, &req));
        checkStatus(env, "LiteRtCreateManagedTensorBufferFromRequirements",
                    lrt.CreateManagedTensorBufferFromRequirements(runner->env, &runner->outputTypes[i], req, &runner->outputBuffers[i]));
    }
    return runner;
}

void prepareInputs(JNIEnv *env, FlowRunner &runner, const std::string &inputDir) {
    if (runner.preparedInputDir == inputDir && !runner.inputBuffers.empty()) {
        return;
    }
    auto &lrt = api();
    for (auto &tb : runner.inputBuffers) {
        if (tb) lrt.DestroyTensorBuffer(tb);
    }
    runner.inputBuffers.clear();
    runner.inputBacking.clear();
    runner.preparedInputDir.clear();

    runner.inputBuffers.resize(runner.inputTypes.size(), nullptr);
    runner.inputBacking.resize(runner.inputTypes.size());

    for (size_t i = 0; i < runner.inputTypes.size(); ++i) {
        const std::string fileName = pickFileForInput(runner.inputNames[i], static_cast<int>(i));
        if (fileName.empty()) {
            throwRuntime(env, "cannot map input file for input index " + std::to_string(i));
        }
        const std::string path = inputDir + "/" + fileName;
        if (!fileExists(path)) {
            throwRuntime(env, "input file missing: " + path);
        }
        auto raw = readAll(path);
        auto bytes = maybeConvertByType(raw, runner.inputTypes[i].element_type, runner.inputNames[i]);
        if (bytes.empty()) {
            throwRuntime(env, "input file is empty: " + path);
        }

        auto mem = std::make_unique<AlignedBuffer>();
        mem->size = bytes.size();
        if (posix_memalign(&mem->ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, mem->size) != 0 || mem->ptr == nullptr) {
            throwRuntime(env, "posix_memalign failed for input " + std::to_string(i));
        }
        std::memcpy(mem->ptr, bytes.data(), mem->size);

        LiteRtTensorBuffer tb = nullptr;
        checkStatus(env, "LiteRtCreateTensorBufferFromHostMemory",
                    lrt.CreateTensorBufferFromHostMemory(&runner.inputTypes[i], mem->ptr, mem->size, nullptr, &tb));
        runner.inputBuffers[i] = tb;
        runner.inputBacking[i] = std::move(mem);
    }

    runner.preparedInputDir = inputDir;
}

std::vector<float> runOnce(JNIEnv *env, FlowRunner &runner, const std::string &inputDir) {
    prepareInputs(env, runner, inputDir);
    auto &lrt = api();
    checkStatus(env, "LiteRtRunCompiledModel",
                lrt.RunCompiledModel(
                        runner.compiled,
                        runner.signatureIndex,
                        runner.inputBuffers.size(),
                        runner.inputBuffers.data(),
                        runner.outputBuffers.size(),
                        runner.outputBuffers.data()));

    if (runner.outputBuffers.empty()) {
        throwRuntime(env, "LiteRT output buffer is empty");
    }
    size_t bytes = 0;
    checkStatus(env, "LiteRtGetTensorBufferSize", lrt.GetTensorBufferSize(runner.outputBuffers[0], &bytes));
    void *ptr = nullptr;
    checkStatus(env, "LiteRtLockTensorBuffer", lrt.LockTensorBuffer(runner.outputBuffers[0], &ptr, kLiteRtTensorBufferLockModeRead));
    const size_t n = bytes / sizeof(float);
    std::vector<float> out(n);
    if (n > 0 && ptr != nullptr) {
        std::memcpy(out.data(), ptr, n * sizeof(float));
    }
    checkStatus(env, "LiteRtUnlockTensorBuffer", lrt.UnlockTensorBuffer(runner.outputBuffers[0]));
    return out;
}

jobject makeCreateResult(JNIEnv *env, jlong handle, const std::string &runtime) {
    jclass cls = env->FindClass("com/example/llama/NativeCreateResult");
    if (cls == nullptr || env->ExceptionCheck()) {
        throwRuntime(env, "FindClass(com/example/llama/NativeCreateResult) failed");
    }
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(JLjava/lang/String;)V");
    if (ctor == nullptr || env->ExceptionCheck()) {
        throwRuntime(env, "GetMethodID(NativeCreateResult.<init>(J,String)) failed");
    }
    jstring runtimeStr = env->NewStringUTF(runtime.c_str());
    if (runtimeStr == nullptr || env->ExceptionCheck()) {
        throwRuntime(env, "NewStringUTF(runtime) failed");
    }
    jobject obj = env->NewObject(cls, ctor, handle, runtimeStr);
    if (obj == nullptr || env->ExceptionCheck()) {
        throwRuntime(env, "NewObject(NativeCreateResult) failed");
    }
    env->DeleteLocalRef(runtimeStr);
    return obj;
}

}  // namespace

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_llama_LiteRtNativeFlowRunner_nativeCreate(JNIEnv *env, jclass, jstring modelPath_) {
    try {
        const char *modelPath = env->GetStringUTFChars(modelPath_, nullptr);
        std::string path = modelPath ? modelPath : "";
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        if (path.empty()) {
            throwRuntime(env, "modelPath is empty");
        }
        auto runner = createRunner(env, path);
        auto *raw = runner.release();
        __android_log_print(ANDROID_LOG_INFO, kTag, "LiteRT native runner created, runtime=%s", raw->runtimeMode.c_str());
        return makeCreateResult(env, reinterpret_cast<jlong>(raw), raw->runtimeMode);
    } catch (const std::exception &e) {
        if (!env->ExceptionCheck()) {
            jclass exClass = env->FindClass("java/lang/RuntimeException");
            env->ThrowNew(exClass, e.what());
        }
        return nullptr;
    }
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_llama_LiteRtNativeFlowRunner_nativeRunFromBin(JNIEnv *env, jclass, jlong handle, jstring inputDir_) {
    try {
        auto *runner = reinterpret_cast<FlowRunner *>(handle);
        if (!runner) {
            throwRuntime(env, "native handle is null");
        }
        const char *inputDir = env->GetStringUTFChars(inputDir_, nullptr);
        std::string dir = inputDir ? inputDir : "";
        env->ReleaseStringUTFChars(inputDir_, inputDir);
        if (dir.empty()) {
            throwRuntime(env, "inputDir is empty");
        }
        auto output = runOnce(env, *runner, dir);
        jfloatArray arr = env->NewFloatArray(static_cast<jsize>(output.size()));
        if (!output.empty()) {
            env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(output.size()), output.data());
        }
        return arr;
    } catch (const std::exception &e) {
        if (!env->ExceptionCheck()) {
            jclass exClass = env->FindClass("java/lang/RuntimeException");
            env->ThrowNew(exClass, e.what());
        }
        return nullptr;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_llama_LiteRtNativeFlowRunner_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    auto *runner = reinterpret_cast<FlowRunner *>(handle);
    delete runner;
}
