#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#define protected public
#include "llm/llm.hpp"
#undef protected

namespace {

constexpr const char *kLogTag = "ai-chat-mnn";
constexpr bool kHifiGanUseCpuBackend = true;
constexpr const char *kHifiGanBackendLabel = kHifiGanUseCpuBackend ? "CPU" : "Vulkan";

using MNN::BackendConfig;
using MNN::Express::Executor;
using MNN::Express::VARP;
using MNN::Interpreter;
using MNN::ScheduleConfig;
using MNN::Session;
using MNN::Tensor;
using MNN::Transformer::Llm;

constexpr int kLlmHiddenSize = 896;

struct LlmHandle {
    Llm *llm = nullptr;

    ~LlmHandle() {
        if (llm != nullptr) {
            Llm::destroy(llm);
            llm = nullptr;
        }
    }
};

struct HifiGanHandle {
    std::unique_ptr<Interpreter, void (*)(Interpreter *)> interpreter{
        nullptr,
        Interpreter::destroy,
    };
    BackendConfig backendConfig{};
    Session *session = nullptr;
};

std::string jstring_to_std(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char *raw = env->GetStringUTFChars(value, nullptr);
    std::string out = raw != nullptr ? raw : "";
    if (raw != nullptr) {
        env->ReleaseStringUTFChars(value, raw);
    }
    return out;
}

void throw_illegal_state(JNIEnv *env, const std::string &message) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    if (ex != nullptr) {
        env->ThrowNew(ex, message.c_str());
    }
}

bool ensure_tokenizer_compat(const std::string &config_path) {
    const auto slash = config_path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? "." : config_path.substr(0, slash);
    const std::string mtok_path = dir + "/tokenizer.mtok";
    const std::string txt_path = dir + "/tokenizer.txt";

    std::ifstream mtok(mtok_path, std::ios::binary);
    if (mtok.good()) {
        return true;
    }
    std::ifstream txt(txt_path, std::ios::binary);
    if (!txt.good()) {
        return false;
    }
    std::ofstream mtok_out(mtok_path, std::ios::binary);
    mtok_out << txt.rdbuf();
    const bool ok = mtok_out.good();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "%s: tokenizer compat copy %s -> %s, ok=%d",
            __func__,
            txt_path.c_str(),
            mtok_path.c_str(),
            ok ? 1 : 0);
    return ok;
}

std::string build_llm_override_json(int num_threads) {
    const int safe_threads = std::max(1, num_threads);
    return std::string("{") +
           "\"thread_num\":" + std::to_string(safe_threads) + "," +
           "\"backend_type\":\"cpu\"," +
           "\"hidden_states\":true" +
           "}";
}

std::vector<int> jint_array_to_vector(JNIEnv *env, jintArray array) {
    const jsize size = array != nullptr ? env->GetArrayLength(array) : 0;
    std::vector<int> out(size);
    if (size > 0) {
        env->GetIntArrayRegion(array, 0, size, out.data());
    }
    return out;
}

std::vector<float> jfloat_array_to_vector(JNIEnv *env, jfloatArray array) {
    const jsize size = array != nullptr ? env->GetArrayLength(array) : 0;
    std::vector<float> out(size);
    if (size > 0) {
        env->GetFloatArrayRegion(array, 0, size, out.data());
    }
    return out;
}

std::string float_vector_stats(const std::vector<float> &values) {
    if (values.empty()) {
        return "size=0";
    }
    double min_v = values.front();
    double max_v = values.front();
    long double sum = 0.0L;
    size_t nan_count = 0;
    for (float v : values) {
        if (std::isnan(v)) {
            nan_count += 1;
            continue;
        }
        min_v = std::min(min_v, static_cast<double>(v));
        max_v = std::max(max_v, static_cast<double>(v));
        sum += v;
    }
    const double mean = values.size() > nan_count ? static_cast<double>(sum / static_cast<long double>(values.size() - nan_count)) : 0.0;
    char buffer[256];
    std::snprintf(
            buffer,
            sizeof(buffer),
            "size=%zu min=%.6f max=%.6f mean=%.6f nan=%zu",
            values.size(),
            min_v,
            max_v,
            mean,
            nan_count);
    return std::string(buffer);
}

jfloatArray vector_to_jfloat_array(JNIEnv *env, const std::vector<float> &values) {
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(values.size()));
    if (out != nullptr && !values.empty()) {
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(values.size()), values.data());
    }
    return out;
}

Tensor *resolve_output_tensor(Interpreter *interpreter, Session *session, int output_index) {
    if (interpreter == nullptr || session == nullptr) {
        return nullptr;
    }
    if (output_index <= 0) {
        Tensor *first = interpreter->getSessionOutput(session, nullptr);
        if (first != nullptr) {
            return first;
        }
    }
    const auto &outputs = interpreter->getSessionOutputAll(session);
    if (outputs.empty()) {
        return nullptr;
    }
    const size_t wanted = std::min<size_t>(std::max(output_index, 0), outputs.size() - 1);
    auto it = outputs.begin();
    std::advance(it, static_cast<long>(wanted));
    return it->second;
}

std::vector<float> tensor_to_vector(Tensor *output_tensor) {
    if (output_tensor == nullptr) {
        return {};
    }
    std::unique_ptr<Tensor> host_output(Tensor::createHostTensorFromDevice(output_tensor, true));
    if (host_output == nullptr) {
        return {};
    }
    if (!output_tensor->copyToHostTensor(host_output.get())) {
        return {};
    }

    const int element_count = host_output->elementSize();
    const float *data = host_output->host<float>();
    if (data == nullptr || element_count <= 0) {
        return {};
    }
    return std::vector<float>(data, data + element_count);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_MnnLlmRunner_nativeLoadModel(JNIEnv *env, jclass, jstring config_path_, jint num_threads) {
    const std::string config_path = jstring_to_std(env, config_path_);
    if (config_path.empty()) {
        throw_illegal_state(env, "MNN LLM config path is empty");
        return 0;
    }

    ensure_tokenizer_compat(config_path);

    std::unique_ptr<LlmHandle> handle(new LlmHandle());
    handle->llm = Llm::createLLM(config_path);
    if (handle->llm == nullptr) {
        throw_illegal_state(env, "Llm::createLLM failed: " + config_path);
        return 0;
    }

    const std::string override_json = build_llm_override_json(num_threads);
    handle->llm->set_config(override_json);
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "%s: config=%s override=%s",
            __func__,
            config_path.c_str(),
            override_json.c_str());

    if (!handle->llm->load()) {
        throw_illegal_state(env, "MNN LLM load failed: " + config_path);
        return 0;
    }

    return reinterpret_cast<jlong>(handle.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_MnnLlmRunner_nativeUnloadModel(JNIEnv *, jclass, jlong handle) {
    delete reinterpret_cast<LlmHandle *>(handle);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_llama_MnnLlmRunner_nativeResetKvCache(JNIEnv *, jclass, jlong handle) {
    auto *runner = reinterpret_cast<LlmHandle *>(handle);
    if (runner == nullptr || runner->llm == nullptr) {
        return -1;
    }
    runner->llm->reset();
    return 0;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_llama_MnnLlmRunner_nativeDecodeEmbeddings(
        JNIEnv *env,
        jclass,
        jlong handle,
        jfloatArray input_embeddings_,
        jint n_past) {
    auto *runner = reinterpret_cast<LlmHandle *>(handle);
    if (runner == nullptr || runner->llm == nullptr) {
        throw_illegal_state(env, "MNN LLM runner is not loaded");
        return nullptr;
    }

    std::vector<float> input_embeddings = jfloat_array_to_vector(env, input_embeddings_);
    if (input_embeddings.empty() || input_embeddings.size() % kLlmHiddenSize != 0) {
        throw_illegal_state(
                env,
                "MNN LLM embedding input must be a multiple of 896, got=" + std::to_string(input_embeddings.size()));
        return nullptr;
    }

    const int seq_len = static_cast<int>(input_embeddings.size() / kLlmHiddenSize);
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "%s: seq_len=%d n_past=%d",
            __func__,
            seq_len,
            static_cast<int>(n_past));

    VARP input_var = MNN::Express::_Const(
            input_embeddings.data(),
            {seq_len, kLlmHiddenSize, 1, 1},
            MNN::Express::NCHW,
            halide_type_of<float>());
    auto outputs = runner->llm->forwardVec(input_var);
    if (outputs.size() < 2 || outputs[1].get() == nullptr || outputs[1]->getInfo() == nullptr) {
        throw_illegal_state(env, "MNN LLM forwardVec did not return hidden_states");
        return nullptr;
    }

    VARP hidden_states = outputs[1];
    const auto *info = hidden_states->getInfo();
    const int element_count = static_cast<int>(info->size);
    const float *data = hidden_states->readMap<float>();
    if (data == nullptr || element_count <= 0) {
        throw_illegal_state(env, "MNN LLM hidden_states output is empty");
        return nullptr;
    }

    return vector_to_jfloat_array(env, std::vector<float>(data, data + element_count));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_MnnHifiGanRunner_nativeLoadModel(JNIEnv *env, jclass, jstring model_path_, jint num_threads) {
    const std::string model_path = jstring_to_std(env, model_path_);
    if (model_path.empty()) {
        throw_illegal_state(env, "HifiGan model path is empty");
        return 0;
    }

    std::unique_ptr<HifiGanHandle> handle(new HifiGanHandle());
    handle->interpreter.reset(Interpreter::createFromFile(model_path.c_str()));
    if (!handle->interpreter) {
        throw_illegal_state(env, "Failed to create MNN interpreter for " + model_path);
        return 0;
    }

    ScheduleConfig schedule{};
    schedule.numThread = std::max(1, static_cast<int>(num_threads));
    if (kHifiGanUseCpuBackend) {
        schedule.type = MNN_FORWARD_CPU;
        schedule.backupType = MNN_FORWARD_CPU;
    } else {
        schedule.type = MNN_FORWARD_VULKAN;
        schedule.mode = MNN_GPU_TUNING_WIDE;

        handle->backendConfig.power = BackendConfig::Power_High;
        handle->backendConfig.memory = BackendConfig::Memory_Normal;
        handle->backendConfig.precision = BackendConfig::Precision_High;
        schedule.backendConfig = &handle->backendConfig;
    }

    handle->session = handle->interpreter->createSession(schedule);
    if (handle->session == nullptr) {
        throw_illegal_state(env, "Failed to create HifiGan session for hifigan: " + model_path);
        return 0;
    }

    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "%s: loaded hifigan model=%s backend=%s",
            __func__,
            model_path.c_str(),
            kHifiGanBackendLabel);
    return reinterpret_cast<jlong>(handle.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_MnnHifiGanRunner_nativeUnloadModel(JNIEnv *, jclass, jlong handle) {
    delete reinterpret_cast<HifiGanHandle *>(handle);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_llama_MnnHifiGanRunner_nativeForward(
        JNIEnv *env,
        jclass,
        jlong handle,
        jfloatArray input_,
        jintArray shape_,
        jint output_index) {
    auto *runner = reinterpret_cast<HifiGanHandle *>(handle);
    if (runner == nullptr || !runner->interpreter || runner->session == nullptr) {
        throw_illegal_state(env, "MNN HifiGan runner is not loaded");
        return nullptr;
    }

    std::vector<float> input = jfloat_array_to_vector(env, input_);
    std::vector<int> shape = jint_array_to_vector(env, shape_);
    if (input.empty() || shape.empty()) {
        throw_illegal_state(env, "HifiGan input/shape cannot be empty");
        return nullptr;
    }

    long long expected = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw_illegal_state(env, "HifiGan shape contains non-positive dimension");
            return nullptr;
        }
        expected *= dim;
    }
    if (expected != static_cast<long long>(input.size())) {
        throw_illegal_state(
                env,
                "HifiGan input size mismatch: input=" + std::to_string(input.size()) +
                        " shape_elems=" + std::to_string(expected));
        return nullptr;
    }

    Tensor *input_tensor = runner->interpreter->getSessionInput(runner->session, nullptr);
    if (input_tensor == nullptr) {
        const auto &all_inputs = runner->interpreter->getSessionInputAll(runner->session);
        if (!all_inputs.empty()) {
            input_tensor = all_inputs.begin()->second;
        }
    }
    if (input_tensor == nullptr) {
        throw_illegal_state(env, "HifiGan session input tensor not found");
        return nullptr;
    }

    {
        std::string input_shape_desc;
        for (int i = 0; i < input_tensor->dimensions(); ++i) {
            if (!input_shape_desc.empty()) {
                input_shape_desc += "x";
            }
            input_shape_desc += std::to_string(input_tensor->length(i));
        }
        __android_log_print(
                ANDROID_LOG_INFO,
                kLogTag,
                "%s: input tensor dims=%d dimType=%d shape=%s",
                __func__,
                input_tensor->dimensions(),
                static_cast<int>(input_tensor->getDimensionType()),
                input_shape_desc.c_str());
    }

    runner->interpreter->resizeTensor(input_tensor, shape);
    runner->interpreter->resizeSession(runner->session);

    std::unique_ptr<Tensor> host_input(Tensor::create<float>(shape, input.data(), input_tensor->getDimensionType()));
    if (!input_tensor->copyFromHostTensor(host_input.get())) {
        throw_illegal_state(env, "Failed to copy HifiGan input tensor to device");
        return nullptr;
    }

    const auto run_code = runner->interpreter->runSession(runner->session);
    if (run_code != MNN::NO_ERROR) {
        throw_illegal_state(env, "HifiGan session run failed");
        return nullptr;
    }

    if (output_index == 3) {
        Tensor *magnitude_tensor = resolve_output_tensor(runner->interpreter.get(), runner->session, 0);
        Tensor *phase_tensor = resolve_output_tensor(runner->interpreter.get(), runner->session, 1);
        if (magnitude_tensor == nullptr || phase_tensor == nullptr) {
            throw_illegal_state(env, "HifiGan magnitude/phase output tensor not found");
            return nullptr;
        }

        std::vector<float> magnitude = tensor_to_vector(magnitude_tensor);
        std::vector<float> phase = tensor_to_vector(phase_tensor);
        if (magnitude.empty() || phase.empty()) {
            throw_illegal_state(env, "HifiGan magnitude/phase output tensor is empty");
            return nullptr;
        }
        if (magnitude.size() != phase.size()) {
            throw_illegal_state(
                    env,
                    "HifiGan magnitude/phase size mismatch: magnitude=" + std::to_string(magnitude.size()) +
                            " phase=" + std::to_string(phase.size()));
            return nullptr;
        }

        __android_log_print(
                ANDROID_LOG_INFO,
                kLogTag,
                "%s: magnitude stats=%s",
                __func__,
                float_vector_stats(magnitude).c_str());
        __android_log_print(
                ANDROID_LOG_INFO,
                kLogTag,
                "%s: phase stats=%s",
                __func__,
                float_vector_stats(phase).c_str());

        std::vector<float> merged;
        merged.reserve(magnitude.size() + phase.size());
        merged.insert(merged.end(), magnitude.begin(), magnitude.end());
        merged.insert(merged.end(), phase.begin(), phase.end());

        __android_log_print(
                ANDROID_LOG_INFO,
                kLogTag,
                "%s: merged magnitude/phase outputs size=%zu",
                __func__,
                merged.size());
        return vector_to_jfloat_array(env, merged);
    }

    Tensor *output_tensor = resolve_output_tensor(runner->interpreter.get(), runner->session, output_index);
    if (output_tensor == nullptr) {
        throw_illegal_state(env, "HifiGan output tensor not found");
        return nullptr;
    }

    std::vector<float> output = tensor_to_vector(output_tensor);
    if (output.empty()) {
        throw_illegal_state(env, "HifiGan output tensor is empty");
        return nullptr;
    }

    return vector_to_jfloat_array(env, output);
}
