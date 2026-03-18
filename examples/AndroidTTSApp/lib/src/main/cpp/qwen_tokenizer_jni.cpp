#include <jni.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers_cpp.h"

namespace {

struct TokenizerHolder {
    explicit TokenizerHolder(std::unique_ptr<tokenizers::Tokenizer> tokenizer)
        : tokenizer(std::move(tokenizer)) {}

    std::mutex mutex;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
};

std::mutex g_tokenizers_mutex;
std::unordered_map<jlong, std::shared_ptr<TokenizerHolder>> g_tokenizers;
std::atomic<jlong> g_next_handle{1};

void throw_java_exception(JNIEnv *env, const char *class_name, const std::string &message) {
    jclass exception_class = env->FindClass(class_name);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

std::string read_file_to_string(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open tokenizer file: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streampos size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to read tokenizer file size: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::string content(static_cast<size_t>(size), '\0');
    if (!content.empty()) {
        file.read(content.data(), static_cast<std::streamsize>(content.size()));
    }
    if (!file.good() && !file.eof()) {
        throw std::runtime_error("Failed to read tokenizer file: " + path);
    }
    return content;
}

std::shared_ptr<TokenizerHolder> find_tokenizer_holder(JNIEnv *env, jlong handle) {
    if (handle <= 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Invalid tokenizer handle");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tokenizers_mutex);
    const auto iter = g_tokenizers.find(handle);
    if (iter == g_tokenizers.end()) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Tokenizer handle not found");
        return nullptr;
    }
    return iter->second;
}

}  // namespace

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_llama_QwenTokenizerService_nativeLoad(
        JNIEnv *env,
        jobject /*unused*/,
        jstring tokenizerJsonPath) {
    if (tokenizerJsonPath == nullptr) {
        throw_java_exception(env, "java/lang/NullPointerException", "tokenizerJsonPath is null");
        return 0;
    }

    const char *path_chars = env->GetStringUTFChars(tokenizerJsonPath, nullptr);
    if (path_chars == nullptr) {
        throw_java_exception(env, "java/lang/RuntimeException", "Failed to read tokenizer path");
        return 0;
    }
    const std::string path(path_chars);
    env->ReleaseStringUTFChars(tokenizerJsonPath, path_chars);

    try {
        const std::string tokenizer_json = read_file_to_string(path);
        auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);
        if (!tokenizer) {
            throw std::runtime_error("Tokenizer::FromBlobJSON returned null");
        }

        const auto holder = std::make_shared<TokenizerHolder>(std::move(tokenizer));
        const jlong handle = g_next_handle.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(g_tokenizers_mutex);
            g_tokenizers[handle] = holder;
        }
        return handle;
    } catch (const std::exception &e) {
        throw_java_exception(env, "java/lang/IllegalStateException", e.what());
        return 0;
    }
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_llama_QwenTokenizerService_nativeEncode(
        JNIEnv *env,
        jobject /*unused*/,
        jlong handle,
        jstring text) {
    if (text == nullptr) {
        throw_java_exception(env, "java/lang/NullPointerException", "text is null");
        return nullptr;
    }

    auto holder = find_tokenizer_holder(env, handle);
    if (holder == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }

    const char *text_chars = env->GetStringUTFChars(text, nullptr);
    if (text_chars == nullptr) {
        throw_java_exception(env, "java/lang/RuntimeException", "Failed to read encode text");
        return nullptr;
    }

    std::vector<int32_t> token_ids;
    {
        std::lock_guard<std::mutex> lock(holder->mutex);
        token_ids = holder->tokenizer->Encode(text_chars);
    }
    env->ReleaseStringUTFChars(text, text_chars);

    const auto length = static_cast<jsize>(token_ids.size());
    jintArray result = env->NewIntArray(length);
    if (result == nullptr) {
        return nullptr;
    }
    if (!token_ids.empty()) {
        env->SetIntArrayRegion(
                result,
                0,
                length,
                reinterpret_cast<const jint *>(token_ids.data()));
    }
    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_llama_QwenTokenizerService_nativeDecode(
        JNIEnv *env,
        jobject /*unused*/,
        jlong handle,
        jintArray tokenIds) {
    if (tokenIds == nullptr) {
        throw_java_exception(env, "java/lang/NullPointerException", "tokenIds is null");
        return nullptr;
    }

    auto holder = find_tokenizer_holder(env, handle);
    if (holder == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }

    const jsize length = env->GetArrayLength(tokenIds);
    std::vector<jint> raw_ids(static_cast<size_t>(length));
    if (length > 0) {
        env->GetIntArrayRegion(tokenIds, 0, length, raw_ids.data());
    }

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(length));
    for (jint id: raw_ids) {
        ids.push_back(static_cast<int32_t>(id));
    }

    std::string decoded;
    {
        std::lock_guard<std::mutex> lock(holder->mutex);
        decoded = holder->tokenizer->Decode(ids);
    }
    return env->NewStringUTF(decoded.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_llama_QwenTokenizerService_nativeDestroy(
        JNIEnv *env,
        jobject /*unused*/,
        jlong handle) {
    if (handle <= 0) return;

    std::lock_guard<std::mutex> lock(g_tokenizers_mutex);
    const auto erased = g_tokenizers.erase(handle);
    if (erased == 0 && !env->ExceptionCheck()) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Tokenizer handle not found");
    }
}
