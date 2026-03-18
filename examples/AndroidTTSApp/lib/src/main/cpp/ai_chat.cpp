#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 896;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;
constexpr int   FLOW_CONTEXT_SIZE       = 2048;
constexpr int   FLOW_BATCH_SIZE         = 896;
constexpr int   FLOW_UBATCH_SIZE        = 896;
constexpr int   FLOW_BATCH_EMBD         = 512;
constexpr int   FLOW_SEQ_SETUP_SIZE     = 512;
constexpr int   HIFT_CONTEXT_SIZE       = 2048;
constexpr int   HIFT_BATCH_SIZE         = 2048;
constexpr int   HIFT_UBATCH_SIZE        = 2048;
constexpr int   HIFT_BATCH_EMBD         = 2048;
constexpr int   HIFT_SEQ_SETUP_SIZE     = 1024;
constexpr int   FLOW_RAND_NOISE_SIZE    = 80 * 50 * 300;
constexpr int   FLOW_EXTEND_PE_SIZE     = 9999 * 512;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
static ggml_backend_dev_t                 g_model_devices[2] = {nullptr, nullptr};
static llama_model                      * g_flow_model;
static llama_context                    * g_flow_context;
static llama_batch                        g_flow_batch;
static llama_model                      * g_hift_model;
static llama_context                    * g_hift_context;
static llama_batch                        g_hift_batch;
static inline void prepare_text_batch_for_decode(llama_batch &b);

// Compatibility shim for llama_batch_init() signature differences across forks.
template <typename Fn = decltype(&llama_batch_init)>
static llama_batch llama_batch_init_compat(int32_t n_tokens, int32_t embd, int32_t n_seq_max, int32_t is_flow = 0) {
    auto fn = static_cast<Fn>(&llama_batch_init);
    if constexpr (std::is_same_v<Fn, llama_batch (*)(int32_t, int32_t, int32_t, int32_t)>) {
        return fn(n_tokens, embd, n_seq_max, is_flow);
    } else {
        return fn(n_tokens, embd, n_seq_max);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == nullptr) {
        // Fallback #1: default search path
        LOGw("%s: CPU backend not found from nativeLibDir, trying default search paths", __func__);
        ggml_backend_load_all();
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == nullptr) {
        // Fallback #2: explicit sonames for Android arm64 variants
        const char *backend_candidates[] = {
                "libggml-cpu-android_armv9.2_2.so",
                "libggml-cpu-android_armv9.2_1.so",
                "libggml-cpu-android_armv9.0_1.so",
                "libggml-cpu-android_armv8.6_1.so",
                "libggml-cpu-android_armv8.2_2.so",
                "libggml-cpu-android_armv8.2_1.so",
                "libggml-cpu-android_armv8.0_1.so",
                "libggml-cpu.so",
        };
        for (const char *candidate : backend_candidates) {
            if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) != nullptr) {
                break;
            }
            LOGw("%s: trying backend candidate: %s", __func__, candidate);
            ggml_backend_load(candidate);
        }
    }
    const auto reg_count = ggml_backend_reg_count();
    const auto dev_count = ggml_backend_dev_count();
    LOGi("%s: backend registry count=%zu, device count=%zu", __func__, reg_count, dev_count);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated; Log handler set. CPU backend available=%s",
         ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) != nullptr ? "true" : "false");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();
    // Force CPU-only model loading to avoid unstable GPU/offload backends on mobile.
    model_params.n_gpu_layers = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;
    model_params.use_mmap = true;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    g_model_devices[0] = cpu_dev;
    g_model_devices[1] = nullptr;
    if (cpu_dev != nullptr) {
        model_params.devices = g_model_devices;
    } else {
        LOGw("%s: CPU backend device not found at load() time; fallback to default device list", __func__);
    }

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        LOGe("%s: Invalid model path", __func__);
        if (model_path != nullptr) {
            env->ReleaseStringUTFChars(jmodel_path, model_path);
        }
        return -1;
    }

    LOGd("%s: Loading model from: \n%s\n", __func__, model_path);

    {
        std::ifstream model_file(model_path, std::ios::binary);
        if (!model_file.good()) {
            LOGe("%s: Model file does not exist or cannot be opened: %s", __func__, model_path);
            env->ReleaseStringUTFChars(jmodel_path, model_path);
            return -2;
        }
    }

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        LOGe("%s: llama_model_load_from_file failed", __func__);
        return -3;
    }
    g_model = model;
    return 0;
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
//    ctx_params.use_mmap = false;
    // Required by decodeEmbeddingsNative -> llama_get_embeddings().
    ctx_params.embeddings = true;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    // Force CPU-only decode path; prevents context init from trying unavailable GPU backends.
    ctx_params.offload_kqv = false;
    ctx_params.op_offload = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static llama_context * init_context_for_model(
        llama_model * model,
        const int n_ctx,
        const int n_batch,
        const int n_ubatch) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    const int n_threads = std::max(
            N_THREADS_MIN,
            std::min(N_THREADS_MAX, (int) sysconf(_SC_NPROCESSORS_ONLN) - N_THREADS_HEADROOM));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.embeddings = true;
    ctx_params.offload_kqv = false;
    ctx_params.op_offload = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    auto * context = llama_init_from_model(model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_init_from_model failed", __func__);
    }
    return context;
}

static void release_flow_resources() {
    if (g_flow_batch.token != nullptr || g_flow_batch.embd != nullptr) {
        if (g_flow_batch.flow_token != nullptr) {
            free(g_flow_batch.flow_token);
            g_flow_batch.flow_token = nullptr;
        }
        if (g_flow_batch.flow_feat != nullptr) {
            free(g_flow_batch.flow_feat);
            g_flow_batch.flow_feat = nullptr;
        }
        if (g_flow_batch.rand_noise != nullptr) {
            free(g_flow_batch.rand_noise);
            g_flow_batch.rand_noise = nullptr;
        }
        if (g_flow_batch.extend_pe != nullptr) {
            free(g_flow_batch.extend_pe);
            g_flow_batch.extend_pe = nullptr;
        }
        llama_batch_free(g_flow_batch);
        g_flow_batch = {};
    }
    if (g_flow_context != nullptr) {
        llama_free(g_flow_context);
        g_flow_context = nullptr;
    }
    if (g_flow_model != nullptr) {
        llama_model_free(g_flow_model);
        g_flow_model = nullptr;
    }
}

static void release_hift_resources() {
    if (g_hift_batch.token != nullptr || g_hift_batch.embd != nullptr) {
        llama_batch_free(g_hift_batch);
        g_hift_batch = {};
    }
    if (g_hift_context != nullptr) {
        llama_free(g_hift_context);
        g_hift_context = nullptr;
    }
    if (g_hift_model != nullptr) {
        llama_model_free(g_hift_model);
        g_hift_model = nullptr;
    }
}

static common_sampler *new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    auto *context = init_context(g_model);
    if (!context) { return 1; }
    g_context = context;
    g_batch = llama_batch_init_compat(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);
    return 0;
}

static int load_aux_model(
        const char * model_path,
        const bool is_flow,
        llama_model ** out_model,
        llama_context ** out_context,
        llama_batch * out_batch,
        const int n_ctx,
        const int n_batch,
        const int n_ubatch,
        const int batch_embd,
        const int batch_is_flow_flag) {
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        return -1;
    }

    if (*out_model != nullptr && *out_context != nullptr) {
        return 0;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;
    model_params.use_mmap = true;
    model_params.is_flow = is_flow;
    model_params.is_hift = !is_flow;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev != nullptr) {
        g_model_devices[0] = cpu_dev;
        g_model_devices[1] = nullptr;
        model_params.devices = g_model_devices;
    }

    {
        std::ifstream model_file(model_path, std::ios::binary);
        if (!model_file.good()) {
            return -2;
        }
    }

    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        return -3;
    }

    auto * context = init_context_for_model(model, n_ctx, n_batch, n_ubatch);
    if (context == nullptr) {
        llama_model_free(model);
        return -4;
    }

    *out_model = model;
    *out_context = context;
    *out_batch = llama_batch_init_compat(n_batch, batch_embd, 1, batch_is_flow_flag);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadFlow(
        JNIEnv * env,
        jobject /*unused*/,
        jstring jmodel_path) {
    const auto * model_path = env->GetStringUTFChars(jmodel_path, 0);
    if (model_path == nullptr) {
        return -1;
    }

    const int result = load_aux_model(
            model_path,
            true,
            &g_flow_model,
            &g_flow_context,
            &g_flow_batch,
            FLOW_CONTEXT_SIZE,
            FLOW_BATCH_SIZE,
            FLOW_UBATCH_SIZE,
            FLOW_BATCH_EMBD,
            1);

    env->ReleaseStringUTFChars(jmodel_path, model_path);
    return result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadHift(
        JNIEnv * env,
        jobject /*unused*/,
        jstring jmodel_path) {
    const auto * model_path = env->GetStringUTFChars(jmodel_path, 0);
    if (model_path == nullptr) {
        return -1;
    }

    const int result = load_aux_model(
            model_path,
            false,
            &g_hift_model,
            &g_hift_context,
            &g_hift_batch,
            HIFT_CONTEXT_SIZE,
            HIFT_BATCH_SIZE,
            HIFT_UBATCH_SIZE,
            HIFT_BATCH_EMBD,
            0);

    env->ReleaseStringUTFChars(jmodel_path, model_path);
    return result;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        prepare_text_batch_for_decode(g_batch);
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }
            prepare_text_batch_for_decode(g_batch);

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * TODO-hyin: implement sliding-window version as a better alternative
 *
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static inline void prepare_text_batch_for_decode(llama_batch &b) {
    b.flow_token = nullptr;
    b.flow_feat = nullptr;
    b.rand_noise = nullptr;
    b.extend_pe = nullptr;
    b.token_len = static_cast<uint32_t>(std::max(0, b.n_tokens));
    b.prompt_token_len = 0;
    b.prompt_feat_len = 0;
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // NOTE:
    // This fork's llama-batch splitter reads token_len/prompt_token_len/prompt_feat_len
    // in the token decode path. Populate those fields before each llama_decode call.

    // Process tokens in chunks with explicit pos/seq assignment.
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        LOGv("%s: Preparing a chunk size of %d starting at: %d", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        common_batch_clear(batch);
        for (int j = 0; j < cur_batch_size; j++) {
            const bool logits = compute_last_logit && (j == cur_batch_size - 1);
            common_batch_add(batch, tokens[i + j], start_pos + i + j, {0}, logits);
        }
        prepare_text_batch_for_decode(batch);

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt_chars = env->GetStringUTFChars(jsystem_prompt, nullptr);
    if (system_prompt_chars == nullptr) {
        LOGe("%s: failed to read system prompt chars", __func__);
        return 1;
    }
    std::string system_prompt(system_prompt_chars);
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt_chars);

    LOGd("%s: System prompt received: \n%s", __func__, system_prompt.c_str());
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt_chars = env->GetStringUTFChars(juser_prompt, nullptr);
    if (user_prompt_chars == nullptr) {
        LOGe("%s: failed to read user prompt chars", __func__);
        return 1;
    }
    std::string user_prompt(user_prompt_chars);
    env->ReleaseStringUTFChars(juser_prompt, user_prompt_chars);

    LOGd("%s: User prompt received: \n%s", __func__, user_prompt.c_str());
    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }

    // Decode formatted user prompts
    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += (int) user_tokens.size();
    stop_generation_position = current_position + n_predict;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPromptTokens(
        JNIEnv *env,
        jobject /*unused*/,
        jintArray jtoken_ids,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    if (jtoken_ids == nullptr) {
        LOGe("%s: null token ids", __func__);
        return 1;
    }

    const jsize token_count = env->GetArrayLength(jtoken_ids);
    std::vector<jint> raw_tokens(static_cast<size_t>(token_count));
    if (token_count > 0) {
        env->GetIntArrayRegion(jtoken_ids, 0, token_count, raw_tokens.data());
    }

    std::vector<llama_token> user_tokens;
    user_tokens.reserve(static_cast<size_t>(token_count));
    for (jint id: raw_tokens) {
        user_tokens.push_back(static_cast<llama_token>(id));
    }
    if (user_tokens.empty()) {
        LOGw("%s: empty token ids", __func__);
        return 1;
    }

    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += (int) user_tokens.size();
    stop_generation_position = current_position + n_predict;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_resetKvCacheNative(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    if (g_context == nullptr) {
        LOGe("%s: context is null", __func__);
        return 1;
    }

    reset_long_term_states(true);
    reset_short_term_states();
    if (g_sampler != nullptr) {
        common_sampler_reset(g_sampler);
    }
    return 0;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_decodeEmbeddingsNative(
        JNIEnv *env,
        jobject /*unused*/,
        jfloatArray jinput_embeddings,
        jint n_past
) {
    if (g_context == nullptr || g_model == nullptr) {
        LOGe("%s: context/model is null", __func__);
        return env->NewFloatArray(0);
    }
    if (jinput_embeddings == nullptr) {
        LOGe("%s: input embeddings is null", __func__);
        return env->NewFloatArray(0);
    }

    const jsize n_values = env->GetArrayLength(jinput_embeddings);
    if (n_values <= 0) {
        return env->NewFloatArray(0);
    }

    const int n_embd = llama_model_n_embd(g_model);
    if (n_embd <= 0) {
        LOGe("%s: invalid n_embd=%d", __func__, n_embd);
        return env->NewFloatArray(0);
    }
    if (n_values % n_embd != 0) {
        LOGe("%s: embedding length %d is not divisible by hidden size %d",
             __func__, (int) n_values, n_embd);
        return env->NewFloatArray(0);
    }

    std::vector<float> input_embeddings(static_cast<size_t>(n_values));
    env->GetFloatArrayRegion(jinput_embeddings, 0, n_values, input_embeddings.data());

    const int n_tokens_total = (int) n_values / n_embd;
    std::vector<float> all_outputs;
    all_outputs.reserve(input_embeddings.size());

    llama_batch embd_batch = llama_batch_init_compat(BATCH_SIZE, n_embd, 1);
    int token_cursor = 0;

    while (token_cursor < n_tokens_total) {
        const int cur_batch_size = std::min(BATCH_SIZE, n_tokens_total - token_cursor);
        common_batch_clear(embd_batch);

        // Fill embedding batch with explicit positions and sequence IDs.
        for (int j = 0; j < cur_batch_size; j++) {
            const int token_index = token_cursor + j;
            std::memcpy(
                    embd_batch.embd + (int64_t) j * n_embd,
                    input_embeddings.data() + (int64_t) token_index * n_embd,
                    sizeof(float) * n_embd
            );
            embd_batch.pos[j] = n_past + token_index;
            embd_batch.n_seq_id[j] = 1;
            embd_batch.seq_id[j][0] = 0;
            embd_batch.logits[j] = 1;
            embd_batch.n_tokens++;
        }
        prepare_text_batch_for_decode(embd_batch);

        const int decode_result = llama_decode(g_context, embd_batch);
        if (decode_result != 0) {
            LOGe("%s: llama_decode failed with %d", __func__, decode_result);
            llama_batch_free(embd_batch);
            return env->NewFloatArray(0);
        }

        float *embeddings = llama_get_embeddings(g_context);
        if (embeddings == nullptr) {
            LOGe("%s: llama_get_embeddings returned null", __func__);
            llama_batch_free(embd_batch);
            return env->NewFloatArray(0);
        }

        const size_t out_chunk = (size_t) cur_batch_size * (size_t) n_embd;
        all_outputs.insert(all_outputs.end(), embeddings, embeddings + out_chunk);
        token_cursor += cur_batch_size;
    }

    llama_batch_free(embd_batch);
    jfloatArray joutput = env->NewFloatArray((jsize) all_outputs.size());
    if (joutput == nullptr) {
        return env->NewFloatArray(0);
    }
    if (!all_outputs.empty()) {
        env->SetFloatArrayRegion(joutput, 0, (jsize) all_outputs.size(), all_outputs.data());
    }
    return joutput;
}

static std::vector<float> take_last_in_time_dim(
        const std::vector<float> & data,
        const int channels,
        const int timesteps,
        const int keep_last) {
    if (channels <= 0 || timesteps <= 0 || keep_last <= 0) {
        return {};
    }
    const int safe_keep = std::min(keep_last, timesteps);
    const int drop = timesteps - safe_keep;
    std::vector<float> out;
    out.reserve((size_t) channels * (size_t) safe_keep);
    for (int c = 0; c < channels; ++c) {
        const int row_start = c * timesteps;
        const int copy_start = row_start + drop;
        const int copy_end = row_start + timesteps;
        out.insert(out.end(), data.begin() + copy_start, data.begin() + copy_end);
    }
    return out;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_encodeFlowNative(
        JNIEnv * env,
        jobject /*unused*/,
        jfloatArray jinput_embeddings,
        jfloatArray jflow_feat,
        jintArray jflow_token,
        jint token_len,
        jint prompt_token_len,
        jint prompt_feat_len,
        jfloatArray jrand_noise,
        jfloatArray jextend_pe) {
    if (g_flow_context == nullptr || g_flow_model == nullptr) {
        LOGe("%s: flow context/model is null", __func__);
        return env->NewFloatArray(0);
    }
    if (jinput_embeddings == nullptr || jflow_feat == nullptr || jflow_token == nullptr ||
        jrand_noise == nullptr || jextend_pe == nullptr) {
        LOGe("%s: null input", __func__);
        return env->NewFloatArray(0);
    }
    if (token_len < 0 || prompt_token_len < 0 || prompt_feat_len < 0) {
        LOGe("%s: invalid flow lengths", __func__);
        return env->NewFloatArray(0);
    }

    const jsize input_count = env->GetArrayLength(jinput_embeddings);
    const jsize flow_feat_count = env->GetArrayLength(jflow_feat);
    const jsize flow_token_count = env->GetArrayLength(jflow_token);
    const jsize rand_noise_count = env->GetArrayLength(jrand_noise);
    const jsize extend_pe_count = env->GetArrayLength(jextend_pe);
    if (input_count <= 0 || flow_feat_count <= 0 || flow_token_count <= 0) {
        return env->NewFloatArray(0);
    }
    if (rand_noise_count < FLOW_RAND_NOISE_SIZE || extend_pe_count < FLOW_EXTEND_PE_SIZE) {
        LOGe("%s: noise/pe sizes too small: rand=%d extend=%d", __func__,
             (int) rand_noise_count, (int) extend_pe_count);
        return env->NewFloatArray(0);
    }

    std::vector<float> input_embeddings((size_t) input_count);
    std::vector<float> flow_feat((size_t) flow_feat_count);
    std::vector<jint> flow_token_j((size_t) flow_token_count);
    std::vector<float> rand_noise((size_t) rand_noise_count);
    std::vector<float> extend_pe((size_t) extend_pe_count);

    env->GetFloatArrayRegion(jinput_embeddings, 0, input_count, input_embeddings.data());
    env->GetFloatArrayRegion(jflow_feat, 0, flow_feat_count, flow_feat.data());
    env->GetIntArrayRegion(jflow_token, 0, flow_token_count, flow_token_j.data());
    env->GetFloatArrayRegion(jrand_noise, 0, rand_noise_count, rand_noise.data());
    env->GetFloatArrayRegion(jextend_pe, 0, extend_pe_count, extend_pe.data());

    g_flow_batch.n_tokens = input_count;
    g_flow_batch.token_len = static_cast<uint32_t>(token_len);
    g_flow_batch.prompt_token_len = static_cast<uint32_t>(prompt_token_len);
    g_flow_batch.prompt_feat_len = static_cast<uint32_t>(prompt_feat_len * 80);
    std::memcpy(g_flow_batch.embd, input_embeddings.data(), sizeof(float) * (size_t) input_count);

    // Match Swift behavior: configure first 512 entries.
    const int seq_setup = std::min(FLOW_SEQ_SETUP_SIZE, FLOW_BATCH_SIZE);
    for (int i = 0; i < seq_setup; ++i) {
        g_flow_batch.n_seq_id[i] = 1;
        g_flow_batch.pos[i] = i;
        g_flow_batch.seq_id[i][0] = 0;
    }

    const int64_t total_tokens = (int64_t) token_len + (int64_t) prompt_token_len;
    if (total_tokens <= 0 || total_tokens > flow_token_count) {
        LOGe("%s: invalid total token count %lld for flow_token_count=%d",
             __func__, (long long) total_tokens, (int) flow_token_count);
        return env->NewFloatArray(0);
    }
    std::vector<llama_token> merged_tokens((size_t) total_tokens);
    for (int64_t i = 0; i < total_tokens; ++i) {
        merged_tokens[(size_t) i] = static_cast<llama_token>(flow_token_j[(size_t) i]);
    }
    std::memcpy(g_flow_batch.flow_token, merged_tokens.data(), sizeof(llama_token) * (size_t) total_tokens);

    const int64_t feat_required = (int64_t) g_flow_batch.prompt_feat_len;
    if (feat_required <= 0 || feat_required > flow_feat_count) {
        LOGe("%s: invalid flow feat required=%lld available=%d",
             __func__, (long long) feat_required, (int) flow_feat_count);
        return env->NewFloatArray(0);
    }
    std::memcpy(g_flow_batch.flow_feat, flow_feat.data(), sizeof(float) * (size_t) feat_required);
    std::memcpy(g_flow_batch.rand_noise, rand_noise.data(), sizeof(float) * FLOW_RAND_NOISE_SIZE);
    std::memcpy(g_flow_batch.extend_pe, extend_pe.data(), sizeof(float) * FLOW_EXTEND_PE_SIZE);
    llama_memory_clear(llama_get_memory(g_flow_context), true);

    if (g_flow_batch.logits != nullptr) {
        const int64_t flow_total_len = 80LL * ((int64_t) (prompt_token_len + token_len) * 2LL - (int64_t) prompt_feat_len);
        const int safe_flow_len = (int) std::max<int64_t>(1LL, flow_total_len);
        std::vector<int8_t> logits_buf((size_t) safe_flow_len, 1);
        int8_t * original_logits = g_flow_batch.logits;
        g_flow_batch.logits = logits_buf.data();
        const int encode_res = llama_encode(g_flow_context, g_flow_batch, 0);
        g_flow_batch.logits = original_logits;
        if (encode_res < 0) {
            LOGe("%s: llama_encode(flow) failed: %d", __func__, encode_res);
            return env->NewFloatArray(0);
        }
    } else {
        const int encode_res = llama_encode(g_flow_context, g_flow_batch, 0);
        if (encode_res < 0) {
            LOGe("%s: llama_encode(flow) failed: %d", __func__, encode_res);
            return env->NewFloatArray(0);
        }
    }

    float * embeddings = llama_get_embeddings(g_flow_context);
    if (embeddings == nullptr) {
        LOGe("%s: llama_get_embeddings(flow) returned null", __func__);
        return env->NewFloatArray(0);
    }

    const int64_t doubled_tokens = (int64_t) (prompt_token_len + token_len) * 2LL;
    const int64_t out_dim = doubled_tokens * 80LL;
    if (out_dim <= 0 || out_dim > INT32_MAX) {
        LOGe("%s: invalid flow out dim: %lld", __func__, (long long) out_dim);
        return env->NewFloatArray(0);
    }

    std::vector<float> flow_raw((size_t) out_dim);
    std::memcpy(flow_raw.data(), embeddings, sizeof(float) * (size_t) out_dim);

    const int keep_last = (int) std::max<int64_t>(0, doubled_tokens - prompt_feat_len);
    std::vector<float> flow_out = take_last_in_time_dim(
            flow_raw,
            /*channels=*/80,
            /*timesteps=*/(int) doubled_tokens,
            /*keep_last=*/keep_last);

    jfloatArray joutput = env->NewFloatArray((jsize) flow_out.size());
    if (joutput == nullptr) {
        return env->NewFloatArray(0);
    }
    if (!flow_out.empty()) {
        env->SetFloatArrayRegion(joutput, 0, (jsize) flow_out.size(), flow_out.data());
    }
    return joutput;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_encodeHiftNative(
        JNIEnv * env,
        jobject /*unused*/,
        jfloatArray jinput_embeddings) {
    if (g_hift_context == nullptr || g_hift_model == nullptr) {
        LOGe("%s: hift context/model is null", __func__);
        return env->NewFloatArray(0);
    }
    if (jinput_embeddings == nullptr) {
        LOGe("%s: null input", __func__);
        return env->NewFloatArray(0);
    }

    const jsize input_count = env->GetArrayLength(jinput_embeddings);
    if (input_count <= 0) {
        return env->NewFloatArray(0);
    }
    if (input_count % 80 != 0) {
        LOGe("%s: hift input count %d is not divisible by 80", __func__, (int) input_count);
        return env->NewFloatArray(0);
    }

    std::vector<float> input_embeddings((size_t) input_count);
    env->GetFloatArrayRegion(jinput_embeddings, 0, input_count, input_embeddings.data());

    const int seq_len = (int) input_count / 80;
    g_hift_batch.n_tokens = seq_len;
    std::memcpy(g_hift_batch.embd, input_embeddings.data(), sizeof(float) * (size_t) input_count);
    llama_memory_clear(llama_get_memory(g_hift_context), true);

    const int seq_setup = std::min(HIFT_SEQ_SETUP_SIZE, HIFT_BATCH_SIZE);
    for (int i = 0; i < seq_setup; ++i) {
        g_hift_batch.n_seq_id[i] = 1;
        g_hift_batch.pos[i] = i;
        g_hift_batch.seq_id[i][0] = 0;
    }
    if (g_hift_batch.logits != nullptr && HIFT_BATCH_SIZE > 0) {
        g_hift_batch.logits[0] = 1;
    }

    const int encode_res = llama_encode(g_hift_context, g_hift_batch, 0);
    if (encode_res < 0) {
        LOGe("%s: llama_encode(hift) failed: %d", __func__, encode_res);
        return env->NewFloatArray(0);
    }

    float * embeddings = llama_get_embeddings(g_hift_context);
    if (embeddings == nullptr) {
        LOGe("%s: llama_get_embeddings(hift) returned null", __func__);
        return env->NewFloatArray(0);
    }

    const double scaled = (double) input_count * 6.0 / 4.0;
    const int final_count = (int) (scaled + 1.0);
    const int64_t out_size = (int64_t) final_count * 18LL;
    if (out_size <= 0 || out_size > INT32_MAX) {
        LOGe("%s: invalid hift out size: %lld", __func__, (long long) out_size);
        return env->NewFloatArray(0);
    }

    jfloatArray joutput = env->NewFloatArray((jsize) out_size);
    if (joutput == nullptr) {
        return env->NewFloatArray(0);
    }
    env->SetFloatArrayRegion(joutput, 0, (jsize) out_size, embeddings);
    return joutput;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unloadFlow(
        JNIEnv * /*unused*/,
        jobject /*unused*/) {
    release_flow_resources();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unloadHift(
        JNIEnv * /*unused*/,
        jobject /*unused*/) {
    release_hift_resources();
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Infinite text generation via context shifting
    if (current_position >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    prepare_text_batch_for_decode(g_batch);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    common_sampler_free(g_sampler);
    g_chat_templates.reset();
    llama_batch_free(g_batch);
    llama_free(g_context);
    llama_model_free(g_model);
    g_batch = {};
    g_context = nullptr;
    g_model = nullptr;

    release_flow_resources();
    release_hift_resources();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    release_flow_resources();
    release_hift_resources();
    llama_backend_free();
}
