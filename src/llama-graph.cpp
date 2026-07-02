#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "llama-model.h"
#include <random>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        if(ubatch->flow_token != nullptr) {
            const int64_t n_embd   = embd->ne[0];
            ggml_backend_tensor_set(embd, ubatch->embd, 0, n_embd*ggml_element_size(embd));
        } else{
            const int64_t n_embd   = embd->ne[0] * embd->ne[1];
            ggml_backend_tensor_set(embd, ubatch->embd, 0, n_embd*ggml_element_size(embd));
            // float buff[10];
            // ggml_backend_tensor_get(embd, buff, 0, sizeof(buff));   // 直接拷 32 字节
            // std::string token_str = "";
            // for (int i = 0; i < 10; ++i)  {
            //     token_str += std::to_string(buff[i]);
            //     token_str += " ";
                
            // }
            // printf("input_prompt_token is: %s\n ", token_str.c_str());
        }
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!tokens && !params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!embd   && !params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_token::set_input(const llama_ubatch * ubatch) {
    if(ubatch->flow_token) {
        const int64_t total_token_len = ubatch->token_len + ubatch->prompt_token_len;

        ggml_backend_tensor_set(input_token, ubatch->flow_token, 0, total_token_len*ggml_element_size(input_token));
        // int32_t buff[total_token_len];
        // ggml_backend_tensor_get(input_token, buff, 0, sizeof(buff));   // 直接拷 32 字节
        // std::string token_str = "";
        // for (int i = 0; i < total_token_len; ++i)  {
        //     token_str += std::to_string(buff[i]);
        //     token_str += " ";
            
        // }
        // printf("input_prompt_token is: %s\n ", token_str.c_str());
    }
}

void llm_graph_input_prompt_token::set_input(const llama_ubatch * ubatch) {
    if(ubatch->flow_token) {
        const int64_t total_token_len = ubatch->prompt_token_len;

        ggml_backend_tensor_set(input_prompt_token, ubatch->flow_token, ubatch->token_len * ggml_element_size(input_prompt_token), total_token_len*ggml_element_size(input_prompt_token));
        
    }
}

void llm_graph_input_prompt_feat::set_input(const llama_ubatch * ubatch) {
    if(ubatch->flow_feat) {
        const int64_t feat_len = ubatch->prompt_feat_len;

        ggml_backend_tensor_set(input_prompt_feat, ubatch->flow_feat, 0, feat_len*ggml_element_size(input_prompt_feat));
    }
}

static bool flow_load_rand_noise_bin(const char * path, std::vector<float> & result) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    constexpr int64_t rand_noise_len = 80 * 50 * 300;
    constexpr size_t expected_size = rand_noise_len * sizeof(float);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size != (std::streamsize) expected_size) {
        LLAMA_LOG_INFO("flow rand_noise bin ignored: %s has %zu bytes, expected %zu\n",
                path, (size_t) size, expected_size);
        return false;
    }

    file.seekg(0, std::ios::beg);
    result.resize(rand_noise_len);
    if (!file.read(reinterpret_cast<char *>(result.data()), expected_size)) {
        result.clear();
        return false;
    }

    LLAMA_LOG_INFO("flow rand_noise loaded from bin: %s\n", path);
    return true;
}

static const std::vector<float> & flow_cached_rand_noise() {
    static const std::vector<float> data = []() {
        constexpr int64_t rand_noise_len = 80 * 50 * 300;
        constexpr float two_pi = 6.28318530717958647692f;

        std::vector<float> result(rand_noise_len);
        if (flow_load_rand_noise_bin(std::getenv("LLAMA_FLOW_RAND_NOISE_BIN"), result) ||
                flow_load_rand_noise_bin("./rand_noise.bin", result)) {
            return result;
        }

        LLAMA_LOG_INFO("flow rand_noise bin not found; using built-in deterministic noise\n");
        uint32_t state = 0x12345678u;

        auto uniform01 = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return ((state >> 8) + 0.5f) * (1.0f / 16777216.0f);
        };

        for (int64_t i = 0; i < rand_noise_len; i += 2) {
            const float u1 = std::max(uniform01(), 1.0e-7f);
            const float u2 = uniform01();
            const float r = std::sqrt(-2.0f * std::log(u1));
            const float theta = two_pi * u2;

            result[i] = r * std::cos(theta);
            if (i + 1 < rand_noise_len) {
                result[i + 1] = r * std::sin(theta);
            }
        }

        return result;
    }();

    return data;
}

static const std::vector<float> & flow_cached_extend_pe() {
    static const std::vector<float> data = []() {
        constexpr int64_t n_ctx  = 5000;
        constexpr int64_t n_embd = 512;
        constexpr int64_t n_pos  = 2 * n_ctx - 1;

        std::vector<float> result(n_pos * n_embd);
        std::vector<float> div_terms(n_embd / 2);

        const float scale = -std::log(10000.0f) / float(n_embd);
        for (int64_t i = 0; i < n_embd / 2; ++i) {
            div_terms[i] = std::exp(float(2 * i) * scale);
        }

        for (int64_t row = 0; row < n_pos; ++row) {
            const bool negative = row >= n_ctx;
            const int64_t pos = negative ? (row - n_ctx + 1) : (n_ctx - 1 - row);
            const float sign = negative ? -1.0f : 1.0f;

            for (int64_t i = 0; i < n_embd / 2; ++i) {
                const float v = float(pos) * div_terms[i];
                result[row * n_embd + 2 * i + 0] = sign * std::sin(v);
                result[row * n_embd + 2 * i + 1] = sign * std::cos(v);
            }
        }

        return result;
    }();

    return data;
}

void llm_graph_input_rand_noise::set_input(const llama_ubatch * ubatch) {
    const int64_t rand_noise_len = 80 * 50 * 300;
    const float * rand_noise = ubatch->rand_noise ? ubatch->rand_noise : flow_cached_rand_noise().data();

    ggml_backend_tensor_set(input_rand_noise, rand_noise, 0, rand_noise_len*ggml_element_size(input_rand_noise));
}

void llm_graph_input_extend_pe::set_input(const llama_ubatch * ubatch) {
    const int64_t extend_pe_len = 9999 * 512;
    const float * extend_pe = ubatch->extend_pe ? ubatch->extend_pe : flow_cached_extend_pe().data();

    ggml_backend_tensor_set(input_extend_pe, extend_pe, 0, extend_pe_len*ggml_element_size(input_extend_pe));
}

void llm_graph_input_stream::set_input(const llama_ubatch * ubatch) {
    
    input_stream = ubatch->stream;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
                }
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && arch == LLM_ARCH_QWEN3) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
        for (int h = 0; h < 1; ++h) {
            for (int i1 = 0; i1 < n_tokens; ++i1) {
                const llama_seq_id s1 = ubatch->seq_id[i1][0];
                const llama_pos    p1 = ubatch->pos[i1];

                const uint64_t idst = h*(n_kv*n_tokens) + i1*n_kv;

                for (int i0 = 0; i0 < n_tokens; ++i0) {
                    const llama_seq_id s0 = ubatch->seq_id[i0][0];
                    const llama_pos p0    = ubatch->pos[i0];

                    // mask different sequences
                    if (s0 != s1) {
                        continue;
                    }

                    // mask future tokens
                    if (cparams.causal_attn && p0 > p1) {
                        continue;
                    }

                    // apply SWA if any
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        continue;
                    }

                    data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
                }
            }
        }
    };

    {
        GGML_ASSERT(self_kq_mask);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

        float * data = (float *) self_kq_mask->data;

        std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);

        fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);

    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));

        float * data = (float *) self_kq_mask_swa->data;

        std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);

        fill_mask(data, hparams.n_swa, hparams.swa_type);

    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_n_kv();
    res &= self_kq_mask->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_base()->get_n_kv();
    res &= self_kq_mask->ne[1] == params.ubatch.n_tokens;

    res &= self_kq_mask_swa->ne[0] == mctx->get_swa()->get_n_kv();
    res &= self_kq_mask_swa->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    float * data = (float *) cross_kq_mask->data;

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[h*(n_enc*n_tokens) + i*n_enc + j] = f;
            }
        }

        for (int i = n_tokens; i < n_tokens; ++i) {
            for (int j = 0; j < n_enc; ++j) {
                data[h*(n_enc*n_tokens) + i*n_enc + j] = -INFINITY;
            }
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= inp_attn->self_kq_mask->ne[0] == mctx->get_attn()->get_n_kv();
    res &= inp_attn->self_kq_mask->ne[1] == params.ubatch.n_tokens;

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_tokens      = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;

    params = llm_graph_params();

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        return false;
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        res = res && cur;
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_layer_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
         float eps,
         std::string blk_type,
         int32_t il) const{
    
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, mw);
    cur = ggml_add(ctx0, cur, mb);
    ggml_set_name(cur, ("layer_norm_" + blk_type + "_" + std::to_string(il)).c_str());
    // cb(cur, ("layer_norm_" + std::to_string(il)).c_str(), -1);
    return cur;

}

ggml_tensor * llm_graph_context::build_F_normalize(
         ggml_tensor * x,
         float        eps) const {
    ggml_tensor * l2_norm = ggml_l2_norm(ctx0, x, eps);
    cb(l2_norm, "F_norm", -1);
    ggml_set_name(l2_norm, "F_norm");
    return l2_norm;
}

ggml_tensor * llm_graph_context::build_flow_embedding(
         ggml_tensor * cur,
         ggml_tensor * embd_w,
                 int   il) const {
    cur = ggml_get_rows(ctx0, embd_w, cur);
    cb(cur, "flow_embd", il);
    return cur;
}


ggml_tensor * llm_graph_context::build_pad_mask(int32_t total_len, int32_t max_len, int32_t il) const {
    
    if(max_len <= 0) {
        max_len = total_len;
    }
    ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, max_len, 1);
    ggml_tensor * zero = ggml_scale(ctx0, mask, 0.0f);
    ggml_tensor * one = ggml_exp(ctx0, zero);
    mask = ggml_sub(ctx0, one, zero);
    ggml_set_name(mask, ("non_pad_mask_" + std::to_string(il)).c_str());
    cb(mask, ("non_pad_mask_" + std::to_string(il)).c_str(), il);
    
    return mask;
}


ggml_tensor * llm_graph_context::build_linear_no_subsampling(
         ggml_tensor * cur,
         ggml_tensor * linear_mw,
         ggml_tensor * linear_mb,
         ggml_tensor * norm_mw,
         ggml_tensor * norm_mb,
         int32_t il) const{
    
    // cur = ggml_mul_mat(ctx0, linear_mw, cur);
    // cur = ggml_add(ctx0, cur, linear_mb);
    cur = ggml_mul_mat_add(ctx0, linear_mw, cur, linear_mb);
    ggml_set_name(cur, ("1st_mul_mat" + std::to_string(il)).c_str());
    cur = build_layer_norm(cur, norm_mw, norm_mb, 1e-05, "sub_sampling", il);
    ggml_set_name(cur, ("embed_linear_no_sub_sample_" + std::to_string(il)).c_str());

    return cur;

}



ggml_tensor * llm_graph_context::build_pos_encoding(
         ggml_tensor * cur,
         size_t offset, 
         size_t size, 
         size_t il) const{
    
    ggml_tensor * new_cur = cur;
    int64_t cols = new_cur->ne[0];           
    int64_t rows = new_cur->ne[1];               
    int64_t start = std::max(int64_t(0), rows / 2 - static_cast<int64_t>(size) - static_cast<int64_t>(offset) + 1);
    int64_t end = std::min(rows, rows / 2 + static_cast<int64_t>(size) + static_cast<int64_t>(offset));
    int64_t new_len = end - start;
    GGML_ASSERT(new_len > 0);

    ggml_tensor * pos_emb = ggml_view_2d(ctx0, new_cur,
                                     cols, new_len,
                                     new_cur->nb[1],
                                     start * new_cur->nb[1]);
    ggml_set_name(pos_emb, ("pos_emb_" + std::to_string(il)).c_str());
    return pos_emb;
}

ggml_tensor * llm_graph_context::build_espnet_pos_encode(
         ggml_tensor * cur,
         int32_t il) const{
        const size_t d_model = 512;
        float xscale = 22.627416997969522f;
        cur = ggml_scale(ctx0, cur, xscale);
        ggml_set_name(cur, ("espnet_pos_encode" + std::to_string(il)).c_str());
        // cb(cur, "espnet_pos_encode", il);
        return cur;
}

ggml_tensor * llm_graph_context::build_flash_attn_encoder(
    ggml_tensor * x,           // [D, T, B]
    const llama_model & model,
    ggml_tensor * mask,   // 可为 nullptr
    ggml_tensor * pos_emb,
    int32_t n_heads, 
    int32_t idx) const {
    
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    const int64_t D = x->ne[0];
    const int64_t d_k = D / n_heads;
    // ---- QKV
    ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[idx + 13].encoders_wq, x);
    ggml_tensor * k = ggml_mul_mat(ctx0, model.layers[idx + 13].encoders_wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx0, model.layers[idx + 13].encoders_wv, x);
    ggml_tensor * p = ggml_mul_mat(ctx0, model.layers[idx + 13].encoders_wpos, pos_emb);

    // q/k/v follow the query length T, while p keeps its own relative-position length.
    auto reshape_qkv = [&](ggml_tensor * t) {
        t = ggml_reshape_4d(ctx0, t, d_k, n_heads, T, B);
        t = ggml_permute(ctx0, t, 0, 2, 1, 3);
        return ggml_cont(ctx0, t);
    };

    q = reshape_qkv(q);
    k = reshape_qkv(k);
    v = reshape_qkv(v);
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    p = ggml_cont(ctx0, p);
    p = ggml_reshape_4d(ctx0, p, d_k, n_heads, p->ne[1], p->ne[2]);
    p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));
    ggml_tensor * q_with_bias_u = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, q, model.layers[idx + 13].encoders_pos_bias_u), 0, 2, 1, 3));
    ggml_tensor * q_with_bias_v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, q, model.layers[idx + 13].encoders_pos_bias_v), 0, 2, 1, 3));
    ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, k, q_with_bias_u);
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_with_bias_v);

    if(!ggml_are_same_shape(matrix_ac, matrix_bd)) {
        matrix_bd = build_rel_shift(matrix_bd);
    }

    ggml_tensor * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
    scores = ggml_scale(ctx0, scores, 0.125f);

    // ---- Flash Attention
    // ggml_tensor * attn_out = ggml_flash_attn_ext(
    //     ctx0,
    //     q, k, v,
    //     mask,
    //     1.0f / sqrtf((float)d_k),
    //     0.0f,
    //     0.0f
    // );
    ggml_tensor * attn_out = build_attn_scores(
        v,
        scores,
        mask,
        model.layers[idx + 13].encoders_wo,
        model.layers[idx + 13].encoders_bo,
        "encoders",
        idx
    );

    // ---- merge heads
    attn_out = ggml_reshape_3d(
        ctx0,
        attn_out,
        d_k * n_heads,
        T,
        B
    );

    // ---- output proj
    // ggml_tensor * out = ggml_mul_mat(ctx0, model.layers[idx + 13].encoders_wo, attn_out);
    // out = ggml_add(ctx0, out, model.layers[idx + 13].encoders_bo);
    ggml_tensor * out = ggml_mul_mat_add(ctx0, model.layers[idx + 13].encoders_wo, attn_out, model.layers[idx + 13].encoders_bo);

    return out;
}

ggml_tensor * llm_graph_context::build_pre_lookahead_layer(
         ggml_tensor * cur,
         ggml_tensor * context,
         const llama_model & model) const{
    const int lookahead = 3;
    ggml_tensor * x = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));  // [B, C, T]
    if (context == nullptr) {
        // x = ggml_reshape_4d(ctx0, x, x->ne[0], x->ne[1], 1, 1);
        x = ggml_pad(ctx0, x, lookahead, 0, 0, 0);
    } else {
        GGML_ASSERT(context->ne[1] == lookahead);
        context = ggml_cont(ctx0, ggml_permute(ctx0, context, 1, 0, 2, 3));
        x = ggml_concat(ctx0, x, context, 0);
        x = ggml_pad(ctx0, x, lookahead - context->ne[0], 0, 0, 0);
    }
    x = ggml_conv_1d(ctx0, model.pre_look_conv1_w, x, 1, 0, 1);
    ggml_tensor * conv1_mb = ggml_reshape_4d(ctx0, model.pre_look_conv1_b, 1, 512, 1, 1);
    x = ggml_add(ctx0, x, conv1_mb);
   
    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 2, x->ne[1], x->ne[2], x->ne[3]);
    zeros = ggml_scale(ctx0, zeros, 0.0f);
    x = ggml_concat(ctx0, zeros, x, 0);
    x = ggml_cont(ctx0, x);
    
    x = ggml_conv_1d(ctx0, model.pre_look_conv2_w, x, 1, 0, 1);
    ggml_tensor * conv2_mb = ggml_reshape_4d(ctx0, model.pre_look_conv2_b, 1, 512, 1, 1);
    x  = ggml_add(ctx0, x, conv2_mb);
    
    x = ggml_permute(ctx0, x, 1, 0, 2, 3);
    x = ggml_cont(ctx0, x);
    x = ggml_add(ctx0, x, cur);

    ggml_set_name(x, "pre_look_res");
    // ggml_tensor * outputs = ggml_conv_1d(ctx0, model.pre_look_conv1_w, x, 1, 0, 1);
    // ggml_tensor * conv1_mb = ggml_reshape_4d(ctx0, model.pre_look_conv1_b, 1, 512, 1, 1);
    // outputs = ggml_add(ctx0, outputs, conv1_mb);
   
    // outputs = ggml_leaky_relu(ctx0, outputs, 0.01f, false);
    
    // ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 2, outputs->ne[1], outputs->ne[2], outputs->ne[3]);
    // zeros = ggml_scale(ctx0, zeros, 0.0f);
    // outputs = ggml_concat(ctx0, zeros, outputs, 0);
    // outputs = ggml_cont(ctx0, outputs);
    
    // outputs = ggml_conv_1d(ctx0, model.pre_look_conv2_w, outputs, 1, 0, 1);
    // ggml_tensor * conv2_mb = ggml_reshape_4d(ctx0, model.pre_look_conv2_b, 1, 512, 1, 1);
    // outputs  = ggml_add(ctx0, outputs, model.pre_look_conv2_b);
    
    // outputs = ggml_permute(ctx0, outputs, 1, 0, 2, 3);
    // outputs = ggml_cont(ctx0, outputs);
    // outputs = ggml_add(ctx0, outputs, cur);

    // ggml_set_name(outputs, "pre_look_res");

    return x;
        
}

ggml_tensor * llm_graph_context::build_rel_pos_attn(
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb
) const{
    ggml_build_forward_expand(gf, cur);

    const int32_t d_k = 64;
    const int32_t heads = 8;
    const int32_t n_batch = cur->ne[2];
    ggml_tensor * flat = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_batch * cur->ne[1]);
    // ggml_tensor * t = ggml_mul_mat(ctx0, mw, flat);
    // t = ggml_add(ctx0, t, mb);
    ggml_tensor * t = ggml_mul_mat_add(ctx0, mw, flat, mb);
    t = ggml_reshape_4d(ctx0, t, d_k, heads, cur->ne[1], n_batch);
    t = ggml_permute(ctx0, t, 0, 2, 1, 3);

    return t;
}

ggml_tensor * llm_graph_context::build_rel_shift(
         ggml_tensor * cur) const{
    const int B = cur->ne[3];   
    const int n_head = cur->ne[2]; 
    const int T = cur->ne[1];    
    const int L = cur->ne[0];

    auto zero_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, T, n_head, B);
    zero_pad = ggml_scale(ctx0, zero_pad, 0.0f);  
    ggml_tensor * x_padded = ggml_concat(ctx0, zero_pad, cur, 0);
    auto reshaped = ggml_reshape_4d(ctx0, x_padded, T, L + 1, n_head, B);
    ggml_tensor* x_sliced = ggml_view_4d(ctx0,
                                         reshaped,
                                         T, L, n_head, B,
                                         reshaped->nb[1],
                                         reshaped->nb[2],
                                         reshaped->nb[3],
                                         reshaped->nb[1]);
    ggml_tensor* x_back = ggml_reshape_4d(ctx0, ggml_cont(ctx0, x_sliced), L, T, n_head, B);
    ggml_tensor* result = ggml_view_4d(ctx0,
                                       x_back,
                                       T, T, n_head, B,
                                       x_back->nb[1],
                                       x_back->nb[2],
                                       x_back->nb[3],
                                       0);
    // result = ggml_cont(ctx0, result);
    return result;
        
}



ggml_tensor * llm_graph_context::build_attn_scores(
         ggml_tensor * cur,
         ggml_tensor * scores,
         ggml_tensor * mask,
         ggml_tensor * mw,
         ggml_tensor * mb,
         std::string attn_type,
         int32_t il) const{
            
    const int B    = cur->ne[3];
    const int n_head = cur->ne[1];
    const int T    = cur->ne[2];
    const int d_k  = cur->ne[0];
    const int d    = n_head * d_k;
    const int time1 = scores->ne[2];
    ggml_tensor * attn = nullptr;
    if (mask) {
        mask = ggml_view_3d(ctx0, mask, scores->ne[0], mask->ne[1], scores->ne[3], mask->nb[1], mask->nb[2], 0);
        mask = ggml_reshape_4d(ctx0, mask, scores->ne[0], mask->ne[1], 1, scores->ne[3]);
        mask = ggml_repeat(ctx0, mask, scores);
        mask = ggml_scale_bias(ctx0, mask, -1.0f, 1.0f);
        ggml_tensor * masked_scores = ggml_add(ctx0, scores, ggml_scale(ctx0, mask, -1.0e9f));
        attn = ggml_soft_max(ctx0, masked_scores);
        attn = ggml_mul(ctx0, attn, mask);
    } else {
        attn = ggml_soft_max(ctx0, scores);
    }
    ggml_tensor * cur_T = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    ggml_tensor * x = ggml_mul_mat(ctx0, cur_T, attn);
    x = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)), time1 * d_k, n_head, B);
    // x = ggml_mul_mat(ctx0, mw, x);
    // x = ggml_add(ctx0, x, mb);
    x = ggml_mul_mat_add(ctx0, mw, x, mb);
    return x;
}


ggml_tensor * llm_graph_context::build_pos_ffn(
             ggml_tensor * cur,
             ggml_tensor * mw_1,
             ggml_tensor * mb_1,
             ggml_tensor * mw_2,
             ggml_tensor * mb_2) const{
    
    // cur = ggml_mul_mat(ctx0, mw_1, cur);
    // cur = ggml_add(ctx0, cur, mb_1);
    cur = ggml_mul_mat_add(ctx0, mw_1, cur, mb_1);
    cur = ggml_silu(ctx0, cur);
    // cur = ggml_mul_mat(ctx0, mw_2, cur);
    // cur = ggml_add(ctx0, cur, mb_2);
    cur = ggml_mul_mat_add(ctx0, mw_2, cur, mb_2);
    return cur;
}

ggml_tensor * llm_graph_context::build_upsample_1d(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb) const
{
    ggml_tensor * up = ggml_interpolate(ctx0, cur, cur->ne[0], cur->ne[1] * 2, cur->ne[2], cur->ne[3], 0);
    cb(up, "interpolate", -1);
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, up->ne[0], 4, up->ne[2], up->ne[3]);
    zeros = ggml_scale(ctx0, zeros, 0.0f);
    ggml_tensor * pad = ggml_concat(ctx0, zeros, up, 1);
    cb(pad, "upsample_pad", -1);
    pad = ggml_cont(ctx0, ggml_permute(ctx0, pad, 1, 0, 2, 3));
    ggml_tensor * out = ggml_conv_1d(ctx0, mw, pad, 1, 0, 1);
    mb = ggml_reshape_3d(ctx0, mb, 1, cur->ne[0], 1);
    out = ggml_add(ctx0, out, mb);
    cb(out, "upsample_conv_1d", -1);
    return out;
}

ggml_tensor * llm_graph_context::build_causal_cond_cfm(
         ggml_cgraph * gf,
         int64_t n_timesteps) const{
    
    ggml_tensor * t_span = ggml_arange(ctx0, 0.0f, 1.1f, 0.1f);
    ggml_tensor * t_cos = ggml_cos(ctx0, ggml_scale(ctx0, t_span, 0.5f * M_PI));
    ggml_tensor * zero = ggml_scale(ctx0, t_span, 0.0f);
    ggml_tensor * one = ggml_exp(ctx0, zero);
    t_span = ggml_sub(ctx0, one, t_cos);
    cb(t_span, "causal_cond_cfm", -1);
    return t_span;
}

ggml_tensor * llm_graph_context::build_sinusoidal_pos_emb(
         ggml_tensor * cur,
         ggml_tensor * emb_row,
         int dim,
         int scale, 
         float emb_div) const{
        
    // const int64_t n_token = cur->ne[1];
    // const int half_dim = dim / 2;

    // // float emb_div = std::log(10000.0f) / (half_dim - 1);
    // ggml_tensor * idx = ggml_arange(ctx0, 0, half_dim, 1);
    // // idx = ggml_cast(ctx0, idx, GGML_TYPE_F32);
    // ggml_tensor * emb = ggml_scale(ctx0, idx, -emb_div);
    // emb = ggml_exp(ctx0, emb);

    ggml_tensor * x_col = ggml_reshape_2d(ctx0, cur, cur->ne[1], cur->ne[0]);
    // ggml_tensor * emb_row = ggml_reshape_2d(ctx0, emb, 1, half_dim);
    ggml_tensor * xx = ggml_scale(ctx0, x_col, scale);
    ggml_tensor * out = ggml_mul_mat(ctx0, xx, emb_row);       

    ggml_tensor * sin_t = ggml_sin(ctx0, out);
    ggml_tensor * cos_t = ggml_cos(ctx0, out);
    ggml_tensor * emb_final = ggml_concat(ctx0, sin_t, cos_t, 1);


    return emb_final;
}

ggml_tensor * llm_graph_context::build_timestep_embedding(
         ggml_tensor * cur,
         ggml_tensor * mw_1,
         ggml_tensor * mb_1,
         ggml_tensor * mw_2,
         ggml_tensor * mb_2) const{
    
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // cur = ggml_mul_mat(ctx0, mw_1, cur);
    // cur = ggml_add(ctx0, cur, mb_1);
    cur = ggml_mul_mat_add(ctx0, mw_1, cur, mb_1);
    cur = ggml_silu(ctx0, cur);
    // cur = ggml_mul_mat(ctx0, mw_2, cur);
    // cur = ggml_add(ctx0, cur, mb_2);
    cur = ggml_mul_mat_add(ctx0, mw_2, cur, mb_2);
    return cur;
}

ggml_tensor * llm_graph_context::build_linear_no_sample(
        ggml_tensor * cur,
        int32_t offset,
        const llama_model & model) const {
    // cur = ggml_mul_mat(ctx0, model.embed_out_0_w, cur);
    // cur = ggml_add(ctx0, cur, model.embed_out_0_b);
    cur = ggml_mul_mat_add(ctx0, model.embed_out_0_w, cur, model.embed_out_0_b);
    cur = build_layer_norm(cur, model.embed_out_1_w, model.embed_out_1_b, 1e-05, "sub_sampling", 0);
    float xscale = 22.627416997969522f;
    cur = ggml_scale(ctx0, cur, xscale);
    return cur;
}

ggml_tensor * llm_graph_context::subsequent_chunk_mask(
        int32_t size,
        int32_t chunk_size,
        int32_t num_left_chunks) const {
    
    GGML_ASSERT(size > 0);
    GGML_ASSERT(chunk_size > 0);
    ggml_tensor * pos_idx = ggml_arange(ctx0, 0.0f, static_cast<float>(size), 1.0f);
    ggml_tensor * div = ggml_scale(ctx0, pos_idx, 1.0f / static_cast<float>(chunk_size));
    ggml_tensor * div_trunc = ggml_floor(ctx0, div);
    ggml_tensor * block_value = ggml_scale_bias(ctx0, div_trunc, static_cast<float>(chunk_size), static_cast<float>(chunk_size));
    
    ggml_tensor * pos_2d = ggml_reshape_2d(ctx0, pos_idx, size, 1);
    ggml_tensor * pos_mat = ggml_repeat_4d(ctx0, pos_2d, size, size, 1, 1);
    ggml_tensor * block_2d = ggml_reshape_2d(ctx0, block_value, 1, size);
    ggml_tensor * block_mat = ggml_repeat_4d(ctx0, block_2d, size, size, 1, 1);
    ggml_tensor * diff = ggml_sub(ctx0, block_mat, pos_mat);
    ggml_tensor * diff_shift = ggml_scale_bias(ctx0, diff, 1.0f, -0.5f);
    ggml_tensor * ret = ggml_step(ctx0, diff_shift);

    return ret;
}

ggml_tensor * llm_graph_context::build_optional_chunk_mask(
        ggml_tensor * cur,
        ggml_tensor * masks,
        bool use_dynamic_chunk,
        bool use_dynamic_left_chunk,
        int32_t decoding_chunk_size,
        int32_t static_chunk_size,
        int32_t num_decoding_left_chunks,
        bool enable_full_context) const{
    
    int num_left_chunks = 0;
    ggml_tensor * chunk_masks = nullptr;
    if (static_chunk_size > 0) {
        num_left_chunks = num_decoding_left_chunks;
        if (masks->ne[0] != cur->ne[1] || masks->ne[2] > 1) {
            GGML_ASSERT(masks->ne[0] >= cur->ne[1]);
            masks = ggml_view_3d(ctx0, masks, cur->ne[1], 1, 1, masks->nb[1], masks->nb[2], 0);
        }
        chunk_masks = subsequent_chunk_mask(cur->ne[1], static_chunk_size, num_left_chunks);
        chunk_masks = ggml_reshape_3d(ctx0, chunk_masks, chunk_masks->ne[0], chunk_masks->ne[1], 1);
        ggml_tensor * masks_expand = ggml_repeat(ctx0, masks, chunk_masks);
        chunk_masks = ggml_mul(ctx0, masks_expand, chunk_masks);
    }else {
        chunk_masks = masks;
    }
    GGML_ASSERT(chunk_masks->ne[0] > 0);
    GGML_ASSERT(chunk_masks->ne[1] > 0);
    GGML_ASSERT(chunk_masks->ne[2] > 0);
    ggml_tensor * row_sum = ggml_sum_rows(ctx0, chunk_masks);
    ggml_tensor * has_any = ggml_step(ctx0, ggml_scale_bias(ctx0, row_sum, 1.0f, -0.5f));
    ggml_tensor * bad_row = ggml_scale_bias(ctx0, has_any, -1.0f, 1.0f);
    ggml_tensor * bad_expand = ggml_repeat(ctx0, bad_row, chunk_masks);
    ggml_tensor * out = ggml_sub(ctx0, ggml_add(ctx0, chunk_masks, bad_expand), ggml_mul(ctx0, chunk_masks, bad_expand));

    return out;
}

ggml_tensor * llm_graph_context::build_encoder(
        ggml_tensor * token,
        int32_t token_len,
        ggml_tensor * context,
        ggml_tensor * extend_pe,
        bool streaming,
        const llama_model & model) const {
    ggml_tensor * x = build_linear_no_sample(token, 0, model);
    ggml_tensor * extend_pe_cpy = ggml_dup(ctx0, extend_pe);
    ggml_tensor * pos_emb = build_pos_encoding(extend_pe_cpy, x->ne[1], 0, 0);
    ggml_tensor * masks, * chunk_masks = nullptr;
    if (streaming) {
        masks = build_pad_mask(token_len, token->ne[1]);
        chunk_masks = build_optional_chunk_mask(x, masks, false, false, 0, 25, -1);
        if(context->ne[1] != 0) {
            context = build_linear_no_sample(context, x->ne[1], model);
        }
    }
    x = build_pre_lookahead_layer(x, context, model);  //✅

    for (int i = 0; i < 6; ++i) {
        ggml_tensor * residual = x;

        x = build_layer_norm(
            x,
            model.layers[i + 13].encoders_normmha_w,
            model.layers[i + 13].encoders_normmha_b,
            1e-12f,
            "encoders",
            i
        );

        ggml_tensor * attn_out = build_flash_attn_encoder(
            x,
            model,
            /* attn_mask = */ chunk_masks,
            /* pos_emb   = */ pos_emb,
            /* n_heads   = */ 8,
            i
        );

        x = ggml_add(ctx0, residual, attn_out);

        // ---- FFN
        ggml_tensor * ffn_res = x;

        x = build_layer_norm(
            x,
            model.layers[i + 13].encoders_normffn_w,
            model.layers[i + 13].encoders_normffn_b,
            1e-12f,
            "encoders",
            i
        );

        x = build_pos_ffn(
            x,
            model.layers[i + 13].encoders_ffn_w1,
            model.layers[i + 13].encoders_ffn_b1,
            model.layers[i + 13].encoders_ffn_w2,
            model.layers[i + 13].encoders_ffn_b2
        );

        x = ggml_add(ctx0, ffn_res, x);
    }
    
    ggml_set_name(x, "after encoders");
    x = build_upsample_1d(gf, x, model.up_layer_conv_w, model.up_layer_conv_b);
    ggml_set_name(x, "after build_upsample_1d");
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    
    x = build_linear_no_subsampling(x, model.up_embed_out_0_w, model.up_embed_out_0_b, model.up_embed_out_1_w, model.up_embed_out_1_b, 2);
    x = build_espnet_pos_encode(x, 2);
    ggml_tensor * extend_pe_cpy2 = extend_pe;
    pos_emb = build_pos_encoding(extend_pe_cpy2, x->ne[1], 0, 2);
    if (streaming) {
        masks = build_pad_mask(x->ne[1] + 6, x->ne[1]);
        chunk_masks = build_optional_chunk_mask(x, masks, false, false, 0, 50, -1);
    } 
    //build up_encoders
    for(int i = 0; i < 4; i++) {
        ggml_tensor * attn_residual = x;
        // ggml_build_forward_expand(gf, attn_residual);
        cb(attn_residual, "up_encoders_x_residual", i);
        x = build_layer_norm(x, model.layers[i + 132].up_encoders_normmha_w, model.layers[i + 132].up_encoders_normmha_b, 1e-12, "up_encoders", 12 + i);
        // x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
        ggml_tensor * query = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wq, model.layers[i + 132].up_encoders_bq);
        cb(query, "up_encoders_attn_q", i);
        ggml_tensor * key = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wk, model.layers[i + 132].up_encoders_bk);
        cb(key, "up_encoders_attn_k", i);
        ggml_tensor * value = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wv, model.layers[i + 132].up_encoders_bv);
        cb(value, "up_encoders_attn_v", i);
        query = ggml_cont(ctx0, ggml_permute(ctx0, query, 0, 2, 1, 3));
        int n_batch_pos = pos_emb->ne[2];
        ggml_tensor * p = ggml_mul_mat(ctx0, model.layers[i + 132].up_encoders_wpos, pos_emb);
        p = ggml_cont(ctx0, p);
        p = ggml_reshape_4d(ctx0, p, 64, 8, p->ne[1], n_batch_pos);
        p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));
        ggml_tensor * q_with_bias_u = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 132].up_encoders_pos_bias_u), 0, 2, 1, 3));
        cb(q_with_bias_u, "up_encoder_q_with_bias_u", i);
        ggml_tensor * q_with_bias_v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 132].up_encoders_pos_bias_v), 0, 2, 1, 3));
        cb(q_with_bias_v, "up_encoder_q_with_bias_v", i);
        ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, key, q_with_bias_u);
        ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_with_bias_v);
        matrix_bd = build_rel_shift(matrix_bd);
        ggml_tensor * scores = ggml_scale(ctx0, ggml_add(ctx0, matrix_ac, matrix_bd), 0.125f);
        cb(scores, "up_scores", i);
        ggml_tensor * x_att = build_attn_scores(value, scores, chunk_masks, model.layers[i + 132].up_encoders_wo, model.layers[i + 132].up_encoders_bo, "up_encoders", i);
        cb(x_att, "up_x_att", i);
        x = ggml_add(ctx0, attn_residual, x_att);
        ggml_tensor * ffn_residual = x;
        x = build_layer_norm(x, model.layers[i + 132].up_encoders_normffn_w, model.layers[i + 132].up_encoders_normffn_b, 1e-12, "up_encoders", 16 + i);
        x = build_pos_ffn(x, model.layers[i + 132].up_encoders_ffn_w1, model.layers[i + 132].up_encoders_ffn_b1, model.layers[i + 132].up_encoders_ffn_w2, model.layers[i + 132].up_encoders_ffn_b2);
        cb(x, "up_fn_out", i);
        x = ggml_add(ctx0, ffn_residual, x);
        cb(x, "up_encoder_out", i);
    }
    return x;
}

ggml_tensor * llm_graph_context::build_basic_attn(
         ggml_tensor * x, 
         ggml_tensor * attn_mask,
         int64_t layer_id,
         std::string mode,
         const llama_model & model
         ) const{
    int64_t input_ndim = ggml_n_dims(x);
    const int64_t sequence_length = x->ne[1];   // T
    const int64_t batch_size      = x->ne[2];   // B
    const int64_t n_heads = 8;
    ggml_tensor * wq, *wk, *wv;
    if (mode == "down_block") {
        wq = model.layers[layer_id].down_block1_wq;
        wk = model.layers[layer_id].down_block1_wk;
        wv = model.layers[layer_id].down_block1_wv;
    } else if (mode == "mid_block") {
        wq = model.mid_block_sub_layers[layer_id].mid_block1_wq;
        wk = model.mid_block_sub_layers[layer_id].mid_block1_wk;
        wv = model.mid_block_sub_layers[layer_id].mid_block1_wv;
    } else {
        wq = model.layers[layer_id].up_block1_wq;
        wk = model.layers[layer_id].up_block1_wk;
        wv = model.layers[layer_id].up_block1_wv;
    }
    ggml_tensor * q = ggml_mul_mat(ctx0, wq, x);
    ggml_tensor * k = ggml_mul_mat(ctx0, wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx0, wv, x);

    int64_t inner_dim = k->ne[0];
    int64_t d_k = inner_dim / n_heads;
    auto reshape_heads = [&](ggml_tensor * t) {
        ggml_tensor * t_4d = ggml_reshape_4d(ctx0, t, d_k, n_heads, sequence_length, batch_size);
        ggml_tensor * t_perm = ggml_permute(ctx0, t_4d, 0, 2, 1, 3);
        return t_perm;
    };
    
    q = reshape_heads(q);
    k = reshape_heads(k);
    v = reshape_heads(v);

    ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, q, k, v, attn_mask, 0.125f, 0.0f, 0.0f);
    ggml_tensor * attn_flat = ggml_reshape_3d(ctx0, attn_out,
                                attn_out->ne[0] * attn_out->ne[1],              
                                attn_out->ne[2],                       
                                attn_out->ne[3]);
    
    ggml_tensor * wo, * bo;
    if (mode == "down_block") {
        wo = model.layers[layer_id].down_block1_wo;
        bo = model.layers[layer_id].down_block1_bo;
    } else if (mode == "mid_block") {
        wo = model.mid_block_sub_layers[layer_id].mid_block1_wo;
        bo = model.mid_block_sub_layers[layer_id].mid_block1_bo;
    } else {
        wo = model.layers[layer_id].up_block1_wo;
        bo = model.layers[layer_id].up_block1_bo;
    }
    // attn_flat = ggml_mul_mat(ctx0, wo, attn_flat);
    // attn_flat = ggml_add(ctx0, attn_flat, bo);
    attn_flat = ggml_mul_mat_add(ctx0, wo, attn_flat, bo);
    
    return attn_flat;
}


ggml_tensor * llm_graph_context::causal_conv1d_forward(
        ggml_tensor * x,
        ggml_tensor * pad,
        ggml_tensor * mask,
        const ConvBias & conv_b,
        std::string mode,
        int32_t layer_id,
        int32_t blk_id,
        const llama_model & model,
        int32_t step) const{
    ggml_tensor * x_pad = ggml_concat(ctx0, pad, x, 0);
    ggml_set_name(x_pad, ("causal_blk1d_conv_pad_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
    ggml_tensor * model_weight;
    ggml_tensor * model_bias;
    if (mode == "down_block") {
        if (blk_id == 1) {
            model_weight = model.down_blk1_conv_w;
            model_bias = conv_b.d1;
        } else if (blk_id == 2){
            model_weight = model.down_blk2_conv_w;
            model_bias = conv_b.d2;
        } else {
            model_weight = model.down_blk_conv_w;
            model_bias = conv_b.d3;
        }
    } else if(mode == "mid_block") {
        int i = layer_id - 280;
        if(blk_id == 1) {
            model_weight = model.layers[layer_id].mid_block1_w;
            model_bias = conv_b.m1[i];
        } else {
            model_weight = model.layers[layer_id].mid_block2_w;
            model_bias = conv_b.m2[i];
        }
    }else if(mode == "up_block"){
        if(blk_id == 1) {
            model_weight = model.up_blk1_conv_w;
            model_bias = conv_b.u1; 
        } else if(blk_id == 2) {
            model_weight = model.up_blk2_conv_w;
            model_bias = conv_b.u2;
        } else {
            model_weight = model.up_blk_conv_w;
            model_bias = conv_b.u3;
        }
    } else if (mode == "final_block"){
        model_weight = model.f_blk_conv_w;
        model_bias = conv_b.f1;
    } 
    ggml_set_name(model_weight, ("causal_blk1d_conv_weight_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
    
    // x_pad = ggml_cont(ctx0, x_pad);
    
    ggml_tensor * y_corrupt = ggml_conv_1d(ctx0, model_weight, x_pad, 1, 0, 1);
    const int64_t W = y_corrupt->ne[0];
    const int64_t C = y_corrupt->ne[1];
    const int64_t B = y_corrupt->ne[2];
    const size_t el_size = ggml_element_size(y_corrupt);
    ggml_tensor * y_fixed = ggml_view_3d(ctx0, y_corrupt, 
                                     W, C, B, 
                                     W * B * el_size,
                                     W * el_size,
                                     0);
    // ggml_tensor * y = ggml_cont(ctx0, y_fixed);
    ggml_tensor * y = ggml_add(ctx0, y_fixed, model_bias);
    return y;
}

ggml_tensor * llm_graph_context::causal_block1d_forward(
    ggml_tensor * x,
    std::vector<ggml_tensor *> pad_list,
    ggml_tensor * mask,
    ggml_tensor * resnet_mish_ones,
    const ConvBias & conv_b,
    std::string mode,
    int32_t layer_id,
    int32_t blk_id,
    const llama_model & model,
    int32_t step) const{
    
    x = ggml_mul(ctx0, x, mask);
    if (x->ne[1] == 256) {
        x = causal_conv1d_forward(x, pad_list[0], mask, conv_b, mode, layer_id, blk_id, model, step);
    } else if (x->ne[1] == 320) {
        x = causal_conv1d_forward(x, pad_list[1], mask, conv_b, mode, layer_id, blk_id, model, step);
    } else {
        x = causal_conv1d_forward(x, pad_list[2], mask, conv_b, mode, layer_id, blk_id, model, step);
    }
    
    ggml_tensor * model_weight;
    ggml_tensor * model_bias;
    if (mode == "down_block") {
        if (blk_id == 1) {
            model_weight = model.down_blk1_norm_w;
            model_bias = model.down_blk1_norm_b;
        } else {
            model_weight = model.down_blk2_norm_w;
            model_bias = model.down_blk2_norm_b;
        }
    } else if(mode == "mid_block") {
        if(blk_id == 1) {
            model_weight = model.layers[layer_id].mid_block1_norm_w;
            model_bias = model.layers[layer_id].mid_block1_norm_b;
        } else {
            model_weight = model.layers[layer_id].mid_block2_norm_w;
            model_bias = model.layers[layer_id].mid_block2_norm_b;
        }
    }else if(mode == "up_block"){
        if(blk_id == 1) {
            model_weight = model.up_blk1_norm_w;
            model_bias = model.up_blk1_norm_b; 
        } else {
            model_weight = model.up_blk2_norm_w;
            model_bias = model.up_blk2_norm_b; 
        } 
    } else {
        model_weight = model.f_blk_norm_w;
        model_bias = model.f_blk_norm_b;
    }
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = build_layer_norm(x, model_weight, model_bias, 1e-5f, "blk_1d", 100);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    // x = ggml_mul_inplace(ctx0, x, ggml_tanh_inplace(ctx0, ggml_softplus(ctx0, x)));
    // x = ggml_silu_inplace(ctx0, x);
    x = ggml_mish(ctx0, x);
    return x;
}

ggml_tensor * llm_graph_context::causal_resnet_block1d_forward(
        ggml_tensor * x,
        std::vector<ggml_tensor *> pad_list,
        ggml_tensor * mask,
        ggml_tensor * t_emb,
        int32_t step,
        int32_t layer_id,
        std::string mode,
        ggml_tensor * t_emb_ones,
        ggml_tensor * resnet_mish_ones,
        std::vector<ggml_tensor *> res_w,
        const ConvBias & conv_b,
        const llama_model & model) const{
    ggml_tensor * x_dup = x;
    x = causal_block1d_forward(x, pad_list, mask, resnet_mish_ones, conv_b, mode, layer_id, 1, model, step);
    if(mode == "down_block") {
        // t_emb = ggml_mul_mat(ctx0, model.down_blk_mlp_w, t_emb);
        // t_emb = ggml_add(ctx0, t_emb, model.down_blk_mlp_b);
        t_emb = ggml_mul_mat_add(ctx0, model.down_blk_mlp_w, t_emb, model.down_blk_mlp_b);
    }else if(mode == "mid_block") {
        // t_emb = ggml_mul_mat(ctx0, model.layers[layer_id].mid_block_mlp_w, t_emb);
        // t_emb = ggml_add(ctx0, t_emb, model.layers[layer_id].mid_block_mlp_b);
        t_emb = ggml_mul_mat_add(ctx0, model.layers[layer_id].mid_block_mlp_w, t_emb, model.layers[layer_id].mid_block_mlp_b);
    } else {
        // t_emb = ggml_mul_mat(ctx0, model.up_blk_mlp_w, t_emb);
        // t_emb = ggml_add(ctx0, t_emb, model.up_blk_mlp_b);
        t_emb = ggml_mul_mat_add(ctx0, model.up_blk_mlp_w, t_emb, model.up_blk_mlp_b);
    }
    t_emb = ggml_reshape_3d(ctx0, t_emb, 1, t_emb->ne[0], t_emb->ne[1]);
    x = ggml_add(ctx0, x, t_emb);
    ggml_set_name(x, ("causal_add_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    x = causal_block1d_forward(x, pad_list, mask, resnet_mish_ones, conv_b, mode, layer_id, 2, model, step);
    ggml_set_name(x, ("causal_blk2_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    
    ggml_tensor * x_mask = ggml_cont(ctx0, ggml_permute(ctx0, x_dup, 1, 0, 2, 3));
    ggml_set_name(x_mask, ("causal_x_mask_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    if(mode == "down_block") {
        // x_mask = ggml_mul_mat(ctx0, res_w[0], x_mask);
        // x_mask = ggml_add(ctx0, x_mask, model.down_blk_res_b);
        x_mask = ggml_mul_mat_add(ctx0, res_w[0], x_mask, model.down_blk_res_b);
        
    } else if(mode == "mid_block") {
        // x_mask = ggml_mul_mat(ctx0, res_w[layer_id - 280 + 1], x_mask);
        // x_mask = ggml_add(ctx0, x_mask, model.layers[layer_id].mid_block_res_b);
        x_mask = ggml_mul_mat_add(ctx0, res_w[layer_id - 280 + 1], x_mask, model.layers[layer_id].mid_block_res_b);
        
    } else {
        // x_mask = ggml_mul_mat(ctx0, res_w[13], x_mask);
        // x_mask = ggml_add(ctx0, x_mask, model.up_blk_res_b);
        x_mask = ggml_mul_mat_add(ctx0, res_w[13], x_mask, model.up_blk_res_b);
    }
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_tensor * result = ggml_add(ctx0, x, x_mask);
    return result;
}

ggml_tensor * llm_graph_context::build_causal_cond_decoder(
         ggml_tensor * x,
         std::vector<ggml_tensor *> pad_list,
         ggml_tensor * mask,
         ggml_tensor * mu,
         ggml_tensor * t,
         ggml_tensor * spks,
         ggml_tensor * cond,
         ggml_tensor * spks_t,
         ggml_tensor * attn_mask,
         ggml_tensor * t_emb_ones,
         ggml_tensor * resnet_mish_ones,
         std::vector<ggml_tensor *> res_w,
         const ConvBias & conv_b,
         ggml_tensor * emb_row,
         const std::vector<ggml_tensor *> & down_w0,
         const std::vector<ggml_tensor *> & down_w2,
         const std::vector<ggml_tensor *> & mid_w0,
         const std::vector<ggml_tensor *> & mid_w2,
         const std::vector<ggml_tensor *> & up_w0,
         const std::vector<ggml_tensor *> & up_w2,
         const llama_model & model,
         int32_t step,
         bool streaming) const{
    
    t = build_sinusoidal_pos_emb(t, emb_row, 320, 1000);
    t = build_timestep_embedding(t, model.time_mlp_1_w, model.time_mlp_1_b, model.time_mlp_2_w, model.time_mlp_2_b);
    // ggml_tensor * t_mish = ggml_mul_inplace(ctx0, t, ggml_tanh_inplace(ctx0, ggml_softplus(ctx0, t)));
    ggml_tensor * t_mish = ggml_mish(ctx0, t);
    // ggml_tensor * t_mish = ggml_silu(ctx0, t);
    x = ggml_concat(ctx0, x, mu, 1);
    if (spks) {
        x = ggml_concat(ctx0, x, spks_t, 1);
    }
    if (cond) {
        x = ggml_concat(ctx0, x, cond, 1);
    }

    std::vector<ggml_tensor *> hiddens;
    std::vector<ggml_tensor *> masks = {mask};
    
    ggml_tensor * mask_down = masks.back();
    x = causal_resnet_block1d_forward(x, pad_list, mask_down, t_mish, step, 0, "down_block", t_emb_ones, resnet_mish_ones, res_w, conv_b, model);
    ggml_set_name(x, ("causal_resnet_blk1d_" + std::to_string(step)).c_str());
    ggml_tensor * tr_x, *h, *attn_out, *ff_out;
    auto apply_streaming_chunk_mask = [&](ggml_tensor * base_mask, ggml_tensor * cur, ggml_tensor * cur_mask) -> ggml_tensor * {
        ggml_tensor * chunk_mask = build_optional_chunk_mask(cur, cur_mask, false, false, 0, 50, -1);
        chunk_mask = ggml_scale_bias(ctx0, chunk_mask, -1.0f, 1.0f);
        chunk_mask = ggml_scale(ctx0, chunk_mask, -1e4f);
        chunk_mask = ggml_reshape_4d(ctx0, chunk_mask, chunk_mask->ne[0], chunk_mask->ne[1], 1, 1);

        const int64_t pad_rows = base_mask->ne[0] - chunk_mask->ne[0];
        if (pad_rows > 0) {
            ggml_tensor * chunk_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, pad_rows, chunk_mask->ne[1], 1, 1);
            chunk_pad = ggml_scale(ctx0, chunk_pad, 0.0f);
            chunk_mask = ggml_concat(ctx0, chunk_mask, chunk_pad, 0);
        }

        const int64_t pad_cols = base_mask->ne[1] - chunk_mask->ne[1];
        if (pad_cols > 0) {
            ggml_tensor * chunk_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, chunk_mask->ne[0], pad_cols, 1, 1);
            chunk_pad = ggml_scale(ctx0, chunk_pad, 0.0f);
            chunk_mask = ggml_concat(ctx0, chunk_mask, chunk_pad, 1);
        }

        chunk_mask = ggml_cast(ctx0, chunk_mask, GGML_TYPE_F16);
        return ggml_add(ctx0, base_mask, chunk_mask);
    };
    if (streaming) {
        attn_mask = apply_streaming_chunk_mask(attn_mask, x, mask_down);
    }
    for (size_t i = 0; i < 4; ++i) {
        tr_x = x;
        h = build_layer_norm(x, model.layers[226 + i].down_block1_norm1_w, model.layers[226 + i].down_block1_norm1_b, 1e-5f, "down_block", 20 + i);
        attn_out = build_basic_attn(h, attn_mask, 226 + i, "down_block", model);
        tr_x = ggml_add(ctx0, attn_out, tr_x);

        h = build_layer_norm(tr_x, model.layers[226 + i].down_block1_norm3_w, model.layers[226 + i].down_block1_norm3_b, 1e-5f, "down_block", 24 + i);
        // ff_out = ggml_mul_mat(ctx0, down_w0[i], h);
        // ff_out = ggml_add(ctx0, ff_out, model.layers[226 + i].down_block1_ffn_b0);
        ff_out = ggml_mul_mat_add(ctx0, down_w0[i], h, model.layers[226 + i].down_block1_ffn_b0);
        ff_out = ggml_gelu_erf(ctx0, ff_out);
        // ff_out = ggml_mul_mat(ctx0, down_w2[i], ff_out);
        // ff_out = ggml_add(ctx0, ff_out, model.layers[226 + i].down_block1_ffn_b2);
        ff_out = ggml_mul_mat_add(ctx0, down_w2[i], ff_out, model.layers[226 + i].down_block1_ffn_b2);
        x = ggml_add(ctx0, ff_out, tr_x);
        // x = tr_x;
    }
    ggml_set_name(x, ("causal_trans_down_block_" + std::to_string(step)).c_str());
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_tensor * hidden = x;
    // hiddens.push_back(x);
    // ggml_tensor * x_mask = ggml_mul(ctx0, x, mask_down);
    if (x->ne[1] == 256) {
        x = causal_conv1d_forward(x, pad_list[0], mask_down, conv_b, "down_block", 0, 3, model, step);
    } else if (x->ne[1] == 320) {
        x = causal_conv1d_forward(x, pad_list[1], mask_down, conv_b, "down_block", 0, 3, model, step);
    } else {
        x = causal_conv1d_forward(x, pad_list[2], mask_down, conv_b, "down_block", 0, 3, model, step);
    }
    
    if (streaming) {
        attn_mask = apply_streaming_chunk_mask(attn_mask, x, mask);
    }
    for(size_t i = 0; i < 12; i++) {
        x = causal_resnet_block1d_forward(x, pad_list, mask, t_mish, step, 280 + i, "mid_block", t_emb_ones, resnet_mish_ones, res_w, conv_b, model);
        ggml_set_name(x, ("causal_resnet_mid_block_" + std::to_string(step) + "_layer_" + std::to_string(i)).c_str());
        
        for (size_t j = 0; j < 4; ++j) {
            tr_x = x;
            h = build_layer_norm(x, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm1_w, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm1_b, 1e-5f, "mid_block", 28 + i + j);
            ggml_set_name(h, ("causal_trans_mid_block_norm_1_" + std::to_string(step) + "_layer_" + std::to_string(i) + "_sub_layer_" + std::to_string(j)).c_str());
            attn_out = build_basic_attn(h, attn_mask, i * 4 + j, "mid_block", model);
            ggml_set_name(attn_out, ("causal_trans_mid_block_attn_" + std::to_string(step) + "_layer_" + std::to_string(i) + "_sub_layer_" + std::to_string(j)).c_str());
            tr_x = ggml_add(ctx0, attn_out, tr_x);
            h = build_layer_norm(tr_x, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm3_w, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm3_b, 1e-5f, "mid_block", 76 + i +j);
            // ff_out = ggml_mul_mat(ctx0,  mid_w0[i * 4 + j], h);
            // ff_out = ggml_add(ctx0, ff_out, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b0);
            ff_out = ggml_mul_mat_add(ctx0, mid_w0[i * 4 + j], h, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b0);
            ff_out = ggml_gelu_erf(ctx0, ff_out);
            // ff_out = ggml_mul_mat(ctx0, mid_w2[i * 4 + j], ff_out);
            // ff_out = ggml_add(ctx0, ff_out, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b2);
            ff_out = ggml_mul_mat_add(ctx0, mid_w2[i * 4 + j], ff_out, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b2);
            x = ggml_add(ctx0, ff_out, tr_x);
            // x = tr_x;
            ggml_set_name(x, ("causal_trans_mid_block_" + std::to_string(step) + "_layer_" + std::to_string(i) + "_sub_layer_" + std::to_string(j)).c_str());
        }
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    }
    ggml_set_name(x, ("causal_trans_mid_block_" + std::to_string(step)).c_str());
    x = ggml_concat(ctx0, x, hidden, 1);
    x = causal_resnet_block1d_forward(x, pad_list, mask, t_mish, step, 0, "up_block", t_emb_ones, resnet_mish_ones, res_w, conv_b, model);
    if (streaming) {
        attn_mask = apply_streaming_chunk_mask(attn_mask, x, mask);
    }
    for (size_t i = 0; i < 4; ++i) {
        tr_x = x;
        h = build_layer_norm(x, model.layers[1059 + i].up_block1_norm1_w, model.layers[1059 + i].up_block1_norm1_b, 1e-5f, "up_block", 124 + i);
        attn_out = build_basic_attn(h, attn_mask, 1059 + i, "up_block", model);
        tr_x = ggml_add(ctx0, attn_out, tr_x);
        h = build_layer_norm(tr_x, model.layers[1059 + i].up_block1_norm3_w, model.layers[1059 + i].up_block1_norm3_b, 1e-5f, "up_block", 128 + i);
        // ff_out = ggml_mul_mat(ctx0, up_w0[i], h);
        // ff_out = ggml_add(ctx0, ff_out, model.layers[1059 + i].up_block1_ffn_b0);
        ff_out = ggml_mul_mat_add(ctx0, up_w0[i], h, model.layers[1059 + i].up_block1_ffn_b0);
        ff_out = ggml_gelu_erf(ctx0, ff_out);
        // ff_out = ggml_mul_mat(ctx0, up_w2[i], ff_out);
        // ff_out = ggml_add(ctx0, ff_out, model.layers[1059 + i].up_block1_ffn_b2);
        ff_out = ggml_mul_mat_add(ctx0, up_w2[i], ff_out, model.layers[1059 + i].up_block1_ffn_b2);
        x = ggml_add(ctx0, ff_out, tr_x);
        // x = tr_x;
    }
    ggml_set_name(x, ("causal_trans_up_block_" + std::to_string(step)).c_str());
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    // x_mask = ggml_mul(ctx0, x, mask_up);
    if (x->ne[1] == 256) {
        x = causal_conv1d_forward(x, pad_list[0], mask, conv_b, "up_block", 0, 3, model, step);
    } else if (x->ne[1] == 320) {
        x = causal_conv1d_forward(x, pad_list[1], mask, conv_b, "up_block", 0, 3, model, step);
    } else {
        x = causal_conv1d_forward(x, pad_list[2], mask, conv_b, "up_block", 0, 3, model, step);
    }
    x = causal_block1d_forward(x, pad_list, mask, resnet_mish_ones, conv_b, "final_block", 0, 3, model, step);
    
    ggml_tensor * weight_t  = model.f_proj_w;
    ggml_tensor * weight_2d = ggml_reshape_2d(ctx0, weight_t, weight_t->ne[1], weight_t->ne[2]);
    ggml_tensor * res_perm = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_tensor * x_2d = ggml_reshape_2d(ctx0, res_perm, res_perm->ne[0], res_perm->ne[1] * res_perm->ne[2]);
    // ggml_tensor * res_mask = ggml_mul_mat(ctx0, weight_2d, x_2d);
    // res_mask = ggml_add(ctx0, res_mask, model.f_proj_b);
    ggml_tensor * res_mask = ggml_mul_mat_add(ctx0, weight_2d, x_2d, model.f_proj_b);
    res_mask = ggml_reshape_4d(ctx0, res_mask, weight_2d->ne[1], res_perm->ne[1], res_perm->ne[2], 1);
    x = ggml_cont(ctx0, ggml_permute(ctx0, res_mask, 1, 0, 2, 3));
    return x;
}

ggml_tensor * llm_graph_context::build_solve_euler(
         ggml_cgraph * gf,
         ggml_tensor * z,
         ggml_tensor * mu,
         ggml_tensor * mask,
         ggml_tensor * spks,
         ggml_tensor * cond,
         const llama_model & model,
         bool streaming) const{
        
    const int64_t B   = z->ne[2];
    const int64_t C   = z->ne[1]; 
    const int64_t T   = z->ne[0];
    const float PI = 3.14159265358979323846f;
    const int N_STEPS = 10; 
    const int64_t spk_dim = spks ? spks->ne[0] : 0;
    std::vector<float> t_span(N_STEPS + 1);
    for (int i = 0; i <= N_STEPS; ++i) {
        t_span[i] = 1.0f - cosf((float)i / N_STEPS * 0.5f * PI);
    }

    ggml_tensor * one = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    one = ggml_exp(ctx0, ggml_scale(ctx0, one, 0.0f));

    // 准备不变的输入
    ggml_tensor * mask_in = ggml_concat(ctx0, mask, mask, 2);
    
    ggml_tensor * mu_zero = ggml_scale(ctx0, mu, 0.0f);
    ggml_tensor * mu_in = ggml_concat(ctx0, mu, mu_zero, 2);
    ggml_tensor * spk_zero = ggml_sub(ctx0, spks, spks);
    ggml_tensor * spks_in = spks ? ggml_concat(ctx0, spks, spk_zero, 1) : nullptr;
    if (spks_in) ggml_set_name(spks_in, ("decoder_spks_in_" + std::to_string(1)).c_str());
    ggml_tensor * spks_t = nullptr;
    if (spks_in) {
        ggml_tensor * spks_reshaped = ggml_reshape_3d(ctx0, spks_in, 80, 1, 2);
        ggml_tensor * zeros = ggml_new_tensor_3d(ctx0, spks_in->type, 80, T, 2);
        zeros = ggml_scale(ctx0, zeros, 0.0f);
        ggml_tensor * spks_broadcasted = ggml_add(ctx0, zeros, spks_reshaped);
        ggml_tensor * spks_permuted = ggml_permute(ctx0, spks_broadcasted, 1, 0, 2, 3);
        spks_t = ggml_cont(ctx0, spks_permuted);
    }
    if (spks_t) ggml_set_name(spks_t, ("decoder_spks_t_" + std::to_string(1)).c_str());
    ggml_tensor * cond_in = nullptr;
    if (cond) {
        ggml_tensor * cond_zero = ggml_scale(ctx0, cond, 0.0f);
        cond_in = ggml_concat(ctx0, cond, cond_zero, 2);
    }
    if (cond_in) ggml_set_name(cond_in, ("decoder_cond_in_" + std::to_string(1)).c_str());
    
    ggml_tensor * z_current = z;
    float t_val = t_span[0];
    float dt = t_span[1] - t_span[0];

    ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, mask->ne[0], mask->ne[0], 1, 1);
    attn_mask = ggml_scale(ctx0, attn_mask, -0.0f);
    ggml_tensor * mask_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, GGML_PAD(mask->ne[0], 64) - mask->ne[0], mask->ne[0], 1, 1);
    mask_pad = ggml_scale(ctx0, mask_pad, 0.0f);
    mask_pad = ggml_scale(ctx0, ggml_exp(ctx0, mask_pad), -1e4f);
    attn_mask = ggml_concat(ctx0, attn_mask, mask_pad, 0);
    attn_mask = ggml_cast(ctx0, attn_mask, GGML_TYPE_F16);
    ggml_tensor * t_tmb_zeros = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1024, 2);
    t_tmb_zeros = ggml_scale(ctx0, t_tmb_zeros, 0.0f);
    ggml_tensor * t_tmb_ones = ggml_exp(ctx0, t_tmb_zeros);
    ggml_tensor * resnet_mish_zeros = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, z->ne[0], 256, 2);
    resnet_mish_zeros = ggml_scale(ctx0, resnet_mish_zeros, 0.0f);
    ggml_tensor * resnet_mish_ones = ggml_exp(ctx0, resnet_mish_zeros);

    std::vector<ggml_tensor *> pad_list;
    ggml_tensor * pad_256 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 256, 2);
    pad_256 = ggml_scale(ctx0, pad_256, 0.0f);
    ggml_tensor * pad_320 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 320, 2);
    pad_320 = ggml_scale(ctx0, pad_320, 0.0f);
    ggml_tensor * pad_512 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 512, 2);
    pad_512 = ggml_scale(ctx0, pad_512, 0.0f);
    pad_list = {pad_256, pad_320, pad_512};

    std::vector<ggml_tensor *> res_w;
    ggml_tensor * res_w_d_blk = ggml_reshape_2d(ctx0, model.down_blk_res_w, model.down_blk_res_w->ne[1], model.down_blk_res_w->ne[2]);
    res_w.push_back(res_w_d_blk);
    for (int i = 0; i < 12; i++) {
        ggml_tensor * res_w_m_blk = ggml_reshape_2d(ctx0, model.layers[280 + i].mid_block_res_w, model.layers[280 + i].mid_block_res_w->ne[1], model.layers[280 + i].mid_block_res_w->ne[2]);
        res_w.push_back(res_w_m_blk);
    }
    ggml_tensor * res_w_u_blk = ggml_reshape_2d(ctx0, model.up_blk_res_w, model.up_blk_res_w->ne[1], model.up_blk_res_w->ne[2]);
    res_w.push_back(res_w_u_blk);

    ConvBias conv_b;
    auto make_bias_3d = [&](ggml_tensor * b) -> ggml_tensor * {
        // int64_t total = ggml_nelements(b);
        return ggml_reshape_3d(ctx0, b, 
            1, b->ne[0], 1);
    };
    conv_b.d1 = make_bias_3d(model.down_blk1_conv_b); 
    conv_b.d2 = make_bias_3d(model.down_blk2_conv_b);
    conv_b.d3 = make_bias_3d(model.down_blk_conv_b);
    for (int i = 0; i < 12; i++) {
        conv_b.m1[i] = make_bias_3d(model.layers[280+i].mid_block1_b);
        conv_b.m2[i] = make_bias_3d(model.layers[280+i].mid_block2_b);
    }
    conv_b.u1 = make_bias_3d(model.up_blk1_conv_b);
    conv_b.u2 = make_bias_3d(model.up_blk2_conv_b);
    conv_b.u3 = make_bias_3d(model.up_blk_conv_b);
    conv_b.f1 = make_bias_3d(model.f_blk_conv_b);

    const int half_dim = 160;
    const float emb_div = 0.05792666900613952f;
    // float emb_div = std::log(10000.0f) / (half_dim - 1);
    ggml_tensor * idx = ggml_arange(ctx0, 0, half_dim, 1);
    // idx = ggml_cast(ctx0, idx, GGML_TYPE_F32);
    ggml_tensor * emb = ggml_scale(ctx0, idx, -emb_div);
    emb = ggml_exp(ctx0, emb);

    ggml_tensor * emb_row = ggml_reshape_2d(ctx0, emb, 1, half_dim);

    // =======================================================
    // [优化] FFN 权重 FP16 化
    // 目的：将 F32 权重转为 F16，减少显存带宽占用
    // 收益：FFN 部分提速，总耗时预计降低 30-50ms
    // =======================================================

    // 1. 转换 Down Blocks 的 FFN 权重
    std::vector<ggml_tensor *> down_w0_f16(4);
    std::vector<ggml_tensor *> down_w2_f16(4);
    for (int i = 0; i < 4; ++i) {
        down_w0_f16[i] = ggml_cast(ctx0, model.layers[226 + i].down_block1_ffn_w0, GGML_TYPE_F16);
        down_w2_f16[i] = ggml_cast(ctx0, model.layers[226 + i].down_block1_ffn_w2, GGML_TYPE_F16);
    }

    // 2. 转换 Mid Blocks 的 FFN 权重 (12 * 4 = 48 个)
    std::vector<ggml_tensor *> mid_w0_f16(48);
    std::vector<ggml_tensor *> mid_w2_f16(48);
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 4; ++j) {
            int idx = i * 4 + j;
            mid_w0_f16[idx] = ggml_cast(ctx0, model.mid_block_sub_layers[idx].mid_block1_ffn_w0, GGML_TYPE_F16);
            mid_w2_f16[idx] = ggml_cast(ctx0, model.mid_block_sub_layers[idx].mid_block1_ffn_w2, GGML_TYPE_F16);
        }
    }

    // 3. 转换 Up Blocks 的 FFN 权重
    std::vector<ggml_tensor *> up_w0_f16(4);
    std::vector<ggml_tensor *> up_w2_f16(4);
    for (int i = 0; i < 4; ++i) {
        up_w0_f16[i] = ggml_cast(ctx0, model.layers[1059 + i].up_block1_ffn_w0, GGML_TYPE_F16);
        up_w2_f16[i] = ggml_cast(ctx0, model.layers[1059 + i].up_block1_ffn_w2, GGML_TYPE_F16);
    }

    // =======================================================

    for (int64_t step = 1; step < 11; ++step) {
        ggml_tensor * t_current = ggml_scale(ctx0, one, t_val);
        ggml_tensor * t_in = ggml_concat(ctx0, t_current, t_current, 0);
        ggml_tensor * z_in = ggml_concat(ctx0, z_current, z_current, 2);
        ggml_tensor * dphi_dt = build_causal_cond_decoder(z_in, pad_list, mask_in, mu_in, t_in, spks_in, cond_in, spks_t, \
                        attn_mask, t_tmb_ones, resnet_mish_ones, res_w, conv_b, emb_row, down_w0_f16, down_w2_f16, mid_w0_f16, mid_w2_f16, up_w0_f16, up_w2_f16, model, step, streaming);
        ggml_tensor * dphi_dt_split   = ggml_view_3d(ctx0, dphi_dt, T, z->ne[1], B, dphi_dt->nb[1], dphi_dt->nb[2], 0);
        ggml_tensor * cfg_dphi_dt  = ggml_view_3d(ctx0, dphi_dt, T, z->ne[1], B, dphi_dt->nb[1], dphi_dt->nb[2], B * dphi_dt->nb[2]);
        ggml_tensor * dphi  = ggml_sub_inplace(ctx0, ggml_scale(ctx0, dphi_dt_split, 1.7f * dt), ggml_scale(ctx0, cfg_dphi_dt, 0.7f * dt));
        z_current = ggml_add(ctx0, z_current, dphi);
        // 更新循环变量（关键！）
        t_val = t_val + dt;
        if (step < N_STEPS) {
            dt = t_span[step + 1] - t_val;
        }
    }
    return z_current;

}

ggml_tensor * llm_graph_context::bulid_f0_predictor(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb) const {

    ggml_tensor * result = ggml_conv_1d(ctx0, mw, cur, 1, 1, 1);
    mb = ggml_reshape_3d(ctx0, mb, 1, mb->ne[0], 1);
    result = ggml_add(ctx0, result, mb);
    result = ggml_elu(ctx0, result);
    return result;
}

static void ggml_compute_snake(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,  // x
    const struct ggml_tensor * src1,  // alpha
    int ith, int nth, void * userdata) {
    
    (void) userdata;
    
    const int64_t ne0 = src0->ne[0];  // L (序列长度)
    const int64_t ne1 = src0->ne[1];  // C (通道数)
    const int64_t ne2 = src0->ne[2];  // B (batch)
    
    const float * x = (const float *) src0->data;
    const float * alpha = (const float *) src1->data;
    float * y = (float *) dst->data;
    
    // 获取 stride (字节转 float 索引)
    const int64_t nb0 = src0->nb[0] / sizeof(float);  // 应该是 1
    const int64_t nb1 = src0->nb[1] / sizeof(float);  // L 或其他
    const int64_t nb2 = src0->nb[2] / sizeof(float);
    
    const int64_t dnb0 = dst->nb[0] / sizeof(float);
    const int64_t dnb1 = dst->nb[1] / sizeof(float);
    const int64_t dnb2 = dst->nb[2] / sizeof(float);
    
    // 按通道并行
    const int64_t dr = (ne1 + nth - 1) / nth;
    const int64_t c0 = dr * ith;
    const int64_t c1 = std::min(c0 + dr, ne1);
    
    for (int64_t b = 0; b < ne2; b++) {
        for (int64_t c = c0; c < c1; c++) {
            const float a = alpha[c];
            const float a_inv = 1.0f / (a + 1e-9f);
            
            for (int64_t l = 0; l < ne0; l++) {
                const int64_t src_idx = b * nb2 + c * nb1 + l * nb0;
                const int64_t dst_idx = b * dnb2 + c * dnb1 + l * dnb0;
                
                const float xi = x[src_idx];
                const float ax = a * xi;
                const float sin_ax = sinf(ax);
                const float sin_sq = sin_ax * sin_ax;
                
                y[dst_idx] = xi + sin_sq * a_inv;
            }
        }
    }
}

ggml_tensor * llm_graph_context::build_snake(ggml_tensor * cur, ggml_tensor * alpha) const {
    // alpha 需要 reshape 为 [C] 1D
    ggml_tensor * alpha_1d = ggml_reshape_1d(ctx0, alpha, alpha->ne[0]);
    ggml_tensor * result = ggml_map_custom2(
        ctx0, cur, alpha_1d, ggml_compute_snake, GGML_N_TASKS_MAX, nullptr);
    
    return result;
}



static void custom_op_mod_1(
      struct ggml_tensor * dst,       // 输出张量
      const struct ggml_tensor * a,   // 输入张量
      int ith,                        // 当前线程 ID
      int nth,                        // 线程总数
      void * userdata)                // 用户数据 (这里不用)
{
// 1. 获取元素总数
    const int ne = ggml_nelements(dst);

    // 2. 计算当前线程负责的数据范围 (分片)
    // 简单的并行策略：每个线程处理一段连续的数据
    const int dr = (ne + nth - 1) / nth; 
    const int ie0 = dr * ith; 
    const int ie1 = std::min(ie0 + dr, ne);

    // 3. 获取数据指针 (假设是 F32 类型)
    const float * src_data = (const float *) a->data;
    float * dst_data = (float *) dst->data;

    // 4. 循环计算
    for (int i = ie0; i < ie1; i++) {
        // 核心逻辑: x % 1 = x - floor(x)
        dst_data[i] = src_data[i] - std::floor(src_data[i]);
    }
}

    

static void custom_op_rand_masked(
    struct ggml_tensor * dst,       
    const struct ggml_tensor * a,   
    int ith, int nth, void * userdata)                
{
    const int dim = dst->ne[0]; 
    const int ne = ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth; 
    const int ie0 = dr * ith; 
    const int ie1 = std::min(ie0 + dr, ne);

    float * dst_data = (float *) dst->data;
    
    // 每个线程独立的真随机生成器
    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = ie0; i < ie1; i++) {
        if (i % dim == 0) {
            dst_data[i] = 0.0f;
        } else {
            dst_data[i] = dist(rng);
        }
    }
}


static void custom_op_cumsum_ne1(
    struct ggml_tensor * dst,       
    const struct ggml_tensor * a,   
    int ith, int nth, void * userdata)                
{
    const int64_t ne0 = dst->ne[0];  // dim
    const int64_t ne1 = dst->ne[1];  // time
    const int64_t ne2 = dst->ne[2];  // batch

    const int64_t n_tasks = ne0 * ne2;
    const int64_t tasks_per_thread = (n_tasks + nth - 1) / nth; 
    const int64_t task_begin = tasks_per_thread * ith; 
    const int64_t task_end = std::min(task_begin + tasks_per_thread, n_tasks);

    const float * __restrict src = (const float *) a->data;
    float * __restrict dst_ptr = (float *) dst->data;

    for (int64_t task = task_begin; task < task_end; task++) {
        const int64_t i0 = task % ne0;  // dim index
        const int64_t i2 = task / ne0;  // batch index
        
        const int64_t offset = i2 * ne1 * ne0 + i0;
        
        // Kahan 求和 (double 精度)
        double sum = 0.0;
        double compensation = 0.0;
        
        for (int64_t i1 = 0; i1 < ne1; i1++) {
            const int64_t idx = offset + i1 * ne0;
            
            const double input = static_cast<double>(src[idx]);
            const double y = input - compensation;
            const double t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
            
            dst_ptr[idx] = static_cast<float>(sum);
        }
    }
}

static void custom_op_threshold_uv(
    struct ggml_tensor * dst,       
    const struct ggml_tensor * a,   
    int ith, int nth, void * userdata)                
{
    // 从 userdata 获取阈值 (float*)
    const float threshold = *(const float *)userdata;

    const int ne = ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth; 
    const int ie0 = dr * ith; 
    const int ie1 = std::min(ie0 + dr, ne);

    const float * src_data = (const float *) a->data;
    float * dst_data = (float *) dst->data;

    for (int i = ie0; i < ie1; i++) {
        // 核心逻辑: f0 > threshold -> 1.0, else -> 0.0
        dst_data[i] = (src_data[i] > threshold) ? 1.0f : 0.0f;
    }
}

// 自定义算子：标准正态分布 (Mean=0, Std=1)
static void custom_op_randn(
    struct ggml_tensor * dst,       
    const struct ggml_tensor * a,   // 这里的 a 只是为了提供 shape (randn_like)
    int ith, int nth, void * userdata)                
{
    const int ne = ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth; 
    const int ie0 = dr * ith; 
    const int ie1 = std::min(ie0 + dr, ne);

    float * dst_data = (float *) dst->data;

    // 线程局部 RNG
    std::mt19937 g_rng(1234);
    std::mt19937 local_rng(g_rng() + ith);
    // 使用 std::normal_distribution 生成高斯噪声
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = ie0; i < ie1; i++) {
        dst_data[i] = dist(local_rng);
    }
}

static void custom_op_randn_broadcast_mul(
    struct ggml_tensor * dst,
    const struct ggml_tensor * sines,     // [9, time, batch] - 只用于获取 shape
    const struct ggml_tensor * noise_amp, // [1, time, batch] - 广播源
    int ith, int nth, void * userdata)
{
    const int64_t ne0 = sines->ne[0];  // 9
    const int64_t ne1 = sines->ne[1];  // time
    const int64_t ne2 = sines->ne[2];  // batch

    const int64_t total = ne0 * ne1 * ne2;
    const int64_t dr = (total + nth - 1) / nth;
    const int64_t ie0 = dr * ith;
    const int64_t ie1 = std::min(ie0 + dr, total);

    const float * amp = (const float *) noise_amp->data;
    float * out = (float *) dst->data;

    thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int64_t i = ie0; i < ie1; i++) {
        // noise_amp 广播: 忽略 dim 维度
        int64_t i1 = (i / ne0) % ne1;  // time index
        int64_t i2 = i / (ne0 * ne1);  // batch index
        
        out[i] = dist(rng) * amp[i1 + i2 * ne1];
    }
}

// rad_values[:, 0, :] += rand_ini
// rand_ini[:, 0] = 0, 其他位置是随机数
static void custom_op_add_rand_first_time(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src,  // rad_values [dim=9, time, batch=1]
    int ith, int nth, void * userdata)
{
    if (ith != 0) return;  // 单线程，保证随机数一致

    const int64_t dim = dst->ne[0];   // 9
    const int64_t time = dst->ne[1];  // 421440
    const int64_t batch = dst->ne[2]; // 1

    const float * src_data = (const float *)src->data;
    float * dst_data = (float *)dst->data;

    // 1. 生成 rand_ini [dim]
    //    rand_ini[0] = 0 (masked)
    //    rand_ini[1..8] = random
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float rand_ini[9];
    rand_ini[0] = 0.0f;  // 第一个 dim 设为 0
    for (int i = 1; i < dim; i++) {
        rand_ini[i] = dist(rng);
    }

    // 2. 复制所有数据
    const int64_t total = dim * time * batch;
    memcpy(dst_data, src_data, total * sizeof(float));

    // 3. 只修改 time=0 的位置
    //    rad_values[d, 0, b] += rand_ini[d]
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            int64_t idx = d + 0 * dim + b * dim * time;  // time=0
            dst_data[idx] += rand_ini[d];
        }
    }
}


ggml_tensor * llm_graph_context::build_m_source(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb) const {
    
    ggml_tensor * arrange_tensor = ggml_arange(ctx0, 1.0f, 10.0f, 1.0f);
    arrange_tensor = ggml_reshape_2d(ctx0, arrange_tensor, 1, 9);
    ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, 1, cur->ne[1]);
    ggml_tensor * fn_res = ggml_mul_mat(ctx0, arrange_tensor, cur_2d);
    fn_res = ggml_reshape_3d(ctx0, fn_res, 9, cur->ne[1], 1);
    ggml_set_name(fn_res, "m_source_fn");
    //_f02sine函数
    ggml_tensor * rad_values = ggml_scale(ctx0, fn_res, 1.0f / 24000.0f);
    rad_values = ggml_map_custom1(ctx0, rad_values, custom_op_mod_1, 1, NULL);
    ggml_set_name(rad_values, "m_source_rad_values");
    
    const int dim = rad_values->ne[0];
    const int time = rad_values->ne[1];
    const int batch = rad_values->ne[2];

    rad_values = ggml_map_custom1(ctx0, rad_values, custom_op_add_rand_first_time, 1, NULL);
    ggml_set_name(rad_values, "m_source_rad_values_rand_ini");


    ggml_tensor * rad_values_downsampled = ggml_interpolate(ctx0, rad_values, dim, time / 480, batch, 1, 1);
    struct ggml_tensor * phase = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, rad_values_downsampled->ne[0], rad_values_downsampled->ne[1], rad_values_downsampled->ne[2]);
    phase = ggml_map_custom1(ctx0, rad_values_downsampled, custom_op_cumsum_ne1, GGML_N_TASKS_MAX, NULL);
    // phase = ggml_cumsum(ctx0, rad_values_downsampled);
    phase = ggml_scale(ctx0, phase, 2.0f * M_PI * 480);
    ggml_tensor * phase_scaled = ggml_interpolate(ctx0, ggml_cont(ctx0, phase), phase->ne[0], phase->ne[1] * 480, phase->ne[2], 1, 1);
    ggml_set_name(phase_scaled, "m_source_phase_scaled");
    ggml_tensor * sines = ggml_sin(ctx0, phase_scaled);
    ggml_set_name(sines, "m_source_sines");
    sines = ggml_scale(ctx0, sines, 0.1f);
    
    //_f02uv
    float threshold_val = 10.0f;
    ggml_tensor * uv = ggml_map_custom1(ctx0, cur, custom_op_threshold_uv, GGML_N_TASKS_MAX, &threshold_val);
    ggml_set_name(uv, "m_source_uv");
    ggml_tensor * zeros = ggml_scale(ctx0, uv, 0.0f);
    ggml_tensor * ones = ggml_exp(ctx0, zeros);

    ggml_tensor * term1 = ggml_scale(ctx0, uv, 0.03f);
    ggml_tensor * term2 = ggml_sub(ctx0, ones, uv);
    term2 = ggml_scale(ctx0, term2, 0.1f / 3.0f);
    ggml_tensor * noise_amp = ggml_add(ctx0, term1, term2);
    ggml_set_name(noise_amp, "m_source_noise_amp");

    ggml_tensor * noise = ggml_map_custom2(ctx0, sines, noise_amp,
                                        custom_op_randn_broadcast_mul, GGML_N_TASKS_MAX, NULL);
    ggml_set_name(noise, "m_source_noise");

    ggml_tensor * sine_waves = ggml_mul(ctx0, sines, uv);
    sine_waves = ggml_add(ctx0, sine_waves, noise);
    ggml_set_name(sine_waves, "m_source_sine_waves");
    ggml_tensor * sine_wavs = ggml_mul_mat(ctx0, mw, sine_waves);
    sine_wavs = ggml_add(ctx0, sine_wavs, mb);

    ggml_tensor * sine_merge = ggml_tanh(ctx0, sine_wavs);
    ggml_set_name(sine_merge, "sine_merge");
    return sine_merge;
}

// ggml_map_custom1
ggml_tensor * llm_graph_context::build_res_blk(
      ggml_tensor * cur,
      int kernel_size,
      int32_t idx,
      const llama_model & model) const {
    
    ggml_tensor * res_cur = cur;
    int64_t expected_len = res_cur->ne[0];
    for(int j = 0; j < 3; j++) {
        ggml_tensor * convs1_mw = model.resblk_sub_layer[idx * 3 + j].resblock_conv1_w;
        ggml_tensor * convs1_mb = model.resblk_sub_layer[idx * 3 + j].resblock_conv1_b;
        ggml_tensor * convs2_mw = model.resblk_sub_layer[idx * 3 + j].resblock_conv2_w;
        ggml_tensor * convs2_mb = model.resblk_sub_layer[idx * 3 + j].resblock_conv2_b;
        ggml_tensor * act1 = model.resblk_sub_layer[idx * 3 + j].resblock_act1;
        ggml_tensor * act2 = model.resblk_sub_layer[idx * 3 + j].resblock_act2;
        int dilation = (j == 0) ? 1 : ((j == 1) ? 3 : 5);
        int padding = (dilation * (kernel_size - 1)) / 2;
        int padding_plain = (1 * (kernel_size - 1)) / 2;
        //-------act1------
        ggml_tensor * act1_res = build_snake(res_cur, act1);
        //-----convs1------
        ggml_tensor * si_res_convs1 = ggml_conv_1d(ctx0, convs1_mw, act1_res, 1, padding, dilation);
        if (si_res_convs1->ne[0] > expected_len) {
            int64_t diff = si_res_convs1->ne[0] - expected_len;
            int64_t offset_idx = diff / 2;
            si_res_convs1 = ggml_view_2d(ctx0, si_res_convs1,
                 expected_len,
                 si_res_convs1->ne[1],
                 si_res_convs1->nb[1],
                 offset_idx * ggml_element_size(si_res_convs1)); // 关键：设置内存偏移
        }
        ggml_tensor * b1_reshaped = ggml_reshape_3d(ctx0, convs1_mb, 1, convs1_mb->ne[0], 1);
        si_res_convs1 = ggml_add(ctx0, si_res_convs1, b1_reshaped);

        //------act2-------
        ggml_tensor * act2_res = build_snake(si_res_convs1, act2);
        //-----convs2-----
        ggml_tensor * si_res_convs2 = ggml_conv_1d(ctx0, convs2_mw, act2_res, 1, padding_plain, 1);
        if (si_res_convs2->ne[0] > expected_len) {
            int64_t diff = si_res_convs2->ne[0] - expected_len;
            int64_t offset_idx = diff / 2;
            
            si_res_convs2 = ggml_view_2d(ctx0, si_res_convs2,
                expected_len,
                si_res_convs2->ne[1],
                si_res_convs2->nb[1],
                offset_idx * ggml_element_size(si_res_convs2));
        }
        ggml_tensor * b2_reshaped = ggml_reshape_3d(ctx0, convs2_mb, 1, convs2_mb->ne[0], 1);
        si_res_convs2 = ggml_add(ctx0, si_res_convs2, b2_reshaped);
        res_cur = ggml_add(ctx0, si_res_convs2, res_cur);
    }

    return res_cur;

}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd = hparams.n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>();

    ggml_tensor * cur = nullptr;
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_tokens = inp->tokens;

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b,
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }
    } else {
        if(arch == LLM_ARCH_COSYVOICEFLOW) {
            inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.spk_embed_dim, 1);
        } 
        else if (arch == LLM_ARCH_COSYVOICEHIFT){
            inp->embd = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ubatch.n_tokens, 80, 1);
        } else {
            inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        }
        // ggml_backend_t backend_cuda = ggml_backend_cuda_init(0);
        // ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend_cuda);
        ggml_set_input(inp->embd);
        // if(hparams.use_flow) {
        //     ggml_backend_tensor_set(inp->embd, ubatch.embd, 0, hparams.spk_embed_dim*ggml_element_size(inp->embd));
        // } else {
        //     ggml_backend_tensor_set(inp->embd, ubatch.embd, 0, ubatch.n_tokens*n_embd*ggml_element_size(inp->embd));
        // }
        cur = inp->embd;
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "inp_embd", -1);
    ggml_set_name(cur, "inp_embd");
    res->add_input(std::move(inp));
    return cur;
}

ggml_tensor * llm_graph_context::build_inp_token() const {

    const uint32_t token_len = ubatch.token_len + ubatch.prompt_token_len;
    
    auto inp = std::make_unique<llm_graph_input_token>();

    ggml_tensor * cur = nullptr;
    inp->input_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_len);
    ggml_set_input(inp->input_token);
    cur = inp->input_token;
    ggml_set_name(cur, "inp_token");
    cb(cur, "inp_token", -1);
    res->add_input(std::move(inp));
    return cur;
}

ggml_tensor * llm_graph_context::build_inp_prompt_token() const {

    const uint32_t token_len = ubatch.prompt_token_len;
    auto inp = std::make_unique<llm_graph_input_prompt_token>();

    ggml_tensor * cur = nullptr;
    inp->input_prompt_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_len);
    // ggml_backend_t backend_cuda = ggml_backend_cuda_init(0);
    // ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend_cpu);
    ggml_set_input(inp->input_prompt_token);
    // ggml_backend_tensor_set(inp->input_prompt_token, ubatch.flow_token + ubatch.token_len, 0, token_len * ggml_element_size(inp->input_prompt_token));
    cur = inp->input_prompt_token;
    cb(cur, "inp_prompt_token", -1);
    res->add_input(std::move(inp));
    return cur;
}

ggml_tensor * llm_graph_context::build_inp_prompt_feat() const {

    const uint32_t feat_len = ubatch.prompt_feat_len / 80;
    
    auto inp = std::make_unique<llm_graph_input_prompt_feat>();
    ggml_tensor * cur = nullptr;
    inp->input_prompt_feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.output_size, feat_len);
    // ggml_backend_t backend_cuda = ggml_backend_cuda_init(0);
    // ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend_cuda);
    ggml_set_input(inp->input_prompt_feat);
    // ggml_backend_tensor_set(inp->input_prompt_feat, ubatch.flow_feat, 0, feat_len * ggml_element_size(inp->input_prompt_feat));
    cur = inp->input_prompt_feat;
    ggml_set_name(cur, "prompt_feat");
    res->add_input(std::move(inp));
    return cur;
}


ggml_tensor * llm_graph_context::build_inp_extend_pe() const {

    const uint32_t pe_len = 9999 * 512;
    
    auto inp = std::make_unique<llm_graph_input_extend_pe>();
    ggml_tensor * cur = nullptr;
    inp->input_extend_pe = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 9999);
    ggml_set_input(inp->input_extend_pe);
    cur = inp->input_extend_pe;
    ggml_set_name(cur, "extend_pe");
    res->add_input(std::move(inp));
    return cur;
}

ggml_tensor * llm_graph_context::build_inp_rand_noise() const {

    
    auto inp = std::make_unique<llm_graph_input_rand_noise>();
    ggml_tensor * cur = nullptr;
    inp->input_rand_noise = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 50 * 300, 80, 1);
    ggml_set_input(inp->input_rand_noise);
    cur = inp->input_rand_noise;
    cb(cur, "inp_rand_noise", -1);
    res->add_input(std::move(inp));
    return cur;
}

int32_t llm_graph_context::build_stream() const {

    const int32_t stream = ubatch.stream;
    auto inp = std::make_unique<llm_graph_input_stream>();
    inp->input_stream = stream;
    res->add_input(std::move(inp));
    return stream;
}

ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        scale_w,
        w_scale,
        gating_op,
        il,
        probs_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    if (up_exps_b) {
        up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
        cb(up, "ffn_moe_up_biased", il);
    }

    ggml_tensor * experts = nullptr;
    if (gate_exps) {
        cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(cur, "ffn_moe_gate", il);
    } else {
        cur = up;
    }

    if (gate_exps_b) {
        cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
        cb(cur, "ffn_moe_gate_biased", il);
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate_exps) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (gate_exps) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (gate_exps) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}


ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interere with
    //       features that require constant topology such as pipline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;
    if (cparams.flash_attn && kq_b == nullptr) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is ineffient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);
    
    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;
    
    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const bool is_swa = hparams.is_swa(il);

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);
    
    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = mctx_cur->get_base()->get_n_kv();

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        const auto n_kv = mctx_cur->get_swa()->get_n_kv();

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_rs - n_seqs), (rs_head + n_seqs)*state_size*ggml_element_size(s))));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || dense_2 == nullptr || dense_3 == nullptr) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    cur = ggml_mul_mat(ctx0, dense_2, cur);
    cur = ggml_mul_mat(ctx0, dense_3, cur);
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_tanh(ctx0, cur);
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
