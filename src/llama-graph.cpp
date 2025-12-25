#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache-unified.h"
#include "llama-kv-cache-unified-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "ggml-cuda.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include "llama-model.h"
#include <random>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <stdio.h>
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
            // LLAMA_LOG_INFO("-------------------------------------- %s:  n_embd is: %d\n", __func__, n_embd);
            ggml_backend_tensor_set(embd, ubatch->embd, 0, n_embd*ggml_element_size(embd));
            LLAMA_LOG_INFO("-------------------------------------- ggml_backend_tensor_set finished!!!!\n");
        }
    }
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
        // LLAMA_LOG_INFO("&&&&&&&&&&&&& check input_prompt_feat\n");
        // float buff[8];
        // ggml_backend_tensor_get(input_prompt_feat, buff, 0, sizeof(buff));   // 直接拷 32 字节
        // for (int i = 0; i < 8; ++i) printf("input_prompt_feat is: %f ", buff[i]);
    }
}

void llm_graph_input_rand_noise::set_input(const llama_ubatch * ubatch) {
    if(ubatch->rand_noise) {
        const int64_t rand_noise_len = 80 * 50 * 300;

        ggml_backend_tensor_set(input_rand_noise, ubatch->rand_noise, 0, rand_noise_len*ggml_element_size(input_rand_noise));
        // float buff[rand_noise_len];
        // ggml_backend_tensor_get(input_rand_noise, buff, 0, sizeof(buff));
        // std::ofstream file("rand_noise_data.txt");
        // if (file.is_open()) {
        //     file << std::fixed << std::setprecision(11); 
        //     for (int i = 0; i < rand_noise_len; ++i)  {
        //         file << buff[i] << " ";
        //         if ((i + 1) % 10 == 0) {  // 每10个数换行，便于阅读
        //             file << "\n";
        //         }
        //     }
        //     file.close();
        //     printf("Data written to rand_noise_data.txt\n");
        // } else {
        //     printf("Failed to open file\n");
        // }
    }
}

void llm_graph_input_extend_pe::set_input(const llama_ubatch * ubatch) {
    if(ubatch->extend_pe) {
        const int64_t extend_pe_len = 9999 * 512;

        ggml_backend_tensor_set(input_extend_pe, ubatch->extend_pe, 0, extend_pe_len*ggml_element_size(input_extend_pe));
        // LLAMA_LOG_INFO("&&&&&&&&&&&&& check input_rand_noise\n");
    }
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

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + 1.0f) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

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
    const int64_t n_seq_tokens = ubatch->n_seq_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
            cparams.pooling_type == LLAMA_POOLING_TYPE_CLS ||
            cparams.pooling_type == LLAMA_POOLING_TYPE_RANK
        )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                data[seq_idx] = i;
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> last_pos(n_seqs_unq, -1);
        std::vector<int> last_row(n_seqs_unq, -1);

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (pos >= last_pos[seq_idx]) {
                    last_pos[seq_idx] = pos;
                    last_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (last_row[s] >= 0) {
                data[s] = last_row[s];
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

    GGML_ASSERT(kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));

    float * data = (float *) kq_mask->data;

    for (int h = 0; h < 1; ++h) {
        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i0]; ++s) {
                    const llama_seq_id s0 = ubatch->seq_id[i0][0];

                    // TODO: reimplement this like in llama_kv_cache_unified
                    if (s0 == s1 && (!cparams.causal_attn || ubatch->pos[i0] <= ubatch->pos[i1])) {
                        if (hparams.use_alibi) {
                            f = -std::abs(ubatch->pos[i0] - ubatch->pos[i1]);
                        } else {
                            f = 0.0f;
                        }
                        break;
                    }
                }

                data[h*(n_kv*n_tokens) + i1*n_kv + i0] = f;
            }
        }
    }
}

void llm_graph_input_attn_kv_unified::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

void llm_graph_input_attn_kv_unified_iswa::set_input(const llama_ubatch * ubatch) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

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

        for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
            for (int j = 0; j < n_enc; ++j) {
                data[h*(n_enc*n_tokens) + i*n_enc + j] = -INFINITY;
            }
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    inp_attn->set_input(ubatch);
    inp_rs->set_input(ubatch);
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
    ctx0             (params.ctx),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    cb_func          (params.cb),
    res              (std::make_unique<llm_graph_result>()) {
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

static void sinusoidal_emb_op(struct ggml_tensor * dst, const struct ggml_tensor * src,
                              int ith, int nth, void * userdata) {
    auto * p = (sinusoidal_params *)userdata;
    const int half_dim = p->dim / 2;
    const float emb_div = std::log(10000.0f) / (half_dim - 1);
    
    const float * x = (const float *)src->data;
    float * out = (float *)dst->data;
    const int64_t n = src->ne[0];
    
    for (int64_t i = ith; i < n; i += nth) {
        for (int j = 0; j < half_dim; j++) {
            float emb = std::exp(-emb_div * j);
            float val = x[i] * p->scale * emb;
            out[i * p->dim + j] = std::sin(val);
            out[i * p->dim + j + half_dim] = std::cos(val);
        }
    }
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
    mask = ggml_scale(ctx0, ggml_exp(ctx0, ggml_scale(ctx0, mask, 0.0f)), 1.0f);
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
    
    cur = ggml_mul_mat(ctx0, linear_mw, cur);
    cur = ggml_add(ctx0, cur, linear_mb);
    ggml_set_name(cur, ("1st_mul_mat" + std::to_string(il)).c_str());
    cur = build_layer_norm(cur, norm_mw, norm_mb, 1e-05, "sub_sampling", il);
    // cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    ggml_set_name(cur, ("embed_linear_no_sub_sample_" + std::to_string(il)).c_str());

    return cur;

}


ggml_tensor * llm_graph_context::build_pos_encoding(
         ggml_tensor * cur,
         size_t offset, 
         size_t size, 
         size_t il) const{
    
    int64_t rows = cur->ne[1], cols = cur->ne[0];
    int64_t start = std::max((int64_t)0, rows/2 - (int64_t)size - (int64_t)offset + 1);
    int64_t end = std::min(rows, rows/2 + (int64_t)size + (int64_t)offset);
    int64_t new_len = end - start;
    ggml_tensor * pos_emb = ggml_view_2d(ctx0, cur, cols, end - start, cur->nb[1], start * cur->nb[1]);
    GGML_ASSERT(new_len > 0);
    ggml_set_name(pos_emb, ("pos_emb_" + std::to_string(il)).c_str());
    return pos_emb;
}

ggml_tensor * llm_graph_context::build_espnet_pos_encode(
         ggml_tensor * cur,
         int32_t il) const{
    cur = ggml_scale(ctx0, cur, std::sqrt(512.0f));
    ggml_set_name(cur, ("espnet_pos_encode" + std::to_string(il)).c_str());
    cb(cur, "espnet_pos_encode", il);
    return cur;
}

ggml_tensor * llm_graph_context::flip_weight(ggml_cgraph * gf, ggml_tensor * conv1_mw) const {
    ggml_tensor * conv1_mw_new = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 4, 512, 512);
    for (int64_t k = 0; k < 4; k++) {
        ggml_tensor * src_slice = ggml_view_3d(ctx0, conv1_mw,
            1, 512, 512,                          // shape: [1, 512, 512]
            conv1_mw->nb[0],             // stride
            conv1_mw->nb[1],
            k * conv1_mw->nb[0]          // offset
        );
        ggml_tensor * dst_slice = ggml_view_3d(ctx0, conv1_mw_new,
            1, 512, 512,
            conv1_mw_new->nb[0],
            conv1_mw_new->nb[1],
            (3 - k) * conv1_mw_new->nb[0]
        );
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src_slice, dst_slice));
    }
}

ggml_tensor * llm_graph_context::build_pre_lookahead_layer(
         ggml_tensor * cur,
         ggml_tensor * conv1_mw,
         ggml_tensor * conv1_mb,
         ggml_tensor * conv2_mw,
         ggml_tensor * conv2_mb) const{
        
    const int lookahead = 3;
    ggml_tensor * x = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));  // [B, C, T]
    x = ggml_reshape_4d(ctx0, x, x->ne[0], x->ne[1], 1, 1);
    x = ggml_pad(ctx0, x, lookahead, 0, 0, 0);
    ggml_set_name(x, "x_pad");
    ggml_tensor * outputs = ggml_conv_1d(ctx0, conv1_mw, x, 1, 0, 1);
    conv1_mb = ggml_reshape_4d(ctx0, ggml_cont(ctx0, conv1_mb), 1, 512, 1, 1);
    outputs = ggml_add(ctx0, outputs, conv1_mb);
    ggml_set_name(outputs, "x_conv_1d");
    outputs = ggml_leaky_relu(ctx0, outputs, 0.01f, false);
    ggml_set_name(outputs, "leaky_relu");
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 2, outputs->ne[1], outputs->ne[2], outputs->ne[3]);
    zeros = ggml_scale(ctx0, zeros, 0.0f);
    outputs = ggml_concat(ctx0, zeros, outputs, 0);
    outputs = ggml_cont(ctx0, outputs);
    ggml_set_name(outputs, "x_pad_2");
    outputs = ggml_conv_1d(ctx0, conv2_mw, outputs, 1, 0, 1);
    conv2_mb = ggml_reshape_4d(ctx0, ggml_cont(ctx0, conv2_mb), 1, 512, 1, 1);
    outputs  = ggml_add(ctx0, outputs, conv2_mb);
    ggml_set_name(outputs, "x_conv_1d_2");
    outputs = ggml_permute(ctx0, outputs, 1, 0, 2, 3);
    outputs = ggml_cont(ctx0, outputs);
    outputs = ggml_add(ctx0, outputs, cur);

    ggml_set_name(outputs, "pre_look_res");

    return outputs;
        
}

ggml_tensor * llm_graph_context::build_rel_shift(
         ggml_cgraph * gf,
         ggml_tensor * cur) const{
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& rel_shift cur shape is: {%d, %d, %d, %d}\n", cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]); 
    // cur = ggml_permute(ctx0, cur, 1,0,2,3);
    const int B = cur->ne[3];   
    const int n_head = cur->ne[2]; 
    const int T = cur->ne[1];    
    const int L = cur->ne[0];

    auto zero_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, T, n_head, B);
    zero_pad = ggml_scale(ctx0, zero_pad, 0.0f);  
    ggml_tensor * x_padded = ggml_concat(ctx0, zero_pad, cur, 0);
    // ggml_tensor * dest_view = ggml_view_4d(ctx0, x_padded, L, T, n_head, B, );
    auto reshaped = ggml_reshape_4d(ctx0, ggml_cont(ctx0, x_padded), T, L + 1, n_head, B);
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
    result = ggml_cont(ctx0, result);
    return result;
        
}

// ===================== Encoder 层统一结构 =====================
ggml_tensor * llm_graph_context::build_qkv_proj(
        ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
    
    int B = x->ne[2], T = x->ne[1];
    ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, x), x->ne[0], B * T);
    ggml_tensor * t = ggml_add(ctx0, ggml_mul_mat(ctx0, w, flat), b);
    t = ggml_reshape_4d(ctx0, ggml_cont(ctx0, t), 64, 8, T, B);
    return ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));
}

ggml_tensor * llm_graph_context::build_encoder_layer(
        ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * pos_emb,
        ggml_tensor * mask, const enc_layer_weights & W, int il) const {
    
    // Self-attention with residual
    ggml_tensor * res = ggml_dup(ctx0, x);
    ggml_build_forward_expand(gf, res);
    
    x = build_layer_norm(x, W.norm_mha_w, W.norm_mha_b, 1e-12f, "enc", il);
    
    ggml_tensor * q = build_qkv_proj(x, W.wq, W.bq);
    ggml_tensor * k = build_qkv_proj(x, W.wk, W.bk);
    ggml_tensor * v = build_qkv_proj(x, W.wv, W.bv);
    
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    
    // Position projection
    ggml_tensor * p = ggml_cont(ctx0, ggml_mul_mat(ctx0, W.wpos, pos_emb));
    p = ggml_reshape_4d(ctx0, p, 64, 8, p->ne[1], p->ne[2]);
    p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));
    // Relative position attention
    auto add_bias = [&](ggml_tensor * bias) {
        return ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, q, bias), 0, 2, 1, 3));
    };
    
    ggml_tensor * q_u = add_bias(W.pos_bias_u);
    ggml_tensor * q_v = add_bias(W.pos_bias_v);
    ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, k, q_u);
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_v);
    matrix_bd = build_rel_shift(gf, matrix_bd);
    ggml_tensor * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
    scores = ggml_scale(ctx0, scores, 1.0f / 8.0f);  // sqrt(64) = 8
    
    x = build_attn_scores(v, scores, mask, W.wo, W.bo, "enc", il);
    x = ggml_add(ctx0, res, x);
    
    // FFN with residual
    res = ggml_dup(ctx0, x);
    ggml_build_forward_expand(gf, res);
    
    x = build_layer_norm(x, W.norm_ffn_w, W.norm_ffn_b, 1e-12f, "enc", il);
    x = build_pos_ffn(x, W.ffn_w1, W.ffn_b1, W.ffn_w2, W.ffn_b2);
    
    return ggml_add(ctx0, res, x);
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
    ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cur), cur->ne[0], n_batch * cur->ne[1]);
    ggml_tensor * t = ggml_mul_mat(ctx0, mw, flat);
    t = ggml_add(ctx0, t, mb);
    t = ggml_reshape_4d(ctx0, ggml_cont(ctx0, t), d_k, heads, cur->ne[1], n_batch);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));

    return t;
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

    if (mask && mask->ne[2] > 0) {
        // ggml_tensor * mask_ext = ggml_reshape_4d(ctx0, mask, mask->ne[0], 1, 1, 1);
        ggml_tensor * mask_ext = ggml_scale(ctx0, ggml_cont(ctx0, mask), 0.0f);
    }
    ggml_tensor * attn = ggml_soft_max(ctx0, scores);
    cb(attn, ("soft_max_attn_" + attn_type).c_str(), il);
    // ggml_tensor * attn_T = ggml_cont(ctx0, ggml_permute(ctx0, attn, 1, 0, 2, 3));
    ggml_tensor * cur_T = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    ggml_tensor * x = ggml_mul_mat(ctx0, cur_T, attn);
    cb(x, ("attn_x_" + attn_type).c_str(), il);
    x = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)), time1 * d_k, n_head, B);
    cb(x, ("reshape_x_" + attn_type).c_str(), il);
    // mw = ggml_cont(ctx0, ggml_transpose(ctx0, mw));
    x = ggml_mul_mat(ctx0, mw, x);
    x = ggml_add(ctx0, x, mb);
    cb(x, ("linear_out_x_" + attn_type).c_str(), il);
    return x;
}


ggml_tensor * llm_graph_context::build_pos_ffn(
             ggml_tensor * cur,
             ggml_tensor * mw_1,
             ggml_tensor * mb_1,
             ggml_tensor * mw_2,
             ggml_tensor * mb_2) const{
    
    cur = ggml_mul_mat(ctx0, mw_1, cur);
    cur = ggml_add(ctx0, cur, mb_1);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, mw_2, cur);
    cur = ggml_add(ctx0, cur, mb_2);
    return cur;
}

ggml_tensor * llm_graph_context::build_upsample_1d(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb) const
{
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& upsample_1d cur shape is: {%d, %d, %d, %d}\n", cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    ggml_tensor * up = ggml_interpolate(ctx0, cur, cur->ne[0], cur->ne[1] * 2, cur->ne[2], cur->ne[3], 0);
    // ggml_tensor * up_reshaped = ggml_cont(ctx0, ggml_permute(ctx0, up, up->ne[1], up->ne[0], cur->ne[2], cur->ne[3]));
    cb(up, "interpolate", -1);
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, up->ne[0], 4, up->ne[2], up->ne[3]);
    zeros = ggml_scale(ctx0, zeros, 0.0f);
    ggml_tensor * pad = ggml_concat(ctx0, zeros, up, 1);
    cb(pad, "upsample_pad", -1);
    pad = ggml_cont(ctx0, ggml_permute(ctx0, pad, 1, 0, 2, 3));
    ggml_tensor * out = ggml_conv_1d(ctx0, mw, pad, 1, 0, 1);
    mb = ggml_reshape_3d(ctx0, ggml_cont(ctx0, mb), 1, cur->ne[0], 1);
    out = ggml_add(ctx0, out, mb);
    cb(out, "upsample_conv_1d", -1);
    return out;
}



// ==================== decoder 自定义算子 ====================
// Mish 激活函数: x * tanh(softplus(x))
static void custom_mish_final(struct ggml_tensor * dst, const struct ggml_tensor * src, 
                              int ith, int nth, void * userdata) {
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src->type == GGML_TYPE_F32); // 强制要求FP32输入
    
    const int64_t n = ggml_nelements(src);
    const float * s = (const float *)src->data;
    float * d = (float *)dst->data;
    for (int64_t i = ith; i < n; i += nth) {
        float x = s[i];
        
        // 使用PyTorch同款边界阈值
        if (x > 20.0f) {
            d[i] = x;  // mish(x) ≈ x
        } else if (x < -20.0f) {
            d[i] = 0.0f; // mish(x) ≈ 0（更精确）
        } else {
            float sp = log1pf(expf(x));  // 使用log1p提升精度
            float tanh_sp = tanhf(sp);
            d[i] = x * tanh_sp;
        }
    }
}

// mask_to_bias 自定义算子
static void custom_mask_to_bias(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    mask_to_bias_params * p = (mask_to_bias_params *)userdata;
    const int64_t n = ggml_nelements(src);
    const float * s = (const float *)src->data;
    float * d = (float *)dst->data;
    
    for (int64_t i = ith; i < n; i += nth) {
        d[i] = (1.0f - s[i]) * p->neg_val;
    }
}


// sinusoidal_pos_emb 自定义算子
static void custom_sinusoidal_pos_emb(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    
    const int dim = 320;
    const int scale = 1000;
    
    // sinusoidal_params * p = (sinusoidal_params *)userdata;
    const int half_dim = dim / 2;
    const float emb_div = logf(10000.0f) / (half_dim - 1);
    const int64_t n_tokens = src->ne[0];
    const float * t_data = (const float *)src->data;
    float * out = (float *)dst->data;
    
    for (int64_t t = ith; t < n_tokens; t += nth) {
        float scaled_t = t_data[t] * scale;
        // printf("&&&&&&&&&& custom_sinusoidal_pos_emb val is: %f\n", t_data[t]);
        for (int i = 0; i < half_dim; i++) {
            float freq = expf(-i * emb_div);
            float val = scaled_t * freq;
            out[t * dim + i] = sinf(val);
            out[t * dim + half_dim + i] = cosf(val);
        }
    }
}

// t_span 生成自定义算子
static void custom_t_span_init(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    (void)src; (void)userdata;
    const int64_t n = ggml_nelements(dst);
    float * d = (float *)dst->data;
    const float PI = 3.14159265358979323846f;
    
    for (int64_t i = ith; i < n; i += nth) {
        float t = (float)i / (n - 1);
        d[i] = 1.0f - cosf(t * 0.5f * PI);
    }
}

ggml_tensor * llm_graph_context::linear(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
        ggml_tensor * out = ggml_mul_mat(ctx0, w, x);
        return ggml_add(ctx0, out, b);
    }

// 统一的 MLP: silu(W1x + b1) @ W2 + b2
ggml_tensor * llm_graph_context::mlp_silu(ggml_tensor * x, ggml_tensor * w1, ggml_tensor * b1, 
                        ggml_tensor * w2, ggml_tensor * b2) const {
    x = ggml_silu(ctx0, linear(x, w1, b1));
    return linear(x, w2, b2);
}

// 统一的 FFN: gelu(W0x + b0) @ W2 + b2
ggml_tensor * llm_graph_context::ffn_gelu(ggml_tensor * x, ggml_tensor * w0, ggml_tensor * b0,
                        ggml_tensor * w2, ggml_tensor * b2, std::string blk_name) const {
    x = linear(x, w0, b0);
    ggml_set_name(x, ("ffn_linear_"+ blk_name).c_str());
    x = ggml_gelu_erf(ctx0, x);
    ggml_set_name(x, ("ffn_gule_"+ blk_name).c_str());
    x = linear(x, w2, b2);
    ggml_set_name(x, ("ffn_gule_linear_"+ blk_name).c_str());
    return x;
}

// Mish 激活
ggml_tensor * llm_graph_context::mish(ggml_tensor * x) const {
    // ggml_tensor * out = ggml_new_tensor_3d(ctx0, x->type, x->ne[0], x->ne[1], x->ne[2]);
    x = ggml_cont(ctx0, x);
    GGML_ASSERT(ggml_is_contiguous(x));
    return ggml_map_custom1(ctx0, x, custom_mish_final, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * llm_graph_context::build_timestep_embedding(ggml_tensor * t, ggml_tensor * w1, ggml_tensor * b1,
                                        ggml_tensor * w2, ggml_tensor * b2) const {
    // t = ggml_cont(ctx0, ggml_permute(ctx0, t, 1, 0, 2, 3));
    return mlp_silu(t, w1, b1, w2, b2);
}

// ==================== 注意力相关 ====================

ggml_tensor * llm_graph_context::scaled_dot_product_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                            ggml_tensor * mask, float scale, std::string blk_name) const {
    const int64_t D = q->ne[0];
    if (scale == 0.0f) scale = 1.0f / sqrtf((float)D);
    
    ggml_tensor * attn = ggml_scale(ctx0, ggml_mul_mat(ctx0, k, q), scale);

    if (mask) {
        attn = ggml_add(ctx0, attn, mask);
    }
    // ggml_set_name(attn, ("sdpa_attn_res_permute_add_"+ blk_name).c_str());
    ggml_tensor * probs = ggml_soft_max(ctx0, attn);
    // ggml_set_name(probs, ("sdpa_attn_res_permute_add_probs_"+ blk_name).c_str());
    v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 0, 2, 3));
    ggml_tensor * res = ggml_mul_mat(ctx0, v, probs);
    // ggml_set_name(res, ("sdpa_attn_res_permute_add_probs_res_"+ blk_name).c_str());
    return res;
}

// 统一的基础注意力块
ggml_tensor * llm_graph_context::build_basic_attn(ggml_tensor * x, ggml_tensor * attn_mask, 
                                const AttnWeights & w, std::string blk_name) const {
    
    const int64_t n_heads = 8;
    const int64_t batch_size = x->ne[2];
    const int64_t seq_len = x->ne[1];
    
    ggml_tensor * thr = x;
    ggml_set_name(thr, ("basic_attn_input_"+ blk_name).c_str());
    // QKV 投影
    ggml_tensor * q = ggml_mul_mat(ctx0, w.wq, x);
    ggml_set_name(q, ("basic_attn_q_"+ blk_name).c_str());
    ggml_tensor * k = ggml_mul_mat(ctx0, w.wk, x);
    ggml_set_name(k, ("basic_attn_k_"+ blk_name).c_str());
    ggml_tensor * v = ggml_mul_mat(ctx0, w.wv, x);
    ggml_set_name(v, ("basic_attn_v_"+ blk_name).c_str());
    
    int64_t d_k = q->ne[0] / n_heads;
    auto reshape_heads = [&](ggml_tensor * t) {
        ggml_tensor * t_4d = ggml_reshape_4d(ctx0, ggml_cont(ctx0, t), d_k, n_heads, seq_len, batch_size);
        ggml_tensor * t_perm = ggml_cont(ctx0, ggml_permute(ctx0, t_4d, 0, 2, 1, 3));
        // ggml_reshape_4d(ctx0, ggml_cont(ctx0, t), d_k, seq_len, n_heads, batch_size);
        return t_perm;
    };
    
    q = reshape_heads(q);
    ggml_set_name(q, ("basic_attn_reshape_q_"+ blk_name).c_str());
    k = reshape_heads(k);
    ggml_set_name(k, ("basic_attn_reshape_k_"+ blk_name).c_str());
    v = reshape_heads(v);
    ggml_set_name(v, ("basic_attn_reshape_v_"+ blk_name).c_str());
    attn_mask = prepare_attention_mask(attn_mask, seq_len, batch_size);
    attn_mask = ggml_reshape_4d(ctx0, ggml_cont(ctx0, attn_mask), attn_mask->ne[0], attn_mask->ne[1], n_heads, batch_size);
    ggml_set_name(attn_mask, ("basic_attn_attn_mask_"+ blk_name).c_str());
    // 注意力计算
    ggml_tensor * attn_out = scaled_dot_product_attention(q, k, v, attn_mask, 0.0f, blk_name);
    ggml_set_name(attn_out, ("basic_attn_attn_out_sdpa_"+ blk_name).c_str());
    ggml_tensor * attn_perm = ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3));
    ggml_tensor * attn_flat = ggml_reshape_3d(ctx0, attn_perm,
                                attn_perm->ne[0] * attn_perm->ne[1], attn_perm->ne[2], attn_perm->ne[3]);
    ggml_set_name(attn_flat, ("basic_attn_attn_out_"+ blk_name).c_str());

    // 输出投影
    ggml_tensor * to_out = ggml_add(ctx0, ggml_mul_mat(ctx0, w.wo, attn_flat), w.bo);
    ggml_set_name(to_out, ("basic_attn_to_out_"+ blk_name).c_str());
    return to_out;
}

// ==================== 卷积相关 ====================
ggml_tensor * llm_graph_context::causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int pad) const {
    
    // 因果填充
    ggml_tensor * zeros = ggml_scale(ctx0, 
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, pad, x->ne[1], x->ne[2]), 0.0f);
    ggml_set_name(zeros, "conv1d_zeros");
    ggml_tensor * x_padded = ggml_concat(ctx0, zeros, ggml_cont(ctx0, x), 0);
    ggml_set_name(x_padded, "conv1d_after_pad");
    ggml_tensor * y = nullptr;
    if (x_padded->ne[2] == 2) {
        ggml_tensor * x_batch0 = ggml_view_3d(ctx0, x_padded, 
            x_padded->ne[0], x_padded->ne[1], 1,  // [1536, 320, 1]
            x_padded->nb[1], x_padded->nb[2], 
            0);

        ggml_tensor * x_batch1 = ggml_view_3d(ctx0, x_padded,
            x_padded->ne[0], x_padded->ne[1], 1,  // [1536, 320, 1]
            x_padded->nb[1], x_padded->nb[2],
            x_padded->nb[2]);
        ggml_tensor * y0 = ggml_conv_1d(ctx0, w, x_batch0, 1, 0, 1);
        ggml_tensor * y1 = ggml_conv_1d(ctx0, w, x_batch1, 1, 0, 1);
        y = ggml_concat(ctx0, y0, y1, 2);
    } else {
        y = ggml_conv_1d(ctx0, w, x_padded, 1, 0, 1);
    }
    
    ggml_tensor * b_reshaped = ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
    y = ggml_add(ctx0, y, b_reshaped);
    
    
    return y;
}

// ==================== Block 构建 ====================

ggml_tensor * llm_graph_context::causal_block1d(ggml_tensor * x, ggml_tensor * mask,
                                ggml_tensor * conv_w, ggml_tensor * conv_b,
                                ggml_tensor * norm_w, ggml_tensor * norm_b, std::string blk_name) const {
    ggml_set_name(x, ("block1d_input_" + blk_name + "_1").c_str());
    ggml_set_name(mask, "block1d_mask");
    x = ggml_mul(ctx0, x, mask);
    ggml_set_name(x, "block1d_after_mask");

    ggml_set_name(conv_w, "conv1d_weight");
    x = causal_conv1d(x, conv_w, conv_b);
    ggml_set_name(x, "block1d_after_conv");
    
    // LayerNorm (需要转置)
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_set_name(x, "block1d_after_permute1");
    x = build_layer_norm(x, norm_w, norm_b, 1e-5f, "blk1d", 0);
    ggml_set_name(x, "block1d_after_norm");
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_set_name(x, "block1d_after_permute2");
    
    // Mish + mask
    x = mish(x);
    ggml_set_name(x, ("block1d_after_mish_" + blk_name).c_str());
    // x = ggml_mul(ctx0, x, mask);
    // ggml_set_name(x, "block1d_output");
    return x;
}

ggml_tensor * llm_graph_context::causal_resnet_block1d(ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * mask, ggml_tensor * t_emb,
                                        const BlockWeights & w, std::string blk_name) const {
    ggml_tensor * residual = ggml_dup(ctx0, x);
    ggml_set_name(residual, ("resnet_input_" + blk_name + "_1").c_str());
    
    // 第一个 block
    ggml_tensor * x_casual = causal_block1d(x, mask, w.conv_w, w.conv_b, w.norm_w, w.norm_b, blk_name + "_1");
    // x = mish(x);
    ggml_set_name(x_casual, ("resnet_after_block1_" + blk_name + "_1").c_str());
    
    // 时间嵌入
    ggml_tensor * t = mish(t_emb);
    ggml_build_forward_expand(gf, t);
    ggml_set_name(t, ("resnet_t_after_mish_"+ blk_name + "_1").c_str());
    
    ggml_tensor * t_linear = linear(t, w.mlp_w, w.mlp_b);
    ggml_build_forward_expand(gf, t_linear);
    ggml_set_name(t_linear, ("resnet_t_after_linear_"+ blk_name + "_1").c_str());
    ggml_tensor * x_permuted = ggml_cont(ctx0, ggml_permute(ctx0, x_casual, 1, 0, 2, 3));
    ggml_tensor * x_p2 = ggml_cont(ctx0, ggml_permute(ctx0, x_permuted, 0, 2, 1, 3));
    ggml_tensor * t_ready = ggml_reshape_4d(ctx0, t_linear, t_linear->ne[0], t_linear->ne[1], 1, 1);
    ggml_tensor * x_added_permuted = ggml_add(ctx0, x_p2, t_ready);
    ggml_tensor * x_rev1 = ggml_permute(ctx0, x_added_permuted, 0, 2, 1, 3);
    x_casual = ggml_cont(ctx0, ggml_permute(ctx0, x_rev1, 1, 0, 2, 3));
    ggml_build_forward_expand(gf, x_casual);
    // printf("迭代 %s: %.1f%% used\n", blk_name.c_str(), 100.0 * ggml_used_mem(ctx0) / ggml_get_mem_size(ctx0));

    
    ggml_set_name(x_casual, ("resnet_after_add_t_"+ blk_name + "_1").c_str());
    // 第二个 block
    x_casual = causal_block1d(x_casual, mask, w.conv2_w, w.conv2_b, w.norm2_w, w.norm2_b, blk_name + "_2");
    ggml_set_name(x_casual, ("resnet_after_block2_"+ blk_name + "_1").c_str());
    // 残差连接
    ggml_tensor * weight_t  = ggml_cont(ctx0, w.res_w);  
    ggml_tensor * weight_2d = ggml_reshape_2d(ctx0, weight_t, weight_t->ne[1], weight_t->ne[2]);
    ggml_tensor * res_perm = ggml_cont(ctx0, ggml_permute(ctx0, residual, 1, 0, 2, 3));
    ggml_tensor * res_mask = ggml_mul_mat(ctx0, weight_2d, res_perm);
    res_mask = ggml_cont(ctx0, ggml_add(ctx0, res_mask, w.res_b));
    res_mask = ggml_cont(ctx0, ggml_permute(ctx0, res_mask, 1, 0, 2, 3));
    ggml_set_name(res_mask, ("resnet_res_conv_"+ blk_name + "_1").c_str());
    ggml_tensor * result = ggml_add(ctx0, x_casual, res_mask);
    ggml_set_name(result, ("resnet_output_"+ blk_name + "_1").c_str());
    result = ggml_cont(ctx0, ggml_permute(ctx0, result, 1, 0, 2, 3));
    return result;
}

// ==================== Transformer Block ====================
ggml_tensor * llm_graph_context::transformer_block(ggml_tensor * x, ggml_tensor * attn_mask,
                                    const TransformerBlockWeights & w, std::string blk_name) const {
    // Self-attention with residual
    ggml_tensor * tr_x = x;
    ggml_set_name(tr_x, ("trans_input_"+ blk_name).c_str());
    ggml_tensor * h = build_layer_norm(x, w.norm1_w, w.norm1_b, 1e-5f, blk_name, 0);
    ggml_set_name(h, ("trans_after_norm1_"+ blk_name).c_str());
    ggml_tensor * attn = build_basic_attn(h, attn_mask, w.attn, blk_name);
    ggml_set_name(attn, ("trans_after_attn_"+ blk_name).c_str());
    tr_x = ggml_add(ctx0, attn, tr_x);
    ggml_set_name(tr_x, ("trans_after_res_connect_"+ blk_name).c_str());
    // FFN with residual
    h = build_layer_norm(tr_x, w.norm3_w, w.norm3_b, 1e-5f, "ffn", 0);
    ggml_set_name(h, ("trans_after_norm3_"+ blk_name).c_str());
    ggml_tensor * ffn_out = ffn_gelu(h, w.ffn_w0, w.ffn_b0, w.ffn_w2, w.ffn_b2, blk_name);
    ggml_set_name(ffn_out, ("trans_after_ffn_"+ blk_name).c_str());
    ffn_out = ggml_cont(ctx0, ggml_add(ctx0, ffn_out, tr_x));
    ggml_set_name(ffn_out, ("trans_after_ffn_add_"+ blk_name).c_str());
    return ffn_out;
}

ggml_tensor * llm_graph_context::build_causal_cond_decoder(ggml_cgraph * gf,
    ggml_tensor * x, ggml_tensor * mask, ggml_tensor * mu,
    ggml_tensor * t, ggml_tensor * spks, ggml_tensor * cond, ggml_tensor * spks_t,
    const llama_model & model, int32_t step) const {
    
    const float neg_big = -1.0e10f;
    
    ggml_build_forward_expand(gf, t);
    ggml_set_name(t, ("causal_t_input_" + std::to_string(step)).c_str());

    // ===== 时间嵌入 =====

    ggml_tensor * t_sin = build_sinusoidal_pos_emb(t, 320, 1000);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&& t_sin shape is: {%d, %d, %d, %d}\n", t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], t_sin->ne[3]);
    ggml_set_name(t_sin, ("causal_t_sinusoidal_" + std::to_string(step)).c_str());
    t = build_timestep_embedding(
        t_sin,
        model.time_mlp_1_w, model.time_mlp_1_b,
        model.time_mlp_2_w, model.time_mlp_2_b);
    ggml_set_name(t, ("causal_t_after_mlp_" + std::to_string(step)).c_str());
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&& t shape is: {%d, %d, %d, %d}\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    // ===== 拼接输入 =====
    x = ggml_concat(ctx0, x, mu, 1);
    ggml_set_name(x, ("causal_x_concat_mu_" + std::to_string(step)).c_str());
    if (spks && spks_t) {
        x = ggml_concat(ctx0, x, spks_t, 1);
        ggml_set_name(x, ("causal_x_concat_spks_" + std::to_string(step)).c_str());
    }
    if (cond) {
        x = ggml_concat(ctx0, x, cond, 1);
        ggml_tensor * cond_x = ggml_dup_tensor(ctx0, x);
        ggml_set_name(cond_x, ("causal_x_concat_cond_" + std::to_string(step)).c_str());
    }
    std::string blk_name = "down_block_step_" + std::to_string(step);
    // ===== Down Block =====
    x = causal_resnet_block1d(gf, x, mask, t, {
        model.down_blk1_conv_w, model.down_blk1_conv_b,
        model.down_blk1_norm_w, model.down_blk1_norm_b,
        model.down_blk_mlp_w, model.down_blk_mlp_b,
        model.down_blk_res_w, model.down_blk_res_b,
        model.down_blk2_conv_w, model.down_blk2_conv_b,
        model.down_blk2_norm_w, model.down_blk2_norm_b}, blk_name);
    ggml_set_name(x, ("causal_x_after_resnet1_" + std::to_string(step)).c_str());
    // 预计算 attention mask（复用）
    const int64_t seq_len = x->ne[1];
    ggml_tensor * attn_mask = mask_to_bias(build_repeat(mask, 1, seq_len, 1), neg_big, x->type);
    ggml_set_name(attn_mask, ("causal_x_down_attn_mask_" + std::to_string(step)).c_str());
    // Down transformer blocks
    for (int i = 0; i < 4; ++i) {
        auto & L = model.layers[226 + i];
        blk_name = "down_block_step_" + std::to_string(step) + "_layer_" + std::to_string(i);
        x = transformer_block(x, attn_mask, {
            L.down_block1_norm1_w, L.down_block1_norm1_b,
            {L.down_block1_wq, L.down_block1_wk, L.down_block1_wv, L.down_block1_wo, L.down_block1_bo},
            L.down_block1_norm3_w, L.down_block1_norm3_b,
            L.down_block1_ffn_w0, L.down_block1_ffn_b0,
            L.down_block1_ffn_w2, L.down_block1_ffn_b2}, blk_name);
        
    }
    ggml_set_name(x, ("causal_x_after_transformer_" + std::to_string(step)).c_str());
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    // 保存 skip connection
    ggml_tensor * hidden = x;
    
    // Downsample（合并 permute + mul + conv）
    x = causal_conv1d(x, model.down_blk_conv_w, model.down_blk_conv_b);
    ggml_set_name(x, ("causal_x_after_downsample_" + std::to_string(step)).c_str());
    // 下采样 mask（只计算一次）
    ggml_tensor * mask_mid = ggml_view_3d(ctx0, mask,
        mask->ne[0] / 2, mask->ne[1], mask->ne[2],
        2 * ggml_type_size(mask->type), mask->nb[2], 0);
    mask_mid = ggml_cont(ctx0, mask_mid);

    // ===== Mid Blocks =====
    for (int i = 0; i < 12; ++i) {
        blk_name = "mid_block_step_" + std::to_string(step) + "_layer_" + std::to_string(i);
        auto & L = model.layers[280 + i];
        x = causal_resnet_block1d(gf, x, mask_mid, t, {
            L.mid_block1_w, L.mid_block1_b,
            L.mid_block1_norm_w, L.mid_block1_norm_b,
            L.mid_block_mlp_w, L.mid_block_mlp_b,
            L.mid_block_res_w, L.mid_block_res_b,
            L.mid_block2_w, L.mid_block2_b,
            L.mid_block2_norm_w, L.mid_block2_norm_b}, blk_name);
        ggml_set_name(x, ("causal_mid_block_resnet_" + std::to_string(step) + "_layer_" + std::to_string(i)).c_str());
        int64_t mid_seq_len = x->ne[1];
        ggml_tensor * mid_attn_mask = mask_to_bias(build_repeat(mask_mid, 1, mid_seq_len, 1), neg_big, x->type);
        for (int j = 0; j < 4; ++j) {
            auto & S = model.mid_block_sub_layers[i * 4 + j];
            blk_name = "mid_block_step_" + std::to_string(step) + "_layer_" + std::to_string(i) + "_sub_layer_" + std::to_string(j);
            x = transformer_block(x, mid_attn_mask, {
                S.mid_block1_norm1_w, S.mid_block1_norm1_b,
                {S.mid_block1_wq, S.mid_block1_wk, S.mid_block1_wv, S.mid_block1_wo, S.mid_block1_bo},
                S.mid_block1_norm3_w, S.mid_block1_norm3_b,
                S.mid_block1_ffn_w0, S.mid_block1_ffn_b0,
                S.mid_block1_ffn_w2, S.mid_block1_ffn_b2}, blk_name);
            ggml_set_name(x, ("causal_mid_block_trans_" + std::to_string(step) + "_layer_" + std::to_string(i) + "_sub_layer_" + std::to_string(j)).c_str());
        }
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    }
    ggml_set_name(x, ("causal_x_after_mid_block_" + std::to_string(step)).c_str());
    // ===== Up Block =====
    // 合并 concat + permute
    x = ggml_concat(ctx0, x, ggml_cont(ctx0, hidden), 1);
    blk_name = "up_block_step_" + std::to_string(step);
    x = causal_resnet_block1d(gf, x, mask, t, {
        model.up_blk1_conv_w, model.up_blk1_conv_b,
        model.up_blk1_norm_w, model.up_blk1_norm_b,
        model.up_blk_mlp_w, model.up_blk_mlp_b,
        model.up_blk_res_w, model.up_blk_res_b,
        model.up_blk2_conv_w, model.up_blk2_conv_b,
        model.up_blk2_norm_w, model.up_blk2_norm_b}, blk_name);
    ggml_set_name(x, ("causal_x_after_up_block_resnet_" + std::to_string(step)).c_str());
    // 复用 down 的 attn_mask
    for (int i = 0; i < 4; ++i) {
        auto & L = model.layers[1059 + i];
        blk_name = "up_block_step_" + std::to_string(step) + "_layer_" + std::to_string(i);
        x = transformer_block(x, attn_mask, {
            L.up_block1_norm1_w, L.up_block1_norm1_b,
            {L.up_block1_wq, L.up_block1_wk, L.up_block1_wv, L.up_block1_wo, L.up_block1_bo},
            L.up_block1_norm3_w, L.up_block1_norm3_b,
            L.up_block1_ffn_w0, L.up_block1_ffn_b0,
            L.up_block1_ffn_w2, L.up_block1_ffn_b2}, blk_name);
    }
    ggml_set_name(x, ("causal_x_after_up_block_transformer_" + std::to_string(step)).c_str());
    // ===== Final Block =====
    blk_name = "final_block_step_" + std::to_string(step);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = causal_conv1d(ggml_mul(ctx0, x, mask), model.up_blk_conv_w, model.up_blk_conv_b);
    x = causal_block1d(x, mask, 
        model.f_blk_conv_w, model.f_blk_conv_b,
        model.f_blk_norm_w, model.f_blk_norm_b, blk_name);
    ggml_set_name(x, ("causal_x_after_final_block_" + std::to_string(step)).c_str());
    // 最终投影（合并操作）
    ggml_tensor * weight_t  = ggml_cont(ctx0, model.f_proj_w);
    ggml_tensor * weight_2d = ggml_reshape_2d(ctx0, weight_t, weight_t->ne[1], weight_t->ne[2]);
    ggml_tensor * res_perm = ggml_cont(ctx0, ggml_permute(ctx0, ggml_cont(ctx0, x), 1, 0, 2, 3));
    ggml_tensor * x_2d = ggml_reshape_2d(ctx0, res_perm, res_perm->ne[0], res_perm->ne[1] * res_perm->ne[2]);
    ggml_tensor * res_mask = ggml_mul_mat(ctx0, weight_2d, x_2d);
    res_mask = ggml_cont(ctx0, ggml_add(ctx0, res_mask, model.f_proj_b));
    res_mask = ggml_reshape_4d(ctx0, res_mask, weight_2d->ne[1], res_perm->ne[1], res_perm->ne[2], 1);
    x = ggml_cont(ctx0, ggml_permute(ctx0, res_mask, 1, 0, 2, 3));
    ggml_build_forward_expand(gf, x);
    
    ggml_set_name(x, ("causal_x_after_final_proj_" + std::to_string(step)).c_str());
    x = ggml_mul(ctx0, x, mask);
    return x;
}


ggml_tensor * llm_graph_context::build_solve_euler(
    ggml_cgraph * gf, ggml_tensor * z, ggml_tensor * mu,
    ggml_tensor * mask, ggml_tensor * spks, ggml_tensor * cond,
    const llama_model & model) const {
    
    const int64_t B = z->ne[2], C = z->ne[1], T = z->ne[0];
    const float PI = 3.14159265358979323846f;
    const int N_STEPS = 10;
    
    // 预计算 t_span
    std::vector<float> t_span(N_STEPS + 1);
    std::string t_span_str = "";
    for (int i = 0; i <= N_STEPS; ++i) {
        t_span[i] = 1.0f - cosf((float)i / N_STEPS * 0.5f * PI);
        t_span_str += std::to_string(t_span[i]);
        t_span_str += " ";
    }
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&&&&&& t_span is: %s\n", t_span_str.c_str());
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
        spks_t = ggml_reshape_3d(ctx0, ggml_cont(ctx0, spks_in), 1, 80, 2);
        spks_t = build_repeat(spks_t, 1, T, 0);
    }
    if (spks_t) ggml_set_name(spks_t, ("decoder_spks_t_" + std::to_string(1)).c_str());
    ggml_tensor * cond_in = nullptr;
    if (cond) {
        ggml_tensor * cond_zero = ggml_scale(ctx0, cond, 0.0f);
        cond_in = ggml_concat(ctx0, cond, cond_zero, 2);
    }
    if (cond_in) ggml_set_name(cond_in, ("decoder_cond_in_" + std::to_string(1)).c_str());
    
    // ODE 循环
    ggml_tensor * z_current = z;
    float t_val = t_span[0];
    float dt = t_span[1] - t_span[0];
    
    for (int step = 1; step <= N_STEPS; ++step) {
        // 创建当前步的 t（关键！）
        // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&&& step is: %d, t_val is: %f\n", step , t_val);
        ggml_tensor * t_current = ggml_scale(ctx0, one, t_val);
        ggml_tensor * t_in = ggml_concat(ctx0, t_current, ggml_dup(ctx0, t_current), 0);
        ggml_set_name(t_in, ("t_in_" + std::to_string(step)).c_str());
        
        // 使用当前的 z（关键！）
        ggml_tensor * z_in = ggml_concat(ctx0, z_current, ggml_dup(ctx0, z_current), 2);
        ggml_set_name(z_in, ("z_in_" + std::to_string(step)).c_str());
        
        // 计算速度场
        ggml_tensor * dphi_dt = build_causal_cond_decoder(gf,
            z_in, mask_in, mu_in, t_in, spks_in, cond_in, spks_t, model, step);
        ggml_set_name(dphi_dt, ("dphi_dt_" + std::to_string(step)).c_str());
        
        ggml_tensor * dphi_dt_clean = ggml_cont(ctx0, dphi_dt);
        // CFG 合并
        ggml_tensor * dphi_cond = ggml_view_3d(ctx0, dphi_dt_clean, T, C, B,
            dphi_dt->nb[1], dphi_dt->nb[2], 0);
        ggml_tensor * dphi_uncond = ggml_view_3d(ctx0, dphi_dt_clean, T, C, B,
            dphi_dt->nb[1], dphi_dt->nb[2], B * dphi_dt->nb[2]);
        
        ggml_tensor * dphi = ggml_sub(ctx0, 
            ggml_scale(ctx0, dphi_cond, 1.7f),
            ggml_scale(ctx0, dphi_uncond, 0.7f));
        ggml_set_name(dphi, ("dphi_" + std::to_string(step)).c_str());
        
        ggml_set_name(z_current, ("z_current_" + std::to_string(step)).c_str());
        // Euler 更新
        // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&&&& step is: %d, dt is: %f, t_val is: %f\n", step, dt, t_val);
        ggml_tensor * dphi_scaled = ggml_scale(ctx0, ggml_cont(ctx0, dphi), dt);
        ggml_set_name(dphi_scaled, ("dphi_scaled_" + std::to_string(step)).c_str());
        ggml_tensor * z_new = ggml_add(ctx0, ggml_cont(ctx0, z_current), dphi_scaled);
        // 更新循环变量（关键！）
        z_current = z_new;
        t_val = t_val + dt;
        if (step < N_STEPS) {
            dt = t_span[step + 1] - t_val;
        }
    }
    
    return z_current;
}

// ==================== 辅助函数 ====================
ggml_tensor * llm_graph_context::build_repeat(ggml_tensor * cur, int32_t current_len, int32_t target_len, int32_t dim) const {
    while (current_len * 2 <= target_len) {
        cur = ggml_concat(ctx0, cur, cur, dim);
        current_len *= 2;
    }
    
    if (current_len < target_len) {
        int64_t remaining = target_len - current_len;
        ggml_tensor * partial;
        
        switch (dim) {
            case 0:
                partial = ggml_view_3d(ctx0, cur, remaining, cur->ne[1], cur->ne[2],
                                        cur->nb[1], cur->nb[2], 0);
                break;
            case 1:
                partial = ggml_view_3d(ctx0, cur, cur->ne[0], remaining, cur->ne[2],
                                        cur->nb[1], cur->nb[2], 0);
                break;
            default:
                partial = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1], remaining,
                                        cur->nb[1], cur->nb[2], 0);
        }
        cur = ggml_concat(ctx0, cur, partial, dim);
    }
    return cur;
}

ggml_tensor * llm_graph_context::build_sinusoidal_pos_emb(
         ggml_tensor * cur,
         int dim,
         int scale) const{
        
    const int64_t n_t = cur->ne[0];
    const int half_dim = dim / 2;

    float emb_div = std::log(10000.0f) / (half_dim - 1);
    ggml_tensor * idx = ggml_arange(ctx0, 0, half_dim, 1);
    idx = ggml_cast(ctx0, idx, GGML_TYPE_F32);
    ggml_tensor * emb = ggml_scale(ctx0, idx, -emb_div);
    emb = ggml_exp(ctx0, emb);
    emb = ggml_reshape_2d(ctx0, emb, 1, half_dim);
    ggml_tensor * x_row = ggml_reshape_2d(ctx0, cur, 1, n_t);
    x_row = ggml_scale(ctx0, x_row, (float)scale);  
    ggml_tensor * out = ggml_mul_mat(ctx0, emb, x_row);

    ggml_tensor * sin_t = ggml_sin(ctx0, out);
    ggml_tensor * cos_t = ggml_cos(ctx0, out);
    ggml_tensor * emb_final = ggml_concat(ctx0, sin_t, cos_t, 0);


    return emb_final;
}

ggml_tensor * llm_graph_context::prepare_attention_mask(
         ggml_tensor * mask,
         int target_len,
         int batch_size) const{
    const int64_t out_dim = 3;
    const int64_t n_heads = 8;
    const int64_t B  = mask->ne[2];   
    const int64_t T  = mask->ne[0];
    // ggml_backend_t backend_cuda = ggml_backend_cuda_init(0);
    if (T != target_len) {
        ggml_tensor * pad = ggml_new_tensor_3d(ctx0, mask->type, target_len - T, mask->ne[1], B);
        pad = ggml_scale(ctx0, pad, 0.0f);  
        mask = ggml_concat(ctx0, mask, pad, 0);
    }
    if (out_dim == 3) {
        ggml_tensor * mask_dst = build_repeat(mask, 2, 16, 2);
        mask = ggml_reshape_3d(ctx0, ggml_cont(ctx0, mask_dst), mask->ne[0], mask->ne[1], B * n_heads);
    } else {
        mask = ggml_reshape_2d(ctx0, ggml_cont(ctx0, mask), mask->ne[0], 1);
        mask = ggml_repeat(ctx0, mask, ggml_new_tensor_4d(ctx0, mask->type, mask->ne[0], mask->ne[1], n_heads, B));
    }
    return mask;
}

ggml_tensor * llm_graph_context::mask_to_bias(
        ggml_tensor * mask,
        float neg_big,
        ggml_type   dtype) const{
    mask = ggml_cast(ctx0, mask, dtype);
    ggml_tensor * mask_one = ggml_dup_tensor(ctx0, mask);
    ggml_tensor * mask_zero = ggml_scale(ctx0, mask_one, 0.0f);
    ggml_tensor * one_tmp = ggml_exp(ctx0, mask_zero);
    return ggml_scale(ctx0, ggml_sub(ctx0, one_tmp, mask), neg_big);
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
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4) {
            // GLM4 seems to have numerical issues with half-precision accumulators
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
                 int   il) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

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

    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    ggml_tensor * experts = nullptr;
    if (gate_exps) {
        cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(cur, "ffn_moe_gate", il);
    } else {
        cur = up;
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
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    // aggregate experts
    ggml_tensor * moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
        ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                experts->nb[2], i*experts->nb[1]);

        if (i == 0) {
            moe_out = cur_expert;
        } else {
            moe_out = ggml_add(ctx0, moe_out, cur_expert);
        }
    }

    if (n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
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
    // LLAMA_LOG_INFO("&&&&&&&&&&&&& token_len is: %d\n", token_len);
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
    // LLAMA_LOG_INFO("&&&&&&&&&&&&& feat_len is: %d\n", feat_len);
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
    // LLAMA_LOG_INFO("&&&&&&&&&&&&& feat_len is: %d\n", feat_len);
    ggml_tensor * cur = nullptr;
    inp->input_rand_noise = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 50 * 300, 80, 1);
    ggml_set_input(inp->input_rand_noise);
    cur = inp->input_rand_noise;
    cb(cur, "inp_rand_noise", -1);
    res->add_input(std::move(inp));
    return cur;
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
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale);

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
    auto inp = std::make_unique<llm_graph_input_cls>(cparams);

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

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd;
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
    const auto * mctx_cur = static_cast<const llama_kv_cache_unified_context *>(mctx);

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
         ggml_cgraph * gf,
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * v_mla,
             float     kq_scale) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    const auto n_tokens = q->ne[1];
    const auto n_head   = q->ne[2];
    const auto n_kv     = k->ne[1];

    ggml_tensor * cur;

    // TODO: replace hardcoded padding with ggml-provided padding
    if (cparams.flash_attn && (n_kv % 256 == 0) && kq_b == nullptr) {
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

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

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
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, 0.08838834764831845f/30.0f));
            kq = ggml_scale(ctx0, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh (ctx0, kq);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);

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
    inp->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->kq_mask);

    inp->kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->kq_mask, GGML_TYPE_F16) : inp->kq_mask;

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_mla, kq_scale);
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

static std::unique_ptr<llm_graph_input_attn_kv_unified> build_attn_inp_kv_unified_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_unified_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv_unified>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_unified_iswa for SWA");

        const auto n_kv = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_kv_unified * llm_graph_context::build_attn_inp_kv_unified() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_unified_context *>(mctx);

    auto inp = build_attn_inp_kv_unified_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv_unified *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_unified * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

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

    
    // for(int i = 0; i < total; i++) {
    //     LLAMA_LOG_DEBUG("***************************** %8.4f ", data[i]);
    // }
    // float * k_data = new float[k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3]];
    // memcpy(k_data, (float *)k->data, k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3] * sizeof(float));
    // for(int i = 0; i< k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3]; i++) {
    //     LLAMA_LOG_DEBUG("***************************** %8.4f ", k_data[i]);  
    // }

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_mla, kq_scale);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4) {
            // GLM4 seems to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_unified_iswa * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
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

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_mla, kq_scale);
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

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
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

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_mla, kq_scale);
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
llm_graph_input_attn_kv_unified_iswa * llm_graph_context::build_attn_inp_kv_unified_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_unified_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_unified_iswa>(hparams, cparams, mctx_cur);

    {
        const auto n_kv = mctx_cur->get_base()->get_n_kv();

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_unified for non-SWA");

        const auto n_kv = mctx_cur->get_swa()->get_n_kv();

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    return (llm_graph_input_attn_kv_unified_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_cgraph * gf,
        ggml_tensor * s,
        ggml_tensor * state_copy,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_kv,
           uint32_t   kv_head,
           uint32_t   kv_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, kv_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between kv_head and kv_head + n_kv
    // {state_size, kv_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, ggml_view_1d(ctx0, state_copy, n_seqs, 0));
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_kv)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, ggml_view_1d(ctx0, state_copy, n_kv - n_seqs, n_seqs*state_copy->nb[0]));
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_kv - n_seqs), (kv_head + n_seqs)*state_size*ggml_element_size(s))));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const auto n_rs = mctx_cur->get_n_rs();

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_cgraph * gf,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(gf, s, inp->s_copy, state_size, n_seqs, kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(), get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
           ggml_cgraph * gf,
    const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, gf, token_shift_all,
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

    auto inp_rs   = build_rs_inp_impl(ctx0, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_unified_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

void llm_graph_context::build_pooling(
        ggml_cgraph * gf,
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
                inp = ggml_get_rows(ctx0, inp, inp_cls);

                if (cls) {
                    // classification head
                    // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                    cur = ggml_mul_mat(ctx0, cls, inp);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_tanh(ctx0, cur);

                    // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                    // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                    if (cls_out) {
                        cur = ggml_mul_mat(ctx0, cls_out, cur);
                        if (cls_out_b) {
                            cur = ggml_add(ctx0, cur, cls_out_b);
                        }
                    }
                } else if (cls_out) {
                    // Single layer classification head (direct projection)
                    // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                    cur = ggml_mul_mat(ctx0, cls_out, inp);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                } else {
                    GGML_ABORT("RANK pooling requires either cls+cls_b or cls_out+cls_out_b");
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
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
