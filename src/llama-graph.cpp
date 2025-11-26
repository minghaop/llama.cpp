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

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
        
    }

    if (ubatch->embd) {
        const int64_t n_embd   = embd->ne[0];
        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_embd*ggml_element_size(embd));
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

// ggml_tensor * llm_graph_context::build_basic_condnet(
//          ggml_tensor * cur,
//          ggml_tensor * conv_weight,
//          ggml_tensor * norm_weight,
//          ggml_tensor * conv_bias,
//          ) const {
    
//     cur = ggml_conv_1d(ctx0, conv_weight, cur);
//     cur = ggml_add(ctx0, cur, conv_bias);
    
//     return cur;

// }
    
    

ggml_tensor * llm_graph_context::build_layer_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
         float eps,
         std::string blk_type,
         int32_t il) const{
    
    // mw = ggml_reshape_4d(ctx0, mw, 1, cur->ne[1], 1, 1);
    // mb = ggml_reshape_4d(ctx0, mb, 1, cur->ne[1], 1, 1);
    // mw = ggml_cast(ctx0, mw, cur->type);
    // mb = ggml_cast(ctx0, mb, cur->type);
    cur = ggml_norm_inplace(ctx0, cur, eps);
    
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
    
    cur = ggml_mul_mat(ctx0, linear_mw, cur);
    cur = ggml_add(ctx0, cur, linear_mb);
    ggml_set_name(cur, ("1st_mul_mat" + std::to_string(il)).c_str());
    cur = build_layer_norm(cur, norm_mw, norm_mb, 1e-05, "sub_sampling", il);
    // cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    ggml_set_name(cur, ("embed_linear_no_sub_sample_" + std::to_string(il)).c_str());

    return cur;

}

ggml_tensor * llm_graph_context::build_pe(ggml_cgraph * gf, int64_t max_len) const{
        
    const int32_t d_model = 512;
    const int64_t seq_len = max_len;
    const int64_t half = d_model / 2;
    const int64_t total_pe_len = 2 * seq_len - 1;

    ggml_tensor * position = ggml_arange(ctx0, 0, seq_len, 1);
    position = ggml_reshape_2d(ctx0, position, 1, seq_len);   // [seq_len, 1]
    

    ggml_tensor * div_term = ggml_arange(ctx0, 0, d_model, 2);
    float inv_den = -logf(10000.0f) / d_model;
    div_term = ggml_scale(ctx0, div_term, inv_den);
    div_term = ggml_exp(ctx0, div_term);                      // [half]
    div_term = ggml_reshape_2d(ctx0, div_term, 1, half);      // [1, half]
    
    ggml_tensor * angle = ggml_mul_mat(ctx0, position, div_term);

    angle = ggml_cont(ctx0, ggml_transpose(ctx0, angle));    // [half, seq_len]
    ggml_set_name(angle, "angle");

    ggml_tensor * pe_sin = ggml_sin(ctx0, angle);
    ggml_tensor * pe_cos = ggml_cos(ctx0, angle);
    ggml_tensor * sin_3d = ggml_reshape_3d(ctx0, pe_sin, half, seq_len, 1);
    ggml_tensor * cos_3d = ggml_reshape_3d(ctx0, pe_cos, half, seq_len, 1);
    ggml_tensor * stacked = ggml_concat(ctx0, sin_3d, cos_3d, 2);
    ggml_tensor * permuted = ggml_permute(ctx0, stacked, 1, 0, 2, 3);
    ggml_tensor * pe_positive = ggml_reshape_2d(ctx0, ggml_cont(ctx0, permuted), d_model, seq_len);      
    ggml_set_name(pe_positive, "pe_positive");

    // 负位置编码使用角度取反
    ggml_tensor * neg_angle = ggml_scale(ctx0, angle, -1.0f);
    ggml_tensor * pe_neg_sin = ggml_sin(ctx0, neg_angle);
    ggml_tensor * pe_neg_cos = ggml_cos(ctx0, neg_angle);
    ggml_tensor * neg_sin_3d = ggml_reshape_3d(ctx0, pe_neg_sin, half, seq_len, 1);
    ggml_tensor * neg_cos_3d = ggml_reshape_3d(ctx0, pe_neg_cos, half, seq_len, 1);
    ggml_tensor * neg_stacked = ggml_concat(ctx0, neg_sin_3d, neg_cos_3d, 2);
    ggml_tensor * neg_permuted = ggml_permute(ctx0, neg_stacked, 1, 0, 2, 3);
    ggml_tensor * pe_negative = ggml_reshape_2d(ctx0, ggml_cont(ctx0, neg_permuted), d_model, seq_len);
    ggml_set_name(pe_negative, "pe_negative");                                       // 输出形状: [seq_len, d_model]

    ggml_tensor * pe_positive_flipped = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pe_positive->ne[0], pe_positive->ne[1]);
    pe_positive_flipped->flags = GGML_TENSOR_FLAG_PARAM;
    for (int64_t i = 0; i < seq_len; ++i) {
        int64_t src_row = seq_len - 1 - i;
        ggml_tensor * src = ggml_view_1d(ctx0, pe_positive, d_model, src_row * pe_positive->nb[1]);
        ggml_tensor * dst = ggml_view_1d(ctx0, pe_positive_flipped, d_model, i * pe_positive_flipped->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
    }

    ggml_tensor * pe_positive_final = ggml_reshape_3d(ctx0, pe_positive_flipped, d_model, seq_len, 1);
    ggml_set_name(pe_positive_final, "pe_positive_final");

    ggml_tensor * pe_negative_sliced = ggml_view_2d(ctx0, pe_negative, d_model, seq_len - 1, pe_negative->nb[1],0);
    ggml_set_name(pe_negative_sliced, "pe_negative_sliced");
    
    ggml_tensor * pe_negative_final = ggml_reshape_3d(ctx0, pe_negative_sliced, d_model, seq_len - 1, 1);
    ggml_set_name(pe_negative_final, "pe_negative_final");
    LLAMA_LOG_INFO("&&&&&&&&&&&&&&& pe_negative_final type is: %d\n", pe_negative_final->type);

    // 8. 转置回 [d_model, total_pe_len] 格式
    ggml_tensor * pe_cat =  ggml_concat(ctx0, pe_positive_final, pe_negative_final, 1);
    ggml_set_name(pe_cat, "espnet_rel_pos_encode");
    cb(pe_cat, "espnet_rel_pos_encode", -1);

    return pe_cat;
}

ggml_tensor * llm_graph_context::build_pos_encoding(
         ggml_tensor * cur,
         size_t offset, 
         size_t size, 
         size_t il) const{
    
    ggml_tensor * new_cur = ggml_dup(ctx0, cur);
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
        float xscale = std::sqrt(static_cast<float>(d_model));
        cur = ggml_scale(ctx0, cur, xscale);
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
        ggml_tensor * mask_ext = ggml_scale(ctx0, mask, 0.0f);
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

ggml_tensor * llm_graph_context::build_rand_noise(
         ggml_tensor * cur,
         float tempture) const{
    // ggml_backend_t backend_cuda = ggml_backend_cuda_init(0);
    
    ggml_tensor * out = ggml_new_tensor_3d(ctx0, cur->type, 50*300, 80, 1);
    ggml_backend_alloc_ctx_tensors(ctx0, backend_cpu);
    const int64_t n_elm = ggml_nelements(out);
    std::vector<float> tmp(n_elm);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int64_t i = 0; i < n_elm; ++i) tmp[i] = dist(gen);
    ggml_backend_tensor_set(out, tmp.data(), 0, tmp.size()*sizeof(float));
    ggml_tensor * z = ggml_view_3d(ctx0, out, 80, cur->ne[0], cur->ne[2], out->nb[0], out->nb[1], 0);
    // z = ggml_scale_inplace(ctx0, z, tempture);
    return z;
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
         int dim,
         int scale) const{
        
    const int64_t n_token = cur->ne[1];
    const int half_dim = dim / 2;

    float emb_div = std::log(10000.0f) / (half_dim - 1);
    ggml_tensor * idx = ggml_arange(ctx0, 0, half_dim, 1);
    idx = ggml_cast(ctx0, idx, GGML_TYPE_F32);
    ggml_tensor * emb = ggml_scale(ctx0, idx, -emb_div);
    emb = ggml_exp(ctx0, emb);

    ggml_tensor * x_col = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cur), cur->ne[1], cur->ne[0]);
    ggml_tensor * emb_row = ggml_reshape_2d(ctx0, ggml_cont(ctx0, emb), 1, half_dim);
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
    cur = ggml_mul_mat(ctx0, mw_1, cur);
    cur = ggml_add(ctx0, cur, mb_1);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, mw_2, cur);
    cur = ggml_add(ctx0, cur, mb_2);
    return cur;
}

ggml_tensor * llm_graph_context::prepare_attention_mask(
         ggml_tensor * mask,
         ggml_tensor * mask_tmpl,
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

ggml_tensor * llm_graph_context::ggml_spda(
         ggml_tensor * q, 
         ggml_tensor * k, 
         ggml_tensor * v, 
         ggml_tensor * mask,
         ggml_tensor * attn_bias,
         float dropout_p, 
         bool is_causal, 
         float scale, 
         bool enable_gqa) const{
    GGML_ASSERT(ggml_n_dims(q) == 4 && ggml_n_dims(k) == 4 && ggml_n_dims(v) == 4);
    int64_t D = q->ne[0];
    int64_t L = q->ne[1];
    int64_t S = k->ne[1];
    int64_t H = q->ne[2];
    int64_t B = q->ne[3];
    const int64_t D_head = q->ne[0];
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(float(D_head));
    }
    attn_bias = ggml_scale(ctx0, attn_bias, 0.0f);
    if (mask) {
        attn_bias = ggml_add(ctx0, mask, attn_bias);
    }
    ggml_tensor* attn_weight = ggml_mul_mat(ctx0, k, q);
    attn_weight = ggml_scale(ctx0, attn_weight, scale);
    attn_weight  = ggml_cont(ctx0, ggml_permute(ctx0, attn_weight, 1, 0, 2, 3));
    attn_weight = ggml_cont(ctx0, attn_weight);
    attn_weight = ggml_reshape_4d(ctx0, ggml_cont(ctx0, attn_weight), L, S, H, B);
    attn_weight = ggml_add(ctx0, attn_weight, attn_bias);
    ggml_tensor* probs = ggml_soft_max(ctx0, attn_weight);
    v  = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx0, v, probs);
    
    return out;
}

ggml_tensor * llm_graph_context::build_basic_attn(
         ggml_tensor * x, 
         ggml_tensor * attn_mask,
         ggml_tensor * attn_bias, 
         ggml_tensor * mask_tmpl,
         int64_t layer_id,
         std::string mode,
         const llama_model & model
         ) const{
    int64_t input_ndim = ggml_n_dims(x);
    const int64_t sequence_length = x->ne[1];   // T
    const int64_t batch_size      = x->ne[2];   // B
    const int64_t n_heads = 8;
    attn_mask = prepare_attention_mask(attn_mask, mask_tmpl, sequence_length, batch_size);
    attn_mask = ggml_reshape_4d(ctx0, ggml_cont(ctx0, attn_mask), attn_mask->ne[0], attn_mask->ne[1], n_heads, batch_size);
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
    q = ggml_reshape_4d(ctx0, ggml_cont(ctx0, q), d_k, q->ne[1], n_heads, batch_size);
    k = ggml_reshape_4d(ctx0, ggml_cont(ctx0, k), d_k, k->ne[1], n_heads, batch_size);
    v = ggml_reshape_4d(ctx0, ggml_cont(ctx0, v), d_k, v->ne[1], n_heads, batch_size);
    ggml_tensor * attn_out = ggml_spda(q, k, v, attn_mask, attn_bias, 0.0f, false, 0.0f, false);
    attn_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, attn_out),
                                attn_out->ne[0] * attn_out->ne[2],              
                                attn_out->ne[1],                       
                                attn_out->ne[3]);
    attn_out = ggml_cast(ctx0, attn_out, q->type);
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
    attn_out = ggml_mul_mat(ctx0, wo, attn_out);
    attn_out = ggml_add(ctx0, attn_out, bo);
    
    return attn_out;
}

ggml_tensor * llm_graph_context::causal_conv1d_forward(
        ggml_tensor * x,
        std::string mode,
        int32_t layer_id,
        int32_t blk_id,
        std::vector<ggml_tensor *> & pad_list,
        const llama_model & model,
        int32_t step) const{
    const int causal_pad = 2;
    
    ggml_tensor * x_cont = ggml_cont(ctx0, x);
    ggml_tensor * x_dup = ggml_dup(ctx0, x_cont);
    ggml_set_name(x_dup, ("causal_blk1d_conv_input_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
    ggml_tensor * pad = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, causal_pad, x->ne[1], x->ne[2]);
    pad = ggml_scale(ctx0, pad, 0.0f);
    ggml_tensor * x_pad = ggml_concat(ctx0, pad, x_dup, 0);

    ggml_tensor * model_weight;
    ggml_tensor * model_bias;
    if (mode == "down_block") {
        if (blk_id == 1) {
            model_weight = model.down_blk1_conv_w;
            model_bias = model.down_blk1_conv_b;
        } else if (blk_id == 2){
            model_weight = model.down_blk2_conv_w;
            model_bias = model.down_blk2_conv_b;
        } else {
            model_weight = model.down_blk_conv_w;
            model_bias = model.down_blk_conv_b;
        }
    } else if(mode == "mid_block") {
        if(blk_id == 1) {
            model_weight = model.layers[layer_id].mid_block1_w;
            model_bias = model.layers[layer_id].mid_block1_b;
        } else {
            model_weight = model.layers[layer_id].mid_block2_w;
            model_bias = model.layers[layer_id].mid_block2_b;
        }
    }else if(mode == "up_block"){
        if(blk_id == 1) {
            model_weight = model.up_blk1_conv_w;
            model_bias = model.up_blk1_conv_b; 
        } else if(blk_id == 2) {
            model_weight = model.up_blk2_conv_w;
            model_bias = model.up_blk2_conv_b; 
        } else {
            model_weight = model.up_blk_conv_w;
            model_bias = model.up_blk_conv_b;
        }
    } else if (mode == "final_block"){
        model_weight = model.f_blk_conv_w;
        model_bias = model.f_blk_conv_b;
    } else {
        model_weight = model.f_blk_norm_w;
        model_bias = model.f_blk_norm_b;
    }
    
    x_pad = ggml_cont(ctx0, x_pad);
    model_weight = ggml_cont(ctx0, model_weight);
    ggml_set_name(model_weight, ("causal_blk1d_conv_weight_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
    GGML_ASSERT(x_pad->type == GGML_TYPE_F32);
    GGML_ASSERT(model_weight->type == GGML_TYPE_F32);
    ggml_set_name(x_pad, ("causal_blk1d_conv_pad_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());

    ggml_tensor * im2col = ggml_im2col(ctx0, model_weight, x_pad, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * weight_permuted = ggml_permute(ctx0, model_weight, 1, 0, 2, 3);
    weight_permuted = ggml_cont(ctx0, weight_permuted);
    ggml_tensor * weight_reshaped = ggml_reshape_2d(ctx0, weight_permuted, 
                                                    weight_permuted->ne[0] * weight_permuted->ne[1],  // IC*K
                                                    weight_permuted->ne[2]);
    ggml_tensor * im2col_reshaped = ggml_reshape_2d(ctx0, im2col, 
                                                     im2col->ne[0],  // IC*K
                                                     im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * y = ggml_mul_mat(ctx0, im2col_reshaped, weight_reshaped);
    y = ggml_reshape_3d(ctx0, y, im2col->ne[1], model_weight->ne[2], im2col->ne[2]);                                            
    // ggml_tensor * y = ggml_conv_1d(ctx0, model_weight, x_pad, 1, 0, 1);
    ggml_tensor * y_safe = ggml_dup(ctx0, ggml_cont(ctx0, y));
    model_bias = ggml_reshape_3d(ctx0, ggml_cont(ctx0, model_bias), 1, model_bias->ne[0], 1);
    model_bias = ggml_cont(ctx0, model_bias);
    ggml_tensor * result = ggml_add(ctx0, y_safe, model_bias);
    result = ggml_cont(ctx0, result);
    return result;
}

ggml_tensor * llm_graph_context::causal_block1d_forward(
        ggml_tensor * x,
        ggml_tensor * mask,
        std::string mode,
        int32_t layer_id,
        int32_t blk_id,
        std::vector<ggml_tensor *> & pad_list,
        const llama_model & model,
        int32_t step) const{
    
    x = ggml_mul(ctx0, x, mask);
    ggml_set_name(x, ("causal_blk1d_x_mask_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& mode is: %s, blk_id is: %d, layer_id is: %d\n", mode.c_str(), blk_id, layer_id);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& causal_block1d_forward x shape is: {%d, %d, %d}\n", x->ne[0], x->ne[1], x->ne[2]);
    x = causal_conv1d_forward(x, mode, layer_id, blk_id, pad_list, model, step);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&&&&& x pos is: %p\n", x);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& causal_block1d_after x shape is: {%d, %d, %d}\n", x->ne[0], x->ne[1], x->ne[2]);
    ggml_set_name(x, ("causal_blk1d_conv_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id) + "_" + std::to_string(blk_id)).c_str());
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
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& causal_blk1d_forward x shape is: {%d, %d, %d}\n", x->ne[0], x->ne[1], x->ne[2]);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& causal_blk1d_forward mask shape is: {%d, %d, %d}\n", mask->ne[0], mask->ne[1], mask->ne[2]);
    x = ggml_mul(ctx0, x, ggml_tanh(ctx0, ggml_log(ctx0, ggml_scale(ctx0, ggml_exp(ctx0, x), 1.0f))));
    if (mask)
        x = ggml_mul(ctx0, x, mask);
    return x;
}

ggml_tensor * llm_graph_context::causal_resnet_block1d_forward(
        ggml_tensor * x,
        ggml_tensor * mask,
        ggml_tensor * t_emb,
        int32_t step,
        int32_t layer_id,
        std::string mode,
        std::vector<ggml_tensor *> & pad_list,
        const llama_model & model) const{
    ggml_tensor * x_dup = ggml_dup(ctx0, x);
    x = causal_block1d_forward(x, mask, mode, layer_id, 1, pad_list, model, step);
    ggml_set_name(x, ("causal_blk1_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    x = ggml_mul(ctx0, x, ggml_tanh(ctx0, ggml_log(ctx0, ggml_scale(ctx0,ggml_exp(ctx0, x), 1.0f))));
    ggml_set_name(x, ("causal_blk1d_mish_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    if(mode == "down_block") {
        t_emb = ggml_mul_mat(ctx0, model.down_blk_mlp_w, t_emb);
        t_emb = ggml_add(ctx0, t_emb, model.down_blk_mlp_b);
    }else if(mode == "mid_block") {
        t_emb = ggml_mul_mat(ctx0, model.layers[layer_id].mid_block_mlp_w, t_emb);
        t_emb = ggml_add(ctx0, t_emb, model.layers[layer_id].mid_block_mlp_b);
    } else {
        t_emb = ggml_mul_mat(ctx0, model.up_blk_mlp_w, t_emb);
        t_emb = ggml_add(ctx0, t_emb, model.up_blk_mlp_b);
    }
    t_emb = ggml_reshape_3d(ctx0, ggml_cont(ctx0, t_emb), 1, t_emb->ne[0], t_emb->ne[1]);
    x = ggml_add(ctx0, x, t_emb);
    ggml_set_name(x, ("causal_mlp_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    // x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
    x = causal_block1d_forward(x, mask, mode, layer_id, 2, pad_list, model, step);
    ggml_set_name(x, ("causal_blk2_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    ggml_tensor * x_mask = ggml_mul(ctx0, x_dup, mask);
    ggml_tensor * x_mask_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, x_mask), x_mask->ne[0] * x_mask->ne[2], x_mask->ne[1]);
    x_mask_flat = ggml_cont(ctx0, ggml_permute(ctx0, x_mask_flat, 1, 0, 2, 3));
    ggml_set_name(x_mask_flat, ("causal_x_mask_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    if(mode == "down_block") {
        ggml_tensor * w = ggml_reshape_2d(ctx0, model.down_blk_res_w, model.down_blk_res_w->ne[1], model.down_blk_res_w->ne[2]);
        x_mask = ggml_mul_mat(ctx0, w, x_mask_flat);
        x_mask = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x_mask), x_mask->ne[0], x_mask->ne[1] / 2, 2);
        x_mask = ggml_add(ctx0, x_mask, model.down_blk_res_b);
    } else if(mode == "mid_block") {
        ggml_tensor * w = ggml_reshape_2d(ctx0, model.layers[layer_id].mid_block_res_w, model.layers[layer_id].mid_block_res_w->ne[1], model.layers[layer_id].mid_block_res_w->ne[2]);
        x_mask = ggml_mul_mat(ctx0, w, x_mask_flat);
        x_mask = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x_mask), x_mask->ne[0], x_mask->ne[1] / 2, 2);
        x_mask = ggml_add(ctx0, x_mask, model.layers[layer_id].mid_block_res_b);
    } else {
        ggml_tensor * w = ggml_reshape_2d(ctx0, model.up_blk_res_w, model.up_blk_res_w->ne[1], model.up_blk_res_w->ne[2]);
        x_mask = ggml_mul_mat(ctx0, w, x_mask_flat);
        x_mask = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x_mask), x_mask->ne[0], x_mask->ne[1] / 2, 2);
        x_mask = ggml_add(ctx0, x_mask, model.up_blk_res_b);
    }
    ggml_set_name(x_mask, ("causal_res_conv_" + mode + "_" + std::to_string(step) + "_" + std::to_string(layer_id)).c_str());
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = ggml_add(ctx0, x, x_mask);
    
    return x;
}

ggml_tensor * llm_graph_context::mask_to_bias(
        ggml_tensor * mask,
        ggml_tensor * one,
        ggml_tensor * neg_big,
        ggml_type   dtype) const{
    mask = ggml_cast(ctx0, mask, dtype);
    ggml_tensor * mask_one = ggml_dup_tensor(ctx0, mask);
    ggml_tensor * mask_zero = ggml_scale(ctx0, mask_one, 0.0f);
    ggml_tensor * one_tmp = ggml_exp(ctx0, mask_zero);
    return ggml_mul(ctx0, ggml_sub(ctx0, one_tmp, mask), neg_big);
}


ggml_tensor * llm_graph_context::build_causal_cond_decoder(
         ggml_tensor * x,
         ggml_tensor * mask,
         ggml_tensor * mu,
         ggml_tensor * t,
         ggml_tensor * spks,
         ggml_tensor * cond,
         ggml_tensor * spks_t,
         ggml_tensor * attn_mask_t,
         ggml_tensor * mask_tmpl,
         ggml_tensor * mask_to_bias_one,
         ggml_tensor * mask_to_bias_neg,
         ggml_tensor * attn_bias,
         std::vector<ggml_tensor *> & pad_list,
         const llama_model & model,
         int32_t step) const{
    
    t = build_sinusoidal_pos_emb(t, 320, 1000);
    ggml_set_name(t, ("sinusoidal_pos_emb_" + std::to_string(step)).c_str());
    t = build_timestep_embedding(t, model.time_mlp_1_w, model.time_mlp_1_b, model.time_mlp_2_w, model.time_mlp_2_b);
    ggml_set_name(t, ("timestep_embedding_" + std::to_string(step)).c_str());
    x = ggml_concat(ctx0, x, mu, 1);
    if (spks) {
        x = ggml_concat(ctx0, x, spks_t, 1);
        ggml_set_name(x, ("spks_" + std::to_string(step)).c_str());
    }
    if (cond) {
        x = ggml_concat(ctx0, x, cond, 1);
        ggml_set_name(x, ("cond_" + std::to_string(step)).c_str());
    }

    std::vector<ggml_tensor *> hiddens;
    std::vector<ggml_tensor *> masks = {mask};
    
    ggml_tensor * mask_down = masks.back();
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& build_causal_cond_decoder x shape is: {%d, %d, %d}\n", x->ne[0], x->ne[1], x->ne[2]);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& build_causal_cond_decoder mask_down shape is: {%d, %d, %d}\n", mask_down->ne[0], mask_down->ne[1], mask_down->ne[2]);
    x = causal_resnet_block1d_forward(x, mask_down, t, step, 0, "down_block", pad_list, model);
    ggml_set_name(x, ("causal_resnet_blk1d_" + std::to_string(step)).c_str());
    ggml_tensor * attn_mask = ggml_view_3d(ctx0, mask_down, mask_down->ne[0], mask_down->ne[1], mask_down->ne[2], mask_down->nb[1], mask_down->nb[2], 0);
    attn_mask = build_repeat(attn_mask, 1, 1584, 1);
    attn_mask = mask_to_bias(attn_mask, mask_to_bias_one, mask_to_bias_neg, x->type);
    ggml_set_name(attn_mask, ("attn_mask_" + std::to_string(step)).c_str());
    for (size_t i = 0; i < 4; ++i) {
        x = build_layer_norm(x, model.layers[226 + i].down_block1_norm1_w, model.layers[226 + i].down_block1_norm1_b, 1e-5f, "down_block", 20 + i);
        ggml_tensor * attn_out = build_basic_attn(x, attn_mask, attn_bias,  mask_tmpl, 226 + i, "down_block", model);
        x = ggml_add(ctx0, attn_out, x);

        x = build_layer_norm(x, model.layers[226 + i].down_block1_norm3_w, model.layers[226 + i].down_block1_norm3_b, 1e-5f, "down_block", 24 + i);
        ggml_tensor * ff_out = ggml_mul_mat(ctx0, model.layers[226 + i].down_block1_ffn_w0, x);
        ff_out = ggml_add(ctx0, ff_out, model.layers[226 + i].down_block1_ffn_b0);
        ff_out = ggml_gelu(ctx0, ff_out);
        ff_out = ggml_mul_mat(ctx0, model.layers[226 + i].down_block1_ffn_w2, ff_out);
        ff_out = ggml_add(ctx0, ff_out, model.layers[226 + i].down_block1_ffn_b2);
        x = ggml_add(ctx0, ff_out, x);
    }
    hiddens.push_back(x);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    ggml_tensor * x_mask = ggml_mul(ctx0, x, mask_down);
    x = causal_conv1d_forward(x_mask, "down_block", 0, 3, pad_list, model, step);
    ggml_tensor * mask_sub = ggml_view_3d(ctx0, mask_down, mask_down->ne[0] / 2, mask_down->ne[1], mask_down->ne[2], 2 * ggml_type_size(mask_down->type), mask_down->nb[2], 0);
    mask_sub = ggml_cont(ctx0, mask_sub);
    masks.push_back(mask_sub);
    masks.pop_back(); 
    ggml_tensor * mask_mid = masks.empty() ? nullptr : masks.back();
    for(size_t i = 0; i < 12; i++) {
        x = causal_resnet_block1d_forward(x, mask_mid, t, step, 280 + i, "mid_block", pad_list, model);
        ggml_tensor * attn_mask = ggml_view_3d(ctx0, mask_mid, mask_mid->ne[0], mask_mid->ne[1], mask_mid->ne[2], mask_mid->nb[1], mask_mid->nb[2], 0);
        attn_mask = build_repeat(attn_mask, 1, 1584, 1);
        attn_mask = mask_to_bias(attn_mask, mask_to_bias_one, mask_to_bias_neg, x->type);
        
        for (size_t j = 0; j < 4; ++j) {
            x = build_layer_norm(x, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm1_w, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm1_b, 1e-5f, "mid_block", 28 + i + j);
            ggml_tensor * attn_out = build_basic_attn(x, attn_mask, attn_bias, mask_tmpl, i * 4 + j, "mid_block", model);
            x = ggml_add(ctx0, attn_out, x);
            x = build_layer_norm(x, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm3_w, model.mid_block_sub_layers[i * 4 + j].mid_block1_norm1_b, 1e-5f, "mid_block", 76 + i +j);
            ggml_tensor * ff_out = ggml_mul_mat(ctx0, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_w0, x);
            ff_out = ggml_add(ctx0, ff_out, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b0);
            ff_out = ggml_gelu(ctx0, ff_out);
            ff_out = ggml_mul_mat(ctx0, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_w2, ff_out);
            ff_out = ggml_add(ctx0, ff_out, model.mid_block_sub_layers[i *4 + j].mid_block1_ffn_b2);
            x = ggml_add(ctx0, ff_out, x);
            
        }

        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    }
    ggml_tensor * mask_up = masks.back();
    ggml_tensor * skip = hiddens.back();
    skip = ggml_permute(ctx0, skip, 1, 0, 2, 3);
    x = ggml_concat(ctx0, x, skip, 1);
    x = causal_resnet_block1d_forward(x, mask_up, t, step, 0, "up_block", pad_list, model);
    attn_mask = mask_to_bias(attn_mask, mask_to_bias_one, mask_to_bias_neg, x->type);
    for (size_t i = 0; i < 4; ++i) {
        x = build_layer_norm(x, model.layers[1059 + i].up_block1_norm1_w, model.layers[1059 + i].up_block1_norm1_b, 1e-5f, "up_block", 124 + i);
        ggml_tensor * attn_out = build_basic_attn(x, attn_mask, attn_bias, mask_tmpl, 1059 + i, "up_block", model);
        x = ggml_add(ctx0, attn_out, x);
        x = build_layer_norm(x, model.layers[1059 + i].up_block1_norm3_w, model.layers[1059 + i].up_block1_norm3_b, 1e-5f, "up_block", 128 + i);
        ggml_tensor * ff_out = ggml_mul_mat(ctx0, model.layers[1059 + i].up_block1_ffn_w0, x);
        ff_out = ggml_add(ctx0, ff_out, model.layers[1059 + i].up_block1_ffn_b0);
        ff_out = ggml_gelu(ctx0, ff_out);
        ff_out = ggml_mul_mat(ctx0, model.layers[1059 + i].up_block1_ffn_w2, ff_out);
        ff_out = ggml_add(ctx0, ff_out, model.layers[1059 + i].up_block1_ffn_b2);
        x = ggml_add(ctx0, ff_out, x);
    }
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x_mask = ggml_mul(ctx0, x, mask_up);
    x = causal_conv1d_forward(x_mask, "up_block", 0, 3, pad_list, model, step);
    x = causal_block1d_forward(x, mask_up, "final_block", 0, 3, pad_list, model, step);
    x = ggml_mul(ctx0, x, mask_up);
    x = ggml_conv_1d(ctx0, model.f_proj_w, x, 1, 0, 0);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = ggml_add(ctx0, x, model.f_proj_b);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    return ggml_mul(ctx0, x, mask);
}

ggml_tensor * llm_graph_context::build_repeat(ggml_tensor * cur, int32_t current_length, int32_t target_length, int32_t dim) const{
    // int64_t current_length = 1;

    // 使用2的幂次快速接近目标
    while (current_length * 2 <= target_length) {
        cur = ggml_concat(ctx0, cur, cur, dim);
        current_length *= 2;
        // printf("Current length after concat: %ld\n", current_length);
    }

    // 添加剩余的部分
    if (current_length < target_length) {
        int64_t remaining = target_length - current_length;
        if (dim == 0) {
            ggml_tensor * partial = ggml_view_3d(ctx0, cur, 
                                        remaining, cur->ne[1], cur->ne[2],
                                        cur->nb[1], 
                                        cur->nb[2], 0);
            cur = ggml_concat(ctx0, cur, partial, dim);
            // printf("Final length: %ld\n", cur->ne[0]);
        } else if (dim == 1) {
            ggml_tensor * partial = ggml_view_3d(ctx0, cur, 
                                        cur->ne[0], remaining, cur->ne[2],
                                        cur->nb[1], 
                                        cur->nb[2], 0);
            cur = ggml_concat(ctx0, cur, partial, dim);
            // printf("Final length: %ld\n", cur->ne[1]);
        }else if (dim == 2) {
            ggml_tensor * partial = ggml_view_3d(ctx0, cur, 
                                        cur->ne[0], cur->ne[1], remaining,
                                        cur->nb[1], 
                                        cur->nb[2], 0);
            cur = ggml_concat(ctx0, cur, partial, dim);
            // printf("Final length: %ld\n", cur->ne[2]);
        }
        
    }
    return cur;
}

ggml_tensor * llm_graph_context::build_solve_euler(
         ggml_cgraph * gf,
         ggml_tensor * z,
         ggml_tensor * mu,
         ggml_tensor * mask,
         ggml_tensor * spks,
         ggml_tensor * cond,
         const llama_model & model) const{
        
    const int64_t B   = z->ne[2];
    const int64_t C   = z->ne[1]; 
    const int64_t T   = z->ne[0];
    // const int64_t N   = t_span->ne[0]; 
    const int64_t spk_dim = spks ? spks->ne[0] : 0;
    // float * ptr = (float *)t_span->data;
    const float PI = 3.14159265358979323846f;
    std::vector<float> t_span(11);
    for (int i = 0; i <= 10; ++i) {
        t_span[i] = static_cast<float>(i) / 10;
    }
    for (auto& val : t_span) {
        val = 1.0f - cosf(val * 0.5f * PI);
    }
    float t0 = t_span[0];
    float t1 = t_span[-1];
    float dt = t_span[1] - t_span[0];
    
    ggml_tensor * common = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_tensor * t = ggml_scale(ctx0, common, t0);
    ggml_tensor * one = ggml_exp(ctx0, t);
    ggml_tensor * dt_t = ggml_scale(ctx0, one, dt);

    ggml_tensor * mu_in  = ggml_new_tensor_3d(ctx0, mu->type, T, 80, 1);
    mu_in = ggml_scale(ctx0, mu_in, 0.0f);
    ggml_tensor * spks_in= ggml_new_tensor_2d(ctx0, spks->type, 80, 1);
    spks_in = ggml_scale(ctx0, spks_in, 0.0f);
    ggml_tensor * cond_in= ggml_new_tensor_3d(ctx0, cond->type, T, 80, 1);
    cond_in = ggml_scale(ctx0, cond_in, 0.0f);
    ggml_tensor * spk_t = ggml_new_tensor_3d(ctx0, spks_in->type, T, spks_in->ne[0], spks_in->ne[1]);
    ggml_tensor * attn_mask_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, T, T, 2);
    ggml_tensor * attn_bias = ggml_new_tensor_4d(ctx0, mask->type, T, T, 8, 2);
    ggml_tensor * neg_big = ggml_scale(ctx0, one, -1.0e10f);
    std::vector<ggml_tensor *> pad_list;
    ggml_tensor * pad_320 = ggml_new_tensor_3d(ctx0, z->type, 2, 320, 2);
    ggml_tensor * pad_256 = ggml_new_tensor_3d(ctx0, z->type, 2, 256, 2);
    ggml_tensor * pad_512 = ggml_new_tensor_3d(ctx0, z->type, 2, 512, 2);
    pad_320 = ggml_scale(ctx0, pad_320, 0.0f);
    pad_256 = ggml_scale(ctx0, pad_256, 0.0f);
    pad_512 = ggml_scale(ctx0, pad_512, 0.0f);
    pad_list.push_back(pad_320);
    pad_list.push_back(pad_256);
    pad_list.push_back(pad_512);
    std::vector<ggml_tensor *> sol;
    ggml_tensor * mask_tmpl = ggml_new_tensor_4d(ctx0, mask->type, mask->ne[0], mask->ne[0], 2 * 8, 1);
    // ggml_backend_alloc_ctx_tensors(ctx0, backend_cpu);
    ggml_tensor * z_in_cpy, * mask_in_cpy, * t_in_cpy, * mu_in_cpy, * spks_in_cpy, * cond_in_cpy;
    for (int64_t step = 1; step < 11; ++step) {
        z_in_cpy = ggml_concat(ctx0, z, z, 2);
        cb(z_in_cpy, "z_in_cpy", step);
        mask_in_cpy = ggml_concat(ctx0, mask, mask, 2);
        cb(mask_in_cpy, "mask_in_cpy", step);
        mu_in_cpy = ggml_concat(ctx0, mu, mu_in, 2);
        cb(mu_in_cpy, "mu_in_cpy", step);
        t_in_cpy = ggml_concat(ctx0, t, t, 0);
        cb(t_in_cpy, "t_in_cpy", step);
        if (spks) {
            spks_in_cpy = ggml_concat(ctx0, spks, spks_in, 1);
            cb(spks_in_cpy, "spks_in_cpy", step);
        }
        if (cond) {
            cond_in_cpy = ggml_concat(ctx0, cond, cond_in, 2);
            cb(cond_in_cpy, "cond_in_cpy", step);
        }
        ggml_tensor * spks_in_reshape = ggml_reshape_3d(ctx0, ggml_cont(ctx0, spks_in_cpy), 1, 80, 2);
        // int64_t current_length = 1;
        int64_t target_length = 1584;
        spks_in_reshape = build_repeat(spks_in_reshape, 1, target_length, 0);
        ggml_tensor * dphi_dt = build_causal_cond_decoder(z_in_cpy, mask_in_cpy, mu_in_cpy, t_in_cpy, spks_in_cpy, cond_in_cpy, spks_in_reshape, attn_mask_t, mask_tmpl, one, neg_big, attn_bias, pad_list, model, step);
        ggml_set_name(dphi_dt, ("dphi_dt_" + std::to_string(step)).c_str());
        ggml_tensor * dphi_dt_split   = ggml_view_3d(ctx0, dphi_dt, T, z->ne[1], B, dphi_dt->nb[1], dphi_dt->nb[2], 0);
        ggml_tensor * cfg_dphi_dt  = ggml_view_3d(ctx0, dphi_dt, T, z->ne[1], B, dphi_dt->nb[1], dphi_dt->nb[2], B * dphi_dt->nb[2]);
        ggml_tensor * dphi  = ggml_sub(ctx0, ggml_scale(ctx0, dphi_dt_split, 1.7f), ggml_scale(ctx0, cfg_dphi_dt, 0.7f));
        z = ggml_add(ctx0, z, ggml_scale(ctx0, dphi, dt));
        t = ggml_add(ctx0, t, dt_t);
        sol.push_back(z);
        if( step < 11 - 1) {
            float t_span_ele = t_span[step + 1] - t_span[step];
            dt_t = ggml_scale(ctx0, one, t_span_ele);
        }
        LLAMA_LOG_INFO("&&&&&&&&&& step is: %d\n", step);
    }
    ggml_tensor * last = sol.back();
    if(last->type != z->type) {
        last = ggml_cast(ctx0, last, z->type);
    }
    LLAMA_LOG_INFO("&&&&&&&&&& last shape is: {%d} {%d} {%d}\n", last->ne[0], last->ne[1], last->ne[2]);
    return last;

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

    ggml_tensor * cur_dup = ggml_dup(ctx0, cur);
    cur_dup = ggml_conv_1d(ctx0, mw, cur_dup);
    cur_dup = ggml_add(ctx0, cur_dup, mb);

    return cur_dup;
}

ggml_tensor * llm_graph_context::build_m_source(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb) const {
    
    ggml_tensor * cur_dup = ggml_dup(ctx0, cur);
    ggml_tensor * arrange_tensor = ggml_arange(ctx0, 1.0, 10.0, 1.0);
    arrange_tensor = ggml_reshape_3d(ctx0, arrange_tensor, arrange_tensor->ne[0], 1, 1);

    cur_dup = ggml_mul(ctx0, cur_dup, arrange_tensor);

    ggml_tensor * rad_values = ggml_scale(ctx0, cur_dup, 1/ 24000);
    int n_tasks = 1;
    ggml_tensor * rad_values = ggml_map_custom1(ctx0, rad_values, custom_op_mod_1, n_tasks, NULL);
    ggml_tensor * dummy_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rad_values->ne[2], rad_values->ne[0]);
    ggml_tensor * rand_ini = ggml_map_custom1(ctx0, dummy_input, custom_op_rand_uniform, 1, NULL);


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
        if(hparams.use_flow) {
            inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.spk_embed_dim, 1);
        } 
        // else if {
        //     inp->embd = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 80, 1);
        // } 
        else {
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

    const uint32_t rand_noise_len = 80 * 50 * 300;
    
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
