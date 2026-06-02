#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define ACC_TYPE8 float8
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#ifndef Q1_WG_SIZE
#define Q1_WG_SIZE 64
#endif
#ifndef STAGE2_CHUNK_PAIRS
#define STAGE2_CHUNK_PAIRS 16
#endif

#if (defined(GGML_FLASH_ATTN_FORCE_SUBGROUP) && (GGML_FLASH_ATTN_FORCE_SUBGROUP != 0)) && \
    (defined(cl_khr_subgroups) || defined(__opencl_c_subgroups))
#define GGML_FLASH_ATTN_USE_SUBGROUP 1
#else
#define GGML_FLASH_ATTN_USE_SUBGROUP 0
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}
__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    const int causal_row_limit = n_kv - n_q + my_query_row;
    const global MASK_DATA_TYPE* mask_ptr_row =
        (mask_base != NULL && my_query_row < n_q)
            ? (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1)
            : NULL;

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

        const int valid_cols = min(BLOCK_N, n_kv - k_start);
        const int work_cols = is_causal ? min(valid_cols, max(0, causal_row_limit - k_start + 1)) : valid_cols;
        if (work_cols <= 0) {
            continue;
        }
        const int valid_even = work_cols & ~1;
        const int has_mask = mask_ptr_row != NULL;
        const int has_softcap = logit_softcap > 0.0f;

        if (!has_mask && !has_softcap) {
            for (int j = 0; j < valid_even; j += 2) {
                ACC_TYPE dot_sum0 = 0.0f;
                ACC_TYPE dot_sum1 = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum0 += dot(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]));
                    dot_sum1 += dot(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]));
                }
                const ACC_TYPE score0 = dot_sum0 * scale;
                const ACC_TYPE score1 = dot_sum1 * scale;
                const ACC_TYPE m_new = max(m_i, max(score0, score1));
                const ACC_TYPE p0 = exp(score0 - m_new);
                const ACC_TYPE p1 = exp(score1 - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);

                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
                }
                l_i = l_i * scale_prev + p0 + p1;
                m_i = m_new;
            }
        } else {
            for (int j = 0; j < valid_even; j += 2) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;

                ACC_TYPE dot_sum0 = 0.0f;
                ACC_TYPE dot_sum1 = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum0 += dot(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]));
                    dot_sum1 += dot(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]));
                }
                ACC_TYPE score0 = dot_sum0 * scale;
                ACC_TYPE score1 = dot_sum1 * scale;

                if (has_mask) {
                    score0 += slope * (ACC_TYPE)mask_ptr_row[k_row0];
                    score1 += slope * (ACC_TYPE)mask_ptr_row[k_row1];
                }
                if (has_softcap) {
                    score0 = logit_softcap * tanh(score0 / logit_softcap);
                    score1 = logit_softcap * tanh(score1 / logit_softcap);
                }

                const ACC_TYPE m_new = max(m_i, max(score0, score1));
                const ACC_TYPE p0 = exp(score0 - m_new);
                const ACC_TYPE p1 = exp(score1 - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);

                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
                }
                l_i = l_i * scale_prev + p0 + p1;
                m_i = m_new;
            }
        }

        if (work_cols & 1) {
            const int j = valid_even;
            const int k_row = k_start + j;
            ACC_TYPE dot_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]));
            }
            ACC_TYPE score = dot_sum * scale;
            if (has_mask) {
                score += slope * (ACC_TYPE)mask_ptr_row[k_row];
            }
            if (has_softcap) {
                score = logit_softcap * tanh(score / logit_softcap);
            }
            const ACC_TYPE m_new = max(m_i, score);
            const ACC_TYPE p = exp(score - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(l_v[j][i]);
            }
            l_i = l_i * scale_prev + p;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }
    const global MASK_DATA_TYPE* mask_ptr_q1 =
        (mask_base != NULL) ? (const global MASK_DATA_TYPE*)(mask_base) : NULL;
    const int has_mask_q1 = (mask_ptr_q1 != NULL);
    const int has_softcap_q1 = (logit_softcap > 0.0f);
    const int fast_path_q1 = (!has_mask_q1 && !has_softcap_q1);

    const ACC_TYPE m_sink = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE m_i = -INFINITY;

    if (fast_path_q1) {
        for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
            const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
            const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
            const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
            const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
            ACC_TYPE dot_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
            }
            const ACC_TYPE score = dot_sum * scale;
            const ACC_TYPE m_new = max(m_i, score);
            const ACC_TYPE p = exp(score - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);
            #pragma unroll
            for (int i = 0; i < DV_VEC; i++) {
                o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
            }
            l_i = l_i * scale_prev + p;
            m_i = m_new;
        }
    } else if (has_mask_q1 && has_softcap_q1) {
        for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
            const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
            const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
            const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
            const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
            ACC_TYPE dot_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
            }
            ACC_TYPE score = dot_sum * scale;
            score += slope * (ACC_TYPE)mask_ptr_q1[k_idx];
            score = logit_softcap * tanh(score / logit_softcap);
            const ACC_TYPE m_new = max(m_i, score);
            const ACC_TYPE p = exp(score - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);
            #pragma unroll
            for (int i = 0; i < DV_VEC; i++) {
                o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
            }
            l_i = l_i * scale_prev + p;
            m_i = m_new;
        }
    } else if (has_mask_q1) {
        for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
            const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
            const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
            const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
            const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
            ACC_TYPE dot_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
            }
            ACC_TYPE score = dot_sum * scale;
            score += slope * (ACC_TYPE)mask_ptr_q1[k_idx];
            const ACC_TYPE m_new = max(m_i, score);
            const ACC_TYPE p = exp(score - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);
            #pragma unroll
            for (int i = 0; i < DV_VEC; i++) {
                o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
            }
            l_i = l_i * scale_prev + p;
            m_i = m_new;
        }
    } else {
        for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
            const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
            const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
            const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
            const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
            ACC_TYPE dot_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
            }
            ACC_TYPE score = dot_sum * scale;
            score = logit_softcap * tanh(score / logit_softcap);
            const ACC_TYPE m_new = max(m_i, score);
            const ACC_TYPE p = exp(score - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);
            #pragma unroll
            for (int i = 0; i < DV_VEC; i++) {
                o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
            }
            l_i = l_i * scale_prev + p;
            m_i = m_new;
        }
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
#if GGML_FLASH_ATTN_USE_SUBGROUP
    const uint sg_lid = get_sub_group_local_id();
    const uint sg_id  = get_sub_group_id();
    const uint sg_n   = get_num_sub_groups();

    const ACC_TYPE sg_m = sub_group_reduce_max(m_i);
    if (sg_lid == 0) {
        local_m[sg_id] = sg_m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg_id == 0) {
        ACC_TYPE v = (sg_lid < sg_n) ? local_m[sg_lid] : -INFINITY;
        v = sub_group_reduce_max(v);
        if (sg_lid == 0) {
            local_m[0] = v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ACC_TYPE m_final = local_m[0];
#else
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ACC_TYPE m_final = local_m[0];
#endif
    m_final = max(m_final, m_sink);
    const ACC_TYPE renorm = exp(m_i - m_final);
    l_i *= renorm;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] *= renorm;
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    __local ACC_TYPE8 local_o_comp8[Q1_WG_SIZE];
    __local ACC_TYPE8 local_chunk[STAGE2_CHUNK_PAIRS * Q1_WG_SIZE];
#if GGML_FLASH_ATTN_USE_SUBGROUP
    ACC_TYPE sg_l = sub_group_reduce_add(l_i);
    if (sg_lid == 0) {
        local_l[sg_id] = sg_l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg_id == 0) {
        ACC_TYPE v = (sg_lid < sg_n) ? local_l[sg_lid] : 0.0f;
        v = sub_group_reduce_add(v);
        if (sg_lid == 0) {
            local_l[0] = v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(m_sink - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
#if GGML_FLASH_ATTN_USE_SUBGROUP
        for (int i = 0; i < DV_VEC; i++) {
            ACC_TYPE4 v = o_acc[i];
            const ACC_TYPE s0 = sub_group_reduce_add(v.s0);
            const ACC_TYPE s1 = sub_group_reduce_add(v.s1);
            const ACC_TYPE s2 = sub_group_reduce_add(v.s2);
            const ACC_TYPE s3 = sub_group_reduce_add(v.s3);
            if (sg_lid == 0) {
                local_o_comp[sg_id] = (ACC_TYPE4)(s0, s1, s2, s3);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (sg_id == 0) {
                ACC_TYPE4 t = (sg_lid < sg_n) ? local_o_comp[sg_lid] : (ACC_TYPE4)(0.0f);
                const ACC_TYPE r0 = sub_group_reduce_add(t.s0);
                const ACC_TYPE r1 = sub_group_reduce_add(t.s1);
                const ACC_TYPE r2 = sub_group_reduce_add(t.s2);
                const ACC_TYPE r3 = sub_group_reduce_add(t.s3);
                if (sg_lid == 0) {
                    local_o_comp[0] = (ACC_TYPE4)(r0, r1, r2, r3);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
#else
        for (int chunk = 0; chunk < DV_VEC; chunk += 2 * STAGE2_CHUNK_PAIRS) {
            const int chunk_end = min(chunk + 2 * STAGE2_CHUNK_PAIRS, DV_VEC);
            const int num_pairs = (chunk_end - chunk) / 2;
            for (int j = 0; j < num_pairs; j++) {
                const ACC_TYPE4 a0 = o_acc[chunk + 2 * j];
                const ACC_TYPE4 a1 = o_acc[chunk + 2 * j + 1];
                local_chunk[tid + j * Q1_WG_SIZE] = (ACC_TYPE8)(
                    a0.s0, a0.s1, a0.s2, a0.s3,
                    a1.s0, a1.s1, a1.s2, a1.s3
                );
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    for (int j = 0; j < num_pairs; j++) {
                        local_chunk[tid + j * Q1_WG_SIZE] += local_chunk[(tid + s) + j * Q1_WG_SIZE];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                for (int j = 0; j < num_pairs; j++) {
                    const ACC_TYPE8 r = local_chunk[j * Q1_WG_SIZE];
                    o_row[chunk + 2 * j]     = CONVERT_O_DATA4((ACC_TYPE4)(r.s0, r.s1, r.s2, r.s3) * l_inv);
                    o_row[chunk + 2 * j + 1] = CONVERT_O_DATA4((ACC_TYPE4)(r.s4, r.s5, r.s6, r.s7) * l_inv);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#endif
     } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

__kernel void flash_attn_f32_f16_q1_splitk_stage1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    const int split_k,
    global void* inter_void
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_group_id(1);
    const int split_k_idx = get_group_id(2);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global MASK_DATA_TYPE* mask_ptr_q1 =
        (mask_base != NULL) ? (const global MASK_DATA_TYPE*)(mask_base) : NULL;
    const int has_mask_q1 = (mask_ptr_q1 != NULL);
    const int has_softcap_q1 = (logit_softcap > 0.0f);
    const int fast_path_q1 = (!has_mask_q1 && !has_softcap_q1);

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE m_i = -INFINITY;

    const int kv_per_split = (n_kv + split_k - 1) / split_k;
    const int k_start = split_k_idx * kv_per_split;
    const int k_end = min(k_start + kv_per_split, n_kv);

    if (k_start < k_end) {
        if (fast_path_q1) {
            for (int k_idx = k_start + tid; k_idx < k_end; k_idx += Q1_WG_SIZE) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
                const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
                const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
                ACC_TYPE dot_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
                }
                const ACC_TYPE score = dot_sum * scale;
                const ACC_TYPE m_new = max(m_i, score);
                const ACC_TYPE p = exp(score - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);
                #pragma unroll
                for (int i = 0; i < DV_VEC; i++) {
                    o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
                }
                l_i = l_i * scale_prev + p;
                m_i = m_new;
            }
        } else if (has_mask_q1 && has_softcap_q1) {
            for (int k_idx = k_start + tid; k_idx < k_end; k_idx += Q1_WG_SIZE) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
                const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
                const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
                ACC_TYPE dot_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
                }
                ACC_TYPE score = dot_sum * scale;
                score += slope * (ACC_TYPE)mask_ptr_q1[k_idx];
                score = logit_softcap * tanh(score / logit_softcap);
                const ACC_TYPE m_new = max(m_i, score);
                const ACC_TYPE p = exp(score - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);
                #pragma unroll
                for (int i = 0; i < DV_VEC; i++) {
                    o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
                }
                l_i = l_i * scale_prev + p;
                m_i = m_new;
            }
        } else if (has_mask_q1) {
            for (int k_idx = k_start + tid; k_idx < k_end; k_idx += Q1_WG_SIZE) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
                const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
                const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
                ACC_TYPE dot_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
                }
                ACC_TYPE score = dot_sum * scale;
                score += slope * (ACC_TYPE)mask_ptr_q1[k_idx];
                const ACC_TYPE m_new = max(m_i, score);
                const ACC_TYPE p = exp(score - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);
                #pragma unroll
                for (int i = 0; i < DV_VEC; i++) {
                    o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
                }
                l_i = l_i * scale_prev + p;
                m_i = m_new;
            }
        } else {
            for (int k_idx = k_start + tid; k_idx < k_end; k_idx += Q1_WG_SIZE) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
                const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
                const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
                ACC_TYPE dot_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_sum += dot(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]));
                }
                ACC_TYPE score = dot_sum * scale;
                score = logit_softcap * tanh(score / logit_softcap);
                const ACC_TYPE m_new = max(m_i, score);
                const ACC_TYPE p = exp(score - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);
                #pragma unroll
                for (int i = 0; i < DV_VEC; i++) {
                    o_acc[i] = o_acc[i] * scale_prev + p * CONVERT_KV_ACC4(v_ptr[i]);
                }
                l_i = l_i * scale_prev + p;
                m_i = m_new;
            }
        }
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ACC_TYPE m_wg = local_m[0];

    const ACC_TYPE renorm = exp(m_i - m_wg);
    l_i *= renorm;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] *= renorm;
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ACC_TYPE l_wg = local_l[0];

    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    __local ACC_TYPE8 local_o_comp8[Q1_WG_SIZE];

    const int n_batch_local = get_num_groups(1) / n_head;
    const int entry_size = DV + 2;
    const int inter_stride = n_batch_local * n_head * entry_size;
    const int idx_base = split_k_idx * inter_stride + batch_idx * n_head * entry_size + head_idx * entry_size;
    global ACC_TYPE* inter_ptr = (global ACC_TYPE*)inter_void;

    if (tid == 0) {
        inter_ptr[idx_base + DV] = l_wg;
        inter_ptr[idx_base + DV + 1] = m_wg;
    }

    for (int i = 0; i + 1 < DV_VEC; i += 2) {
        const ACC_TYPE4 a0 = o_acc[i];
        const ACC_TYPE4 a1 = o_acc[i + 1];
        local_o_comp8[tid] = (ACC_TYPE8)(
            a0.s0, a0.s1, a0.s2, a0.s3,
            a1.s0, a1.s1, a1.s2, a1.s3
        );
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o_comp8[tid] += local_o_comp8[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            const ACC_TYPE8 r = local_o_comp8[0];
            inter_ptr[idx_base + i * 4 + 0] = r.s0;
            inter_ptr[idx_base + i * 4 + 1] = r.s1;
            inter_ptr[idx_base + i * 4 + 2] = r.s2;
            inter_ptr[idx_base + i * 4 + 3] = r.s3;
            inter_ptr[idx_base + i * 4 + 4] = r.s4;
            inter_ptr[idx_base + i * 4 + 5] = r.s5;
            inter_ptr[idx_base + i * 4 + 6] = r.s6;
            inter_ptr[idx_base + i * 4 + 7] = r.s7;
        }
    }
    if (DV_VEC & 1) {
        const int i = DV_VEC - 1;
        local_o_comp[tid] = o_acc[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            const ACC_TYPE4 r = local_o_comp[0];
            inter_ptr[idx_base + i * 4 + 0] = r.s0;
            inter_ptr[idx_base + i * 4 + 1] = r.s1;
            inter_ptr[idx_base + i * 4 + 2] = r.s2;
            inter_ptr[idx_base + i * 4 + 3] = r.s3;
        }
    }
}

__kernel void flash_attn_f32_f16_q1_splitk_stage2(
    const global void* inter_void,
    const global void* sinks_void,
    const ulong sinks_offset,
    const global void* o_void,
    const ulong o_offset,
    const int n_head,
    const int n_batch,
    const int split_k,
    const ulong o_nb1,
    const ulong o_nb2,
    const ulong o_nb3
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_group_id(0);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int entry_size = DV + 2;
    const int inter_stride = n_batch * n_head * entry_size;
    const int block_base = batch_idx * n_head * entry_size + head_idx * entry_size;

    const global ACC_TYPE* inter_ptr = (const global ACC_TYPE*)inter_void;

    const ACC_TYPE m_sink = (sinks_void != NULL)
        ? ((const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset))[head_idx]
        : -INFINITY;

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)((global char*)o_void + o_offset + o_row_offset);

    if (tid == 0) {
        ACC_TYPE m_max = -INFINITY;
        for (int k = 0; k < split_k; k++) {
            m_max = max(m_max, inter_ptr[k * inter_stride + block_base + DV + 1]);
        }
        if (sinks_void != NULL) m_max = max(m_max, m_sink);

        ACC_TYPE l_sum = 0.0f;
        ACC_TYPE sink_scale = 1.0f;
        if (sinks_void != NULL && m_sink > m_max) sink_scale = exp(m_max - m_sink);
        for (int k = 0; k < split_k; k++) {
            const int idx = k * inter_stride + block_base;
            l_sum += exp(inter_ptr[idx + DV + 1] - m_max) * inter_ptr[idx + DV];
        }
        if (sinks_void != NULL) {
            ACC_TYPE vs = (m_sink <= m_max) ? exp(m_sink - m_max) : 1.0f;
            l_sum = l_sum * sink_scale + vs;
        }

        ACC_TYPE l_inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;

        if (l_sum > 0.0f) {
            for (int i = 0; i + 1 < DV_VEC; i += 2) {
                ACC_TYPE4 s0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 s1 = (ACC_TYPE4)(0.0f);
                const int b0 = i * 4;
                const int b1 = (i + 1) * 4;
                for (int k = 0; k < split_k; k++) {
                    const int idx = k * inter_stride + block_base;
                    ACC_TYPE sc = exp(inter_ptr[idx + DV + 1] - m_max) * sink_scale;
                    s0 += (ACC_TYPE4)(inter_ptr[idx + b0 + 0] * sc, inter_ptr[idx + b0 + 1] * sc, inter_ptr[idx + b0 + 2] * sc, inter_ptr[idx + b0 + 3] * sc);
                    s1 += (ACC_TYPE4)(inter_ptr[idx + b1 + 0] * sc, inter_ptr[idx + b1 + 1] * sc, inter_ptr[idx + b1 + 2] * sc, inter_ptr[idx + b1 + 3] * sc);
                }
                o_row[i]   = CONVERT_O_DATA4(s0 * l_inv);
                o_row[i + 1] = CONVERT_O_DATA4(s1 * l_inv);
            }
            if (DV_VEC & 1) {
                const int i = DV_VEC - 1;
                ACC_TYPE4 s0 = (ACC_TYPE4)(0.0f);
                const int b0 = i * 4;
                for (int k = 0; k < split_k; k++) {
                    const int idx = k * inter_stride + block_base;
                    ACC_TYPE sc = exp(inter_ptr[idx + DV + 1] - m_max) * sink_scale;
                    s0 += (ACC_TYPE4)(inter_ptr[idx + b0 + 0] * sc, inter_ptr[idx + b0 + 1] * sc, inter_ptr[idx + b0 + 2] * sc, inter_ptr[idx + b0 + 3] * sc);
                }
                o_row[i] = CONVERT_O_DATA4(s0 * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

