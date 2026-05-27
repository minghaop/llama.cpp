#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GGML_FUSED_UNARY_GELU       0
#define GGML_FUSED_UNARY_GELU_ERF   1
#define GGML_FUSED_UNARY_GELU_QUICK 2
#define GGML_FUSED_UNARY_SILU       3
#define GGML_FUSED_UNARY_MISH       4

#define GELU_COEF_A      0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI   0.79788456080286535587989211986876f
#define SQRT_2_INV       0.70710678118654752440084436210484f

inline float ggml_apply_fused_unary(float x, int op) {
    if (op == GGML_FUSED_UNARY_GELU) {
        return 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
    }
    if (op == GGML_FUSED_UNARY_GELU_ERF) {
        return 0.5f*x*(1.0f + erf(x*SQRT_2_INV));
    }
    if (op == GGML_FUSED_UNARY_GELU_QUICK) {
        return x*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x)));
    }
    if (op == GGML_FUSED_UNARY_MISH) {
        const float sp = log(1.0f + exp(x));
        return x * tanh(sp);
    }
    // GGML_FUSED_UNARY_SILU
    return x / (1.0f + exp(-x));
}

kernel void kernel_add_row_unary_f32(
    global float * src0,
    ulong offset0,
    global float * src1,
    ulong offset1,
    global float * dst,
    ulong offsetd,
    int ne0,
    int nrows,
    int unary_op
) {
    src0 = (global float *) ((global char *) src0 + offset0);
    src1 = (global float *) ((global char *) src1 + offset1);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int i0 = (int) get_global_id(0);
    const int row = (int) get_global_id(1);
    if (i0 >= ne0 || row >= nrows) {
        return;
    }

    const int idx = row * ne0 + i0;
    const float x = src0[idx] + src1[i0];
    dst[idx] = ggml_apply_fused_unary(x, unary_op);
}

kernel void kernel_add_unary_f32(
    global float * src0,
    ulong offset0,
    global float * src1,
    ulong offset1,
    global float * dst,
    ulong offsetd,
    int ne,
    int unary_op
) {
    src0 = (global float *) ((global char *) src0 + offset0);
    src1 = (global float *) ((global char *) src1 + offset1);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int i = (int) get_global_id(0);
    if (i >= ne) {
        return;
    }

    const float x = src0[i] + src1[i];
    dst[i] = ggml_apply_fused_unary(x, unary_op);
}

kernel void kernel_add_add_f32(
    global float * src0,
    ulong offset0,
    global float * src1,
    ulong offset1,
    global float * src2,
    ulong offset2,
    global float * dst,
    ulong offsetd,
    int ne
) {
    src0 = (global float *) ((global char *) src0 + offset0);
    src1 = (global float *) ((global char *) src1 + offset1);
    src2 = (global float *) ((global char *) src2 + offset2);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int i = (int) get_global_id(0);
    if (i >= ne) {
        return;
    }

    dst[i] = src0[i] + src1[i] + src2[i];
}

kernel void kernel_add_add_unary_f32(
    global float * src0,
    ulong offset0,
    global float * src1,
    ulong offset1,
    global float * src2,
    ulong offset2,
    global float * dst,
    ulong offsetd,
    int ne,
    int unary_op
) {
    src0 = (global float *) ((global char *) src0 + offset0);
    src1 = (global float *) ((global char *) src1 + offset1);
    src2 = (global float *) ((global char *) src2 + offset2);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int i = (int) get_global_id(0);
    if (i >= ne) {
        return;
    }

    const float x = src0[i] + src1[i] + src2[i];
    dst[i] = ggml_apply_fused_unary(x, unary_op);
}
