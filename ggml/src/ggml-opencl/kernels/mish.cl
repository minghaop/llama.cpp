#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// mish
//------------------------------------------------------------------------------
kernel void kernel_mish(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    float sp = log(1.0f + exp(x));      // softplus(x)
    dst[get_global_id(0)] = x * tanh(sp);
}

kernel void kernel_mish_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    float4 sp = log(1.0f + exp(x));    // softplus(x)
    dst[get_global_id(0)] = x * tanh(sp);
}

kernel void kernel_mish_f32_nd(
    global void * p_src0_base, ulong off_src0_abs,
    global void * p_dst_base,  ulong off_dst_abs,
    int ne00, int ne01, int ne02, int ne03,
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
    int ne10, int ne11, int ne12, int ne13,
    ulong nb10, ulong nb11, ulong nb12, ulong nb13
) {
    int i0 = get_global_id(0);
    int i1 = get_global_id(1);
    int i2 = get_global_id(2);

    if (i0 < ne10 && i1 < ne11 && i2 < ne12) {
        for (int i3 = 0; i3 < ne13; ++i3) {
            ulong src_offset = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
            global const float *src_ptr = (global const float *)((global char *)p_src0_base + off_src0_abs + src_offset);
            float x = *src_ptr;
            float sp = log(1.0f + exp(x));

            ulong dst_offset = (ulong)i0*nb10 + (ulong)i1*nb11 + (ulong)i2*nb12 + (ulong)i3*nb13;
            global float *dst_ptr = (global float *)((global char *)p_dst_base + off_dst_abs + dst_offset);
            *dst_ptr = x * tanh(sp);
        }
    }
}
