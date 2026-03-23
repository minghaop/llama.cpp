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