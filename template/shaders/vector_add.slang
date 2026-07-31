// Vector add kernel: c[i] = a[i] + b[i].
//
// Each RWStructuredBuffer binds to one GpuBuffer. The vk::binding(N, 0) slot must
// match the order of the descriptorInfo() entries passed to the pipeline, and
// [numthreads(x, y, z)] is the local workgroup size. SV_DispatchThreadID is the
// global thread index across the whole dispatch.

[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    c[tid.x] = a[tid.x] + b[tid.x];
}
