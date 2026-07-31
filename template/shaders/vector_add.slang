// Vector add kernel: c[i] = a[i] + b[i].
//
// Each RWStructuredBuffer binds to one GpuBuffer. The register(uN) slot must
// match the order of the descriptorInfo() entries passed to the pipeline, and
// [numthreads(x, y, z)] is the local workgroup size. SV_DispatchThreadID is the
// global thread index across the whole dispatch.

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> c : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    c[tid.x] = a[tid.x] + b[tid.x];
}
