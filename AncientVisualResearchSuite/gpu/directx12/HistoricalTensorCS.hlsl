// DirectX 12 / Shader Model 6 compute kernel.
RWStructuredBuffer<float> Output : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Output[id.x] = A[id.x] + B[id.x];
}
