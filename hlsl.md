# hlsl语法
```
线程组结构
[numthreads(THREAD_COUNT,1,1)]

类型
SV_VertexID
SV_POSITION
SV_TARGET
SV_InstanceID
SV_GroupID
SV_DispatchThreadID
SV_GroupThreadID
TEXCOORD0
SV_Target0
SV_Target1
SV_RenderTargetArrayIndex
SV_ClipDistance1
SV_ViewPortArrayIndex
StructuredBuffer<float4>
RWStructuredBuffer<float4>

register(t4, space0)
register(u0, space1)

Texture2D

groupshared

函数
GroupMemoryBarrier();

变量
GGroupThreadId.x

```

# compute shader
- https://docs.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader






