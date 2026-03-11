## 图形编程介绍
[官方文档](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/graphics-programming-for-unreal-engine)

- 入门
```
入手之处： FDeferredShadingSceneRenderer::Render
profilegpu 命令可查看绘制事件，然后在 Visual Studio 中对绘制事件名称进行 Find in Files 操作，找出对应的 C++ 实现。

控制台命令
stat unit	显示整体帧长、游戏线程时长、渲染线程时长、GPU 时长。最长者为瓶颈。不过，CPU 时间包含空闲时间，所以只有在其为最长者且为独立时才会成为瓶颈。
Ctrl+Shift+. 或 recompileshaders changed	重新编译上次保存 .usf 文件后发生变化的着色器。这将在加载后自动进行。
Ctrl+Shift+; 或 profilegpu	测量渲染视图的 GPU 时间。可在弹出的 UI 或引擎日志中查看结果。
Vis 或 VisualizeTexture	可视化显示多种渲染目标的内容，并可保存为 bmp 文件。
show x	切换特定的显示标记。使用 show 来列出各种 showflag 及其当前状态。在编辑器中，使用视口 UI 作为替代。
pause	暂停游戏，但继续渲染。任何模拟渲染工作都将停止。
slomo x	变更游戏速度。此命令有助于在进行分析时减缓时间而不跳过模拟工作。例如 slomo .01
debugcreateplayer 1	用于测试分屏游戏。
r.CompositionGraphDebug	执行后可对某一帧的复合图形进行单帧转储（后期处理及光照）。
r.DumpShaderDebugInfo	设为 1 时会把所有被编译的着色器的调试信息转储到 GameName/Saved/ShaderDebugInfo 中。
r.RenderTargetPoolTest	清除 rendertarget 池返回的、带有特殊颜色的纹理，以便追踪颜色泄漏 bug。
r.SetRes	设置当前游戏视图的显示分辨率。在编辑器中不起作用。
r.ViewportTest	可用于测试不同视口矩形配置（仅在游戏中），因为使用 Matinee/Editor 时可能会出现这些情况。

命令行参数
-d3ddebug	启用 D3D11 调试层，可用于捕捉 API 错误。
-sm4	强制功能层 SM4 使用 D3D11 RHI。
-opengl3 / -opengl4	在特定功能层强制使用 OpenGL RHI。
-dx11	当前 Windows 的默认设置
-dx12	实验性
-featureleveles2	使用编辑器时忽略，此时必须使用 UI
-featureleveles31	使用编辑器时忽略，此时必须在 Editor Preferences 中将其启用
-ddc=noshared	防止使用网络（共享）派生数据缓存。可用于调试着色器缓存问题。
```

- 模块
```
渲染器模块和RHI 模块
RHI 模块：渲染 API 的接口

IRendererModule
FSceneInterface
```

### Scene Representation
- 主要场景类
```
FScene
  Primitive Components
  Octree

UWorld	包含多个可交互的 Actor 和组件的世界场景。关卡可以流送进入和退出世界场景，且程序中可以同时有多个世界场景处于激活状态。
ULevel	一同加载/卸载并保存在同一地图文件中的 Actor 和组件合集。
USceneComponent	需要添加到 FScene 中的任意对象的基础类，如光照、网格体、雾等。
UPrimitiveComponent	可渲染或进行物理交互的任意资源的基础类。也可以作为可视性剔除的粒度和渲染属性规范（投射阴影等）。与所有 UObjects 一样，游戏线程拥有所有变量和状态，渲染线程不应直接对其进行访问。
ULightComponent	代表光源。渲染器负责计算和添加其对场景的贡献。
FScene	UWorld 的渲染器版本。对象仅在其被添加到 FScene（注册组件时调用）后才会存在于渲染器中。渲染线程拥有 FScene 中的所有状态，游戏线程无法直接对其进行修改。
FPrimitiveSceneProxy	UPrimitiveComponent 的渲染器版本，为渲染线程映射 UPrimitiveComponent 状态。存在于引擎模块中，用于划分为子类以支持不同类型的基元（骨架、刚体、BSP 等）。实现某些非常重要的函数，如 GetViewRelevance、DrawDynamicElements 等。
FPrimitiveSceneInfo	内部渲染器状态（FRendererModule 实现专有），对应于 UPrimitiveComponent 和 FPrimitiveSceneProxy。存在于渲染器模块中，因此引擎看不到它。
FSceneView	单个视图到一个 FScene 的引擎代表。视图可以通过对 FSceneRenderer::Render 的不同调用的不同视图来渲染（多编辑器视口）或通过对 FSceneRenderer::Render 的同一调用中的多个视图来渲染（分屏游戏）。为每个帧构建新视图。
FViewInfo	视图的内部渲染器代表，存在于渲染器模块中。
FSceneViewState	ViewState 存储有关在多个帧中需要的某个视图的私有渲染器信息。在游戏中，每个 ULocalPlayer 只有一个视图状态。
FSceneRenderer	为每个帧创建的类，用于封装跨帧的临时对象。

代码所在的模块
  引擎模块	                                渲染器模块
  UWorld	                                  FScene
  UPrimitiveComponent/FPrimitiveSceneProxy	FPrimitiveSceneInfo
  FSceneView	                              FViewInfo
  ULocalPlayer	                            FSceneViewState
  ULightComponent/FLightSceneProxy	        FLightSceneInfo

所属的线程

  游戏线程	                                  渲染线程
  UWorld	                                    FScene
  UPrimitiveComponent	                        FPrimitiveSceneProxy / FPrimitiveSceneInfo
   	                                          FSceneView / FViewInfo
  ULocalPlayer	                              FSceneViewState
  ULightComponent	                            FLightSceneProxy / FLightSceneInfo
```

- 材质类
```
FMaterial	An interface to a material used for rendering. Provides access to material properties (e.g. blend mode). Contains a shader map used by the renderer to retrieve individual shaders.
FMaterialResource	UMaterial's implementation of the FMaterial interface.
FMaterialRenderProxy	A material's representation on the rendering thread. Provides access to an FMaterial interface and the current value of each scalar, vector, and texture parameter.
UMaterialInterface	[abstract] Game thread interface for material functionality. Used to retrieve the FMaterialRenderProxy used for rendering and the UMaterial that is used as the source.
UMaterial	A material asset. Authored as a node graph. Computes material attributes used for shading, sets blend mode, etc.
UMaterialInstance	[abstract] An instance of a UMaterial. Uses the node graph in the UMaterial but provides different parameters (scalars, vectors, textures, static switches). Each instance has a parent UMaterialInterface. Therefore a material instance's parent may be a UMaterial or another UMaterialInstance. This creates a chain that will eventually lead to a UMaterial.
UMaterialInstanceConstant	A UMaterialInstance that may only be modified in the editor. May provide scalar, vector, texture, and static switch parameters.
UMaterialInstanceDynamic	A UMaterialInstance that may be modified at runtime. May provide scalar, vector, and texture parameters. It cannot provide static switch parameters and it cannot be the parent of another UMaterialInstance.
```

- Primitive components and proxies 基元组件和代理
```
```

- FPrimitiveSceneProxy and FPrimitiveSceneInfo
```
```

- Scene rendering order 场景渲染顺序
```
渲染器按照其希望将数据整合给渲染目标的顺序处理场景。例如，仅 Depth 的通道会比 Base 通道先渲染，先填充 Heirarchical Z (HiZ)，从而降低基础通道中的着色消耗。此顺序是按通道函数在 C++ 中调用的顺序静态决定的。
```

- Relevance 相关性
```
FPrimitiveViewRelevance is the information on what effects (and therefore passes) are relevant to the primitive. A primitive may have multiple elements with different relevance, so FPrimitiveViewRelevance is effectively a logical OR of all the element's relevancies. This means that a primitive can have both opaque and translucent relevance, or dynamic and static relevance; they are not mutually exclusive.
FPrimitiveViewRelevance also indicates whether a primitive needs to use the dynamic and/or static rendering path with bStaticRelevance and bDynamicRelevance.
```

- Drawing Policies 绘制规则
```
绘制规则包括通过通道特定的着色器渲染网格体的逻辑。它们使用 FVertexFactory 接口来抽取网格体类型，并使用 FMaterial 接口来抽取材质详情。在最底层，一条绘制规则会负责一组网格体材质着色器以及一个顶点工厂，将顶点工厂的缓冲区与渲染硬件接口 (RHI) 绑定，将网格体材质着色器与 RHI 绑定，设置适当的着色器参数，然后执行 RHI 绘制调用。
```

- 绘制规则方法
```
函数	描述
Constructor	从给定的顶点工厂和材质着色器地图，并存储这些引用。
CreateBoundShaderState	为绘制规则创建 RHI 边界着色器状态。
Matches/Compare	提供排列绘制规则与静态绘制列表中的其他项目的方法。Matches 必须比较 DrawShared 依赖的所有因素。
DrawShared	设置在从 Matches 返回 True 的绘制规则之间一致的 RHI 状态。例如，大多数绘制规则会为材质和顶点工厂排序，因此着色器参数只依赖可以设置的材质，并且可以绑定特定于该顶点工厂的顶点缓冲区。应尽可能在此处设置状态，而非在 SetMeshRenderState 设置，因为 DrawShared 在静态渲染路径中调用较少。
SetMeshRenderState	设置特定于此网格体的 RHI 状态，或 DrawShared 中未设置的任何项目。这比 DrawShared 调用的次数多得多，因此此处性能非常重要。
DrawMesh	实际发出 RHI 绘制调用。
```

## 并行渲染
- ![并行渲染图](https://d1iv7db44yhgxn.cloudfront.net/documentation/images/54910761-17ed-4578-af6c-3638b5043a9e/parallel_rendering_00.png)

## 渲染依赖图
https://dev.epicgames.com/documentation/zh-cn/unreal-engine/render-dependency-graph-in-unreal-engine

- RDG编程指南
```

1.着色器参数结构体
  BEGIN_SHADER_PARAMETER_STRUCT(FMyShaderParameters, /** MODULE_API_TAG */)
  		SHADER_PARAMETER(FVector2D, ViewportSize)
  		SHADER_PARAMETER(FVector4, Hello)
  		SHADER_PARAMETER(float, World)
  		SHADER_PARAMETER_ARRAY(FVector, FooBarArray, [16])
   
  		SHADER_PARAMETER_TEXTURE(Texture2D, BlueNoiseTexture)
  		SHADER_PARAMETER_SAMPLER(SamplerState, BlueNoiseSampler)
   
  		SHADER_PARAMETER_TEXTURE(Texture2D, SceneColorTexture)
  		SHADER_PARAMETER_SAMPLER(SamplerState, SceneColorSampler)
   
  		SHADER_PARAMETER_UAV(RWTexture2D, SceneColorOutput)
  	END_SHADER_PARAMETER_STRUCT()

2.编译时反射元数据：
  const FShaderParametersMetadata* ParameterMetadata = FMyShaderParameters::FTypeInfo::GetStructMetadata();

3.着色器绑定:
  着色器参数结构体与 FShader 成对提供，以生成提交到RHI命令列表所需的绑定。
  
  你可以通过在 FShader 派生类中将参数结构体声明为 FParameters 类型来实现。
  
  它可以作为内联定义或通过using / typedef指令实现。然后，使用 SHADER_USE_PARAMETER_STRUCT 宏为将注册绑定的类生成一个构造函数。
  
  第一个着色器类：
  
  	class FMyShaderCS : public FGlobalShader
  	{
  		DECLARE_GLOBAL_SHADER(FMyShaderCS);
   
  		// 生成一个构造函数，该构造函数将使用此FShader实例注册FParameter绑定。
  		SHADER_USE_PARAMETER_STRUCT(FMyShaderCS, FGlobalShader);
   
  		// 将FParameters类型分配给着色器——使用内联定义或using指令。
  		using FParameters = FMyShaderParameters;
  	};


4.将着色器参数绑定到RHI命令列表是通过实例化结构体、填充数据并调用 SetShaderParameters 辅助函数来完成的。
  TShaderMapRef<FMyShaderCS> ComputeShader(View.ShaderMap);
	RHICmdList.SetComputeShader(ComputeShader.GetComputeShader());
 
	FMyShaderCS::FParameters ShaderParameters;
 
	// 参数赋值。
	ShaderParameters.ViewportSize = View.ViewRect.Size();
	ShaderParameters.World = 1.0f;
	ShaderParameters.FooBarArray[4] = FVector(1.0f, 0.5f, 0.5f);
 
	// 参数提交。
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
 
	RHICmdList.DispatchComputeShader(GroupCount.X, GroupCount.Y, GroupCount.Z);

5.统一缓冲区
  统一缓冲区（Uniform Buffer） 将着色器参数作为一组RHI资源，本身将作为着色器参数绑定。每个统一缓冲区都在HLSL中定义了一个全局命名空间。使用 BEGIN_UNIFORM_BUFFER_STRUCT 和 END_UNIFORM_BUFFER_STRUCT 宏声明统一缓冲区。
  
  定义统一缓冲区：
  
  	BEGIN_UNIFORM_BUFFER_STRUCT(FSceneTextureUniformParameters, RENDERER_API)
  		SHADER_PARAMETER_TEXTURE(Texture2D, SceneColorTexture)
  		SHADER_PARAMETER_SAMPLER(SamplerState, SceneColorTextureSampler)
  		SHADER_PARAMETER_TEXTURE(Texture2D, SceneDepthTexture)
  		SHADER_PARAMETER_SAMPLER(SamplerState, SceneDepthTextureSampler)
   
  		// ...
  	END_UNIFORM_BUFFER_STRUCT()


  在C++源文件中使用 IMPLEMENT_UNIFORM_BUFFER_STRUCT 向着色器系统注册统一缓冲区定义并生成其HLSL定义。
    实现统一缓冲区：
    	IMPLEMENT_UNIFORM_BUFFER_STRUCT(FSceneTextureUniformParameters, "SceneTexturesStruct")
 
  统一缓冲区参数由着色器自动生成，使用 UniformBuffer.Member 语法编译和访问。
  
  HLSL中的统一缓冲区：
  
  	// 包含统一缓冲区声明的生成文件。由Common.ush自动包含。
  	#include "/Engine/Generated/GeneratedUniformBuffers.ush"
   
  	// 引用统一缓冲区成员（类似于结构体）。
  	Texture2DSample(SceneTexturesStruct.SceneColorTexture, SceneTexturesStruct.SceneColorTextureSampler);

  现在，SHADER_PARAMTER_STRUCT_REF 宏可用于将统一缓冲区作为参数包含在父着色器参数结构体中。
    	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
    		// ...
     
    		// 定义一个引用计数的TUniformBufferRef<FSceneTextureUniformParameters>实例。
    		SHADER_PARAMETER_STRUCT_REF(FSceneTextureUniformParameters, SceneTextures)
    	END_SHADER_PARAMETER_STRUCT()



6.静态绑定
每个着色器的着色器参数都是唯一绑定的，每个着色器阶段（例如，顶点和像素）都需要自己的着色器。使用 Set{Graphics, Compute}PipelineState ，在RHI命令列表中将着色器作为管线状态对象（Pipeline State Object）（PSO）绑定在一起。
注意：在命令列表绑定一个管线状态 会使所有着色器绑定 无效。

设置PSO后，需要绑定所有着色器参数。例如，考虑让一组典型绘制调用的命令流共享PSO。

设置PSO A
对于每个绘制调用
  设置顶点着色器参数
  设置像素着色器参数
  绘制
设置PSO B
对于每个绘制调用
  设置顶点着色器参数
  设置像素着色器参数
  绘制
这种方法的一个问题是渲染器中的网格绘制命令会被缓存并在多个通道和视图之间共享。为每帧的每个通道/视图组合生成一组独特的绘制命令是非常低效的。但是，网格绘制命令还需要知道通道/视图统一缓冲区资源，以便正确绑定它们。为了解决此问题，统一缓冲区使用了一个 静态 绑定模型。

使用静态绑定声明时，统一缓冲区直接绑定到RHI命令列表的 静态插槽，而不是为每个单独的着色器提供的 唯一插槽。当着色器请求统一缓冲区时，命令列表直接从静态插槽中提取绑定。现在，绑定以 通道 频率发生，而非 PSO 频率。

采用与上面相同的示例，但着色器输入来自静态统一缓冲区：

设置静态统一缓冲区
设置PSO A
对于每个绘制调用
  绘制
设置PSO B
  对于每个绘制调用
绘制
此模型允许每个绘制调用从命令列表继承着色器绑定。

定义静态统一缓冲区
要使用静态绑定来定义统一缓冲区，请使用 IMPLEMENT_STATIC_UNIFORM_BUFFER_STRUCT 宏。需要额外的插槽声明。它由 IMPLEMENT_STATIC_UNIFORM_BUFFER_SLOT 宏指定。
多个静态统一缓冲区定义可以引用同一个静态插槽，但一次只能绑定其中一个。最好尽可能重用插槽，以减少引擎中插槽的总数。
静态统一缓冲区：

	// 按名称定义一个唯一的静态插槽。
	IMPLEMENT_STATIC_UNIFORM_BUFFER_SLOT(SceneTextures);
 
	// 使用SceneTextures插槽的静态绑定定义SceneTexturesStruct统一缓冲区。
	IMPLEMENT_STATIC_UNIFORM_BUFFER_STRUCT(FSceneTextureUniformParameters, "SceneTexturesStruct", SceneTextures);
 
	// 定义具有相同静态插槽的MobileSceneTextures统一缓冲区。一次只能绑定一个。
	IMPLEMENT_STATIC_UNIFORM_BUFFER_STRUCT(FMobileSceneTextureUniformParameters, "MobileSceneTextures", SceneTextures);

  使用 RHICmdList.SetStaticUniformBuffers 方法绑定静态统一缓冲区。RDG在执行每个通道之前自动将静态统一缓冲区绑定到命令列表。任何静态统一缓冲区都应包含在通道参数结构体中。


7.渲染图生成器
{
		FRDGBuilder GraphBuilder(RHICmdList);
 
		FMyShaderCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMyShaderCS::FParameters>();
		//...
		PassParameters->SceneColorTexture = SceneColor;
		PassParameters->SceneColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp>::GetRHI();
		PassParameters->SceneColorOutput = GraphBuilder.CreateUAV(NewSceneColor);
 
		GraphBuilder.AddPass(
			// 使用printf语义，用于分析器的通道友好名称。
			RDG_EVENT_NAME("MyShader %d%d", View.ViewRect.Width(), View.ViewRect.Height()),
			// 提供给RDG的参数。
			PassParameters,
			// 发出计算命令。
			ERDGPassFlags::Compute,
			// 推迟到执行。可以与其他通道并行执行。
			[PassParameters, ComputeShader, GroupCount] (FRHIComputeCommandList& RHICmdList)
		{
			FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, PassParameters, GroupCount);
		});
 
		// 执行图。
		GraphBuilder.Execute();
	}


8.RDG资源和视图
// 创建一个新的临时纹理实例。此时未分配GPU内存，仅分配了描述符。
	FRDGTexture* Texture = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(...), TEXT("MyTexture"));
 
	// 无效！将触发断言。如果在通道上声明，则仅允许在通道Lambda中使用！
	FRHITexture* TextureRHI = Texture->GetRHI();
 
	// 创建一个新的UAV，引用特定mip级别的纹理。
	FRDGTextureUAV* TextureUAV = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(Texture, MipLevel));
 
	// 无效！
	FRHIUnorderedAccessView* UAVRHI = TextureUAV->GetRHI();
 
	// 创建一个新的临时结构化缓冲区实例。
	FRDGBuffer* Buffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(...), TEXT("MyBuffer"));
 
	// 无效！
	FRHIBuffer* BufferRHI= Buffer->GetRHI();
 
	// 创建一个新的SRV，引用具有R32浮点格式的缓冲区。
	FRDGBufferSRV* BufferSRV = GraphBuilder.CreateSRV(Buffer, PF_R32_FLOAT);
 
	// 无效！
	FRHIShaderResourceView* SRVRHI = TextureSRV->GetRHI();


```


