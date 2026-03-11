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
- 
