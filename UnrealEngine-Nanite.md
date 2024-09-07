# Nanite 虚拟几何系统 virtual geometry
Ref: https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf

Outline

- [1.VirtualGeometryInvestigation](#1VirtualGeometryInvestigation)
- [2.GPUDrivenPipeline](#2GPUDrivenPipeline)
- [3.TriangleClusterCulling](#3TriangleClusterCulling)
- [4.DecoupleVisibilityFromMaterial](#4DecoupleVisibilityFromMaterial)
- [5.ClusteringLOD](#5ClusteringLOD)

## 1.VirtualGeometryInvestigation

```
比Virtual texturing要难得多，因为它不单单是一个memory management问题；Geometry detail直接影响rendering cost；几何的filtering没那么简单。

是否能用Voxels?
  要实现最大的稀疏性，细节/sharp edges的自适应，不能牺牲ray casting performance，不能在平滑的地方浪费samples， 总会丢失细节， 那么是否仍然要存着原始的mesh来绘制它， 不能完全改变CG工作流，应该要支持各种各样的网格格式，
  必须支持UVs和tiling detail maps，
  虽然艺术家都讨厌UV mapping，但是不可否认它是一个procedural texturing surfaces的强大工具。
  ...
  整个管线都得改，这可能需要多个领域的研究才能替换显示曲面。


细分曲面？
  细分曲面只会更细，不会更简单。 细分曲面的case有的时候比典型的game low polys面数更多。
  为啥没调研LOD ？
    
Displacement maps?
  有缺点： 不能增加genus of a surface。 那对有些高genus的曲面比如自行车链条，the base mesh仍然会有大量的面； 另外生成normal map或者displacement map其实是某种uniform resampling， 对hard surface features会有破坏性； 对一般目的的简化来说，还不够好。

Points?
  点需要非常多，会出来很多洞, 怎么区分一个地方是洞还是应该填洞，

Triangles?
  对某种艺术风格也许能找到一种表示方法，但是Unreal 不能强加一种艺术风格。
```

## 2.GPUDrivenPipeline
```
UE's Renderer is retained mode design

  GPU scene representation persists across frames： 结合后面PPT的内容， 这里的意思应该是场景是GPU场景动态更新。
  Sparsely updated where things change.
  All vertex/index data in single large resource. i.e. all Nanite mesh data is stored in single large resources.

  每帧能决定哪些是可见的实例。
如果只绘制深度的话，在一次DrawIndirect中。 可在UE代码中搜索DrawIndirect, 可以找到很多相关的内容。

```

## 3.TriangleClusterCulling
```
即使有GPU Driven Pipeline，仍然要处理不大不小的工作（三角形还是太多）， triangle cluster culling是用来修剪掉不必要的工作。

● Group triangles into clusters
● Build bounding data for each cluster
● Cull clusters based on bounds
● Frustum culling
● Occlusion culling
  ● Occlusion cull against Hierarchical Z-Buffer (HZB)
  ● Calculate screen rect from bounds
  ● Test against lowest mip where screen rect <= 4x4 pixels
  ● What HZB? Haven't rendered anything yet
  ● Reproject z-buffer from previous frame into current frame? 近似及不保守
  ● 上一帧可见的物体很可能这一阵也可见。
  ● Two pass solution
    ● Draw what was visible in the previous frame
    ● Build HZB from that
    ● Draw what is visible now but wasn't in the last frame : Test the HZB to determine what is visible now but wasn’t in the last frame and draw anything that’s new
```

## 4. DecoupleVisibilityFromMaterial
??
```
需要解决的问题：
● Switching shaders during rasterization
● Overdraw/Overshade for material eval
● Depth prepass to avoid overdraw
● Pixel quad inefficiencies from dense meshes

解决办法调研：
● REYES : Object space shading
● Texture space shading : 
● Deferred materials

Visibility Buffer：
● Write geometry data to screen
  ● Depth : InstanceID : TriangleID
● Material shader per pixel:
  ● Load VisBuffer
  ● Load instance transform
  ● Load 3 vert indexes
  ● Load 3 positions
  ● Transform positions to screen
  ● Derive barycentric coordinates for pixel
  ● Load and lerp attributes
```

## 5.ClusteringLOD
```
cost 应该要跟屏幕分辨率成正比，而不是跟三角形数成正比。

Memory bandwidth

Cluster hierarchy
● Decide LOD on a cluster basis
● Build a hierarchy of LODs
  ● Simplest is tree of clusters
  ● Parents are the simplified versions of their children

LOD run-time
● Find cut of the tree for desired LOD
● View dependent based on perceptual difference

At run time we find a cut of the tree that matches the desired LOD. That means different parts of the same mesh can be at different levels of detail based on what’s needed.

Streaming
● Entire tree doesn’t need to be in memory at once
● Can mark any cut of the tree as leaves and toss丢弃 the rest
● Request data on demand during rendering
  ● Like virtual texturing

LOD cracks
● If each cluster decides LOD independent from neighbors, cracks!
● Naive solution:
  ● Lock shared boundary edges during simplification: 这个之前在某个项目中曾经做过这个事情，没记错的话是在Hypereal时期做的对SLAM mesh进行并行实时简化。但是在边界上还是有许多三角形。
  ● Independent clusters will always match at boundaries

Locked boundaries
● Can detect these cases during build
● Group clusters
  ● Force them to make the same LOD decision
  ● Now free to unlock shared edges and collapse them

LOD cracks solution




```

## Keywords
```
HZB
```
## Reference
