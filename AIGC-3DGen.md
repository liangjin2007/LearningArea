## 技术流

- 两步法: VAE + Latent Diffusion
```
3D Supervised VAE: [2024]Dora, [2024]Craftsman, [2024]Triposr, [2025]Hunyuan3d 2.0
UDF, SDF, MC, DMC

2D Supervised VAE: [2024]Trellis
  Dense Volumetric grids for 2D projections限制了分辨率。
  缺少3D拓扑约束，导致内部几何不正确，生成的网格可能是开的。
  使用了DINOv2 1B参数模型ViT features to SDF，导致非常重的注意力机制，增加了模型复杂度和导致不连贯的风险。
```

- 3D Shape VAEs
```
VecSet-based VAEs
  Sampled points and normals -> Transformer encoder -> VecSet -> Transformer decoder -> Output SDF
  监督学习
  [2023]3dshape2vecset
  [2024]CLAY
  [2025]TripoSG
  [2024]Dora
  [2025]Hunyuan3d 2.0


Sparse voxel-based VAEs
  [2024]XCube
  [2024]Trellis  enriches this representation with aggregated DINOv2 [22] features 把纹理和几何一起表示
  [2025]TripoSF 高分辨率
  mapping point normals or DINOv2 descriptors to continuous SDF fields remains a key bottleneck.是瓶颈所在。

```

## hitem3d相关文论
- [2025]Sparc3D
```
  Raw Mesh to watertight surface: Sparcubes(Sparse Deformable Marching Cubes)
    没理解：Our method begins by identifying a sparse set of activated voxels from the input mesh and performing a flood-fill to assign coarse signed labels. We then optimize grid-vertex deformations via gradient descent and refine them using a
    view-dependent 2D rendering loss
  在Sparcubes基础之上，引入Sparconv-VAE


```

