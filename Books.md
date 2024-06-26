# 书单和课程目录
- An Introduction to Physics based animation
- RayTracingGems_v1.4
- Realtime Rendering 4th Edition
- Fluid Simulation for Computer Graphics, Second Edition
- Contact and Friction Simulation for Computer Graphics
- 3D Game Design with Unreal Engine 4 and Blender
- Computational Geometry Algorithms and Applications
- GPU Gems 1
- GPU Gems 2
- GPU Gems 3
- GPU Pro 1
- GPU Pro 2
- GPU Pro 3
- GPU Pro 4
- GPUZen Advanced Rendering Techniques
- Numerical Methods for Engineers
- ComputerAnimationAlgorithmsAndTechniques.3rdEdition
- Video Tracking : theory and practise
- ComputerVision-cs4476
- Practical Rendering And Computation With Direct3D11
- 统计学习方法 李航
- Physical Based Rendering
- OpenGL https://learnopengl.com/book/book_pdf.pdf
- UnrealEngine官方程序设计范式分析


## An Introduction to Physics based animation
```
I. A Simple Start: Particle Dynamics
A. A Passive Particle in a Velocity Field
B. A Particle with Mass
C. Spring-Mass Systems
II. Mathematical Models
A. Physical Laws
1. Newton’s Laws of Motion
2. Conservation of Mass, Momentum, Energy
B. Materials
1. Rigid Bodies
2. Soft Bodies (Elasticity)
3. Fluids
III. Spatial Discretization
A. Lagrangian vs. Eulerian
B. Grids, Meshes, Particles
C. Interpolation
D. Finite Differences
E. Finite Elements
V. Temporal Discretization
A. Explicit
1. Trapezoidal Rule vs. Midpoint
Method
2. Symplectic Euler
B. Implicit Integration
VI. Constraints
A. Bilateral and Unilateral Constraints
B. Soft vs. Hard Constraints
C. Penalty Forces, Lagrange Multipliers, Generalized Coordinates
D. Practical Rigid Body Systems
E. Non-penetration Constraints, Collisions, and Contact
```


## RayTracingGems_v1.4
```
PART I: Ray Tracing Basics ��������������������������������������������������������������� 5
Chapter 1: Ray Tracing Terminology���������������������������������������������������������� 7
1.1 Historical Notes ......................................................................................7
1.2 Definitions ...............................................................................................8
Chapter 2: What is a Ray? ������������������������������������������������������������������������ 15
2.1 Mathematical Description of a Ray .......................................................15
2.2 Ray Intervals..........................................................................................17
2.3 Rays in DXR ...........................................................................................18
2.4 Conclusion .............................................................................................19
Chapter 3: Introduction to DirectX Raytracing ����������������������������������������� 21
3.1 Introduction ...........................................................................................21
3.2 Overview ................................................................................................21
3.3 Getting Started ......................................................................................22
3.4 The DirectX Raytracing Pipeline ...........................................................23
3.5 New HLSL Support for DirectX Raytracing ...........................................25
3.6 A Simple HLSL Ray Tracing Example ...................................................28
3.7 Overview of Host Initialization for DirectX Raytracing ..........................30
3.8 Basic DXR Initialization and Setup ........................................................31
 3.9 Ray Tracing Pipeline State Objects .....................................................37
3.10 Shader Tables ......................................................................................41
3.11 Dispatching Rays .................................................................................43
3.12 Digging Deeper and Additional Resources .........................................44
3.13 Conclusion ...........................................................................................45
Chapter 4: A Planetarium Dome Master Camera������������������������������������� 49
4.1 Introduction ...........................................................................................49
4.2 Methods .................................................................................................50
4.3 Planetarium Dome Master Projection Sample Code ...........................58
Chapter 5: Computing Minima and Maxima of Subarrays ������������������������ 61
5.1 Motivation ..............................................................................................61
5.2 Naive Full Table Lookup ........................................................................62
5.3 The Sparse Table Method ......................................................................62
5.4 The (Recursive) Range Tree Method .....................................................64
5.5 Iterative Range Tree Queries ................................................................66
5.6 Results ..................................................................................................69
5.7 Summary ...............................................................................................69
PART II: Intersections and Efficiency ���������������������������������������������� 75
Chapter 6: A Fast and Robust Method for Avoiding Self-Intersection ������ 77
6.1 Introduction ...........................................................................................77
6.2 Method ...................................................................................................78
6.3 Conclusion .............................................................................................84
Chapter 7: Precision Improvements for Ray/Sphere Intersection ���������� 87
7.1 Basic Ray/Sphere Intersection .............................................................87
7.2 Floating-Point Precision Considerations ..............................................89
7.3 Related Resources ................................................................................93
Chapter 8: Cool Patches: A Geometric Approach to 
Ray/Bilinear Patch Intersections������������������������������������������������������������� 95
8.1 Introduction and Prior Art .....................................................................95
8.2 GARP Details .......................................................................................100
8.3 Discussion of Results ..........................................................................102
8.4 Code.....................................................................................................105
Chapter 9: Multi-Hit Ray Tracing in DXR ������������������������������������������������ 111
9.1 Introduction .........................................................................................111
9.2 Implementation ...................................................................................113
9.3 Results ................................................................................................119
9.4 Conclusions .........................................................................................124
Chapter 10: A Simple Load-Balancing Scheme with 
High Scaling Efficiency �������������������������������������������������������������������������� 127
10.1 Introduction .......................................................................................127
10.2 Requirements ....................................................................................128
10.3 Load Balancing..................................................................................128
10.4 Results ..............................................................................................132
PART III: Reflections, Refractions, and Shadows �������������������������� 137
Chapter 11: Automatic Handling of Materials in Nested Volumes���������� 139
11.1 Modeling Volumes .............................................................................139
11.2 Algorithm ..........................................................................................142
11.3 Limitations ........................................................................................146
Chapter 12: A Microfacet-Based Shadowing Function to 
Solve the Bump Terminator Problem����������������������������������������������������� 149
12.1 Introduction .......................................................................................149
12.2 Previous Work ...................................................................................150
12.3 Method ...............................................................................................151
12.4 Results ..............................................................................................157
Chapter 13: Ray Traced Shadows: Maintaining Real-Time 
Frame Rates ������������������������������������������������������������������������������������������ 159
13.1 Introduction .......................................................................................159
13.2 Related Work .....................................................................................161
13.3 Ray Traced Shadows .........................................................................162
13.4 Adaptive Sampling ............................................................................164
13.5 Implementation .................................................................................171
13.6 Results ..............................................................................................175
13.7 Conclusion and Future Work ............................................................179
Chapter 14: Ray-Guided Volumetric Water Caustics in 
Single Scattering Media with DXR���������������������������������������������������������� 183
14.1 Introduction .......................................................................................183
14.2 Volumetric Lighting and Refracted Light ..........................................186
14.3 Algorithm ..........................................................................................189
14.4 Implementation Details.....................................................................197
14.5 Results ..............................................................................................198
14.6 Future Work .......................................................................................200
14.7 Demo .................................................................................................200
PART IV: Sampling ������������������������������������������������������������������������ 205
Chapter 15: On the Importance of Sampling ������������������������������������������ 207
15.1 Introduction .......................................................................................207
15.2 Example: Ambient Occlusion ............................................................208
15.3 Understanding Variance ....................................................................213
15.4 Direct Illumination ............................................................................216
15.5 Conclusion .........................................................................................221
Chapter 16: Sampling Transformations Zoo ������������������������������������������ 223
16.1 The Mechanics of Sampling ..............................................................223
16.2 Introduction to Distributions .............................................................224
16.3 One-Dimensional Distributions ........................................................226
16.4 Two-Dimensional Distributions ........................................................230
16.5 Uniformly Sampling Surfaces ...........................................................234
16.6 Sampling Directions ..........................................................................239
16.7 Volume Scattering .............................................................................243
16.8 Adding to the Zoo Collection .............................................................244
Chapter 17: Ignoring the Inconvenient When Tracing Rays�������������������� 247
17.1 Introduction .......................................................................................247
17.2 Motivation ..........................................................................................247
17.3 Clamping ...........................................................................................250
17.4 Path Regularization...........................................................................251
17.5 Conclusion .........................................................................................252
Chapter 18: Importance Sampling of Many Lights on the GPU��������������� 255
18.1 Introduction .......................................................................................255
18.2 Review of Previous Algorithms .........................................................257
18.3 Foundations .......................................................................................259
18.4 Algorithm ..........................................................................................265
18.5 Results ..............................................................................................271
18.6 Conclusion .........................................................................................280
PART V: Denoising and Filtering ��������������������������������������������������� 287
Chapter 19: Cinematic Rendering in UE4 with Real-Time 
Ray Tracing and Denoising��������������������������������������������������������������������� 289
19.1 Introduction .......................................................................................289
19.2 Integrating Ray Tracing in Unreal Engine 4 ......................................290
19.3 Real-Time Ray Tracing and Denoising .............................................300
19.4 Conclusions .......................................................................................317
Chapter 20: Texture Level of Detail Strategies for 
Real-Time Ray Tracing �������������������������������������������������������������������������� 321
20.1 Introduction .......................................................................................321
20.2 Background .......................................................................................323
20.3 Texture Level of Detail Algorithms ...................................................324
20.4 Implementation .................................................................................336
20.5 Comparison and Results ...................................................................338
20.6 Code ...................................................................................................342
Chapter 21: Simple Environment Map Filtering Using 
Ray Cones and Ray Differentials������������������������������������������������������������ 347
21.1 Introduction .......................................................................................347
21.2 Ray Cones ..........................................................................................348
21.3 Ray Differentials ................................................................................349
21.4 Results ..............................................................................................349
Chapter 22: Improving Temporal Antialiasing with 
Adaptive Ray Tracing ����������������������������������������������������������������������������� 353
22.1 Introduction .......................................................................................353
22.2 Previous Temporal Antialiasing ........................................................355
22.3 A New Algorithm ...............................................................................356
22.4 Early Results .....................................................................................363
22.5 Limitations ........................................................................................366
22.6 The Future of Real-Time Ray Traced Antialiasing ............................367
22.7 Conclusion .........................................................................................368
PART VI: Hybrid Approaches and Systems������������������������������������ 375
Chapter 23: Interactive Light Map and Irradiance Volume 
Preview in Frostbite������������������������������������������������������������������������������� 377
23.1 Introduction .......................................................................................377
23.2 GI Solver Pipeline ..............................................................................378
23.3 Acceleration Techniques ...................................................................393
23.4 Live Update ........................................................................................398
23.5 Performance and Hardware ..............................................................400
23.6 Conclusion .........................................................................................405
Chapter 24: Real-Time Global Illumination with Photon Mapping ��������� 409
24.1 Introduction .......................................................................................409
24.2 Photon Tracing ..................................................................................411
24.3 Screen-Space Irradiance Estimation ................................................418
24.4 Filtering .............................................................................................425
24.5 Results ..............................................................................................430
24.6 Future Work .......................................................................................434
Chapter 25: Hybrid Rendering for Real-Time Ray Tracing��������������������� 437
25.1 Hybrid Rendering Pipeline Overview ................................................437
25.2 Pipeline Breakdown ..........................................................................439
25.3 Performance .....................................................................................468
25.4 Future ................................................................................................469
25.5 Code ...................................................................................................469
Chapter 26: Deferred Hybrid Path Tracing��������������������������������������������� 475
26.1 Overview ............................................................................................475
26.2 Hybrid Approach ................................................................................476
26.3 BVH Traversal ....................................................................................478
26.4 Diffuse Light Transport .....................................................................481
26.5 Specular Light Transport ..................................................................485
26.6 Transparency .....................................................................................487
26.7 Performance .....................................................................................488
Chapter 27: Interactive Ray Tracing Techniques for 
High-Fidelity Scientific Visualization����������������������������������������������������� 493
27.1 Introduction .......................................................................................493
27.2 Challenges Associated with Ray Tracing Large Scenes ...................494
27.3 Visualization Methods .......................................................................500
27.4 Closing Thoughts ..............................................................................512
PART VII: Global Illumination�������������������������������������������������������� 519
Chapter 28: Ray Tracing Inhomogeneous Volumes�������������������������������� 521
28.1 Light Transport in Volumes ...............................................................521
28.2 Woodcock Tracking ...........................................................................522
28.3 Example: A Simple Volume Path Tracer ...........................................524
28.4 Further Reading ................................................................................530
Chapter 29: Efficient Particle Volume Splatting in a Ray Tracer ������������ 533
29.1 Motivation ..........................................................................................533
29.2 Algorithm ..........................................................................................534
29.3 Implementation .................................................................................535
29.4 Results ..............................................................................................539
29.5 Summary ...........................................................................................539
Chapter 30: Caustics Using Screen-Space Photon Mapping������������������� 543
30.1 Introduction .......................................................................................543
30.2 Overview ............................................................................................544
30.3 Implementation .................................................................................545
30.4 Results ..............................................................................................552
30.5 Code ...................................................................................................553
Chapter 31: Variance Reduction via Footprint Estimation in 
the Presence of Path Reuse������������������������������������������������������������������� 557
31.1 Introduction .......................................................................................557
31.2 Why Assuming Full Reuse Causes a Broken MIS Weight ................559
31.3 The Effective Reuse Factor ...............................................................560
31.4 Implementation Impacts ...................................................................565
31.5 Results ..............................................................................................566
Chapter 32: Accurate Real-Time Specular Reflections with 
Radiance Caching����������������������������������������������������������������������������������� 571
32.1 Introduction .......................................................................................571
32.2 Previous Work ...................................................................................573
32.3 Algorithm ..........................................................................................575
32.4 Spatiotemporal Filtering ...................................................................587
32.5 Results ..............................................................................................598
32.6 Conclusion .........................................................................................604
32.7 Future Work .......................................................................................605
```

## Realtime Rendering 4th Edition
```
1 Introduction 1
1.1 Contents Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Notation and Definitions . . . . . . . . . . . . . . . . . . . . . . . . 5
2 The Graphics Rendering Pipeline 11
2.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.2 The Application Stage . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.3 Geometry Processing . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.4 Rasterization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2.5 Pixel Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
2.6 Through the Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . 25
3 The Graphics Processing Unit 29
3.1 Data-Parallel Architectures . . . . . . . . . . . . . . . . . . . . . . 30
3.2 GPU Pipeline Overview . . . . . . . . . . . . . . . . . . . . . . . . 34
3.3 The Programmable Shader Stage . . . . . . . . . . . . . . . . . . . 35
3.4 The Evolution of Programmable Shading and APIs . . . . . . . . . 37
3.5 The Vertex Shader . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.6 The Tessellation Stage . . . . . . . . . . . . . . . . . . . . . . . . . 44
3.7 The Geometry Shader . . . . . . . . . . . . . . . . . . . . . . . . . 47
3.8 The Pixel Shader . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
3.9 The Merging Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
3.10 The Compute Shader . . . . . . . . . . . . . . . . . . . . . . . . . . 54
4 Transforms 57
4.1 Basic Transforms . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
4.2 Special Matrix Transforms and Operations . . . . . . . . . . . . . . 70
4.3 Quaternions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
4.4 Vertex Blending . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
4.5 Morphing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
4.6 Geometry Cache Playback . . . . . . . . . . . . . . . . . . . . . . . 92
4.7 Projections . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
5 Shading Basics 103
5.1 Shading Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
5.2 Light Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
5.3 Implementing Shading Models . . . . . . . . . . . . . . . . . . . . . 117
5.4 Aliasing and Antialiasing . . . . . . . . . . . . . . . . . . . . . . . . 130
5.5 Transparency, Alpha, and Compositing . . . . . . . . . . . . . . . . 148
5.6 Display Encoding . . . . . . . . . . . . . . . . . . . . . . . . . . . . 160
6 Texturing 167
6.1 The Texturing Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . 169
6.2 Image Texturing . . . . . . . . . . . . . . . . . . . . . . . . . . . . 176
6.3 Procedural Texturing . . . . . . . . . . . . . . . . . . . . . . . . . . 198
6.4 Texture Animation . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
6.5 Material Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . 201
6.6 Alpha Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
6.7 Bump Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 208
6.8 Parallax Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . 214
6.9 Textured Lights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 221
7 Shadows 223
7.1 Planar Shadows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 225
7.2 Shadows on Curved Surfaces . . . . . . . . . . . . . . . . . . . . . . 229
7.3 Shadow Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . 230
7.4 Shadow Maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 234
7.5 Percentage-Closer Filtering . . . . . . . . . . . . . . . . . . . . . . 247
7.6 Percentage-Closer Soft Shadows . . . . . . . . . . . . . . . . . . . . 250
7.7 Filtered Shadow Maps . . . . . . . . . . . . . . . . . . . . . . . . . 252
7.8 Volumetric Shadow Techniques . . . . . . . . . . . . . . . . . . . . 257
7.9 Irregular Z-Buffer Shadows . . . . . . . . . . . . . . . . . . . . . . 259
7.10 Other Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . 262
8 Light and Color 267
8.1 Light Quantities . . . . . . . . . . . . . . . . . . . . . . . . . . . . 267
8.2 Scene to Screen . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 281
9 Physically Based Shading 293
9.1 Physics of Light . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 293
9.2 The Camera . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 307
9.3 The BRDF . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 308
9.4 Illumination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 315
9.5 Fresnel Reflectance . . . . . . . . . . . . . . . . . . . . . . . . . . . 316
9.6 Microgeometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 327
9.7 Microfacet Theory . . . . . . . . . . . . . . . . . . . . . . . . . . . 331
9.8 BRDF Models for Surface Reflection . . . . . . . . . . . . . . . . . 336
9.9 BRDF Models for Subsurface Scattering . . . . . . . . . . . . . . . 347
9.10 BRDF Models for Cloth . . . . . . . . . . . . . . . . . . . . . . . . 356
9.11 Wave Optics BRDF Models . . . . . . . . . . . . . . . . . . . . . . 359
9.12 Layered Materials . . . . . . . . . . . . . . . . . . . . . . . . . . . . 363
9.13 Blending and Filtering Materials . . . . . . . . . . . . . . . . . . . 365
10 Local Illumination 375
10.1 Area Light Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . 377
10.2 Environment Lighting . . . . . . . . . . . . . . . . . . . . . . . . . 391
10.3 Spherical and Hemispherical Functions . . . . . . . . . . . . . . . . 392
10.4 Environment Mapping . . . . . . . . . . . . . . . . . . . . . . . . . 404
10.5 Specular Image-Based Lighting . . . . . . . . . . . . . . . . . . . . 414
10.6 Irradiance Environment Mapping . . . . . . . . . . . . . . . . . . . 424
10.7 Sources of Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 433
11 Global Illumination 437
11.1 The Rendering Equation . . . . . . . . . . . . . . . . . . . . . . . . 437
11.2 General Global Illumination . . . . . . . . . . . . . . . . . . . . . . 441
11.3 Ambient Occlusion . . . . . . . . . . . . . . . . . . . . . . . . . . . 446
11.4 Directional Occlusion . . . . . . . . . . . . . . . . . . . . . . . . . . 465
11.5 Diffuse Global Illumination . . . . . . . . . . . . . . . . . . . . . . 472
11.6 Specular Global Illumination . . . . . . . . . . . . . . . . . . . . . 497
11.7 Unified Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . 509
12 Image-Space Effects 513
12.1 Image Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . 513
12.2 Reprojection Techniques . . . . . . . . . . . . . . . . . . . . . . . . 522
12.3 Lens Flare and Bloom . . . . . . . . . . . . . . . . . . . . . . . . . 524
12.4 Depth of Field . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 527
12.5 Motion Blur . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 536
13 Beyond Polygons 545
13.1 The Rendering Spectrum . . . . . . . . . . . . . . . . . . . . . . . . 545
13.2 Fixed-View Effects . . . . . . . . . . . . . . . . . . . . . . . . . . . 546
13.3 Skyboxes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 547
13.4 Light Field Rendering . . . . . . . . . . . . . . . . . . . . . . . . . 549
13.5 Sprites and Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . 550
13.6 Billboarding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 551
13.7 Displacement Techniques . . . . . . . . . . . . . . . . . . . . . . . . 564
13.8 Particle Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . 567
13.9 Point Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 572
13.10 Voxels . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 578
14 Volumetric and Translucency Rendering 589
14.1 Light Scattering Theory . . . . . . . . . . . . . . . . . . . . . . . . 589
14.2 Specialized Volumetric Rendering . . . . . . . . . . . . . . . . . . . 600
14.3 General Volumetric Rendering . . . . . . . . . . . . . . . . . . . . . 605
14.4 Sky Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 613
14.5 Translucent Surfaces . . . . . . . . . . . . . . . . . . . . . . . . . . 623
14.6 Subsurface Scattering . . . . . . . . . . . . . . . . . . . . . . . . . . 632
14.7 Hair and Fur . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 640
14.8 Unified Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . 648
15 Non-Photorealistic Rendering 651
15.1 Toon Shading . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 652
15.2 Outline Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . 654
15.3 Stroke Surface Stylization . . . . . . . . . . . . . . . . . . . . . . . 669
15.4 Lines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 673
15.5 Text Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 675
16 Polygonal Techniques 681
16.1 Sources of Three-Dimensional Data . . . . . . . . . . . . . . . . . . 682
16.2 Tessellation and Triangulation . . . . . . . . . . . . . . . . . . . . . 683
16.3 Consolidation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 690
16.4 Triangle Fans, Strips, and Meshes . . . . . . . . . . . . . . . . . . . 696
16.5 Simplification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 706
16.6 Compression and Precision . . . . . . . . . . . . . . . . . . . . . . . 712
17 Curves and Curved Surfaces 717
17.1 Parametric Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . 718
17.2 Parametric Curved Surfaces . . . . . . . . . . . . . . . . . . . . . . 734
17.3 Implicit Surfaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . 749
17.4 Subdivision Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . 753
17.5 Subdivision Surfaces . . . . . . . . . . . . . . . . . . . . . . . . . . 756
17.6 Efficient Tessellation . . . . . . . . . . . . . . . . . . . . . . . . . . 767
18 Pipeline Optimization 783
18.1 Profiling and Debugging Tools . . . . . . . . . . . . . . . . . . . . . 784
18.2 Locating the Bottleneck . . . . . . . . . . . . . . . . . . . . . . . . 786
18.3 Performance Measurements . . . . . . . . . . . . . . . . . . . . . . 788
18.4 Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 790
18.5 Multiprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 805
19 Acceleration Algorithms 817
19.1 Spatial Data Structures . . . . . . . . . . . . . . . . . . . . . . . . 818
19.2 Culling Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . 830
19.3 Backface Culling . . . . . . . . . . . . . . . . . . . . . . . . . . . . 831
19.4 View Frustum Culling . . . . . . . . . . . . . . . . . . . . . . . . . 835
19.5 Portal Culling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 837
19.6 Detail and Small Triangle Culling . . . . . . . . . . . . . . . . . . . 839
19.7 Occlusion Culling . . . . . . . . . . . . . . . . . . . . . . . . . . . . 840
19.8 Culling Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 850
19.9 Level of Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 852
19.10 Rendering Large Scenes . . . . . . . . . . . . . . . . . . . . . . . . 866
20 Efficient Shading 881
20.1 Deferred Shading . . . . . . . . . . . . . . . . . . . . . . . . . . . . 883
20.2 Decal Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . 888
20.3 Tiled Shading . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 892
20.4 Clustered Shading . . . . . . . . . . . . . . . . . . . . . . . . . . . 898
20.5 Deferred Texturing . . . . . . . . . . . . . . . . . . . . . . . . . . . 905
20.6 Object- and Texture-Space Shading . . . . . . . . . . . . . . . . . . 908
21 Virtual and Augmented Reality 915
21.1 Equipment and Systems Overview . . . . . . . . . . . . . . . . . . 916
21.2 Physical Elements . . . . . . . . . . . . . . . . . . . . . . . . . . . 919
21.3 APIs and Hardware . . . . . . . . . . . . . . . . . . . . . . . . . . . 924
21.4 Rendering Techniques . . . . . . . . . . . . . . . . . . . . . . . . . 932
22 Intersection Test Methods 941
22.1 GPU-Accelerated Picking . . . . . . . . . . . . . . . . . . . . . . . 942
22.2 Definitions and Tools . . . . . . . . . . . . . . . . . . . . . . . . . . 943
22.3 Bounding Volume Creation . . . . . . . . . . . . . . . . . . . . . . 948
22.4 Geometric Probability . . . . . . . . . . . . . . . . . . . . . . . . . 953
22.5 Rules of Thumb . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 954
22.6 Ray/Sphere Intersection . . . . . . . . . . . . . . . . . . . . . . . . 955
22.7 Ray/Box Intersection . . . . . . . . . . . . . . . . . . . . . . . . . . 959
22.8 Ray/Triangle Intersection . . . . . . . . . . . . . . . . . . . . . . . 962
22.9 Ray/Polygon Intersection . . . . . . . . . . . . . . . . . . . . . . . 966
22.10 Plane/Box Intersection . . . . . . . . . . . . . . . . . . . . . . . . . 970
22.11 Triangle/Triangle Intersection . . . . . . . . . . . . . . . . . . . . . 972
22.12 Triangle/Box Intersection . . . . . . . . . . . . . . . . . . . . . . . 974
22.13 Bounding-Volume/Bounding-Volume Intersection . . . . . . . . . . 976
22.14 View Frustum Intersection . . . . . . . . . . . . . . . . . . . . . . . 981
22.15 Line/Line Intersection . . . . . . . . . . . . . . . . . . . . . . . . . 987
22.16 Intersection between Three Planes . . . . . . . . . . . . . . . . . . 990
23 Graphics Hardware 993
23.1 Rasterization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 993
23.2 Massive Compute and Scheduling . . . . . . . . . . . . . . . . . . . 1002
23.3 Latency and Occupancy . . . . . . . . . . . . . . . . . . . . . . . . 1004
23.4 Memory Architecture and Buses . . . . . . . . . . . . . . . . . . . 1006
23.5 Caching and Compression . . . . . . . . . . . . . . . . . . . . . . . 1007
23.6 Color Buffering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1009
23.7 Depth Culling, Testing, and Buffering . . . . . . . . . . . . . . . . 1014
23.8 Texturing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1017
23.9 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1019
23.10 Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1024
23.11 Ray Tracing Architectures . . . . . . . . . . . . . . . . . . . . . . . 1039
24 The Future 1041
24.1 Everything Else . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1042
24.2 You . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1046
```

## Fluid Simulation for Computer Graphics, Second Edition
```
 The Basics 1
1 The Equations of Fluids 3
1.1 Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 The Momentum Equation . . . . . . . . . . . . . . . . . . 4
1.3 Lagrangian and Eulerian Viewpoints . . . . . . . . . . . . 7
1.4 Incompressibility . . . . . . . . . . . . . . . . . . . . . . . 11
1.5 Dropping Viscosity . . . . . . . . . . . . . . . . . . . . . . 13
1.6 Boundary Conditions . . . . . . . . . . . . . . . . . . . . . 13
2 Overview of Numerical Simulation 17
2.1 Splitting . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
2.2 Splitting the Fluid Equations . . . . . . . . . . . . . . . . 19
2.3 Time Steps . . . . . . . . . . . . . . . . . . . . . . . . . . 20
2.4 Grids . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2.5 Dynamic Sparse Grids . . . . . . . . . . . . . . . . . . . . 25
2.6 Two Dimensional Simulations . . . . . . . . . . . . . . . . 27
3 Advection Algorithms 29
3.1 Semi-Lagrangian Advection . . . . . . . . . . . . . . . . . 29
3.2 Boundary Conditions . . . . . . . . . . . . . . . . . . . . . 33
3.3 Time Step Size . . . . . . . . . . . . . . . . . . . . . . . . 34
3.4 Diffusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
3.5 Reducing Numerical Diffusion . . . . . . . . . . . . . . . . 39
4 Level Set Geometry 43
4.1 Signed Distance . . . . . . . . . . . . . . . . . . . . . . . . 44
4.2 Discretizing Signed Distance Functions . . . . . . . . . . . 47
4.3 Computing Signed Distance . . . . . . . . . . . . . . . . . 49
4.4 Recomputing Signed Distance . . . . . . . . . . . . . . . . 54
4.5 Operations on Level Sets . . . . . . . . . . . . . . . . . . . 55
4.6 Contouring . . . . . . . . . . . . . . . . . . . . . . . . . . 59
4.7 Limitations of Level Sets . . . . . . . . . . . . . . . . . . . 64
4.8 Extrapolating Data . . . . . . . . . . . . . . . . . . . . . . 64
5 Making Fluids Incompressible 67
5.1 The Discrete Pressure Gradient . . . . . . . . . . . . . . . 68
5.2 The Discrete Divergence . . . . . . . . . . . . . . . . . . . 72
5.3 The Pressure Equations . . . . . . . . . . . . . . . . . . . 74
5.4 Projection . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
5.5 More Accurate Curved Boundaries . . . . . . . . . . . . . 91
5.6 The Compatibility Condition . . . . . . . . . . . . . . . . 96
6 Smoke 99
6.1 Temperature and Smoke Concentration . . . . . . . . . . 99
6.2 Buoyancy . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
6.3 Variable Density Solves . . . . . . . . . . . . . . . . . . . 103
6.4 Divergence Control . . . . . . . . . . . . . . . . . . . . . . 105
7 Particle Methods 107
7.1 Advection Troubles on Grids . . . . . . . . . . . . . . . . 107
7.2 Particle Advection . . . . . . . . . . . . . . . . . . . . . . 109
7.3 Transferring Particles to the Grid . . . . . . . . . . . . . . 111
7.4 Particle Seeding . . . . . . . . . . . . . . . . . . . . . . . . 114
7.5 Diffusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
7.6 Particle-in-Cell Methods . . . . . . . . . . . . . . . . . . . 116
II More Types of Fluids 121
8 Water 123
8.1 Marker Particles and Voxels . . . . . . . . . . . . . . . . . 123
8.2 More Accurate Pressure Solves . . . . . . . . . . . . . . . 127
8.3 Topology Change and Wall Separation . . . . . . . . . . . 129
8.4 Volume Control . . . . . . . . . . . . . . . . . . . . . . . . 130
8.5 Surface Tension . . . . . . . . . . . . . . . . . . . . . . . . 131
9 Fire 133
9.1 Thin Flames . . . . . . . . . . . . . . . . . . . . . . . . . 134
9.2 Volumetric Combustion . . . . . . . . . . . . . . . . . . . 137
10 Viscous Fluids 139
10.1 Stress . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 139
10.2 Applying Stress . . . . . . . . . . . . . . . . . . . . . . . . 141
10.3 Strain Rate and Newtonian Fluids . . . . . . . . . . . . . 142
10.4 Boundary Conditions . . . . . . . . . . . . . . . . . . . . . 147
10.5 Implementation . . . . . . . . . . . . . . . . . . . . . . . . 148
III More Algorithms 161
11 Turbulence 163
11.1 Vorticity . . . . . . . . . . . . . . . . . . . . . . . . . . . . 163
11.2 Vorticity Confinement . . . . . . . . . . . . . . . . . . . . 167
11.3 Procedural Turbulence . . . . . . . . . . . . . . . . . . . . 168
11.4 Simulating Sub-Grid Turbulence . . . . . . . . . . . . . . 172
12 Shallow Water 175
12.1 Deriving the Shallow Water Equations . . . . . . . . . . . 176
12.2 The Wave Equation . . . . . . . . . . . . . . . . . . . . . 180
12.3 Discretization . . . . . . . . . . . . . . . . . . . . . . . . . 182
13 Ocean Modeling 185
13.1 Potential Flow . . . . . . . . . . . . . . . . . . . . . . . . 185
13.2 Simplifying Potential Flow for the Ocean . . . . . . . . . 188
13.3 Evaluating the Height Field Solution . . . . . . . . . . . . 193
13.4 Unsimplifying the Model . . . . . . . . . . . . . . . . . . . 195
13.5 Wave Parameters . . . . . . . . . . . . . . . . . . . . . . . 198
13.6 Eliminating Periodicity . . . . . . . . . . . . . . . . . . . . 199
14 Vortex Methods 201
14.1 Velocity from Vorticity . . . . . . . . . . . . . . . . . . . . 202
14.2 Biot-Savart and Streamfunctions . . . . . . . . . . . . . . 206
14.3 Vortex Particles . . . . . . . . . . . . . . . . . . . . . . . . 207
15 Coupling Fluids and Solids 217
15.1 One-Way Coupling . . . . . . . . . . . . . . . . . . . . . . 217
15.2 Weak Coupling . . . . . . . . . . . . . . . . . . . . . . . . 219
15.3 The Immersed Boundary Method . . . . . . . . . . . . . . 222
15.4 General Sparse Matrices . . . . . . . . . . . . . . . . . . . 223
15.5 Strong Coupling . . . . . . . . . . . . . . . . . . . . . . . 225
A Background 231
A.1 Vector Calculus . . . . . . . . . . . . . . . . . . . . . . . . 231
A.2 Numerical Methods . . . . . . . . . . . . . . . . . . . . . . 239
B Derivations 243
B.1 The Incompressible Euler Equations . . . . . . . . . . . . 243
```


## 3D Game Design with Unreal Engine 4 and Blender
```
Chapter 1: Unreal, My Friend, I'd Like You to Meet Blender 1
Installing Blender 2
Exploring the interface 3
Customizing your settings 6
Working with modes 8
Jumping into our first project 8
Getting things started in Unreal Engine 4 9
Summary 11
Chapter 2: Starting Our First Project 13
Using the Content Browser to start building the level 15
Using different types of light 28
Adding interactive elements using Triggers and Blueprints 31
Playtesting our level 41
Summary 42
Chapter 3: It's Time to Customize! 43
Getting started making game assets 44
Using the basic tools of polygon modeling 46
How to use UV mapping and why it's important 57
UV unwrapping our game asset 60
Basic texturing techniques 66
Summary 70
Chapter 4: Getting the Assets to the Level 71
Exporting our object from Blender 71
What is FBX? 75
Importing our object into Unreal 76
Setting up and using our new 3D asset 79
Summary 86
Chapter 5: Taking This Level Up a Notch 87
Planning a more complex level 87
Whiteboxing a level for better asset creation 89
Level design principles 93
Advanced scripting techniques 95
Win conditions 109
Summary 112
Chapter 6: Monster Assets – The Level Totally Needs 
One of These 113
Designing our asset – inspiration and concept art 114
Advanced tools: Subdivide, Knife, Bridge Edge Loops, and more 115
Subdivide tool 116
Knife tool 117
Bridge Edge Loops tool 118
Triangulate modifier tool 118
Using multiple shapes within Blender 119
Summary 143
Chapter 7: Let's Dress to Impress! 145
Unwrapping complex objects 145
Marking seams 146
Unwrap tool 147
Stitch 148
Average Island Scale 149
Pack Islands 150
Using Smart UV Project 150
Custom Marking Seams 151
Unwrapping Cubes 152
Unwrapping Cylinders 154
Using different maps to create a more realistic look 165
Summary 184
Chapter 8: Lights, Camera, Animation! 185
How does Blender handle animation? 185
Rigging and using keyframes 188
Using Blender's suite of animation tools 200
Summary 204
Chapter 9: Bang Bang – Let's Make It Explode 205
Creating a class blueprint to tie it all together 206
Using sound effects 219
Using particle effects 224
```


## Computational Geometry Algorithms and Applications
```
1 Computational Geometry 1
Introduction
1.1 An Example: Convex Hulls 2
1.2 Degeneracies and Robustness 8
1.3 Application Domains 10
1.4 Notes and Comments 13
1.5 Exercises 15
2 Line Segment Intersection 19
Thematic Map Overlay
2.1 Line Segment Intersection 20
2.2 The Doubly-Connected Edge List 29
2.3 Computing the Overlay of Two Subdivisions 33
2.4 Boolean Operations 39
2.5 Notes and Comments 40
2.6 Exercises 41
3 Polygon Triangulation 45
Guarding an Art Gallery
3.1 Guarding and Triangulations 46
3.2 Partitioning a Polygon into Monotone Pieces 49
3.3 Triangulating a Monotone Polygon 55
3.4 Notes and Comments 59
3.5 Exercises 60
4 Linear Programming 63
Manufacturing with Molds
4.1 The Geometry of Casting 64
4.2 Half-Plane Intersection 66
4.3 Incremental Linear Programming 71
4.4 Randomized Linear Programming 76
4.5 Unbounded Linear Programs 79
4.6* Linear Programming in Higher Dimensions 82
4.7* Smallest Enclosing Discs 86
4.8 Notes and Comments 89
4.9 Exercises 91
5 Orthogonal Range Searching 95
Querying a Database
5.1 1-Dimensional Range Searching 96
5.2 Kd-Trees 99
5.3 Range Trees 105
5.4 Higher-Dimensional Range Trees 109
5.5 General Sets of Points 110
5.6* Fractional Cascading 111
5.7 Notes and Comments 115
5.8 Exercises 117
6 Point Location 121
Knowing Where You Are
6.1 Point Location and Trapezoidal Maps 122
6.2 A Randomized Incremental Algorithm 128
6.3 Dealing with Degenerate Cases 137
6.4* A Tail Estimate 140
6.5 Notes and Comments 143
6.6 Exercises 144
7 Voronoi Diagrams 147
The Post Office Problem
7.1 Definition and Basic Properties 148
7.2 Computing the Voronoi Diagram 151
7.3 Voronoi Diagrams of Line Segments 160
7.4 Farthest-Point Voronoi Diagrams 163
7.5 Notes and Comments 167
7.6 Exercises 170
8 Arrangements and Duality 173
Supersampling in Ray Tracing
8.1 Computing the Discrepancy 175
8.2 Duality 177
8.3 Arrangements of Lines 179
8.4 Levels and Discrepancy 185
8.5 Notes and Comments 186
8.6 Exercises 188
9 Delaunay Triangulations 191
Height Interpolation
9.1 Triangulations of Planar Point Sets 193
9.2 The Delaunay Triangulation 196
9.3 Computing the Delaunay Triangulation 199
9.4 The Analysis 205
9.5* A Framework for Randomized Algorithms 208
9.6 Notes and Comments 214
9.7 Exercises 215
10 More Geometric Data Structures 219
Windowing
10.1 Interval Trees 220
10.2 Priority Search Trees 226
10.3 Segment Trees 231
10.4 Notes and Comments 237
10.5 Exercises 239
11 Convex Hulls 243
Mixing Things
11.1 The Complexity of Convex Hulls in 3-Space 244
11.2 Computing Convex Hulls in 3-Space 246
11.3* The Analysis 250
11.4* Convex Hulls and Half-Space Intersection 253
11.5* Voronoi Diagrams Revisited 254
11.6 Notes and Comments 256
11.7 Exercises 257
12 Binary Space Partitions 259
The Painter’s Algorithm
12.1 The Definition of BSP Trees 261
12.2 BSP Trees and the Painter’s Algorithm 263
12.3 Constructing a BSP Tree 264
12.4* The Size of BSP Trees in 3-Space 268
12.5 BSP Trees for Low-Density Scenes 271
12.6 Notes and Comments 278
12.7 Exercises 279
13 Robot Motion Planning 283
Getting Where You Want to Be
13.1 Work Space and Configuration Space 284
13.2 A Point Robot 286
13.3 Minkowski Sums 290
13.4 Translational Motion Planning 297
13.5* Motion Planning with Rotations 299
13.6 Notes and Comments 303
13.7 Exercises 305
14 Quadtrees 307
Non-Uniform Mesh Generation
14.1 Uniform and Non-Uniform Meshes 308
14.2 Quadtrees for Point Sets 309
14.3 From Quadtrees to Meshes 315
14.4 Notes and Comments 318
14.5 Exercises 320
15 Visibility Graphs 323
Finding the Shortest Route
15.1 Shortest Paths for a Point Robot 324
15.2 Computing the Visibility Graph 326
15.3 Shortest Paths for a Translating Polygonal Robot 330
15.4 Notes and Comments 331
15.5 Exercises 332
16 Simplex Range Searching 335
Windowing Revisited
16.1 Partition Trees 336
16.2 Multi-Level Partition Trees 343
16.3 Cutting Trees 346
16.4 Notes and Comments 352
16.5 Exercises 353
```


## GPU Gems 1
```
第一章 用物理模型进行高效的水模拟
第二章 水刻蚀的渲染
第三章 Dawn演示中的皮肤
第四章 Dawn演示中的动画
第五章 改良的Perlin噪声的实现
第六章 Vulcan演示中的火
第七章 无数波动草叶的渲染
第八章 衍射的模拟
第九章 有效的阴影体渲染
第十章 电影级的光照
第十一章 阴影贴图反走样 
11.2 靠近的百分比过滤
第十二章 全方位的阴影映射
第十三章 使用遮挡区间产生模拟的阴影
第十四章 透视阴影贴图
第十五章 逐像素光照的可见性管理
第十六章 次表面散射的实时近似
第十七章 环境遮挡
第十八章 空间的BRDFs
第十九章 基于图像的光照
第二十章 纹理爆炸
第二十一章 实时辉光
第二十二章 颜色控制
第二十三章 景深：技术综述
第二十四章 高质量的过滤
第二十五章 用纹理贴图进行快速过滤宽度的计算
第二十六章 OpenEXR图像文件格式
第二十七章 图像处理的框架
第二十八章 图形流水线性能
第二十九章 有效的遮挡剔除
第三十章 FX Composer的设计
第三十一章 FX Composer的使用
第三十二章 Shader接口入门
第三十三章 将产品的RenderMan Shader转成实时Shader
第三十四章 将硬件着色整合进Cinema 4D
第三十五章 在实时引用程序中用高质软件渲染效果
第三十六章 将Shader整合到应用程序中去。
第三十七章 用于GPU计算的工具箱
第三十八章 在GPU上的快速流体动力学模拟
第三十九章 体渲染技术
第四十章 用于三维超声波可视化的实时着色
第四十一章 实时立体图
第四十二章 变形
```


## GPU Gems 2
```
Part I: Geometric Complexity
o Chapter 1. Toward Photorealism in Virtual Botany
o Chapter 2. Terrain Rendering Using GPU-Based Geometry Clipmaps
o Chapter 3. Inside Geometry Instancing
o Chapter 4. Segment Buffering
o Chapter 5. Optimizing Resource Management with Multistreaming
o Chapter 6. Hardware Occlusion Queries Made Useful
o Chapter 7. Adaptive Tessellation of Subdivision Surfaces with Displacement Mapping
o Chapter 8. Per-Pixel Displacement Mapping with Distance Functions
 Part II: Shading, Lighting, and Shadows
o Chapter 9. Deferred Shading in S.T.A.L.K.E.R.
o Chapter 10. Real-Time Computation of Dynamic Irradiance Environment Maps
o Chapter 11. Approximate Bidirectional Texture Functions
o Chapter 12. Tile-Based Texture Mapping
o Chapter 13. Implementing the mental images Phenomena Renderer on the GPU
o Chapter 14. Dynamic Ambient Occlusion and Indirect Lighting
o Chapter 15. Blueprint Rendering and "Sketchy Drawings"
o Chapter 16. Accurate Atmospheric Scattering
o Chapter 17. Efficient Soft-Edged Shadows Using Pixel Shader Branching
o Chapter 18. Using Vertex Texture Displacement for Realistic Water Rendering
o Chapter 19. Generic Refraction Simulation
 Part III: High-Quality Rendering
o Chapter 20. Fast Third-Order Texture Filtering
o Chapter 21. High-Quality Antialiased Rasterization
o Chapter 22. Fast Prefiltered Lines
o Chapter 23. Hair Animation and Rendering in the Nalu Demo
o Chapter 24. Using Lookup Tables to Accelerate Color Transformations
o Chapter 25. GPU Image Processing in Apple's Motion
o Chapter 26. Implementing Improved Perlin Noise
o Chapter 27. Advanced High-Quality Filtering
o Chapter 28. Mipmap-Level Measurement
 Part IV: General-Purpose Computation on GPUS: A Primer
o Chapter 29. Streaming Architectures and Technology Trends
o Chapter 30. The GeForce 6 Series GPU Architecture
o Chapter 31. Mapping Computational Concepts to GPUs
o Chapter 32. Taking the Plunge into GPU Computing
o Chapter 33. Implementing Efficient Parallel Data Structures on GPUs
o Chapter 34. GPU Flow-Control Idioms
o Chapter 35. GPU Program Optimization
o Chapter 36. Stream Reduction Operations for GPGPU Applications
 Part V: Image-Oriented Computing
o Chapter 37. Octree Textures on the GPU
o Chapter 38. High-Quality Global Illumination Rendering Using Rasterization
o Chapter 39. Global Illumination Using Progressive Refinement Radiosity
o Chapter 40. Computer Vision on the GPU
o Chapter 41. Deferred Filtering: Rendering from Difficult Data Formats
o Chapter 42. Conservative Rasterization
 Part VI: Simulation and Numerical Algorithms
o Chapter 43. GPU Computing for Protein Structure Prediction
o Chapter 44. A GPU Framework for Solving Systems of Linear Equations
o Chapter 45. Options Pricing on the GPU
o Chapter 46. Improved GPU Sorting
o Chapter 47. Flow Simulation with Complex Boundaries
o Chapter 48. Medical Image Reconstruction with the FFT
```


## GPU Gems 3
```
第一部分 几何体
第一章 使用GPU生成复杂的程序化地形
第二章 群体动画渲染
第三章 DX10混合形状 打破限制
第四章 下一代SpeedTree渲染
第五章 普遍自适应的网格优化
第六章 GPU生成的树的过程式风动画
第七章 GPU上基于点的变形球可视化
第二部分 光照和阴影
第八章 区域求和的差值阴影贴图
第九章 使用全局照明实现互动的电影级重光照
第十章 在可编程GPU中实现并行分割的阴影贴图
第十一章 使用层次化的遮挡剔除和几何体着色器得到高效鲁棒的阴影体
第十二章 高质量的环境遮挡
第十三章 作为后置处理的体积光照散射
第三部分 渲染
第十四章 用于真实感实时皮肤渲染的高级技术
第十五章 可播放的全方位捕捉
第十六章 Crysis中植被的过程化动画和着色
第十七章 鲁棒的多镜面反射和折色
第十八章 用于浮雕映射的松散式锥形步进
第十九章 Tabula Rasa中的延迟着色
第二十章 基于GPU的重要性采样
第四部分 图像效果
第二十一章 真正的Impostor
第二十二章 在GPU上处理法线贴图
第二十三章 高速的离屏粒子
第二十四章 保持线性的重要性
第二十五章 在GPU上渲染向量图
第二十六章 通过颜色进行对象探测：使用GPU进行实时视频图像处理
第二十七章 作为后置处理效果的运动模糊
第二十八章 实用景深后期处理
第五部分 物理仿真 
第二十九章 GPU上实时刚体仿真
第三十章 实时仿真与3D流体渲染
第三十一章 使用CUDA进行快速N-body仿真
第三十二章 使用CUDA进行宽阶段碰撞检测
第三十三章 用于碰撞检测的LCP算法的CUDA实现
第三十四章 使用单过程GPU扫描和四面体转换的有向距离场
第六部分 GPU计算
第三十五章 使用GPU进行病毒特征的快速匹配
第三十六章 用GPU进行AES加密和解密
第三十七章 使用CUDA进行高效的随机数生成及应用
第三十八章 使用CUDA进行地球内部成像
第三十九章 使用CUDA的并行前缀和扫描方法
第四十章 高斯函数的增量计算
第四十一章 使用几何体着色器处理紧凑和可变长度的GPU反馈

```


## GPU Pro 2
```
I Geometry Manipulation 1
Wolfgang Engel, editor
1 Terrain and Ocean Rendering with Hardware Tessellation 3
Xavier Bonaventura
1.1 DirectX 11 Graphics Pipeline . . . . . . . . . . . . . . . . . . . 4
1.2 Definition of Geometry . . . . . . . . . . . . . . . . . . . . . . . 7
1.3 Vertex Position, Vertex Normal, and Texture Coordinates . . . 10
1.4 Tessellation Correction Depending on the Camera Angle . . . . 12
1.5 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2 Practical and Realistic Facial Wrinkles Animation 15
Jorge Jimenez, Jose I. Echevarria, Christopher Oat, and Diego Gutierrez
2.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
2.2 Our Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
2.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
2.4 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
2.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
2.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 26
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
3 Procedural Content Generation on the GPU 29
Aleksander Netzel and Pawel Rohleder
3.1 Abstract . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
3.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
3.3 Terrain Generation and Rendering . . . . . . . . . . . . . . . . 30
3.4 Environmental Effects . . . . . . . . . . . . . . . . . . . . . . . 32
3.5 Putting It All Together . . . . . . . . . . . . . . . . . . . . . . 34
3.6 Conclusions and Future Work . . . . . . . . . . . . . . . . . . . 35
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
II Rendering 39
Christopher Oat, editor
1 Pre-Integrated Skin Shading 41
Eric Penner and George Borshukov
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
1.2 Background and Previous Work . . . . . . . . . . . . . . . . . . 42
1.3 Pre-Integrating the Effects of Scattering . . . . . . . . . . . . . 42
1.4 Scattering and Diffuse Light . . . . . . . . . . . . . . . . . . . . 44
1.5 Scattering and Normal Maps . . . . . . . . . . . . . . . . . . . 47
1.6 Shadow Scattering . . . . . . . . . . . . . . . . . . . . . . . . . 48
1.7 Conclusion and Future Work . . . . . . . . . . . . . . . . . . . 51
1.8 Appendix A: Lookup Textures . . . . . . . . . . . . . . . . . . 52
1.9 Appendix B: Simplified Skin Shader . . . . . . . . . . . . . . . 53
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
2 Implementing Fur Using Deferred Shading 57
Donald Revie
2.1 Deferred Rendering . . . . . . . . . . . . . . . . . . . . . . . . . 57
2.2 Fur . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
2.3 Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
2.4 Fur Implementation Details . . . . . . . . . . . . . . . . . . . . 68
2.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
2.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 74
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
3 Large-Scale Terrain Rendering for Outdoor Games 77
Ferenc Pint´er
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
3.2 Content Creation and Editing . . . . . . . . . . . . . . . . . . . 79
3.3 Runtime Shading . . . . . . . . . . . . . . . . . . . . . . . . . . 84
3.4 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
3.5 Possible Extensions . . . . . . . . . . . . . . . . . . . . . . . . . 91
3.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 93
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
4 Practical Morphological Antialiasing 95
Jorge Jimenez, Belen Masia, Jose I. Echevarria, Fernando Navarro,
and Diego Gutierrez
4.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 97
4.2 Detecting Edges . . . . . . . . . . . . . . . . . . . . . . . . . . 98
4.3 Obtaining Blending Weights . . . . . . . . . . . . . . . . . . . . 100
4.4 Blending with the Four-Neighborhood . . . . . . . . . . . . . . 105
4.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
4.6 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
4.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 111
4.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 112
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
5 Volume Decals 115
Emil Persson
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
5.2 Decals as Volumes . . . . . . . . . . . . . . . . . . . . . . . . . 115
5.3 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
III Global Illumination Effects 121
Carsten Dachsbacher, editor
1 Temporal Screen-Space Ambient Occlusion 123
Oliver Mattausch, Daniel Scherzer, and Michael Wimmer
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123
1.2 Ambient Occlusion . . . . . . . . . . . . . . . . . . . . . . . . . 124
1.3 Reverse Reprojection . . . . . . . . . . . . . . . . . . . . . . . . 126
1.4 Our Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . 127
1.5 SSAO Implementation . . . . . . . . . . . . . . . . . . . . . . . 134
1.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 137
1.7 Discussion and Limitations . . . . . . . . . . . . . . . . . . . . 140
1.8 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 140
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
2 Level-of-Detail and Streaming Optimized Irradiance Normal Mapping 143
Ralf Habel, Anders Nilsson, and Michael Wimmer
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
2.2 Calculating Directional Irradiance . . . . . . . . . . . . . . . . 144
2.3 H-Basis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 146
2.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 149
2.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
2.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
2.7 Appendix A: Spherical Harmonics Basis Functions
without Condon-Shortley Phase . . . . . . . . . . . . . . . . . . 157
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 157
3 Real-Time One-Bounce Indirect Illumination and Shadows
using Ray Tracing 159
Holger Gruen
3.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 159
3.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 159
3.3 Phase 1: Computing Indirect Illumination without Indirect Shadows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 161
3.4 Phase 2: Constructing a 3D Grid of Blockers . . . . . . . . . . 165
3.5 Phase 3: Computing the Blocked Portion of Indirect Light . . . 168
3.6 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 170
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 171
4 Real-Time Approximation of Light Transport in
Translucent Homogenous Media 173
Colin Barr´e-Brisebois and Marc Bouchard
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 173
4.2 In Search of Translucency . . . . . . . . . . . . . . . . . . . . . 174
4.3 The Technique: The Way Out is Through . . . . . . . . . . . . 175
4.4 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 179
4.5 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 181
4.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 182
4.7 Demo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 183
4.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 183
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 183
5 Diffuse Global Illumination with Temporally Coherent
Light Propagation Volumes 185
Anton Kaplanyan, Wolfgang Engel, and Carsten Dachsbacher
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 185
5.2 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 186
5.3 Algorithm Detail Description . . . . . . . . . . . . . . . . . . . 187
5.4 Injection Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . 189
5.5 Optimizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
5.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
5.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
5.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 203
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203
IV Shadows 205
Wolfgang Engel, editor
1 Variance Shadow Maps Light-Bleeding Reduction Tricks 207
Wojciech Sterna
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
1.2 VSM Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
1.3 Light-Bleeding . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
1.4 Solutions to the Problem . . . . . . . . . . . . . . . . . . . . . . 210
1.5 Sample Application . . . . . . . . . . . . . . . . . . . . . . . . . 213
1.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 213
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 214
2 Fast Soft Shadows via Adaptive Shadow Maps 215
Pavlo Turchyn
2.1 Percentage-Closer Filtering with Large Kernels . . . . . . . . . 215
2.2 Application to Adaptive Shadow Maps . . . . . . . . . . . . . . 218
2.3 Soft Shadows with Variable Penumbra Size . . . . . . . . . . . 221
2.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 223
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 224
3 Adaptive Volumetric Shadow Maps 225
Marco Salvi, Kiril Vidimˇce, Andrew Lauritzen, Aaron Lefohn, and Matt Pharr
3.1 Introduction and Previous Approaches . . . . . . . . . . . . . . 225
3.2 Algorithm and Implementation . . . . . . . . . . . . . . . . . . 227
3.3 Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . 234
3.4 Conclusions and Future Work . . . . . . . . . . . . . . . . . . . 239
3.5 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 240
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 241
4 Fast Soft Shadows with Temporal Coherence 243
Daniel Scherzer, Michael Schw¨arzler and Oliver Mattausch
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 243
4.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 244
4.3 Comparison and Results . . . . . . . . . . . . . . . . . . . . . . 252
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 254
5 Mipmapped Screen-Space Soft Shadows 257
Alberto Aguado and Eugenia Montiel
5.1 Introduction and Previous Work . . . . . . . . . . . . . . . . . 257
5.2 Penumbra Width . . . . . . . . . . . . . . . . . . . . . . . . . 259
5.3 Screen-Space Filter . . . . . . . . . . . . . . . . . . . . . . . . 260
5.4 Filtering Shadows . . . . . . . . . . . . . . . . . . . . . . . . . . 263
5.5 Mipmap Level Selection . . . . . . . . . . . . . . . . . . . . . . 265
5.6 Multiple Occlusions . . . . . . . . . . . . . . . . . . . . . . . . 268
5.7 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 271
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 272
V Handheld Devices 275
Kristof Beets, editor
1 A Shader-Based eBook Renderer 277
Andrea Bizzotto
1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 277
1.2 Page-Peeling Effect . . . . . . . . . . . . . . . . . . . . . . . . . 278
1.3 Enabling Two Pages Side-by-Side . . . . . . . . . . . . . . . . . 283
1.4 Improving the Look and Antialiasing Edges . . . . . . . . . . . 285
1.5 Direction-Aligned Triangle Strip . . . . . . . . . . . . . . . . . 286
1.6 Performance Optimizations and Power Consumption . . . . . . 287
1.7 Putting it Together . . . . . . . . . . . . . . . . . . . . . . . . . 287
1.8 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 288
1.9 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 288
1.10 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 289
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 289
2 Post-Processing Effects on Mobile Devices 291
Marco Weber and Peter Quayle
2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 291
2.2 Technical Details . . . . . . . . . . . . . . . . . . . . . . . . . . 294
2.3 Case Study: Bloom . . . . . . . . . . . . . . . . . . . . . . . . . 296
2.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 298
2.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 304
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 305
3 Shader-Based Water Effects 307
Joe Davis and Ken Catterall
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 307
3.2 Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 307
3.3 Optimizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 318
3.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 325
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 325
VI 3D Engine Design 327
Wessam Bahnassi, editor
1 Practical, Dynamic Visibility for Games 329
Stephen Hill and Daniel Collin
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 329
1.2 Surveying the Field . . . . . . . . . . . . . . . . . . . . . . . . . 329
1.3 Query Quandaries . . . . . . . . . . . . . . . . . . . . . . . . . 330
1.4 Wish List . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 333
1.5 Conviction Solution . . . . . . . . . . . . . . . . . . . . . . . . 333
1.6 Battlefield Solution . . . . . . . . . . . . . . . . . . . . . . . . . 340
1.7 Future Development . . . . . . . . . . . . . . . . . . . . . . . . 342
1.8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 345
1.9 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 346
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 346
2 Shader Amortization using Pixel Quad Message Passing 349
Eric Penner
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 349
2.2 Background and Related Work . . . . . . . . . . . . . . . . . . 349
2.3 Pixel Derivatives and Pixel Quads . . . . . . . . . . . . . . . . 350
2.4 Pixel Quad Message Passing . . . . . . . . . . . . . . . . . . . . 352
2.5 PQA Initialization . . . . . . . . . . . . . . . . . . . . . . . . . 353
2.6 Limitations of PQA . . . . . . . . . . . . . . . . . . . . . . . . 354
2.7 Cross Bilateral Sampling . . . . . . . . . . . . . . . . . . . . . 356
2.8 Convolution and Blurring . . . . . . . . . . . . . . . . . . . . . 357
2.9 Percentage Closer Filtering . . . . . . . . . . . . . . . . . . . . 359
2.10 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 365
2.11 Appendix A: Hardware Support . . . . . . . . . . . . . . . . . . 366
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 366
3 A Rendering Pipeline for Real-Time Crowds 369
Benjam´ın Hern´andez and Isaac Rudomin
3.1 System Overview . . . . . . . . . . . . . . . . . . . . . . . . . . 369
3.2 Populating the Virtual Environment and Behavior . . . . . . . 371
3.3 View-Frustum Culling . . . . . . . . . . . . . . . . . . . . . . . 371
3.4 Level of Detail Sorting . . . . . . . . . . . . . . . . . . . . . . . 377
3.5 Animation and Draw Instanced . . . . . . . . . . . . . . . . . . 379
3.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 379
3.7 Conclusions and Future Work . . . . . . . . . . . . . . . . . . . 382
3.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 383
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 383
VII GPGPU 385
Sebastien St-Laurent, editor
1 2D Distance Field Generation with the GPU 387
Philip Rideout
1.1 Vocabulary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 388
1.2 Manhattan Grassfire . . . . . . . . . . . . . . . . . . . . . . . . 390
1.3 Horizontal-Vertical Erosion . . . . . . . . . . . . . . . . . . . . 392
1.4 Saito-Toriwaki Scanning with OpenCL . . . . . . . . . . . . . . 394
1.5 Signed Distance with Two Color Channels . . . . . . . . . . . . 402
1.6 Distance Field Applications . . . . . . . . . . . . . . . . . . . . 404
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 407
2 Order-Independent Transparency using Per-Pixel Linked Lists 409
Nicolas Thibieroz
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 409
2.2 Algorithm Overview . . . . . . . . . . . . . . . . . . . . . . . . 409
2.3 DirectX 11 Features Requisites . . . . . . . . . . . . . . . . . . 410
2.4 Head Pointer and Nodes Buffers . . . . . . . . . . . . . . . . . 411
2.5 Per-Pixel Linked List Creation . . . . . . . . . . . . . . . . . . 413
2.6 Per-Pixel Linked Lists Traversal . . . . . . . . . . . . . . . . . . 416
2.7 Multisampling Antialiasing Support . . . . . . . . . . . . . . . 421
2.8 Optimizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 425
2.9 Tiling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 427
2.10 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 430
2.11 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 431
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 431
3 Simple and Fast Fluids 433
Martin Guay, Fabrice Colin, and Richard Egli
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 433
3.2 Fluid Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . 434
3.3 Solver’s Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . 436
3.4 Code . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 440
3.5 Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 441
3.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 442
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 444
4 A Fast Poisson Solver for OpenCL using Multigrid Methods 445
Sebastien Noury, Samuel Boivin, and Olivier Le Maˆıtre
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 445
4.2 Poisson Equation and Finite Volume Method . . . . . . . . . . 446
4.3 Iterative Methods . . . . . . . . . . . . . . . . . . . . . . . . . . 451
4.4 Multigrid Methods (MG) . . . . . . . . . . . . . . . . . . . . . 457
4.5 OpenCL Implementation . . . . . . . . . . . . . . . . . . . . . . 460
4.6 Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 468
4.7 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 470
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 470
```

## GPU Pro 3
```
I Geometry Manipulation 1
Wolfgang Engel, editor
1 Vertex Shader Tessellation 3
Holger Gruen
1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.3 The Basic Vertex Shader Tessellation Algorithm . . . . . . . . 4
1.4 Per-Edge Fractional Tessellation Factors . . . . . . . . . . . . . 7
1.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2 Real-Time Deformable Terrain Rendering with DirectX 11 13
Egor Yusov
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.2 Algorithm Overview . . . . . . . . . . . . . . . . . . . . . . . . 15
2.3 Compressed Multiresolution Terrain Representation . . . . . . 15
2.4 Hardware-Accelerated Terrain Tessellation . . . . . . . . . . . 23
2.5 Texturing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
2.6 Dynamic Modifications . . . . . . . . . . . . . . . . . . . . . . 33
2.7 Implementation Details . . . . . . . . . . . . . . . . . . . . . . 34
2.8 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
2.9 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
2.10 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . 38
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
3 Optimized Stadium Crowd Rendering 41
Alan Chambers
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
3.2 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.3 Content Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . 43
3.4 Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
3.5 Further Optimizations . . . . . . . . . . . . . . . . . . . . . . . 60
3.6 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
3.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
3.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 68
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
4 Geometric Antialiasing Methods 71
Emil Persson
4.1 Introduction and Previous Work . . . . . . . . . . . . . . . . . 71
4.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
4.3 Conclusion and Future Work . . . . . . . . . . . . . . . . . . . 86
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
II Rendering 89
Christopher Oat, editor
1 Practical Elliptical Texture Filtering on the GPU 91
Pavlos Mavridis and Georgios Papaioannou
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
1.2 Elliptical Filtering . . . . . . . . . . . . . . . . . . . . . . . . . 92
1.3 Elliptical Footprint Approximation . . . . . . . . . . . . . . . . 97
1.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
1.5 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
1.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 103
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
2 An Approximation to the Chapman Grazing-Incidence Function for
Atmospheric Scattering 105
Christian Sch¨uler
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
2.2 Atmospheric Scattering . . . . . . . . . . . . . . . . . . . . . . 105
2.3 The Chapman Function . . . . . . . . . . . . . . . . . . . . . . 107
2.4 Towards a Real-Time Approximation . . . . . . . . . . . . . . . 109
2.5 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 111
2.6 Putting the Chapman Function to Use . . . . . . . . . . . . . . 114
2.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
2.8 Appendix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
3 Volumetric Real-Time Water and Foam Rendering 119
Daniel Scherzer, Florian Bagar, and Oliver Mattausch
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 119
3.2 Simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
3.3 Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 122
3.4 Artist Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . 129
3.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 131
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 131
4 CryENGINE 3: Three Years of Work in Review 133
Tiago Sousa, Nickolay Kasyan, and Nicolas Schulz
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
4.2 Going Multiplatform . . . . . . . . . . . . . . . . . . . . . . . . 134
4.3 Physically Based Rendering . . . . . . . . . . . . . . . . . . . . 137
4.4 Forward Shading Passes . . . . . . . . . . . . . . . . . . . . . . 155
4.5 Batched HDR Postprocessing . . . . . . . . . . . . . . . . . . . 158
4.6 Stereoscopic 3D . . . . . . . . . . . . . . . . . . . . . . . . . . . 163
4.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 167
4.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 167
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 167
5 Inexpensive Antialiasing of Simple Objects 169
Mikkel Gjøl and Mark Gjøl
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 169
5.2 Antialiasing via Smoothed Lines . . . . . . . . . . . . . . . . . 169
5.3 Rendering Lines . . . . . . . . . . . . . . . . . . . . . . . . . . 171
5.4 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 175
5.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 176
5.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 177
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 177
III Global Illumination Effects 179
Carsten Dachsbacher, editor
1 Ray-Traced Approximate Reflections Using a Grid of Oriented Splats 181
Holger Gruen
1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 181
1.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . 181
1.3 The Basic Algorithm . . . . . . . . . . . . . . . . . . . . . . . . 182
1.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
1.5 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 189
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 190
2 Screen-Space Bent Cones: A Practical Approach 191
Oliver Klehm, Tobias Ritschel, Elmar Eisemann, and Hans-Peter Seidel
2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 191
2.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 191
2.3 Ambient Occlusion . . . . . . . . . . . . . . . . . . . . . . . . . 192
2.4 Our Technique . . . . . . . . . . . . . . . . . . . . . . . . . . . 195
2.5 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
2.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
2.7 Discussion and Conclusion . . . . . . . . . . . . . . . . . . . . . 205
2.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 206
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 206
3 Real-Time Near-Field Global Illumination Based on a Voxel Model 209
Sinje Thiedemann, Niklas Henrich, Thorsten Grosch, and Stefan M¨uller
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
3.2 Binary Boundary Voxelization . . . . . . . . . . . . . . . . . . . 210
3.3 Hierarchical Ray/Voxel Intersection Test . . . . . . . . . . . . . 215
3.4 Near-Field Indirect Illumination . . . . . . . . . . . . . . . . . . 222
3.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 225
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 228
IV Shadows 231
Wolfgang Engel, editor
1 Efficient Online Visibility for Shadow Maps 233
Oliver Mattausch, Jiri Bittner, Ari Silvennoinen, Daniel Scherzer,
and Michael Wimmer
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 233
1.2 Algorithm Overview . . . . . . . . . . . . . . . . . . . . . . . . 234
1.3 Detailed Description . . . . . . . . . . . . . . . . . . . . . . . . 236
1.4 Optimization: Shadow-Map Focusing . . . . . . . . . . . . . . . 238
1.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 239
1.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 241
1.7 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . 241
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 241
2 Depth Rejected Gobo Shadows 243
John White
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 243
2.2 Basic Gobo Shadows . . . . . . . . . . . . . . . . . . . . . . . . 243
2.3 Depth Rejected Gobo Shadows . . . . . . . . . . . . . . . . . . 245
2.4 Extensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 247
2.5 Failure Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 248
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 248
V 3D Engine Design 249
Wessam Bahnassi, editor
1 Z3 Culling 251
Pascal Gautron, Jean-Eudes Marvie, and Ga¨el Sourimant
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 251
1.2 Principle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 253
1.3 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 253
1.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 254
1.5 Performance Analysis . . . . . . . . . . . . . . . . . . . . . . . 261
1.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 263
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 263
2 A Quaternion-Based Rendering Pipeline 265
Dzmitry Malyshau
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 265
2.2 Spatial Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . 265
2.3 Handedness Bit . . . . . . . . . . . . . . . . . . . . . . . . . . . 266
2.4 Facts about Quaternions . . . . . . . . . . . . . . . . . . . . . . 267
2.5 Tangent Space . . . . . . . . . . . . . . . . . . . . . . . . . . . 268
2.6 Interpolation Problem with Quaternions . . . . . . . . . . . . . 270
2.7 KRI Engine: An Example Application . . . . . . . . . . . . . . 271
2.8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 272
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 273
3 Implementing a Directionally Adaptive Edge AA Filter Using
DirectX 11 275
Matthew Johnson
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 275
3.2 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 285
3.3 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 288
3.4 Appendix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 289
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 290
4 Designing a Data-Driven Renderer 291
Donald Revie
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 291
4.2 Problem Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . 292
4.3 Solution Development . . . . . . . . . . . . . . . . . . . . . . . 304
4.4 Representational Objects . . . . . . . . . . . . . . . . . . . . . 307
4.5 Pipeline Objects . . . . . . . . . . . . . . . . . . . . . . . . . . 310
4.6 Frame Graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . 313
4.7 Case Study: Praetorian Tech . . . . . . . . . . . . . . . . . . . 314
4.8 Further Work and Considerations . . . . . . . . . . . . . . . . . 317
4.9 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 318
4.10 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 318
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 318
VI GPGPU 321
Sebastien St-Laurent, editor
1 Volumetric Transparency with Per-Pixel Fragment Lists 323
L´aszl´o Sz´ecsi, P´al Barta, and Bal´azs Kov´acs
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 323
1.2 Light Transport Model . . . . . . . . . . . . . . . . . . . . . . . 324
1.3 Ray Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . 325
1.4 Finding Intersections with Ray Casting . . . . . . . . . . . . . 327
1.5 Application for Particle System Rendering . . . . . . . . . . . . 330
1.6 Finding Intersections with Rasterization . . . . . . . . . . . . . 331
1.7 Adding Surface Reflection . . . . . . . . . . . . . . . . . . . . . 333
1.8 Shadow Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . 333
1.9 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 334
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 335
2 Practical Binary Surface and Solid Voxelization with Direct3D 11 337
Michael Schwarz
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 337
2.2 Rasterization-Based Surface Voxelization . . . . . . . . . . . . . 338
2.3 Rasterization-Based Solid Voxelization . . . . . . . . . . . . . . 342
2.4 Conservative Surface Voxelization with DirectCompute . . . . . 344
2.5 Solid Voxelization with DirectCompute . . . . . . . . . . . . . . 349
2.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 351
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 352
3 Interactive Ray Tracing Using the Compute Shader in DirectX 11 353
Arturo Garc´ıa, Francisco ´Avila, Sergio Murgu´ıa, and Leo Reyes
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 353
3.2 Ray Tracing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 354
3.3 Our Implementation . . . . . . . . . . . . . . . . . . . . . . . . 357
3.4 Primary Rays Stage . . . . . . . . . . . . . . . . . . . . . . . . 361
3.5 Intersection Stage . . . . . . . . . . . . . . . . . . . . . . . . . . 363
3.6 Color Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 367
3.7 Multipass Approach . . . . . . . . . . . . . . . . . . . . . . . . 371
3.8 Results and Discussion . . . . . . . . . . . . . . . . . . . . . . . 371
3.9 Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 373
3.10 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 374
3.11 Further Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 374
3.12 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 375
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 376
```

## GPU Pro 4
```
I Geometry Manipulation 1
Wolfgang Engel
1 GPU Terrain Subdivision and Tessellation 3
Benjamin Mistal
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 The Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
1.4 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
2 Introducing the Programmable Vertex Pulling Rendering Pipeline 21
Christophe Riccio and Sean Lilley
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2.2 Draw Submission Limitations and Objectives . . . . . . . . . . 22
2.3 Evaluating Draw Call CPU Overhead and the GPU Draw
Submission Limitation . . . . . . . . . . . . . . . . . . . . . . . 23
2.4 Programmable Vertex Pulling . . . . . . . . . . . . . . . . . . . 28
2.5 Side Effects of the Software Design . . . . . . . . . . . . . . . . 34
2.6 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
2.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
3 A WebGL Globe Rendering Pipeline 39
Patrick Cozzi and Daniel Bagnell
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
3.2 Rendering Pipeline Overview . . . . . . . . . . . . . . . . . . . 39
3.3 Filling Cracks in Screen Space . . . . . . . . . . . . . . . . . . . 40
3.4 Filling Poles in Screen Space . . . . . . . . . . . . . . . . . . . 42
3.5 Overlaying Vector Data . . . . . . . . . . . . . . . . . . . . . . 44
3.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
3.7 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 47
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
II Rendering 49
Christopher Oat and Carsten Dachsbacher
1 Practical Planar Reflections Using Cubemaps and Image Proxies 51
S´ebastien Lagarde and Antoine Zanuttini
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
1.2 Generating Reflection Textures . . . . . . . . . . . . . . . . . . 52
1.3 Using Reflection Textures . . . . . . . . . . . . . . . . . . . . . 63
1.4 Conclusion and Future Work . . . . . . . . . . . . . . . . . . . 66
1.5 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 67
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
2 Real-Time Ptex and Vector Displacement 69
Karl Hillesland
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
2.2 Packed Ptex . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
2.3 Runtime Implementation . . . . . . . . . . . . . . . . . . . . . . 72
2.4 Adding Displacement . . . . . . . . . . . . . . . . . . . . . . . . 75
2.5 Performance Costs . . . . . . . . . . . . . . . . . . . . . . . . . 76
2.6 Memory Costs . . . . . . . . . . . . . . . . . . . . . . . . . . . 78
2.7 Alternatives and Future Work . . . . . . . . . . . . . . . . . . . 79
2.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 79
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
3 Decoupled Deferred Shading on the GPU 81
G´abor Liktor and Carsten Dachsbacher
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
3.2 Decoupled Sampling in a Rasterization Pipeline . . . . . . . . . 82
3.3 Shading Reuse for Deferred Shading . . . . . . . . . . . . . . . 84
3.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
3.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
3.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 97
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 97
4 Tiled Forward Shading 99
Markus Billeter, Ola Olsson, and Ulf Assarsson
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 99
4.2 Recap: Forward, Deferred, and Tiled Shading . . . . . . . . . . 101
4.3 Tiled Forward Shading: Why? . . . . . . . . . . . . . . . . . . 104
4.4 Basic Tiled Forward Shading . . . . . . . . . . . . . . . . . . . 104
4.5 Supporting Transparency . . . . . . . . . . . . . . . . . . . . . 106
4.6 Support for MSAA . . . . . . . . . . . . . . . . . . . . . . . . . 109
4.7 Supporting Different Shaders . . . . . . . . . . . . . . . . . . . 111
4.8 Conclusion and Further Improvements . . . . . . . . . . . . . . 111
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 113
5 Forward+: A Step Toward Film-Style Shading in Real Time 115
Takahiro Harada, Jay McKee, and Jason C. Yang
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
5.2 Forward+ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 116
5.3 Implementation and Optimization . . . . . . . . . . . . . . . . 117
5.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123
5.5 Forward+ in the AMD Leo Demo . . . . . . . . . . . . . . . . . 124
5.6 Extensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 127
5.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
5.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 134
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 134
6 Progressive Screen-Space Multichannel Surface Voxelization 137
Athanasios Gaitatzes and Georgios Papaioannou
6.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 137
6.2 Overview of Voxelization Method . . . . . . . . . . . . . . . . . 138
6.3 Progressive Voxelization for Lighting . . . . . . . . . . . . . . . 144
6.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 145
6.5 Performance and Evaluation . . . . . . . . . . . . . . . . . . . . 145
6.6 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 151
6.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 153
6.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 153
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 153
7 Rasterized Voxel-Based Dynamic Global Illumination 155
Hawar Doghramachi
7.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
7.2 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
7.3 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 156
7.4 Handling Large Environments . . . . . . . . . . . . . . . . . . . 168
7.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 168
7.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 169
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 171
III Image Space 173
Michal Valient
1 The Skylanders SWAP Force Depth-of-Field Shader 175
Michael Bukowski, Padraic Hennessy, Brian Osman, and Morgan McGuire
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 175
1.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 177
1.3 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
2 Simulating Partial Occlusion in Post-Processing
Depth-of-Field Methods 187
David C. Schedl and Michael Wimmer
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
2.2 Depth of Field . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
2.3 Algorithm Overview . . . . . . . . . . . . . . . . . . . . . . . . 189
2.4 Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 190
2.5 Scene Decomposition . . . . . . . . . . . . . . . . . . . . . . . . 190
2.6 Blurring and Composition . . . . . . . . . . . . . . . . . . . . . 194
2.7 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 197
2.8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 198
2.9 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 199
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
3 Second-Depth Antialiasing 201
Emil Persson
3.1 Introduction and Previous Work . . . . . . . . . . . . . . . . . 201
3.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
3.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
3.4 Conclusion and Future Work . . . . . . . . . . . . . . . . . . . 211
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 211
4 Practical Framebuffer Compression 213
Pavlos Mavridis and Georgios Papaioannou
4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 213
4.2 Color Space Conversion . . . . . . . . . . . . . . . . . . . . . . 214
4.3 Chrominance Multiplexing . . . . . . . . . . . . . . . . . . . . . 215
4.4 Chrominance Reconstruction . . . . . . . . . . . . . . . . . . . 217
4.5 Antialiasing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 221
4.6 Blending . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 222
4.7 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 222
4.8 Conclusion and Discussion . . . . . . . . . . . . . . . . . . . . . 224
4.9 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 225
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 225
5 Coherence-Enhancing Filtering on the GPU 227
Jan Eric Kyprianidis and Henry Kang
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 227
5.2 Local Orientation Estimation . . . . . . . . . . . . . . . . . . . 229
5.3 Flow-Guided Smoothing . . . . . . . . . . . . . . . . . . . . . . 238
5.4 Shock Filter . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 243
5.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 248
5.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 248
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 248
IV Shadows 251
Wolfgang Engel
1 Real-Time Deep Shadow Maps 253
Ren´e F¨urst, Oliver Mattausch, and Daniel Scherzer
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 253
1.2 Transmittance Function . . . . . . . . . . . . . . . . . . . . . . 255
1.3 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 255
1.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 262
1.5 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 263
1.6 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 264
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 264
V Game Engine Design 265
Wessam Bahnassi
1 An Aspect-Based Engine Architecture 267
Donald Revie
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 267
1.2 Rationale . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 267
1.3 Engine Core . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 268
1.4 Aspects . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 272
1.5 Common Aspects . . . . . . . . . . . . . . . . . . . . . . . . . . 275
1.6 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 277
1.7 Aspect Interactions . . . . . . . . . . . . . . . . . . . . . . . . . 278
1.8 Praetorian: The Brief History of Aspects . . . . . . . . . . . . . 281
1.9 Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 282
1.10 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 283
1.11 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 283
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 284
2 Kinect Programming with Direct3D 11 285
Jason Zink
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 285
2.2 Meet the Kinect . . . . . . . . . . . . . . . . . . . . . . . . . . 285
2.3 Mathematics of the Kinect . . . . . . . . . . . . . . . . . . . . . 289
2.4 Programming with the Kinect SDK . . . . . . . . . . . . . . . . 292
2.5 Applications of the Kinect . . . . . . . . . . . . . . . . . . . . . 300
2.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 302
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 302
3 A Pipeline for Authored Structural Damage 303
Homam Bahnassi and Wessam Bahnassi
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 303
3.2 The Addressed Problem . . . . . . . . . . . . . . . . . . . . . . 303
3.3 Challenges and Previous Work . . . . . . . . . . . . . . . . . . 304
3.4 Implementation Description and Details . . . . . . . . . . . . . 305
3.5 Level of Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . 313
3.6 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 314
3.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 314
3.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 314
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 315
VI GPGPU 317
S´ebastien St-Laurent
1 Bit-Trail Traversal for Stackless LBVH on DirectCompute 319
Sergio Murgu´ıa, Francisco ´
Avila, Leo Reyes, and Arturo Garc´ıa
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 319
1.2 Ray Tracing Rendering . . . . . . . . . . . . . . . . . . . . . . . 320
1.3 Global Illumination . . . . . . . . . . . . . . . . . . . . . . . . . 320
1.4 Stackless LBVH . . . . . . . . . . . . . . . . . . . . . . . . . . . 322
1.5 The SLBVH in Action . . . . . . . . . . . . . . . . . . . . . . . 331
1.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 334
1.7 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 335
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 335
2 Real-Time JPEG Compression Using DirectCompute 337
Stefan Petersson
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 337
2.2 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 342
2.3 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 352
2.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 355
2.5 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 355
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 355
```

## GPUZen Advanced Rendering Techniques
```
I Geometry Manipulation 1
Christopher Oat, editor
1 Attributed Vertex Clouds 3
Willy Scheibel, Stefan Buschmann, Matthias Trapp, and J¨urgen D¨ollner
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Concept . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.3 Applications for Attributed Vertex Clouds . . . . . . . . . . . . 5
1.4 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
1.5 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2 Rendering Convex Occluders with Inner Conservative
Rasterization 23
Marcus Svensson and Emil Persson
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
2.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
2.3 Conclusions and Future Work . . . . . . . . . . . . . . . . . . . 29
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
II Lighting 31
Carsten Dachsbacher, editor
1 Stable Indirect Illumination 33
Holger Gruen and Louis Bavoil
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
1.2 Deinterleaved Texturing for RSM Sampling . . . . . . . . . . . 34
1.3 Adding Sub-image Blurs . . . . . . . . . . . . . . . . . . . . . . 38
1.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
2 Participating Media Using Extruded Light Volumes 45
Nathan Hoobler, Andrei Tatarinov, and Alex Dunn
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
2.2 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
2.3 Directional Lights . . . . . . . . . . . . . . . . . . . . . . . . . 51
2.4 Local Lights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
2.5 Additional Optimizations . . . . . . . . . . . . . . . . . . . . . 67
2.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
2.7 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
III Rendering 75
Mark Chatfield, editor
1 Deferred+ 77
Hawar Doghramachi and Jean-Normand Bucci
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
1.2 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
1.3 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
1.4 Comparison with Hierarchical
Depth Buffer-based Culling . . . . . . . . . . . . . . . . . . . . 96
1.5 Pros and Cons . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
1.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
1.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
2 Programmable Per-pixel Sample Placement with
Conservative Rasterizer 105
Rahul P. Sathe
2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
2.2 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
2.3 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
2.4 Demo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 114
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 114
3 Mobile Toon Shading 115
Felipe Lira, Felipe Chaves, Fl´avio Villalva, Jesus Sosa,
Kl´everson Paix˜ao and Te´ofilo Dutra
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
3.2 Technique Overview . . . . . . . . . . . . . . . . . . . . . . . . 116
3.3 Flat Shading . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
3.4 Soft Light Blending . . . . . . . . . . . . . . . . . . . . . . . . . 117
3.5 Halftone-based Shadows . . . . . . . . . . . . . . . . . . . . . . 118
3.6 Threshold-based Inverted Hull Outline . . . . . . . . . . . . . . 119
3.7 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
3.8 Final Considerations . . . . . . . . . . . . . . . . . . . . . . . . 121
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 121
4 High Quality GPU-efficient Image Detail Manipulation 123
Kin-Ming Wong and Tien-Tsin Wong
4.1 Image Detail Manipulation Pipeline . . . . . . . . . . . . . . . 124
4.2 Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . 126
4.3 GLSL Compute Shader-based Implementation . . . . . . . . . . 129
4.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
4.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
4.6 Acknowledgement . . . . . . . . . . . . . . . . . . . . . . . . . . 136
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 136
5 Linear-Light Shading with Linearly Transformed Cosines 137
Eric Heitz and Stephen Hill
5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 137
5.2 The Linear-Light Shading Model . . . . . . . . . . . . . . . . . 140
5.3 Line-integral of a Diffuse Material . . . . . . . . . . . . . . . . 147
5.4 Line-Integral of a Glossy Material with LTCs . . . . . . . . . . 150
5.5 Adding the End Caps . . . . . . . . . . . . . . . . . . . . . . . 154
5.6 Rectangle-Like Linear Lights . . . . . . . . . . . . . . . . . . . 157
5.7 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 160
5.8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 160
5.9 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 160
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 160
6 Profiling and Optimizing WebGL Applications Using Google
Chrome 163
Gareth Morgan
6.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 163
6.2 Browser Profiling Tools . . . . . . . . . . . . . . . . . . . . . . 167
6.3 Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . 174
6.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 180
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 180
IV Screen Space 181
Wessam Bahnassi, editor
1 Scalable Adaptive SSAO 183
Filip Strugar
1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 183
1.2 Problem Statement . . . . . . . . . . . . . . . . . . . . . . . . . 183
1.3 ASSAO—A High-level Overview . . . . . . . . . . . . . . . . . 184
1.4 SSAO—A Quick Refresh . . . . . . . . . . . . . . . . . . . . . . 185
1.5 Scaling the SSAO . . . . . . . . . . . . . . . . . . . . . . . . . . 186
1.6 Sampling Kernel . . . . . . . . . . . . . . . . . . . . . . . . . . 197
1.7 Adaptive SSAO . . . . . . . . . . . . . . . . . . . . . . . . . . . 197
1.8 Putting It All Together . . . . . . . . . . . . . . . . . . . . . . 199
1.9 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
2 Robust Screen Space Ambient Occlusion 203
Wojciech Sterna
2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203
2.2 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . 203
2.3 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 205
2.4 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
2.5 Possible Improvements . . . . . . . . . . . . . . . . . . . . . . . 213
2.6 Demo Application . . . . . . . . . . . . . . . . . . . . . . . . . 214
2.7 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 215
2.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 215
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 216
3 Practical Gather-based Bokeh Depth of Field 217
Wojciech Sterna
3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
3.2 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . 217
3.3 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 221
3.4 Implementation Details . . . . . . . . . . . . . . . . . . . . . . 227
3.5 Per-pixel Kernel Scale . . . . . . . . . . . . . . . . . . . . . . . 235
3.6 Demo Application . . . . . . . . . . . . . . . . . . . . . . . . . 236
3.7 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 237
3.8 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . 237
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 237
V Virtual Reality 239
Eric Haines, editor
1 Efficient Stereo and VR Rendering 241
´I˜nigo Qu´ılez
1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 241
1.2 Engine Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . 241
1.3 Stereo Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . 247
1.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 251
2 Understanding, Measuring, and Analyzing VR
Graphics Performance 253
James Hughes, Reza Nourai, and Ed Hutchins
2.1 VR Graphics Overview . . . . . . . . . . . . . . . . . . . . . . . 253
2.2 Trace Collection . . . . . . . . . . . . . . . . . . . . . . . . . . 259
2.3 Analyzing Traces . . . . . . . . . . . . . . . . . . . . . . . . . . 263
2.4 The Big Picture . . . . . . . . . . . . . . . . . . . . . . . . . . . 274
VI Compute 275
Wolfgang Engel, editor
1 Optimizing the Graphics Pipeline with Compute 277
Graham Wihlidal
1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 277
1.2 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 278
1.3 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 280
1.4 Draw Association . . . . . . . . . . . . . . . . . . . . . . . . . . 283
1.5 Draw Compaction . . . . . . . . . . . . . . . . . . . . . . . . . 287
1.6 Cluster Culling . . . . . . . . . . . . . . . . . . . . . . . . . . . 291
1.7 Triangle Culling . . . . . . . . . . . . . . . . . . . . . . . . . . . 293
1.8 Batch Scheduling . . . . . . . . . . . . . . . . . . . . . . . . . . 308
1.9 De-interleaved Vertex Buffers . . . . . . . . . . . . . . . . . . . 310
1.10 Hardware Tessellation . . . . . . . . . . . . . . . . . . . . . . . 311
1.11 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 315
1.12 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 317
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 319
2 Real Time Markov Decision Processes for Crowd Simulation 321
Sergio Ruiz and Benjam´ın Hern´andez
2.1 Modeling Agent Navigation using a
Markov Decision Process . . . . . . . . . . . . . . . . . . . . . . 321
2.2 Crowd Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . 332
2.3 Coupling the MDP Solver with Crowd
Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 333
2.4 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 335
2.5 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 338
Bibliography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 338
```

## Numerical Methods for Engineers
https://www.math.hkust.edu.hk/~machas/numerical-methods-for-engineers.pdf
```
I Scientific Computing 1
1 Binary numbers 2
2 Double precision 4
3 Matlab as a calculator 6
4 Scripts and functions 8
5 Vectors 10
6 Line plots 13
7 Matrices 16
8 Logicals 20
9 Conditionals 22
10 Loops 24
11 Project I: Logistic map (Part A) 26
12 Project I: Logistic map (Part B) 28
II Root Finding 30
13 Bisection method 31
14 Newton’s method 33
15 Secant method 35
16 Order of convergence 37
17 Convergence of Newton’s method 39
18 Fractals from Newton’s method 41
19 Coding the Newton fractal 43
20 Root finding in Matlab 46
21 Project II: Feigenbaum delta (Part A) 48
ivCONTENTS v
22 Project II: Feigenbaum delta (Part B) 50
23 Project II: Feigenbaum delta (Part C) 51
III Matrix Algebra 53
24 Gaussian elimination without pivoting 54
25 Gaussian elimination with partial pivoting 56
26 LU decomposition with partial pivoting 58
27 Operation counts 61
28 Operation counts for Gaussian elimination 63
29 Operation counts for forward and backward substitution 65
30 Eigenvalue power method 67
31 Eigenvalue power method (example) 69
32 Matrix algebra in Matlab 71
33 Systems of nonlinear equations 74
34 Systems of nonlinear equations (example) 76
35 Project III: Fractals from the Lorenz equations 78
IV Quadrature and Interpolation 80
36 Midpoint rule 81
37 Trapezoidal rule 83
38 Simpson’s rule 85
39 Composite quadrature rules 87
40 Gaussian quadrature 89
41 Adaptive quadrature 91
42 Quadrature in Matlab 93
43 Interpolation 95
44 Cubic spline interpolation (Part A) 97CONTENTS vi
45 Cubic spline interpolation (Part B) 99
46 Interpolation in Matlab 102
47 Project IV: Bessel functions and their zeros 104
V Ordinary Differential Equations 106
48 Euler method 107
49 Modified Euler method 109
50 Runge-Kutta methods 111
51 Second-order Runge-Kutta methods 112
52 Higher-order Runge-Kutta methods 114
53 Higher-order odes and systems 116
54 Adaptive Runge-Kutta methods 118
55 Integrating odes in Matlab (Part A) 120
56 Integrating odes in Matlab (Part B) 121
57 Shooting method for boundary value problems 124
58 Project V: Two-body problem (Part A) 126
59 Project V: Two-body problem (Part B) 128
VI Partial Differential Equations 130
60 Boundary and initial value problems 131
Practice quiz: Classify partial differential equations 132
61 Central difference approximation 133
62 Discrete Laplace equation 135
63 Natural ordering 137
64 Matrix formulation 139
65 Matlab solution of the Laplace equation (direct method) 141
66 Jacobi, Gauss-Seidel and SOR methods 144CONTENTS vii
67 Red-black ordering 146
68 Matlab solution of the Laplace equation (iterative method) 147
69 Explicit methods for solving the diffusion equation 149
70 Von Neumann stability analysis 151
71 Implicit methods for solving the diffusion equation 153
72 Crank-Nicolson method for the diffusion equation 155
73 Matlab solution of the diffusion equation 157
74 Project VI: Two-dimensional diffusion equation 160
```
