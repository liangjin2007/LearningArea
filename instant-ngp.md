# Neural Geometry Primitive

## paper 1 related to Neural Radiance Caching https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf
- 有关工作
```
Radiance Caching
  1. irradiance caching, irradiance在场景中是比较smoothly变化， 可通过albedo modulation恢复texture细节。 [1988]
  2. irradiance probe volume [1998]难点在probe之间的插值及probe位置的选择
  3. [2008、2009]有一波改进probe之间的插值的改进
  4. radiance cache： representing the directional domain with spherical harmonics[2005], 因为glossy surface不满足 ** irradiance在场景中是比较smoothly变化 ** 这一条。 
     这个radiance cache是一种数据结构，用来预计算lighting信息。 offline rendering[2017、2018、2019], realtime rendering （compression, sparse interpolation[2017], pre-convolved environment   maps[2014、2012], spatial hashing[2018、2020]）
```
```
Precomputation-based techniques
  lightmap : 当场景中光和geometry都不动时，可以把irradiance预计算到texture中。[1997]
  light probes : [2005]
  iterated dynamic lighting updates [2010]
  combine with other technique like precomputed radiance transfer or visibility [2017]
```
```
Fully dynamic techniques : build upon efficient rendering algorithms that reuse shading and visibility computation across pixels, extracting further efficiency through various approximations
   photon mapping[1996]
   many-light rendering[1997]
   radiosity maps[2004]
         Use point cloud to approximate scene geometry, the point cloud is rasterized into shadow map
   ray tracing hardware
      diffuse interreflections
      glossy reflection
   以上都有个问题是需要更新一个dual scene representation动态地连续地更新。   

```
```
Path guiding
   importance sampling
```
```

```

- neural radiance cache


$$
\textit{L}_{S}\left(\textbf{x}, \omega\right)
$$


$$
=\(\int_{S^2}f_{S}(\textbf{x}, \omega,\omega_{i}) L_{i}(\textbf{x}, \omega_{i})|dot(\omega_{i}, \textbf{n})| d\omega_{i}\)
$$


  - scattered radiance is across $\textbf{x} $ in the direction of $\omega$

  - $\omega_{i}$ is incident direction

  - $f_{S}(\textbf{x}, \omega,\omega_{i})$ is BSDF

  - $L_{i}(\textbf{x}, \omega_{i})$ is incident radiance入射辐度



### Diffuse interreflections : 说的是反射折射引起的简介光照
```
Diffuse interreflections occur when light is reflected and scattered by surfaces in a scene, resulting in indirect lighting. 
This occurs when light from a light source bounces off a surface and illuminates other surfaces in the scene. 
The light that is scattered in various directions by the illuminated surfaces then further reflects off other surfaces, creating a chain reaction of indirect lighting.

Diffuse interreflections are an important factor to consider in computer graphics and computer vision, as they play a significant role in determining the overall appearance of a scene. 
Accurately modeling diffuse interreflections can help create more realistic and physically accurate renderings of scenes.
```



### Albedo modulation : 指调节surface亮度
```
Albedo modulation refers to the process of modifying the reflectivity or albedo of a surface to achieve a desired effect. 
Albedo is a measure of the fraction of incoming light that a surface reflects, with a value of 1 indicating perfect reflection and a value of 0 indicating complete absorption.

In computer graphics and computer vision, albedo modulation is often used to adjust the appearance of surfaces in a scene.
For example, by increasing the albedo of a surface, the surface will reflect more light and appear brighter, while decreasing the albedo will result in a darker surface. 
This can be used to create different lighting effects, simulate the appearance of different materials, or enhance the contrast in an image.

Albedo modulation can be achieved through various techniques, such as adjusting the surface material properties, applying post-processing effects, or using image-based lighting. 
In addition, machine learning algorithms can be used to automatically adjust the albedo of surfaces based on the desired output.

```


### Sphere Harmonics for Radiance Cache 
```
Spherical harmonics are a set of mathematical functions that can be used to represent functions on the surface of a sphere.
In computer graphics, spherical harmonics are often used to precompute and store lighting information in a radiance cache.

A radiance cache is a data structure that stores precomputed lighting information for a scene, which can be used to accelerate rendering by avoiding expensive lighting calculations.
Using spherical harmonics to represent the lighting information in the radiance cache allows for efficient storage and interpolation of lighting values across the surface of the sphere.

To use spherical harmonics for radiance caching, the lighting information for a scene is first represented as a set of coefficients corresponding to the basis functions of the spherical harmonics.
These coefficients are then stored in the radiance cache, along with the positions and orientations of the cache points.
During rendering, the lighting value at a given point in the scene can be quickly approximated by interpolating the coefficients of the nearest cache points using the spherical harmonics basis functions.

Spherical harmonics offer an efficient and accurate way to represent and store lighting information for radiance caching, making them a popular choice in many computer graphics applications.
```
