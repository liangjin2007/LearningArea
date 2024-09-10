# PBR 

About: Physics based realtime rendering. 

Reference: github PBR source code https://github.com/Nadrin/PBR/tree/master

参考的PPT：https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
参考的PPT2： [2022]Real-Time Global Illumination in Unreal Engine 5, A Master's Thesis



## Lambert模型
```
Lambert模型在计算机图形学中通常用于模拟物体表面的光照效果。在早期的Lambert模型中，一个物体的最终颜色（或者说是像素的颜色）通常由以下几个分量组成：
  环境光（Ambient）：这是场景中均匀照亮物体的光。它不考虑光源的位置和方向，为物体提供基础亮度，使得物体不会完全黑暗。
  漫反射（Diffuse）：这部分模拟了光线击中粗糙表面后，光线在各个方向均匀反射的现象。漫反射的强度取决于光线与表面法线的夹角，当光线垂直于表面时，漫反射最强。
  镜面反射（Specular）：这部分模拟了光线在光滑表面上的直接反射。镜面反射通常表现为亮点或高光，其强度取决于观察角度和表面的光滑程度。
除了上述分量，还有一些其他分量可能会被包括在更复杂的Lambert模型或其扩展模型中：
  高光（Specular Highlight）：虽然镜面反射中已经包含了高光的概念，但在一些模型中，高光可能会被单独处理，以更精细地模拟光线在非常光滑表面上的反射。
  发射（Emission）：某些物体可以自己发光，如光源或屏幕上的发光物体。发射分量直接为物体添加颜色，不考虑任何外部光照。
  透明度（Transparency）/折射（Refraction）：如果物体是透明的，那么光线会穿过物体并可能发生折射。这通常需要额外的计算来模拟。
  阴影（Shadows）：虽然不是颜色分量的一部分，但阴影会影响物体的最终外观。在考虑最终颜色时，通常需要考虑物体是否被其他物体遮挡，从而接收不到某些光线。
  反射（Reflection）：某些高级模型可能会考虑物体表面的反射效果，即物体反射其周围环境的能力。
需要注意的是，Lambert模型本身主要考虑的是漫反射分量，而其他分量通常是在此基础上扩展或结合其他光照模型来实现的。随着计算机图形学的发展，这些模型已经变得更加复杂和真实，能够模拟更多细节的光照效果。
```

## 预计算lightmap
```
// VS
void main() {
    // 计算世界空间中的位置和法线
    vec4 worldPosition = modelMatrix * vec4(inPosition, 1.0);
    vec3 worldNormal = normalize(mat3(modelMatrix) * inNormal);
    // 计算光照强度（简单的Lambert漫反射）
    float diff = dot(worldNormal, -lightDirection);
    // 输出光照强度作为纹理坐标，用于后续烘焙到lightmap
    // 这里简化处理，直接使用光照强度作为R通道，其他通道为0
    outTexCoord = vec2(diff, 0.0);
    // 进行实际的顶点变换
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
}

// FS
// 输入：光照强度
in vec2 inTexCoord;
// 输出：lightmap颜色
layout(location = 0) out vec4 outColor;
void main() {
    // 简化处理，直接使用光照强度作为颜色输出
    outColor = vec4(inTexCoord.x, 0.0, 0.0, 1.0); // 只使用R通道，其他通道为0或1
}
```

## 源代码解读
```
三个program用来调用drawcall的时候使用。skybox program, pbr program, tonemap program

每帧的render函数做如下的事情:
  关掉depth test先渲染天空盒，
  开启depth test渲染pbr
  全屏triangle的tone mapping/post processing。

天空盒
  skybox_fs.glsl
  skybox_vs.glsl
  layout(location=0) in vec3 localPosition;
  layout(binding=0) uniform samplerCube envTexture;
  vec3 envVector = normalize(localPosition);
	color = textureLod(envTexture, envVector, 0);


Tone mapping：
    vs: 画了一个能覆盖屏幕的三角形。
    fs:
      const float gamma     = 2.2;
      const float exposure  = 1.0;
      const float pureWhite = 1.0;
      vec3 color = texture(sceneColor, screenPosition).rgb * exposure;
      vec3 luminance = dot(color, vec3(0.2126, 0.7152, 0.0722))/*luminance*/ ;
      vec3 mappedLuminance = (luminance*(1+luminance/(pureWhite*pureWhite)))/(1+luminance);
      vec3 mappedColor = (mappedLuminance / luminance) * color;
      	// Gamma correction.
	    outColor = vec4(pow(mappedColor, vec3(1.0/gamma)), 1.0);   // color^(1/gamma)



PBR material：//总体上是这三部分： Physically Based shading model: Lambetrtian diffuse BRDF + Cook-Torrance microfacet specular BRDF + IBL for ambient.

// Constant normal incidence Fresnel factor for all dielectrics.
const vec3 Fdielectric = vec3(0.04);
const int NumLights = 3;

// Fresnel: Shlick's approximation of the Fresnel factor.
vec3 fresnelSchlick(vec3 F0, float cosTheta)
{
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Distribution: GGX/Towbridge-Reitz normal distribution function.
// Uses Disney's reparametrization of alpha = roughness^2.
float ndfGGX(float cosLh, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}


// Single term for separable Schlick-GGX below.
float gaSchlickG1(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method.
float gaSchlickGGX(float cosLi, float cosLo, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0; // Epic suggests using this roughness remapping for analytic lights.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// fragment shader
void main()
{
	vec3 albedo = texture(albedoTexture, vin.texcoord).rgb;			// Diffuse related
	float metalness = texture(metalnessTexture, vin.texcoord).r;
	float roughness = texture(roughnessTexture, vin.texcoord).r;
	
	// Outgoing light direction (vector from world-space fragment position to the "eye").
	vec3 Lo = normalize(eyePosition - vin.position);
	
	// Get current fragment's normal and transform to world space.
	vec3 N = normalize(2.0 * texture(normalTexture, vin.texcoord).rgb - 1.0); // normal map中存的是[0, 1]这个公式将其转换为[-1, 1]
	N = normalize(vin.tangentBasis * N); // 变换到世界空间
	
	// Angle between surface normal and outgoing light direction.
	float cosLo = max(0.0, dot(N, Lo));
	
	// Fresnel reflectance at normal incidence (for metals use albedo color).
	vec3 F0 = mix(Fdielectric, albedo, metalness);
	
	// Specular reflection vector.
	vec3 Lr = 2.0 * cosLo * N - Lo;

	// Direct lighting calculation for analytical lights.
	vec3 directLighting = vec3(0);
	for each light:
	{
		//
		// Various variables appeared in shader
		// 
		vec3 Li = -lights[i].direction;
		vec3 Lradiance = lights[i].radiance;

		// Half-vector between Li and Lo.
		vec3 Lh = normalize(Li + Lo);

		// Calculate angles between surface normal and various light vectors.
		float cosLi = max(0.0, dot(N, Li));
		float cosLh = max(0.0, dot(N, Lh));
		// Specular reflection vector.
		vec3 Lr = 2.0 * cosLo * N - Lo;


		//
		// PBR specific temporary variable
		//

		// Microfacet Specular BRDF: 受albedo, metal, roughness, normal N, Li, Lo, Lh影响及常量Fdielectric影响
		{
			// Specular F(Fresnel)
			vec3 F = fresnelSchlick(F0, max(0.0, dot(Lh, Lo)));  // 当Lh与Lo夹角大于90度时，此值等于vec3(1, 1, 1)

			// Specular D(Distribution): normal distribution 
			float D = ndfGGX(cosLh, roughness);

			// Specular Geometry attenuation: Calculate geometric attenuation for specular BRDF.
			float G = gaSchlickGGX(cosLi, cosLo, roughness);

			// Cook-Torrance specular microfacet BRDF.
			vec3 specularBRDF = (F * D * G) / max(Epsilon, 4.0 * cosLi * cosLo);
		}


		// Diffuse BRDF: use Lambert Diffuse BRDF f(l, v) = c_diff/PI, c_diff is the diffuse albedo
		// 受
		{
			// Diffuse scattering happens due to light being refracted multiple times by a dielectric medium.
			// Metals on the other hand either reflect or absorb energy, so diffuse contribution is always zero.
			// To be energy conserving we must scale diffuse BRDF contribution based on Fresnel factor & metalness.
			vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metalness); // kd与F可以理解为是总和小于等于1的权重作用于Diffuse盒Specular这两个分量。

			// Lambert diffuse BRDF.
			// We don't scale by 1/PI for lighting & material units to be more convenient.
			// See: https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
			vec3 diffuseBRDF = kd * albedo;
		}

		// Total contribution for this light.
		directLighting += (diffuseBRDF + specularBRDF) * Lradiance * cosLi; // integrate( Li(l) f(l, v) cosTheta dl)
	};
		
	// Ambient lighting (IBL). 可以理解为整个环境映射Cube都是光源, 那计算量比上述的Diffuse + Specular要大得多。需要一种加速技术（LUT），这也是为什么需要下面的**setup**部分
	vec3 ambientLighting;
	{
		// Ambient DiffuseIBL 
		{
			// Sample diffuse irradiance at normal direction. 应该记录的是从cube map来的radiance(Split Sum中前面那个求和)。
			vec3 irradiance = texture(irradianceTexture, N).rgb;
	
			// Calculate Fresnel term for ambient lighting.
			// Since we use pre-filtered cubemap(s) and irradiance is coming from many directions
			// use cosLo instead of angle with light's half-vector (cosLh above).
			// See: https://seblagarde.wordpress.com/2011/08/17/hello-world/
			vec3 F = fresnelSchlick(F0, cosLo);


			// Get diffuse contribution factor (as with direct lighting).
			vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metalness);

			// Irradiance map contains exitant radiance assuming Lambertian BRDF, no need to scale by 1/PI here either.
			vec3 diffuseIBL = kd * albedo * irradiance;
		}

		// Ambient SpecularIBL： 
		{
			// Sample pre-filtered specular reflection environment at correct mipmap level.
			int specularTextureLevels = textureQueryLevels(specularTexture);
			vec3 specularIrradiance = textureLod(specularTexture, Lr, roughness * specularTextureLevels).rgb;

			// Split-sum approximation factors for Cook-Torrance specular BRDF.
			vec2 specularBRDF = texture(specularBRDF_LUT, vec2(cosLo, roughness)).rg; // 事先计算了不同的roughness下的LUT。

			// Total specular IBL contribution.
			vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;

			// Total ambient lighting contribution.
			ambientLighting = diffuseIBL + specularIBL;
		}
	}
	// Final fragment color.
	color = vec4(directLighting + ambientLighting, 1.0);
}



**setup**阶段按顺序有几个compute shader, 
	1.equirect2cube_cs.glsl
	
	  environment.hdr是一个球面的equirect map表示，是一个2D texture。 代码中在Setup阶段使用compute shader将其转化为cube map。 i.e. shaders/glsl/equirect2cube_cs.glsl。
	  在计算机图形学中，equirectangular map（通常简称为equirect map，又称圆柱投影地图）是一种将三维球面映射到二维平面的方法。这种映射方式保持了两极之间的角度关系，使得经线和纬线在二维平面上呈现为等距的直线。
	  具体来说，equirectangular map将球面按照经度和纬度划分成小格子，然后将这些格子展开成一个矩形。在这种映射中，经线（纵向线条）和纬线（横向线条）是均匀分布的，因此它非常适合表示地球表面的地图。然而，这种映射方式在两极附近会出现较大的畸变，因为靠近极点的区域在展开时会被拉得非常长。
	  在计算机图形学和虚拟现实领域，equirectangular map常用于全景图像的表示。全景图像通常捕捉了360度水平视角和180度垂直视角的图像，可以用来创建环绕观者的虚拟环境。由于equirectangular map保持了角度关系，因此它可以用来创建无缝的全景视图，尽管在两极附近会有畸变，但整体上它是一种简单且易于实现的映射方法。

	// 这三个貌似PPT中只是简单提了一嘴。
	2.spmap_cs.glsl: 没看明白。这个的输出texture会被irmap_cs.glsl使用
	3.irmap_cs.glsl： 计算diffuse irradiance cubemap convolution for image-based lighting. Uses quasi Monte Carlo sampling with Hammersley sequence.
	4.spbrdf_cs.glsl： Pre-integrates Cook-Torrance specular BRDF for varying roughness and viewing directions.

```




