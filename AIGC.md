- 博客 https://antkillerfarm.github.io/#GAN%20&%20VAE
- RBM DBM DBN : Restricted Boltzmann Machines etc from https://antkillerfarm.github.io/dl/2018/01/04/Deep_Learning_27.html


# topic

立绘 LoRA https://civitai.com/models/13090

AI作图软件 MidJourney https://discord.com/invite/midjourney

Diffusion https://github.com/huggingface/diffusers

https://www.speech.kth.se/research/listen-denoise-action/

https://keras.io/examples/

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

https://github.com/ultralytics/ultralytics

https://github.com/XingangPan/DragGAN

https://github.com/Shimingyi/MotioNet

https://github.com/anandpawara/Real_Time_Image_Animation

https://github.com/liangjin2007/EverybodyDanceNow

https://github.com/yzhq97/transmomo.pytorch

# yolov8 
- https://github.com/liangjin2007/ultralytics
- 需要独立再创建一个environment

# stable-diffusion-webui 107k Star https://github.com/AUTOMATIC1111/stable-diffusion-webui










# 安装tensorflow/pytorch/etc

### 安装 tensorflow-gpu 2.4.1
- 安装：别照着官网来
- 照着这个video来安装 youtube.com/watch?v=toJe8ZbFhEc
  - 使用Anaconda prompt打开命令行
  - 使用conda create -n tf_gpu python=3.8来创建一个新的环境。
  - conda activate tf_gpu
  - 安装cuda和cudnn ： conda install cudatoolkit=11.0 cudnn=8.0 -c=conda-forge
  - 安装tensorflow-gpu 2.4.1 :   pip install --upgrade tensorflow-gpu==2.4.1，
    -  对国内来说，需要添加一个国内镜像 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -upgrade tensorflow-gpu==2.4.1

- 检查使用支持GPU
  - python
  - import tensorflow as tf
  - tf.__version__
  - tf.test.is_gpu_avaliable() # deprecated 这个会返回True
  - tf.config.list_physical_devices('GPU') # 这个会返回GPU信息



# 安装pytorch 
- 既然已经安装了tensorflow gpu版，那么pytorch最好也安装在一起
- 比较简单 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110 # 使用相同的cuda版本
- 检查GPU支持
  - python
  - import torch
  - torch.cuda.is_available() # 返回True表示cuda支持



# 如何用UE做一个自己的虚拟Avatar







# 3D Reconstruction
- LLFF [2019] https://github.com/fyusion/llff
- nerf [2020] https://www.matthewtancik.com/nerf
- nerf ppt https://www.slideshare.net/taeseonryu/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis-251398636
  - MLP networks 表示5D neural radiance field $(x, y, z, \theta, \phi) \to (r, g, b, \sigma)$
  - related to Volume based Differential Rendering, hierarchical sampling
  - A positional encoding, map 5D coordinates to higher dimensional space
- [2022]instant-ngp 能輸出density field, 但精度比较低
- [2023]PlenVDB: VDB + Nerf，更快的训练 + 更快的渲染 https://plenvdb.github.io/
- [2023]Wonder3D https://arxiv.org/pdf/2310.15008.pdf
- [2023]Gaussian Splatting https://github.com/graphdeco-inria/gaussian-splatting
- [2023]Sparse3D: Distilling Multiview-Consistent Diffusion for Object Reconstruction from Sparse Views https://arxiv.org/pdf/2308.14078.pdf
- 
- Input video to instant-ngp https://neuralradiancefields.io/how-to-create-a-nerf-using-a-input-video-through-ffmpeg/
- https://neuralradiancefields.io/velox-xr-announces-instant-ngp-to-unreal-engine-5/
- https://neuralradiancefields.io/nerfs-in-unreal-engine-5-alpha-announced-by-luma-ai/ 注意 luma-ai的官网暂时访问不了
- UE Sparse Volume Textures https://zhuanlan.zhihu.com/p/643961497   

# Learn high-frequency signal
- Coordinate-MLPs
  - encode continuous signals $ f : R^n \to R^m$  as their weights
    - 比如： 输入是低维坐标(x, y) positions, 输出是采样的信号值at each coordinates e.g. pixel intensities.
  - 跟普通MLPs的区别是它可以encode 高频信号 ——减轻spectral bias of the MLPs
  - 有三种coordinate-MLPs
    - Random Fourier Feature MLPs
      - positional embedding layer 


# [2020]Differential Rendering 《Differentiable Rendering: A Survey》从图像观察3D场景参数
- 绘制函数 Rendering Function $I = (I_c, I_d) = R(\Phi_s, \Phi_m, \Phi_l, \Phi_c)$
  - 输入$`\Phi = \left\{\Phi_s, \Phi_m, \Phi_l, \Phi_c\right\}`$
  - 输出$`I = \left\{I_c, I_d\right\}`$  分别代表RGB image or depth image
- Try to learn $\Phi_s, \Phi_m, \Phi_l, \Phi_c$ ， 分别代表形状参数，材质参数， 光照参数， 相机参数
- 为了能学习，需要知道$\frac{\partial I}{\partial \Phi}$ 怎么（近似）算，并尽量好。
- **2.1.Mesh**
  - 包括两步 对每个pixel（1） 分配一个最近的三角形给它 （2）根据三角形顶点颜色计算pixel 颜色。
    - 第一步涉及到离散selection，不可微
    - 第二步可微 why??
      - Popular reflection models such as Phong [47], Lambertian [48] and Spherical Harmonics [49] are all differentiable. 
  - 近似梯度 approximated gradients $\frac{\partial I}{\partial \Phi}$， backward pass时需要用。
    - [2014]OpenDR
      - $\frac{\partial I_c}{\partial \Phi_s}$ is approximated by differential filters e.g. Sobel filter.
    - [2018]Neural 3D mesh renderer(NMR)
      - non-local approximated gradients with 也利用了 $\frac{\partial loss}{\partial \Phi_s}$
    - [2018]rasterization derivatives using the barycentric coordinates of each triangle with respect to each pixel, 负值重心坐标。
  - approximated rendering： 这是另一个思路:对物体的hard boundary近似成光滑过度，也就是近似forward pass(rasterization)。
    - [2014]一种方法是给三维物体定义density parameter，渲染出来在物体边界上会比较模糊和光滑
    - [2019]Soft Rasterizer
      - 替换z-buffer-based triangle selection of rasterization with probabilistic approach
      - aggregation function ??
  - Global illumination
    - try to resolve the discontinuity in the rendering equation 
      - [2018]Differential Monte Carlo ray tracing by edge sampling.
    - try to estimate the derivatives of the path integral formulation
      - [2020] Path-Space DR
- **2.2.Voxel**
  - Volume, 二值或者非二值，
  - Occupancy probablility $P_O \in [p_min, p_max]$ 表示a ray's absorption(transparency) at a certain point.
  - Material information
  - ray marching
  - Shapes
    - Distance function DF
    - SDF
    - TSDF
  - **绘制过程也分两步**
    - 1. Collecting the voxels that are located along a ray
      - [2019][2020]world space
      - [2019]project the the screen space and perform bilinear sampling similar to [2015]Spatial Transformer Networks
      - [2019]warp field, inverse warp
    - 2. Aggregating voxels along a ray
      - [2016] occupancy probability to each pixel
      - EA, VH, and AO models.
- **2.3.Point Cloud**
- **2.4.Implicit Representations**
  - neural implicit representation F(P)
  - occupancy
  - transparency
  - probability of occupancy
  - [2019]learning to infer implicit surfaces without 3d supervision.
- **2.5.Neural Rendering**
- **4. Application**
  - Object Reconstruction
    - 标注数据不好弄
    - shape network
    - camera network ?
  - Human reconstruction
    - Body Shape and Pose Reconstruction
    - Hand Shape and Pose Reconstruction
    - Face Reconstruction
  - 3D Adversarial Examples
  - Other Application
- **5.Libraries**
  - PyTorch3D
  - Mitsuba 2
  - Kaolin
  - TensorFlow Graphics
- **Find some source code to read**
  - https://github.com/thalesfm/differentiable-renderer 
  
  - scene
  ```
  template<typename T>
  class Shape {
  ...
  virtual bool intersect(Vector<T, 3> orig, Vector<T, 3> dir, double& t) const = 0;
  virtual Vector<T, 3> normal(Vector<T, 3> point) const = 0;
  };
  
  
  template<typename T>
  using Scene = std::vector<Shape<T>*>;
  ```
  
  - camera
  ```
  template<typename T>
  class Camera {
    Camera(width, height, vfov, eye_position = (0, 0, 0), forward = (0, 0, -1), right = (1, 0, 0), up = (0, 1, 0));

    void look_at(eye, at, up = (0, 1, 0));  // Setup m_eye, m_forward, m_right, m_up

    int width() const;
    int height() const;
    Vector<T, 3> eye() const;
    
    double aspect() const{return double(m_width)/m_height; }
  
    // ray tracing related sample ray
    std::tuple<Vector<T, 3>, double> sample(x, y) const{
      // 1. y is from top to down, that's why * -m_up
      // 2. fov is in the field of view angle in the y direction, that's why m_right need multiply aspect() which is width/height
      double s = (x + random::uniform()) / m_width; // [0.0, 1.0]
      double t = (y + random::uniform()) / m_height; // [0.0, 1.0]
      Vector<T, 3> dir = m_forward;
      dir += (2*s - 1) * aspect() * tan(vfov/2) * m_right;
      dir += (2*t - 1) * tan(vfov/2) * -m_up;
      dir = normalize(dir);
      return std::make_tuple(dir, 1);
    }
  ```

# [2023]Wonder3D
- Score Distillation Sampling SDS method
  - 2D diffusion prior: means the diffusion implicit feature representation? instead of CNN representation
  - reconstruction 3d from 2D diffusion prior
  - cross-domain diffusion model to predict normal map and the corresponding color images

# [2023]Sparse3D


# Expanding Wave Function Collapse with Growing Grids for Procedural Content Generation
- PCG
  - L-System
    - was made for describing plant growth 
  - Perlin noise
  - online and offline generation
```
PCG algorithms can be categorized in six different
categories [1]. Pseudo random number generators, generative grammars, image
filters, spatial algorithms, modeling, simulation of complex systems and finally
artificial intelligence where constraint satisfaction algorithms such as WFC are
assigned.
```






# GAN
- https://antkillerfarm.github.io/#GAN%20&%20VAE, 这个地方还有很多有意思的博文
- pix2pix ppt http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf
- cycleGAN https://www.jianshu.com/p/5bf937a0d993
- 比cycleGAN训练更快 https://github.com/taesungp/contrastive-unpaired-translation








# 人像抠图
- MODNet: Trimap-Free Portrait Matting in Real Time
- https://github.com/ZHKKKe/MODNet






# HairModeling
- HairSteps最新代码已经发布了，但是并不是所有代码，比如生成strand部分的代码就没有， 另外只有linux环境+Mac环境， windows环境下试了一下，依赖包装不上去。 得看看pretrained model是否存在
- pix2pix pytorch版实践 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
- HairNet
- Neuralhaircut
```
code is for linux, how to inference on windows??

git clone xxx
cd Neuralhaircut
conda create -n neuralhaircut python=3.9.16
conda activate neuralhaircut

// 
conda install cudatoolkit=11.6 cudnn=8.3.2 -c=conda-forge

安裝pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116 

git submodule update --init --recursive


pip install gdown   # https://github.com/wkentaro/gdown
```





# HairStep
- 0. 由于开源代码是linux/mac上的，所以-f environment.yml通常失败，按照如下步骤来一步步步构建环境
```
首先看一下environment.yml中python的版本是3.6.13
创建基本环境 conda create -n hairstep python=3.6.13
conda activate hairstep

安装pytorch pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


git clone --recursive https://github.com/GAP-LAB-CUHK-SZ/HairStep.git
cd HairStep


接下来需要编译3DDFA_V2，一个3D人脸预测模型.
  原始指令
    cd external/3DDFA_V2
    sh ./build.sh
      cd FaceBoxes
      sh ./build_cpu_nms.sh
      cd ..
      
      cd Sim3DR
      sh ./build_sim3dr.sh
      cd ..
      
      cd utils/asset
      gcc -shared -Wall -O3 render.c -o render.so -fPIC
      cd ../..
    cd ../../
  需要修改为
    ./build.bat
      cd FaceBoxes
      ./build_cpu_nms.bat
      cd ..
      
      cd Sim3DR
      ./build_sim3dr.bat
      cd ..
      
      cd utils/asset
      set header1="%MSVCDir%\include"
      set header2="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\ucrt"
      set header3="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\um"
      set header4="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\winrt"	
      set lib1="%MSVCDir%\lib\x64\LIBCMT.lib"
      set lib2="%MSVCDir%\lib\x64\oldnames.lib"
      set lib3="%MSVCDir%\lib\x64\libvcruntime.lib"
      set lib4="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\ucrt\x64\libucrt.lib"
      set lib5="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x64\kernel32.lib"
      set lib6="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x64\Uuid.lib"
      cl.exe render.c /O2 /LD /MD /I %header1% /I %header2% /I %header3% /I %header4% %lib1% %lib2% %lib3% %lib4% %lib5% %lib6% 
      
      cd ../..
      pause 

windows上build_cpu_nms.bat编译不过， 参考https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173
  1. f you got "cl : Command line error D8021 : invalid numeric argument '/Wno-cpp'" error modify 47 line of build.py to extra_compile_args=['std=c99'],


执行 python -m scripts.img2hairstep
  pip install tqdm

执行 python scripts/get_lmk.py
  pip install pyyaml
  change the code in FaceBoxes/utils/nms/cpu_nms.pyx near line 18 to

      def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
          cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
          cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
          cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
          cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
          cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
      
          cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
          cdef np.ndarray[np.int64_t, ndim=1] order = scores.argsort()[::-1]
      
          cdef int ndets = dets.shape[0]
          cdef np.ndarray[np.int64_t, ndim=1] suppressed = \
                  np.zeros((ndets), dtype=np.int64)
  re-execute build_cpu_nms.bat


执行 python -m scripts.opt_cam
执行 python -m scripts.recon3D
   pip install rtree
   pip install pyglet==1.2.4
   add mesh.show() to visualize trimesh object.
   发现卡死了 卡在export_hair_real上了。
    修改lib/hair_util.py中的save_strands_with_mesh把所有10000根头发先存出来会快很多。
    改成这样
      def save_strands_with_mesh(strands, mesh_path, outputpath, err=0.3, is_eval=False):
          mesh = trimesh.load(mesh_path, process=False)
          #for coarse mesh /1000.0
          if is_eval:
              mesh.vertices = mesh.vertices/1000.0
          print("save_strands_with_mesh start...")
      
          lst_pc_all_valid = []
          lst_num_pt = []
          pc_all_valid = []
          lines = []
          sline = 0
      
          strands_new = strands.reshape(-1, 3)
          print(strands_new.shape)
      
          lines = [[0, 0]] * strands.shape[0]*(strands.shape[1] - 1)
          
          seg_index = 0
          for i in range(strands.shape[0]):
              for j in range(strands.shape[1]-1):
                  lines[seg_index] = [sline + j, sline + j + 1]
                  seg_index += 1
              sline += strands.shape[1]
          print("\tfinish collect lines and pc_all_valid...")
      
          line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(strands_new), lines=o3d.utility.Vector2iVector(lines))
          o3d.io.write_line_set(outputpath, line_set)
      
          print("\tfinish write_line_set...")
      
          #o3d.visualization.draw_geometries([line_set])

```




- 1. Change several xxx.sh to xxx.bat

- 2. linux code to windows bat
- cl.exe to compile c/c++ files instead of gcc xxx
```
rem cl.exe compile options -- https://max.book118.com/html/2017/0610/113214867.shtm
rem example of cl.exe compiling a main.cpp file -- https://blog.csdn.net/caozhenyu/article/details/130666879

set header1="%MSVCDir%\include"
set header2="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\ucrt"
set header3="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\um"
set header4="C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\winrt"	

set lib1="%MSVCDir%\lib\x64\LIBCMT.lib"
set lib2="%MSVCDir%\lib\x64\oldnames.lib"
set lib3="%MSVCDir%\lib\x64\libvcruntime.lib"
set lib4="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\ucrt\x64\libucrt.lib"
set lib5="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x64\kernel32.lib"
set lib6="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x64\Uuid.lib"
rem if build exe program call this line -- cl.exe render.c /O2 /I %header1% /I %header2% /I %header3% /I %header4% %lib1% %lib2% %lib3% %lib4% %lib5% %lib6% 
cl.exe render.c /O2 /LD /MD /I %header1% /I %header2% /I %header3% /I %header4% %lib1% %lib2% %lib3% %lib4% %lib5% %lib6% 

cd ../..
```
  - Press Ctrl-Q in notepad++ to add/remove rem for each line








