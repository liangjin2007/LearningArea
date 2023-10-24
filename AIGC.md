![image](https://github.com/liangjin2007/LearningArea/assets/4059558/278e2fa2-b269-499d-b376-4d7b7078c659)# 博客 https://antkillerfarm.github.io/#GAN%20&%20VAE
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
- nerf [2020] https://www.matthewtancik.com/nerf
- nerf ppt https://www.slideshare.net/taeseonryu/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis-251398636
  - related to Differential Rendering  
- [2022]instant-ngp
- [2023]https://github.com/graphdeco-inria/gaussian-splatting
- Input video to instant-ngp https://neuralradiancefields.io/how-to-create-a-nerf-using-a-input-video-through-ffmpeg/
- https://neuralradiancefields.io/velox-xr-announces-instant-ngp-to-unreal-engine-5/
- https://neuralradiancefields.io/nerfs-in-unreal-engine-5-alpha-announced-by-luma-ai/ 注意 luma-ai的官网暂时访问不了
- UE Sparse Volume Textures https://zhuanlan.zhihu.com/p/643961497   

# Differential Rendering 《Differentiable Rendering: A Survey》
- 绘制函数 Rendering Function $I = (I_c, I_d) = R(\Phi_s, \Phi_m, \Phi_l, \Phi_c)$
  - 输入$`\Phi = \left\{\Phi_s, \Phi_m, \Phi_l, \Phi_c\right\}`$
  - 输出$I = \left\{I_c, I_d\right\}$ 
- Try to learn $\Phi_s, \Phi_m, \Phi_l, \Phi_c$ ， 分别代表形状参数，材质参数， 光照参数， 相机参数
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








