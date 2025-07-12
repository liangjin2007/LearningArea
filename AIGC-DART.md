# DART
- [1.编译](#1编译)
- [2.集成进UE](#2集成进UE)
  - [2.1.任务拆解](#21任务拆解)
## 1.编译
```
如果发现conda装东西特别慢， 删除C:/Users/xxx/.condarc, 删除C:/Users/xxx/.conda, 卸载Anaconda3， 重装Anaconda3

打开Anaconda Powershell

conda create -n dart python=3.8.8
conda activate dart

可选
  conda config --add channels https://pypi.tuna.tsinghua.edu.cn/simple
  conda config --set show_channel_urls yes

安装cuda
使用CUDA 11.8， 如果没安装，先安装CUDA，可以看顶部如何安装CUDA 11.8

安装pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


download data, model, etc.
unzip 3 of them and put into DART-main

cd DART-main
run the following command to see what packages are not installed:
    python -m mld.rollout_demo --denoiser_checkpoint "D:/T2M_Runtime/DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt" --batch_size 1 --guidance_param 5 --use_predicted_joints 1 --device cuda
    Note: use absolute path for the checkpoint


    // 安装git+xxx需要执行下面这行。
    git config --global http.sslverify false
    pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install git+https://github.com/openai/CLIP.git
新建requirements.txt
  tyro
  pyyaml==6.0.1
  tensorboard
  tornado
  tqdm
  pyrender==0.1.45
  loralib
  smplx
  omegaconf==2.3.0
  torch_dct

pip install -r requirements.txt
pip install spacy==2.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib==3.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

    about pytorch3d, it's hard to install if following the steps below:
      download pytorch3d-0.7.8  from https://github.com/facebookresearch/pytorch3d/tree/V0.7.8
      unzip
      follow https://github.com/facebookresearch/pytorch3d/blob/V0.7.8/INSTALL.md to install pytorch3d
        download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice.
        Define the environment variable CUB_HOME before building and point it to the directory that contains CMakeLists.txt for CUB. Dont Need to compile cub before building pytorch3d.
      Setup CUDA_HOME and CUDA_PATH to the path of CUDA 11.8
      conda install -c iopath iopath
      添加C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64到环境变量PATH
      cd pytorch3d-0.7.8
      pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

See https://github.com/facebookresearch/pytorch3d/discussions/1752
  pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu118
  or
    download pytorch3d-0.7.8+pt2.4.1cu118-cp38-cp38-win_amd64.whl from https://miropsota.github.io/torch_packages_builder
    pip install .\pytorch3d-0.7.8+pt2.4.1cu118-cp38-cp38-win_amd64.whl

Succeeded, but the python code has errors related to smplx, numpy version.

about smplx:
    download SMPL-X from https://smpl-x.is.tue.mpg.de/download.php, my registered account is jl5400@163.com.
    unzip it
    put it to data directory.
    modify ./config_files/data_paths.py, change to body_model_dir = dataset_root_dir / 'smplx-models', where 'smplx-models' contain a sub-dir called smplx.


修改\DART-main\data_loaders\humanml\common\quaternion.py line 13 from np.float to float

修改rollout_demo.py line 321
     import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)
   
    pathlib.PosixPath = temp

Succeeded to resolve all the problems.

Now: its downloading ViT-B-32.pt which is used by clip package.
  D:\Anaconda3\envs\dart\lib\site-packages\clip\clip.py:57: UserWarning: C:\Users\liangjin/.cache/clip\ViT-B-32.pt exists, but the SHA256 checksum does not match; re-downloading the file

After downling clip's ViT model.
Opened the viewer.
```

## 2.集成进UE
### 2.1.任务拆解
```
1. 将DART模型用到的几个模型导出onnx-ort模型。
2. 将onnx-ort模型集成到UE的一个运行时节点中。
3. 预测输出，生成单帧的bvh数据，用bvh数据通过retarget去驱动我们的角色
4. 学习相关paper
```
### 具体实施
1.将DART模型用到的几个模型导出onnx-ort模型
- 1.1. VSCode配置Python调试环境，并使得Powershell terminal能识别conda环境
```
1.1.1. 选择Python时能看到列表中有dart环境的Python.exe，选择它。
1.1.2. Run And Debugger中可能需要创建一个launch.json如下。
  {
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "python",
            "request": "launch",
            "program": "",
            "console": "integratedTerminal",
            "args": [
                "-m", "mld.rollout_demo",
                "--denoiser_checkpoint", "D:/T2M_Runtime/DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt",
                "--batch_size", "1",
                "--guidance_param", "5",
                "--use_predicted_joints", "1",
                "--device", "cuda"
            ]
        }
    ]
  }
1.1.3. 在VS code底下的terminal中执行conda init powershell
```

- 1.2. 调试得到有几个模型，输入的维度等信息
```

```  
- 1.3. 添加代码来写出onnx-ort模型 https://onnxruntime.ai/docs/tutorials/mobile/helpers/
```
安装 https://onnxruntime.ai/docs/install/#install-on-web-and-mobile
Install ONNX Runtime CPU
  pip install onnxruntime # first try

pytorch api https://docs.pytorch.org/docs/stable/index.html

```




- 1.4. 学习相关paper
```
[2022]两万+引 High-Resolution Image Synthesis with Latent Diffusion Models https://arxiv.org/pdf/2112.10752
有关工作：
  GAN： 质量高，但是难于训练
  基于似然的方法：
    VAE, Flow based model： 高效，但是质量没GAN高，
    自回归模型autoregressive model（ARM）： 图像分辨率不高，计算量太大，顺序采样过程。
```  
