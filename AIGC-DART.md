# DART
- [1.编译和执行mld.rollout_demo](#1编译和执行mld.rollout_demo)
- [2.执行Inbetween_demo](#2执行Inbetween_demo)
- [3.集成进UE](#3集成进UE)
  - [3.1.导入smpl-x模型到UE](#21导入smpl-x模型到UE)
  - [3.2.将DART模型用到的几个模型导出onnx-ort模型](#22将DART模型用到的几个模型导出onnx-ort模型)
    


## 1.编译和执行mld.rollout_demo
原始链接 https://github.com/zkf1997/DART

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

- VSCode launch.json
```
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
```

## 2.执行Inbetween_demo
Python命令：
```
python -m mld.optim_mld --denoiser_checkpoint "D:/T2M_Runtime/DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt" --optim_input "D:/T2M_Runtime/DART-main/data/inbetween/pace_in_circles/babel_2f.pkl" --text_prompt "pace in circles" --optim_lr 0.05 --optim_steps 100 --batch_size 4 --guidance_param 5 --respacing "ddim10" --export_smpl 0  --use_predicted_joints 1  --optim_unit_grad 1  --optim_anneal_lr 1  --weight_jerk 0.0  --weight_floor 0.0 --seed_type 'history'  --init_noise_scale 0.1 --device cuda
```
VSCode launch.json
```
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
                "-m", "mld.optim_mld",
                "--denoiser_checkpoint", "D:/T2M_Runtime/DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt",
                "--optim_input","D:/T2M_Runtime/DART-main/data/inbetween/pace_in_circles/babel_2f.pkl",
                "--text_prompt","pace in circles",
                "--optim_lr","0.05",
                "--optim_steps","100",
                "--batch_size", "4",
                "--guidance_param", "5",
                "--respacing","ddim10",
                "--export_smpl","0",
                "--use_predicted_joints", "1",
                "--optim_unit_grad","1",
                "--optim_anneal_lr","1",
                "--weight_jerk","0.0",
                "--weight_floor","0.0",
                "--seed_type","history",
                "--init_noise_scale","0.1",
                "--device", "cuda"
            ]
        }
    ]
  }
```


## 3.集成进UE
```
1. 导入smpl-x模型到UE
2. 将DART模型用到的几个模型导出onnx-ort模型。
3. 将onnx-ort模型集成到UE的一个运行时节点中。
4. 预测输出，生成单帧的bvh数据，用bvh数据通过retarget去驱动我们的角色
```

### 3.1.导入smpl-x模型到UE
```
smpl-x官方网站：
  https://smpl-x.is.tue.mpg.de/download.php
  https://agora.is.tuebingen.mpg.de/login.php

1. 从smpl-x官网下载smpl-x blender Add on带数据那个zip， 大概600多M，不要下载Code那个链接里的。

2. 参考Code中的README.md https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon/-/blob/master/README.md?ref_type=heads的Installation部分。

	Register at https://smpl-x.is.tue.mpg.de and download the SMPL-X for Blender add-on. The ZIP release file will include the required SMPL-X model which is not included in the code repository.
	Blender>Edit>Preferences>Add-ons>Install
	Select downloaded SMPL-X for Blender add-on ZIP file (smplx_blender_addon-YYYYMMDD.zip) and install
	Enable SMPL-X for Blender add-on
	Enable sidebar in 3D Viewport>View>Sidebar
	SMPL-X tool will show up in sidebar

3.选择version, 选择性别，点Add

4.点Export Fbx   (选择Unreal)

5.导入到UE

6.导入的骨骼网格体有200多M。 需要考虑把不必要的数据删除，这一步可能能在UE编辑器中操作。
可在MorphTarget中删除Shape，Exp, Pose等。

```







### 3.2.将DART模型用到的几个模型导出onnx-ort模型

#### 3.2.1.模型可视化
DART有三个模型: vae_model, diffusion, 和denoiser_model。
可使用torchvis包做可视化。
- 可视化vae_model:
```
DART中在预测的代码vae_model.decode(...)附近添加代码：
vae_model的输入的维度为[1, B, D]
```

### 3.2.2.导出onnx模型
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

通过https://onnxruntime.ai/docs/tutorials/mobile/helpers/#pytorch-export-helpers在适当位置添加导出代码







```
- pytorch api https://docs.pytorch.org/docs/stable/index.html



### 3.2.3.模型优化 
- https://github.com/ShusenTang/Dive-into-DL-PyTorch
- https://github.com/TingsongYu/PyTorch_Tutorial
- https://github.com/microsoft/MMdnn

```  
graph LR 
A[原始模型] --> B(知识蒸馏) 
B --> C{评估精度}
C -->|达标| D[结构化剪枝]
C -->|未达标| B 
D --> E[量化训练]
E --> F[部署至移动端]
 四、附加资源
视频教程：B站搜索《PyTorch最全实用教程》第5章（模型压缩实战）4
论文精读：
《Distilling the Knowledge in a Neural Network》 (Hinton, 2015)
《Learning Efficient Convolutional Networks through Network Slimming》 (ICCV 2017)
模型分析工具：
Netron：可视化压缩后模型结构（支持ONNX）
TensorBoard：监控蒸馏/剪枝过程中的损失变化 3
提示：优先运行GitHub项目的示例代码（如Optimum的蒸馏Pipeline），再结合PDF教程理解理论细节。工业部署建议导出ONNX后使用MNN/TNN等移动端引擎加速

```
