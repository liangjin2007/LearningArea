# About how to encoding Motion Matching Into Networks and how to apply into UE.

## Reference
- dataset
  - ubisoft-laforge dataset https://github.com/ubisoft/ubisoft-laforge-animation-dataset
  - HumanML3D dataset https://github.com/EricGuo5513/HumanML3D
  - Kit dataset https://motion-annotation.humanoids.kit.edu/dataset/
  - How to generate wanted dataset https://github.com/zju3dv/EasyMocap
  - https://github.com/IDEA-Research/Motion-X
- https://github.com/pau1o-hs/Learned-Motion-Matching/tree/master
- https://github.com/orangeduck/Motion-Matching/blob/main/resources/train_decompressor.py
- Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices
- Robust Motion Inbetween (LSTM based) Robust motion in-betweening
- SIG 2024 https://github.com/setarehc/diffusion-motion-inbetweening
- simulator
  - c++ human simulator https://github.com/google-deepmind/mujoco?tab=readme-ov-file
  - https://github.com/HoangGiang93/URoboViz?tab=readme-ov-file
  - MuJoCo-Unreal-Engine-Plugin https://lab.uwa4d.com/lab/67e6900f333affa84f3e3bac
  - XCCQuinn
  - mujoco_mpc real-time behaviour synthesis with MuJoCo, using Predictive Control https://github.com/google-deepmind/mujoco_mpc
  - A collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind. https://github.com/google-deepmind/mujoco_menagerie
  - Imitation learning benchmark focusing on complex locomotion tasks using MuJoCo  https://github.com/robfiras/loco-mujoco
## 目标
- 避免滑步：Runtime节点中只要脚是着地的必须不移动。整个运动的移动应该交给RootMotion来控制。
  - 其他参考链接： https://dev.epicgames.com/documentation/en-us/unreal-engine/fix-foot-sliding-with-ik-retargeter-in-unreal-engine
- 快速牵引角色的各个部位到目标
  - Motion Matching模型预测出Pose。
- 约束：
  - 按照角色的关节物理约束运动
  - 自己维护物理模拟状态求解物理资产中的胶囊体的碰撞约束

## 记录
- gym
```
import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

- humenv tutorial
```
https://github.com/facebookresearch/humenv/blob/main/tutorial.ipynb
```

- metamotivo
```
https://github.com/facebookresearch/metamotivo/tree/main?tab=readme-ov-file

conda create -n metamotivo python=3.10
conda init powershell
restart powershell

conda activate metamotivo

pip install huggingface
pip install huggingface-hub

cd xxx/metamotivo
pip install -e .

cd xxx/humenv
pip install -e .


pip install mediapy

pip install torchsummary


python

from metamotivo.fb_cpr.huggingface import FBcprModel
model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")

cd xxx/metamotivo/metamotivo
create a directory called data

create the following code as a download_buffers.py
    from huggingface_hub import hf_hub_download
    from metamotivo.buffers.buffers import DictBuffer
    import h5py
    
    local_dir = "metamotivo-S-1-datasets"
    dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference
    # dataset = "buffer.hdf5"  # the full training buffer of the model
    buffer_path = hf_hub_download(
            repo_id="facebook/metamotivo-S-1",
            filename=f"data/{dataset}",
            repo_type="model",
            local_dir=local_dir,
        )
    hf = h5py.File(buffer_path, "r")
    print(hf.keys())
    
    # create a DictBuffer object that can be used for sampling
    data = {k: v[:] for k, v in hf.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)

python download_buffers.py download the hdf5 buffer.

```

- 导出pytorch模型为onnx 
```
https://zhuanlan.zhihu.com/p/498425043
https://docs.pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
```  




- 可视化模型
```
pip install torchsummary
summary(model._backward_map, input_size=input_goal.shape)

# 输出以下内容
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1               [-1, 1, 256]          91,904
         LayerNorm-2               [-1, 1, 256]             512
              Tanh-3               [-1, 1, 256]               0
            Linear-4               [-1, 1, 256]          65,792
              Norm-5               [-1, 1, 256]               0
================================================================
Total params: 158,208
Trainable params: 0
Non-trainable params: 158,208
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.60
Estimated Total Size (MB): 0.61
```


```
Windows：下载 Graphviz 并添加路径到环境变量。 https://graphviz.org/download/
pip install torchviz

import torch 
from torchviz import make_dot 
from torch.autograd  import Variable 
 
# 使用上面定义的 SimpleCNN 模型 
model = SimpleCNN() 
 
# 创建输入张量（需带梯度，否则无法追踪计算图） 
x = Variable(torch.randn(1,  3, 32, 32))  # batch_size=1, channels=3, height=32, width=32 
y = model(x)  # 前向传播 
 
# 生成计算图（保存为 PDF 或 PNG） 
dot = make_dot(y, params=dict(model.named_parameters()))  
dot.render("cnn_model",  format="png")  # 保存为 cnn_model.png  
dot.view()   # 直接显示 



```


- awass
https://amass.is.tue.mpg.de/download.php

- Metahuman
```
https://dev.epicgames.com/community/learning/tutorials/GDMx/metahuman-animation-tutorial-series
https://www.bilibili.com/video/BV1E4MNzUEat?spm_id_from=333.788.player.switch&vd_source=38a71595005700b6ff304d0a48055f82
https://dev.epicgames.com/community/learning/tutorials/B3Yy/metahuman-living-notebook-ue5-6?locale=zh-cn
https://dev.epicgames.com/documentation/zh-cn/metahuman/metahuman-5-6-release-notes
```
