立绘 LoRA https://civitai.com/models/13090

AI作图软件 MidJourney https://discord.com/invite/midjourney

Diffusion https://github.com/huggingface/diffusers

https://www.speech.kth.se/research/listen-denoise-action/



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





