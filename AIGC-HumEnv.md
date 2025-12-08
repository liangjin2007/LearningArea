# Humenv

- 问题：如何处理amass数据集 ？
## humenv
- https://github.com/facebookresearch/humenv/tree/main/data_preparation
- 尝试转化amass数据集以使得能生成部分调试数据
```
创建humenv环境
conda create -n humenv python=3.10

安装SMPLSim
下载https://github.com/ZhengyiLuo/SMPLSim
cd SMPLSim-master
conda activate humenv
pip install -r .\requirements.txt

发现https://github.com/ZhengyiLuo/smplx装不上。
先下载下来
cd smplx-master
pip install .

继续安装SMPLSim
cd ../SMPLSim-master
在requirements.txt中注释掉smplx那一行（前面加#）。
pip install .

尝试安装PHC
从https://github.com/ZhengyiLuo/PHC下载PHC
放到SMPLSim-master同级目录
将https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/process_amass_db.py文件拷贝到humenv-master/data_preparation里

配置vscode
用vscode打开humenv-master代码
Ctrl+Shift+P，点击Python Interpretor, 选择humenv那个python。
打开data_preparation/process_amass.py
安装h5py
安装rich
pip install rich h5py


```

## SMPL
- build qpos https://github.com/ZhengyiLuo/SMPLSim/blob/master/examples/motion_test.py


## Genesis
- 中文文档 https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/overview/what_is_genesis.html

## PHC and SMPLSim
- Perpetual Humanoid Control for Real-time Simulated Avatars https://github.com/ZhengyiLuo/PHC
- SMPLSim: Simulating SMPL/SMPLX Humanoids in MUJOCO and Isaac Gym https://github.com/ZhengyiLuo/SMPLSim
## process_dataset.sh
```
配置环境:
  先下载humenv的源码
  cd humenv
  conda create -n humenv python=3.10
  conda activate humenv
  pip install .
  用vscode打开humenv源码
  ctrl+shift+p选择python interpretor为humenv对应的python
  新建一个launch.json，调试当前active python file

下载amass到data_preparation

看process_dataset.sh中的内容，对比了一下源码，有些步骤可以跳过：

cd data_preparation
git clone https://github.com/ZhengyiLuo/PHC.git
git clone https://github.com/ZhengyiLuo/SMPLSim.git

以下这部分打补丁的不用处理
   cd PHC
    git checkout 34fa3a1c42c519895bc33ae47a10a1ef61a39520
    git apply ../phc_patch.patch 
    bash download_data.sh
    cd ..

    cd SMPLSim
    git checkout 3bcc506d92bf15329b2d68efcf429725b67f3a06
    git apply ../smplsim_patch.patch 
    cd ..

尝试安装python第三方依赖库 cd PHC, pip install ., cd SMPLSim, pip install .。


```


