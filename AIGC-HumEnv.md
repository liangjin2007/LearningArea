# Humenv

- 问题：如何处理amass数据集 ？

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


