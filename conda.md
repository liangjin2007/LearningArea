```
conda update -n base -c defaults conda
conda env list
conda create -n xxx python=xxx
conda create --file xxx.yml
conda remove -n ct2hair --all
powershell: conda init powershell

// setup packages directory to another path.
// setup env to another path
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting

// export env

```

- 如何清理一个env
```
要将一个Anaconda环境恢复到初始状态，你可以创建一个新的环境并将旧环境的依赖导出，然后删除旧环境并重命名新环境。但是，Anaconda不提供直接的方式来“重置”一个环境到初始状态。

以下是一个简单的步骤来实现这个目标：

创建一个新的Anaconda环境。

导出旧环境的依赖。

删除旧环境。

重命名新环境到旧环境的名字。

# 创建新环境
conda create --name new_env
 
# 激活旧环境
conda activate old_env
 
# 导出环境依赖到文件
conda env export > environment.yml
 
# 退出旧环境
conda deactivate
 
# 删除旧环境
conda remove --name old_env --all
 
# 基于environment.yml文件创建新环境并重命名为旧环境名
conda env create --name old_env --file environment.yml
 
# 删除导出的环境文件
rm environment.yml
请注意，这个过程不会删除旧环境中的任何数据或配置文件，只是将其重置为当前的初始状态，即一个全新的环境，没有任何额外的包或者配置。如果你需要一个全新的、没有任何包的环境，你可以直接创建一个新的环境，不需要先创建一个旧环境，然后再删除它。


# 注意base 环境是无法通过conda remove删除的。
```

- 添加channels
```
conda如何设置channels
在conda中设置channels，你需要编辑.condarc配置文件或者使用命令行。以下是如何通过命令行设置channels的步骤：

查看当前配置的channels：

conda config --show channels

添加一个新的channel：

conda config --add channels new_channel_name

移除一个已有的channel：

conda config --remove channels channel_name_to_remove

设置channel的优先级（可选）：

conda config --set channel_priority true

如果需要，可以通过以下命令清除所有channels并从头开始设置：

conda config --remove-key channels

也可以直接编辑.condarc文件（通常位于用户的主目录下），添加或修改channels列表。例如：

channels:
  - defaults
  - conda-forge

以上步骤可以帮助你管理conda的channels。记得在添加或移除channels后尝试更新环境以确保变更生效。

目前可设置的几个channels
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
conda config --show channels
```  



