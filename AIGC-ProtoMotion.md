- Step 1. Clone ProtoMotion code
```
Fork ProtoMotions
git config --global user.name xxx
git config --global user.emal xxx
sshkey-rsa xxx
copy pub key to git ssh-key
git clone https://github.com/liangjin2007/ProtoMotions.git
```

- Step 2. git lfs big files
```
Following Installing Documentation https://protomotions.github.io/getting_started/installation.html

su feelingai
sudo apt install git-lfs
su liangjin
git lfs fetch --all
```

- Step 3. Clone newton
```
git clone https://github.com/newton-physics/newton
```

- Step 4. Install uv
```
Ref https://docs.astral.sh/uv/#highlights
安装 curl -LsSf https://astral.sh/uv/install.sh | sh

```

- Step 5. Install newton based protomotion
```
Newton
Newton (currently in beta) is a GPU-accelerated physics simulator built on NVIDIA Warp. We recommend using uv for installation. For full installation details, see the Newton Installation Guide.
Requirements: Python 3.10+, NVIDIA GPU (compute capability >= 5.0), driver 545+

Clone Newton and create a virtual environment:

git clone git@github.com:newton-physics/newton.git
cd newton
uv venv
source .venv/bin/activate
Install Newton dependencies:

uv pip install mujoco --pre -f https://py.mujoco.org/
uv pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
uv pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
uv pip install -e .[examples]
Install ProtoMotions and dependencies:

uv pip install -e /path/to/protomotions
uv pip install -r /path/to/protomotions/requirements_newton.txt
```

- Step 6. Resolve the installation of openmesh==1.2.1
```
安装这个包需要安装cmake及用cmake来编译c++ openmesh，所以需要给linux安装c++环境
sudo apt-get install g++ gcc make
uv pip install cmake ninja

参考这个方式 https://blog.csdn.net/wang1zhong1quan/article/details/147615829
  wget https://github.com/liangjin2007/LearningArea/raw/refs/heads/master/openmesh-1.2.1.tar.gz
  tar -zxvf openmesh-1.2.1.tar.gz
  cd openmesh-1.2.1中的几个CMakeLists.txt的cmake_minimum_required(VERSION xxx)为cmake_minimum_required(VERSION 4.2.1)
  uv pip install -e .
  安装openmesh==1.2.1成功
```

- Step 7. 继续安装uv pip install -r /path/to/protomotions/requirements_newton.txt
```
成功
```

- Step 8. scp Send Windows AMASS npz tar.gz files to Ubuntu Server
```
scp D:\Downloads\project.zip liangjin@172.19.4.52:/home/liangjin/
```

- Step 9. Get ready smpl model pkl datas and use scp to send to Ubuntu Server
```
scp D:\ue-simulator\Training\humenv\data_preparation\AMASS\models\smpl_models.zip liangjin@172.19.4.52:/home/liangjin/
```

- Step 10. unzip AMASS.zip 
```
unzip AMASS.zip
find . -maxdepth 1 -name "*.tar.bz2" -print0 | xargs -0 -I {} tar jxf {} -C /home/liangjin/AMASS_npz/
```















