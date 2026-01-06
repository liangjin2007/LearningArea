- Step 1. Clone code
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


碰到问题: cmake找不到
  使用uv pip install cmake ninja 安装cmake

碰到问题openmesh 1.2.1装不上
  wget https://github.com/liangjin2007/LearningArea/raw/refs/heads/master/openmesh-python-1.2.1.zip
  unzip openmesh-python-1.2.1.zip
  cd openmesh-python-1.2.1 Ref https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/openmesh-python/-/tree/1.2.1?ref_type=tags
  

  




```
