## Isaacgym Simulator
- Installation
```
conda create -n isaacgym python=3.8
conda activate isaacgym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
pip install -e isaacgym/python
pip install -e /path/to/protomotions
pip install -r /path/to/protomotions/requirements_isaacgym.txt
```

- Data Preprocessing
```

```
- Train
```
train.sh
# To resolve isaacgym problem
export LD_LIBRARY_PATH=/home/liangjin/anaconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

export PYTHONPATH="/home/liangjin/ProtoMotions:/home/liangjin/newton:$PYTHONPATH"

python /home/liangjin/ProtoMotions/protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaacgym \
    --experiment-path /home/liangjin/ProtoMotions/examples/experiments/mimic/mlp.py \
    --experiment-name smpl_mlp_mimic \
    --motion-file /home/liangjin/AMASS_pt/amass_smpl_train.pt \
    --num-envs 512 \
    --batch-size 1024 \
    --ngpu 2 \
    --use-wandb
```

- Inferece
```
Ref https://protomotions.github.io/tutorials/workflows/amass_smpl.html, part Evaluation
https://protomotions.github.io/user_guide/experiments.html

Run inference on trained model:

python protomotions/inference_agent.py \
    --checkpoint results/smpl_amass_flat/last.ckpt \
    --simulator isaacgym
Full evaluation over all motions:

python protomotions/inference_agent.py \
    --checkpoint results/smpl_amass_flat/last.ckpt \
    --simulator isaacgym \
    --num-envs 1024 \
    --full-eval

```

- Log Train

## Newton ??
尝试失败

- Step 1. Clone ProtoMotion code
```
Fork ProtoMotions
git config --global user.name xxx
git config --global user.emal xxx
sshkey-rsa xxx
copy pub key to git ssh-key
git clone https://github.com/liangjin2007/ProtoMotions.git
注意 目前必须是main branch, main branch才有newton
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
注意newton必须使用beta-1-1 git checkout tags/beta-1-1

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
git checkout tags/beta-1-1
cd newton
uv venv
source .venv/bin/activate


uv pip install mujoco --pre -f https://py.mujoco.org/
uv pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
uv pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
uv pip install -e .[examples]
uv pip install . # install newton


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

- Step 10. Process AMASS
```
unzip AMASS.zip
mkdir AMASS_npz
mkdir AMASS_pt

# tar jxf all the tar.bz2 files
find . -maxdepth 1 -name "*.tar.bz2" -print0 | xargs -0 -I {} tar jxf {} -C /home/liangjin/AMASS_npz/

# convert AMASS npz files to ProtoMotions' .pt files.

create a convert_data.sh in /home/liangjin/ProtoMotions
set the content to:
source ~/newton/.venv/bin/activate
export PYTHONPATH="/home/liangjin/ProtoMotions:$PYTHONPATH"
python ./data/scripts/convert_amass_to_motionlib.py /home/liangjin/AMASS_npz/ /home/liangjin/AMASS_pt --motion-config data/yaml_files/amass_smpl_train.yaml --motion-config data/yaml_files/amass_smpl_test.yaml --motion-config data/yaml_files/amass_smpl_validation.yaml

chmod 777 convert_data.sh

./convert_data.sh

```

- Step 11. Train
```

1. wandb
  Use google account to login https://wandb.ai/liangjin2007-ff
  wandb key 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
  wandb login
  Paste the wandb key

2. Train

cd ProtoMotions
cp convert_data.sh train.sh
vim train.sh
remove the convert_amass_to_motionlib.py related code
replace with:

source ~/newton/.venv/bin/activate

export PYTHONPATH="/home/liangjin/ProtoMotions:/home/liangjin/newton:$PYTHONPATH"

python /home/liangjin/ProtoMotions/protomotions/train_agent.py \
    --robot-name smpl \
    --simulator newton \
    --experiment-path /home/liangjin/ProtoMotions/examples/experiments/mimic/mlp.py \
    --experiment-name smpl_mlp_mimic \
    --motion-file /home/liangjin/AMASS_pt/amass_smpl_train.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 2 \
    --use-wandb

```














