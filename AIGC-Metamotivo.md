# Metamotivo




### FBModel
- https://slideslive.com/38968883/learning-one-representation-to-optimize-all-rewards?ref=speaker-17548
- [2021]Learning One Representation to Optimize All Rewards
- https://arxiv.org/pdf/2404.05695 

### 修复 网络中有一些代码导出onnx时会出错的问题（一般为非torch代码导致）
nn_models.py
```
class ResidualActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1, 
                 embedding_layers: int = 2) -> None:
        super().__init__()

        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = residual_embedding(obs_dim, hidden_dim, embedding_layers)

        seq = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)] + [Block(hidden_dim, action_dim, False)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # bs x h_dim // 2
        s_embedding = self.embed_s(obs) # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        #dist = TruncatedNormal(mu, std) # 注释掉这一行
        return mu, std
```

### 导出onnx
```
from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.nn_models import eval_mode
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

import torch
import mediapy as media
import math
import h5py
from pathlib import Path
import numpy as np


from onnxruntime import tools
from torchsummary import summary

import onnx


# Model download
model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
print(model)


# Run a policy from Meta Motivo:
device = "cpu"

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        )
else:
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
        )

env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        transform_obs_wrapper,
    ],
    state_init="Default",
)

observation, _ = env.reset() # [1, 358]

init_obs = env.unwrapped.get_obs()
print("init_obs:", init_obs)

model.to(device)


# Goal and Tracking prompts
goal_qpos = np.array([0.13769039,-0.20029453,0.42305034,0.21707786,0.94573617,0.23868944
,0.03856998,-1.05566834,-0.12680767,0.11718296,1.89464102,-0.01371153
,-0.07981451,-0.70497424,-0.0478,-0.05700732,-0.05363342,-0.0657329
,0.08163511,-1.06263979,0.09788937,-0.22008936,1.85898192,0.08773695
,0.06200327,-0.3802791,0.07829525,0.06707749,0.14137152,0.08834448
,-0.07649805,0.78328658,0.12580912,-0.01076061,-0.35937259,-0.13176489
,0.07497022,-0.2331914,-0.11682692,0.04782308,-0.13571422,0.22827948
,-0.23456622,-0.12406075,-0.04466465,0.2311667,-0.12232673,-0.25614032
,-0.36237662,0.11197906,-0.08259534,-0.634934,-0.30822742,-0.93798716
,0.08848668,0.4083417,-0.30910404,0.40950143,0.30815359,0.03266103
,1.03959336,-0.19865537,0.25149713,0.3277561,0.16943092,0.69125975
,0.21721349,-0.30871948,0.88890484,-0.08884043,0.38474549,0.30884107
,-0.40933304,0.30889523,-0.29562966,-0.6271498])
env.unwrapped.set_physics(qpos=goal_qpos, qvel=np.zeros(75))
goal_obs = torch.tensor(env.unwrapped.get_obs()["proprio"].reshape(1,-1), device=model.cfg.device, dtype=torch.float32)
print("goal_obs:", goal_obs.shape, goal_obs)
media.show_image(env.render()) 


# Ref: https://zhuanlan.zhihu.com/p/498425043.
# Need containing batch dimension.


# Export _obs_normalizer
model._obs_normalizer.train(False)
#summary(model._obs_normalizer, input_size=(1,358,), device=device)
torch.onnx.export(model._obs_normalizer,(goal_obs,),f"D:/T2M_Runtime/onnx_models/fbcpr_obsnormalizer_M1.onnx", input_names=("goal_obs",), output_names=("normed_goal_obs",))
onnx_model = onnx.load(f"D:/T2M_Runtime/onnx_models/fbcpr_obsnormalizer_M1.onnx")
onnx.checker.check_model(onnx_model)


# Export _backward_map
model._backward_map.train(False)
backward_trace = torch.jit.trace(model._backward_map, goal_obs)
backward_script = torch.jit.script(model._backward_map, goal_obs)
summary(model._backward_map, input_size=goal_obs.shape, device=device)
torch.onnx.export(backward_trace, goal_obs, f"D:/T2M_Runtime/onnx_models/fbcpr_backward_M1.onnx", input_names=["goal_obs"], output_names=["z"])
onnx_model = onnx.load(f"D:/T2M_Runtime/onnx_models/fbcpr_backward_M1.onnx")
onnx.checker.check_model(onnx_model)



z = model.goal_inference(next_obs=goal_obs) # [1, 256], use model._backward_map
observation, _ = env.reset() # [1, 358]

print("z:", z)

# Export actor model. Need to comment out "return dist" in function Actor::forward of nn_models.py and replace with "return mu, std"
# Also need to modify FBModel::act()
input_std = torch.zeros((1, 1,), dtype=torch.float32, device=device)
model._actor.train(False)
torch.onnx.export(model._actor,(observation, z, input_std,),"D:/T2M_Runtime/onnx_models/fbcpr_actor_M1.onnx", input_names=("observation", "z", "std",), output_names=("mu","std",))
onnx_model = onnx.load(f"D:/T2M_Runtime/onnx_models/fbcpr_actor_M1.onnx")
onnx.checker.check_model(onnx_model)

print("Done")

```

### Goal policy
```
from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.nn_models import eval_mode
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

import torch
import mediapy as media
import math
import h5py
from pathlib import Path
import numpy as np


from onnxruntime import tools
from torchsummary import summary

import onnx


# Model download
model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
print(model)





# Run a policy from Meta Motivo:
device = "cpu"

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        )
else:
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
        )

env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        transform_obs_wrapper,
    ],
    state_init="Default",
)

observation, _ = env.reset() # [1, 358]

init_obs = env.unwrapped.get_obs()
print("init_obs:", init_obs)

model.to(device)

# Goal and Tracking prompts
goal_qpos = np.array([0.13769039,-0.20029453,0.42305034,0.21707786,0.94573617,0.23868944
,0.03856998,-1.05566834,-0.12680767,0.11718296,1.89464102,-0.01371153
,-0.07981451,-0.70497424,-0.0478,-0.05700732,-0.05363342,-0.0657329
,0.08163511,-1.06263979,0.09788937,-0.22008936,1.85898192,0.08773695
,0.06200327,-0.3802791,0.07829525,0.06707749,0.14137152,0.08834448
,-0.07649805,0.78328658,0.12580912,-0.01076061,-0.35937259,-0.13176489
,0.07497022,-0.2331914,-0.11682692,0.04782308,-0.13571422,0.22827948
,-0.23456622,-0.12406075,-0.04466465,0.2311667,-0.12232673,-0.25614032
,-0.36237662,0.11197906,-0.08259534,-0.634934,-0.30822742,-0.93798716
,0.08848668,0.4083417,-0.30910404,0.40950143,0.30815359,0.03266103
,1.03959336,-0.19865537,0.25149713,0.3277561,0.16943092,0.69125975
,0.21721349,-0.30871948,0.88890484,-0.08884043,0.38474549,0.30884107
,-0.40933304,0.30889523,-0.29562966,-0.6271498])
env.unwrapped.set_physics(qpos=goal_qpos, qvel=np.zeros(75))
goal_obs = torch.tensor(env.unwrapped.get_obs()["proprio"].reshape(1,-1), device=model.cfg.device, dtype=torch.float32)
print("goal_obs:", goal_obs.shape, goal_obs)
media.show_image(env.render()) 

z = model.goal_inference(next_obs=goal_obs) # [1, 256], use model._backward_map
observation, _ = env.reset() # [1, 358]

print("z:", z)

#frames = [env.render()]
for i in range(30):
    action = model.act(observation, z, mean=True) # [1,69]
    print("action:",action)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    print("env observation: ", observation.shape, observation) # [1, 358]
    print("env reward:", reward)
    print("env terminated:", terminated)
    print("env truncated:", truncated)
    print("env info:", info)
    #frames.append(env.render())

#media.show_video(frames, fps=30)

```

### Motion tracking policy
```
import os
os.environ["OMP_NUM_THREADS"] = "1"
from metamotivo.wrappers.humenvbench import TrackingWrapper 
from pathlib import Path
from humenv.misc.motionlib import MotionBuffer
from metamotivo.fb_cpr.huggingface import FBcprModel

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
track_model = TrackingWrapper(model=model)
motion_buffer = MotionBuffer(files=ADD_THE_DESIRED_MOTION, base_path=ADD_YOUR_MOTION_ROOT, keys=["qpos", "qvel", "observation"])
ep_ = motion_buffer.get(motion_buffer.get_motion_ids()[0])
ctx = track_model.tracking_inference(next_obs=ep_["observation"][1:])
observation, info = env.reset(options={"qpos": ep_["qpos"][0], "qvel": ep_["qvel"][0]})
done = False
observation, info = env.reset()
frames = [env.render()]
for t in range(len(ctx)):
    obs = torch.tensor(observation.reshape(1,-1), dtype=torch.float32, device=track_model.device)
    action = track_model.act(obs=obs, z=ctx[t]).ravel()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

media.show_video(frames, fps=30)
```

### Reward based tracking policy
```
import os
os.environ["OMP_NUM_THREADS"] = "1"
from humenv import STANDARD_TASKS
import mediapy as media


from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper 
from huggingface_hub import hf_hub_download
from metamotivo.buffers.buffers import DictBuffer
from humenv import make_humenv
from humenv.rewards import RewardFunction, LocomotionReward
import h5py
import torch
import gymnasium
import h5_to_json
import json
local_dir = "metamotivo-S-1-datasets"
dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference

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


# json_file = 'D:/metamotivo/metamotivo-S-1-datasets/data/buffer_inference_1000.ibjson'
# with open(json_file,  'r') as f:
#     data = json.load(f)
# data = {k: torch.Tensor(v[:]) for k, v in data.items()}    

buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)

#
#
# data = buffer.sample(20000)
# data = {k: v[:].tolist() for k, v in data.items()}
# json_str = json.dumps(data)
# json_binary = json_str.encode("utf-8")
# with open('D:/metamotivo/metamotivo-S-1-datasets/data/buffer_inference_20000.ibjson',  'wb') as f:
#     f.write(json_binary) 



task = STANDARD_TASKS[0]
model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
rew_model = RewardWrapper(
        model=model,
        inference_dataset=buffer, # see above how to download and create a buffer
        num_samples_per_inference=1_000,
        inference_function="reward_wr_inference",
        max_workers=1,
        process_executor=False,
        process_context="spawn"
    )
z = rew_model.reward_inference(task)
print("z: ", z)
env, _ = make_humenv(num_envs=1, task=task, state_init="DefaultAndFall", wrappers=[gymnasium.wrappers.FlattenObservation])
done = False
observation, info = env.reset()
frames = [env.render()]
while not done:
    obs = torch.tensor(observation.reshape(1,-1), dtype=torch.float32, device=rew_model.device)
    action = rew_model.act(obs=obs, z=z).ravel()
    observation, reward, terminated, truncated, info = env.step(action)
    a = env.render()
    frames.append(a)
    done = bool(terminated or truncated)

media.show_video(frames, fps=30)
print("Done")

```


### 尝试执行训练
```
python fbcpr_train_humenv.py --compile --motions D:/T2M_Runtime/humenv-main/data_preparation/test_train_split/large1_small1_train_0.1.txt --motions_root D:/T2M_Runtime/AMASS_humenv --prioritization

vscode launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--compile",
                "--motions",".../humenvdata_preparation/test_train_split/large1_small1_train_0.1.txt",
                "--motions_root","D:/T2M_Runtime/AMASS_humenv",
                "--prioritization"
            ],
			"justMyCode": false
        }
    ]
}

Select examples/fbcpr_train_humenv.py

F5

pip install pot
cuda is not available for torch，重新安装torch pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

triton未安装：
	git clone https://github.com/openai/triton.git
	cd triton
	pip install cmake
	// comment out the cmake line in python\requirements.txt
	pip install -r python\requirements.txt
	pip install -e .


```


Victory Plugin  https://forums.unrealengine.com/t/ramas-extra-blueprint-nodes-for-ue5-no-c-required/231476




