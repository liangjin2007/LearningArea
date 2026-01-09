# 目录
- [1.执行训练](1执行训练)
- [2.代码阅读](#2代码阅读)

## 1.执行训练

- 源代码地址 https://github.com/xbpeng/MimicKit?tab=readme-ov-file
- 部署
```
安装isaacgym
安装requirements.txt
下载data放到MimicKit/data/中。这一步在wsl中，我是直接从Windows下载，并拷贝到WSL Ubuntu的MimicKit/data/中。
```
- 使用WSL训练（因为isaacgym只能linux系统）
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
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--mode", "train",
                "--num_envs", "1",
                "--env_config", "/home/liangjin/Desktop/MimicKit-main/data/envs/deepmimic_humanoid_env.yaml",
                "--agent_config", "/home/liangjin/Desktop/MimicKit-main/data/agents/deepmimic_humanoid_ppo_agent.yaml",
                "--visualize", "false",
                "--log_file", "output/model.pt"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
            },
        }
    ]
}
```
- 问题
```
## Mimickit
```
碰到以下代码报错:
    gt = torch.utils.cpp_extension.load(name="gymtorch", sources=sources, extra_cflags=cflags, verbose=True)

    尝试安装gcc，没解决问题:
        # 安装GCC 10
        sudo apt update
        sudo apt install gcc-10 g++-10 -y
        
        # 设置为默认编译器
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
        
        # 验证版本
        gcc --version  # 应显示gcc (Ubuntu 10.3.0-1ubuntu1~20.04) 10.3.0
    
    安装编译需要的东西：
        sudo apt update && sudo apt install -y build-essential
        c++ --version成功
        which c++ 返回/usr/bin/c++

libcuda.so的问题：
    launch.json添加
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
    },

```

```
- 选择mimickit/run.py 按F5启动训练

## 2.代码阅读

### build_env
```
build_env(args, num_envs, device, visualize)
    env = env_builder.build_env(deepmimic_humanoid_env.yaml, num_envs, device, visualize)
        DeepMimicEnv.__init__
            _enable_early_termination: True
            _num_phase_encoding: 4
            _pose_termination: True
            _pose_termination_dist: 1.0
            _enable_phase_obs: False
            _enable_tar_obs: True
            _rand_reset : True
            _ref_char_offset : 2, 0, 0
            self._reward_pose_w = env_config.get("reward_pose_w") # 0.5
            self._reward_vel_w = env_config.get("reward_vel_w") # 0.1
            self._reward_root_pose_w = env_config.get("reward_root_pose_w") # 0.15
            self._reward_root_vel_w = env_config.get("reward_root_vel_w")   # 0.1
            self._reward_key_pos_w = env_config.get("reward_key_pos_w") # 0.15
    
            self._reward_pose_scale = env_config.get("reward_pose_scale") # 0.25
            self._reward_vel_scale = env_config.get("reward_vel_scale") # 0.01
            self._reward_root_pose_scale = env_config.get("reward_root_pose_scale") # 5.0
            self._reward_root_vel_scale = env_config.get("reward_root_vel_scale") # 1.0
            self._reward_key_pos_scale = env_config.get("reward_key_pos_scale") # 10.0

            CharEnv.__init__
                self._global_obs: True
                self._root_height_obs: True
                self._zero_center_action: False

                SimEnv.__init__
                    self._episode_length: 10
                    self._engine= build_engine()
                    self._build_envs()
                        DeepMimicEnv._build_envs()
                            CharEnv._build_envs()
                            DeepMimicEnv._load_motions(motion_file) # load data/motions/humanoid/humanoid_spinkick.pkl, 才24k?
                                self._motion_lib = motion_lib.MotionLib(motion_file, ...)
                                    MotionLib.__init__
                                        MotionLib._load_motions(motion_file)
                                            MotionLib._load_motion_pkl(motion_file)
                                            num_motions = self.get_num_motions()
                                            total_len = self.get_total_length()
                    self._engine.initialize_sim()
                    self._action_space = self._build_action_space()
                    self._build_data_buffers()
                        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
                        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
                        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
                        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
                        obs_space = self.get_obs_space()
```


### _reset_envs会从motion进行采样一帧。
```
BaseAgent._reset_envs
    DeepmimicEnv.reset()
        2.1. CharEnv._reset_envs(env_ids)
            2.1.1. DeepMimicEnv._reset_char # 多态
                    DeepMimicEnv._reset_ref_motion
                        注意这里是只取一帧
                        motion_ids, motion_times = self._sample_motion_times(n) # rand reset
                        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times) ? # dof_vel is local or global ? 
                        self._ref_root_pos[env_ids] = root_pos # [1, 3]
                        self._ref_root_rot[env_ids] = root_rot # [1, 4]
                        self._ref_root_vel[env_ids] = root_vel # [1, 3]
                        self._ref_root_ang_vel[env_ids] = root_ang_vel # [1, 3]
                        self._ref_joint_rot[env_ids] = joint_rot # [1, 14, 4]
                        self._ref_dof_vel[env_ids] = dof_vel # [1, 28]
                        
                        ref_body_pos, _ = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                                                 self._ref_joint_rot)
                        self._ref_body_pos[:] = ref_body_pos # [1, 15, 3]
                
                        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot) # [1, 28]
                        self._ref_dof_pos[env_ids] = dof_pos
                    DeepMimicEnv._ref_state_init
                        self._engine.set_root_pos(env_ids, char_id, self._ref_root_pos[env_ids])
                        self._engine.set_root_rot(env_ids, char_id, self._ref_root_rot[env_ids])
                        self._engine.set_root_vel(env_ids, char_id, self._ref_root_vel[env_ids])
                        self._engine.set_root_ang_vel(env_ids, char_id, self._ref_root_ang_vel[env_ids])
                        
                        self._engine.set_dof_pos(env_ids, char_id, self._ref_dof_pos[env_ids])
                        self._engine.set_dof_vel(env_ids, char_id, self._ref_dof_vel[env_ids])
                        
                        self._engine.set_body_vel(env_ids, char_id, 0.0)
                        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
                        self._reset_ref_char if enable ref char # for visualization
            2.1.2. CharEnv._reset_char_rigid_body_state
                root_pos = self._engine.get_root_pos(char_id)[env_ids]
                root_rot = self._engine.get_root_rot(char_id)[env_ids]
                dof_pos = self._engine.get_dof_pos(char_id)[env_ids]
        
                joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
                body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        
                self._engine.set_body_pos(env_ids, char_id, body_pos)
                self._engine.set_body_rot(env_ids, char_id, body_rot)
    
        2.2. SimEnv._engine.update_sim_state()
                _gym.set_actor_root_state_tensor_indexed
                _gym.set_dof_state_tensor_indexed
                _gym.set_dof_position_target_tensor_indexed
    
        2.3. SimEnv._update_observations(env_ids)
            DeepmimicEnv._compute_obs
                motion_ids = self._motion_ids
                motion_times = self._get_motion_times(env_ids)
                char_id = self._get_char_id()
                root_pos = self._engine.get_root_pos(char_id)
                root_rot = self._engine.get_root_rot(char_id)
                root_vel = self._engine.get_root_vel(char_id)
                root_ang_vel = self._engine.get_root_ang_vel(char_id)
                dof_pos = self._engine.get_dof_pos(char_id)
                dof_vel = self._engine.get_dof_vel(char_id)
                body_pos = self._engine.get_body_pos(char_id)
                joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
                tar_root_pos, tar_root_rot, tar_joint_rot = DeepmimicEnv._fetch_tar_obs_data(motion_ids, motion_times)
                    num_steps = self._tar_obs_steps.shape[0] # 3, _tar_obs_steps = [1, 2, 3]
                    motion_times = motion_times.unsqueeze(-1)
                    time_steps = self._engine.get_timestep() * self._tar_obs_steps # [0.0333333, 0.066666, 0.10000]
                    motion_times = motion_times + time_steps # [0.0333333, 0.066666, 0.10000]
                    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids_tiled, motion_times)
                    root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]]) # [1, 3, 3]
                    root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]]) # [1, 3, 4]
                    joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]]) # [1, 3, 14, 4]
                    return root_pos, root_rot, joint_rot

                obs = compute_deepmimic_obs(root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel,
                                    joint_rot=joint_rot,
                                    dof_vel=dof_vel,
                                    key_pos=key_pos,
                                    global_obs=self._global_obs,
                                    root_height_obs=self._root_height_obs,
                                    phase=motion_phase,
                                    num_phase_encoding=self._num_phase_encoding,
                                    enable_phase_obs=self._enable_phase_obs,
                                    enable_tar_obs=self._enable_tar_obs,
                                    tar_root_pos=tar_root_pos,
                                    tar_root_rot=tar_root_rot,
                                    tar_joint_rot=tar_joint_rot,
                                    tar_key_pos=tar_key_pos)


        2.4. SimEnv._update_info(env_ids)
```

### build_agent
```
build_agent
    init_std = 0.05, init_output_scale = 0.01
    logstd = [-2.9957]
    self._build_actor(config, env)
    self._build_critic(config, env)

    DistributionGaussianDiagBuilder(torch.nn.Module)
        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)
        def forward(self, input):
            mean = self._mean_net(input)
    
            if (self._std_type == StdType.FIXED or self._std_type == StdType.CONSTANT):
                logstd = torch.broadcast_to(self._logstd_net, mean.shape)
            elif (self._std_type == StdType.VARIABLE):
                logstd = self._logstd_net(input)
            else:
                assert(False), "Unsupported StdType: {}".format(self._std_type)
    
            dist = DistributionGaussianDiag(mean=mean, logstd=logstd)
            return dist
```


### DistributionGaussianDiag
```
class DistributionGaussianDiag():
    def __init__(self, mean, logstd):
        self._mean = mean
        self._logstd = logstd
        self._std = torch.exp(self._logstd)
        self._dim = self._mean.shape[-1]
        return

    @property
    def stddev(self):
        return self._std
        
    @property
    def logstd(self):
        return self._logstd
        
    @property
    def mean(self):
        return self._mean
        
    @property
    def mode(self):
        return self._mean

    def sample(self):
        noise = torch.normal(torch.zeros_like(self._mean), torch.ones_like(self._std))
        x = self._mean + self._std * noise
        return x

    def log_prob(self, x):
        diff = x - self._mean
        logp = -0.5 * torch.sum(torch.square(diff / self._std), dim=-1)
        logp += -0.5 * self._dim * np.log(2.0 * np.pi) - torch.sum(self._logstd, dim=-1)
        return logp

    def entropy(self):
        ent = torch.sum(self._logstd, dim=-1)
        ent += 0.5 * self._dim * np.log(2.0 * np.pi * np.e)
        return ent
        
    def kl(self, other):
        assert(isinstance(other, DistributionGaussianDiag))
        other_var = torch.square(other.stddev)
        res = torch.sum(other.logstd - self._logstd + (torch.square(self._std) + torch.square(self._mean - other.mean)) / (2.0 * other_var), dim=-1)
        res += -0.5 * self._dim
        return res

    def param_reg(self):
        # only regularize mean, covariance is regularized via entropy reg
        reg = torch.sum(torch.square(self._mean), dim=-1)
        return reg
```

### Train _rollout_train(num_steps)

```
Train
    _train_iter
        _rollout_train(num_steps)
            for i in range(num_steps):
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)
                    norm_obs = self._obs_norm.normalize(obs)
                    norm_action_dist = self._model.eval_actor(norm_obs)
                    norm_a_rand = norm_action_dist.sample()
                    norm_a_mode = norm_action_dist.mode
                    exp_prob = self._get_exp_prob()
                    exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
                    rand_action_mask = torch.bernoulli(exp_prob)
                    norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
                    rand_action_mask = rand_action_mask.squeeze(-1)
        
                    # calculate logp of norm_a
                    norm_a_logp = norm_action_dist.log_prob(norm_a)
                    norm_a = norm_a.detach()
                    norm_a_logp = norm_a_logp.detach()
                    a = self._a_norm.unnormalize(norm_a)
            
                    a_info = {
                        "a_logp": norm_a_logp,
                        "rand_action_mask": rand_action_mask
                    }
                    return a, a_info
                self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

                # next_obs, r, done, next_info分别是怎么计算的
                next_obs, r, done, next_info = self._step_env(action)
                    obs, r, done, info = self._env.step(action)
                        SimEnv.step(action)
                            SimEnv._pre_physics_step(action)
                                CharEnv._apply_action(action) # 对比于我那边的update_action
                                    char_id = self._get_char_id()
                                    clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low), self._action_bound_high)
                                    self._engine.set_cmd(char_id, clip_action)
                            SimEnv._physics_step()
                                SimEnv._step_sim() 
                                    IsaacGymEngine.step()
                                        _apply_cmd()
                                       for i in range(self._sim_steps): # sim_step = 4
                                            self._pre_sim_step()
                                            self._sim_step()
                                            
                                        self._refresh_sim_tensors()
                            SimEnv._post_physics_step()
                                self._update_time()
                                    self._timestep_buf += 1 # [4]
                                    self._time_buf[:] = self._engine.get_timestep() * self._timestep_buf
                                    return
                                self._update_misc()
                                    DeepMimicEnv._update_misc()
                                        self._update_ref_motion()
                                        motion_ids = self._motion_ids
                                        motion_times = self._get_motion_times()
                                        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
                                        
                                        self._ref_root_pos[:] = root_pos
                                        self._ref_root_rot[:] = root_rot
                                        self._ref_root_vel[:] = root_vel
                                        self._ref_root_ang_vel[:] = root_ang_vel
                                        self._ref_joint_rot[:] = joint_rot
                                        self._ref_dof_vel[:] = dof_vel
                                
                                        ref_body_pos, _ = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                                                                 self._ref_joint_rot)
                                        self._ref_body_pos[:] = ref_body_pos
                                self._update_observations()
                                    if (env_ids is None or len(env_ids) > 0):
                                    obs = self._compute_obs(env_ids)            # _compute_obs
                                    if (env_ids is None):
                                        self._obs_buf[:] = obs
                                    else:
                                        self._obs_buf[env_ids] = obs
    
                                self._update_info()
                                    if (self._mode == base_env.EnvMode.TEST):
                                    if (self._log_tracking_error):
                                        self._record_tracking_error(env_ids)
                                self._update_reward()
                                    DeepMimicEnv._update_reward()
                                    
                                self._update_done()
                            return self._obs_buff, self._reward_buf, self._done_buf, self._info
                    return obs, r, done, info

```

### train _build_train_data()
```
    self.eval()
        
    obs = self._exp_buffer.get_data("obs")
    next_obs = self._exp_buffer.get_data("next_obs")
    r = self._exp_buffer.get_data("reward")
    done = self._exp_buffer.get_data("done")
    rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
    
    norm_next_obs = self._obs_norm.normalize(next_obs)
    next_critic_inputs = {"obs": norm_next_obs}
    next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
    next_vals = next_vals.squeeze(-1).detach()

    succ_val = self._compute_succ_val()
    succ_mask = (done == base_env.DoneFlags.SUCC.value)
    next_vals[succ_mask] = succ_val

    fail_val = self._compute_fail_val()
    fail_mask = (done == base_env.DoneFlags.FAIL.value)
    next_vals[fail_mask] = fail_val

    new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

    norm_obs = self._obs_norm.normalize(obs)
    critic_inputs = {"obs": norm_obs}
    vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
    vals = vals.squeeze(-1).detach()
    adv = new_vals - vals
    
    rand_action_mask = (rand_action_mask == 1.0).flatten()
    adv_flat = adv.flatten()
    rand_action_adv = adv_flat[rand_action_mask]
    adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
    norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
    norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
    
    self._exp_buffer.set_data("tar_val", new_vals)
    self._exp_buffer.set_data("adv", norm_adv)
    
    info = {
        "adv_mean": adv_mean,
        "adv_std": adv_std
    }
    return info

```

### train _update_models()
```
PPOAgent._update_model()
    self.train() # set the module in training mode
        
    num_envs = self.get_num_envs() # 1
    num_samples = self._exp_buffer.get_sample_count() # 32
    batch_size = self._batch_size * num_envs # 4
    num_batches = int(np.ceil(float(num_samples) / batch_size)) # 8
    train_info = dict()

    for i in range(self._update_epochs): # 5
        for b in range(num_batches): # 
            batch = self._exp_buffer.sample(batch_size) # {'obs': [4, 464], 'obs_next' : xxx, ....}
            loss_info = self._compute_loss(batch)

                batch["norm_obs"] = self._obs_norm.normalize(batch["obs"]) # normalize obs
                batch["norm_action"] = self._a_norm.normalize(batch["action"]) # normalize action
        
                critic_info = self._compute_critic_loss(batch)
                        norm_obs = batch["norm_obs"]
                        tar_val = batch["tar_val"]
                        pred = self._model.eval_critic(norm_obs)
                        pred = pred.squeeze(-1)
                
                        diff = tar_val - pred
                        loss = torch.mean(torch.square(diff))
                
                        info = {
                            "critic_loss": loss
                        }
                        return info


                actor_info = self._compute_actor_loss(batch)
                    norm_obs = batch["norm_obs"]
                    norm_a = batch["norm_action"]
                    old_a_logp = batch["a_logp"]
                    adv = batch["adv"]
                    rand_action_mask = batch["rand_action_mask"]
            
                    # loss should only be computed using samples with random actions
                    rand_action_mask = (rand_action_mask == 1.0)
                    norm_obs = norm_obs[rand_action_mask]
                    norm_a = norm_a[rand_action_mask]
                    old_a_logp = old_a_logp[rand_action_mask]
                    adv = adv[rand_action_mask]
            
                    a_dist = self._model.eval_actor(norm_obs)
                    a_logp = a_dist.log_prob(norm_a)
            
                    a_ratio = torch.exp(a_logp - old_a_logp)
                    actor_loss0 = adv * a_ratio
                    actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
                    actor_loss = torch.minimum(actor_loss0, actor_loss1)
                    actor_loss = -torch.mean(actor_loss)
                    
                    clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
                    clip_frac = torch.mean(clip_frac)
                    imp_ratio = torch.mean(a_ratio)
                    
                    info = {
                        "actor_loss": actor_loss,
                        "clip_frac": clip_frac.detach(),
                        "imp_ratio": imp_ratio.detach()
                    }
            
                    if (self._action_bound_weight != 0):
                        action_bound_loss = self._compute_action_bound_loss(a_dist)
                        if (action_bound_loss is not None):
                            action_bound_loss = torch.mean(action_bound_loss)
                            actor_loss += self._action_bound_weight * action_bound_loss
                            info["action_bound_loss"] = action_bound_loss.detach()
            
                    if (self._action_entropy_weight != 0):
                        action_entropy = a_dist.entropy()
                        action_entropy = torch.mean(action_entropy)
                        actor_loss += -self._action_entropy_weight * action_entropy
                        info["action_entropy"] = action_entropy.detach()
                    
                    if (self._action_reg_weight != 0):
                        action_reg_loss = a_dist.param_reg()
                        action_reg_loss = torch.mean(action_reg_loss)
                        actor_loss += self._action_reg_weight * action_reg_loss
                        info["action_reg_loss"] = action_reg_loss.detach()
                    
                    return info


                critic_loss = critic_info["critic_loss"]
                actor_loss = actor_info["actor_loss"]
        
                loss = actor_loss + self._critic_loss_weight * critic_loss
        
                info = {"loss":loss, **critic_info, **actor_info}
                return info

            loss = loss_info["loss"]
            self._optimizer.step(loss)

            torch_util.add_torch_dict(loss_info, train_info)
    
    num_steps = self._update_epochs * num_batches
    torch_util.scale_torch_dict(1.0 / num_steps, train_info)

    return train_info


```



















