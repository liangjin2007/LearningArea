# 目录
- [1.执行训练](1执行训练)
- [2.代码阅读](#2代码阅读)

## 1.执行训练

- 源代码地址 https://github.com/xbpeng/MimicKit?tab=readme-ov-file
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

### Train
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




```























