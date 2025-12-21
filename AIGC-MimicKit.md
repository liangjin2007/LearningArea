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
- run(...)
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

BaseAgent._reset_envs
    CharEnv._reset_envs
        DeepMimicEnv._reset_char # 多态
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
                self._reset_ref_char if enable ref char
        CharEnv._reset_char_rigid_body_state
    





```























