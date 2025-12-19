import gymnasium as gym

from . import agents, flat_env_cfg, sonic_flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="MultiTracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatMultiTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiG1FlatPPORunnerCfg",
    },
)

#######################################################
# Register Deploy environments. Wo-State-Estimation
#######################################################

gym.register(
    id="Deploy-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Deploy_G1FlatTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Deploy-MultiTracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Deploy_G1FlatMultiTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiG1FlatPPORunnerCfg",
    },
)


#################################
# SONIC
#################################

gym.register(
    id="SONIC-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SONIC_G1FlatTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="SONIC-Tracking-Flat-G1-VAE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SONIC_G1FlatTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_G1Flat_VAE_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-Tracking-Flat-G1-FSQVAE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SONIC_G1FlatTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_G1Flat_FSQVAE_PPORunnerCfg",
    },
)

# multi-motion sonic envs
gym.register(
    id="SONIC-MultiTracking-Flat-G1-FSQVAE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_FSQVAE_PPORunnerCfg",
    },
)

##################################################
# all exp settings
# Register SONIC env with SMPLX observations
gym.register(
    id="SONIC-MultiTracking-Flat-G1-VQVAE-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VQVAE_Scratch_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-VQVAE-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VQVAE_Scratch_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-VQVAE-SMPLX-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VQVAE_Scratch_SMPLX_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-Projection-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_Projection_Scratch_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-Projection-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_Projection_Finetune_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-Projection-SMPLX-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_Projection_Scratch_SMPLX_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-VAE-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VAE_Scratch_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-VAE-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VAE_Finetune_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-VAE-SMPLX-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_VAE_Scratch_SMPLX_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-DualAE-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_DualAE_Scratch_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-DualAE-Scratch-SMPLX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_DualAE_Scratch_SMPLX_PPORunnerCfg",
    },
)



################
# triple AE
################

gym.register(
    id="SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-Robot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX_wKeypoints,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_TripleAE_Scratch_Robot_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-SMPLX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX_wKeypoints,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_TripleAE_Scratch_SMPLX_PPORunnerCfg",
    },
)

gym.register(
    id="SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-Keypoints-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sonic_flat_env_cfg.SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX_wKeypoints,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SONIC_Multi_G1Flat_TripleAE_Scratch_Keypoints_PPORunnerCfg",
    },
)
