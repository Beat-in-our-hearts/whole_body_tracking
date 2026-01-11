from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from whole_body_tracking.rsl_rl import (
    RslRl_Triple_AE_PPOPolicyCfg,
    RslRl_Triple_AE_PPOAlgorithmCfg,
)

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class MultiTracking_G1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 50_000
    experiment_name = "multi_g1_flat"

@configclass
class GAEMimic_G1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 50_000
    experiment_name = "gaemimic_g1_flat"
    empirical_normalization = False # disable empirical normalization for high-dim input
    policy = RslRl_Triple_AE_PPOPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Triple_AE specific configs
        actor_sg_dim=290,
        actor_sh_dim=1260, # human state dimension 126x10
        actor_sk_dim=450,  # keypoints state dimension 5x9x10 (5 keypoints, 9 dims per keypoint, 10 frames)
        latent_dim=64,
        activate_signals="smplx", # Use robot signals for zero-shot training
        robot_encoder_hidden_dims=[512, 256, 128],
        human_encoder_hidden_dims=[512, 256, 128],
        keypoints_encoder_hidden_dims=[512, 256, 128],
        robot_decoder_hidden_dims=[128, 256, 512],
        human_decoder_hidden_dims=[128, 256, 512],
        keypoints_decoder_hidden_dims=[128, 256, 512],
    )
    algorithm = RslRl_Triple_AE_PPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # specific to Triple_AE_PPO
        reconstruction_loss_coef_sg=5e-1,
        reconstruction_loss_coef_sh=1e-1,
        reconstruction_loss_coef_sk=1e-1,
        alignment_loss_coef=1.0,
        consistency_loss_coef=5.0,
        finetune_human_encoder=False, # not finetune in scratch training
        finetune_robot_encoder=False, # not finetune in scratch training
        finetune_keypoints_encoder=False, # not finetune in scratch training
    )


@configclass
class GAEMimic_Large_G1FlatPPORunnerCfg(GAEMimic_G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.robot_encoder_hidden_dims = [1024, 512, 256]
        self.policy.human_encoder_hidden_dims = [1024, 512, 256]
        self.policy.keypoints_encoder_hidden_dims = [1024, 512, 256]
        self.policy.robot_decoder_hidden_dims = [256, 512, 1024]
        self.policy.human_decoder_hidden_dims = [256, 512, 1024]
        self.policy.keypoints_decoder_hidden_dims = [256, 512, 1024]
        
@configclass
class GAEMimic_G1FlatPPORunnerCfg_V2(GAEMimic_G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [512, 256, 128]
