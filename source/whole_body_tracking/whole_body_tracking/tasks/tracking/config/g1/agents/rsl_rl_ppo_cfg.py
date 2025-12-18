from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from whole_body_tracking.rsl_rl import (
    RslRl_VAE_PPOPolicyCfg, 
    RslRl_VAE_PPOAlgorithmCfg,
    RslRl_FSQVAE_PpoPolicyCfg,
    RslRl_FSQVAE_PpoAlgorithmCfg,
    RslRl_SONIC_PpoPolicyCfg,
    RslRl_SONIC_PpoAlgorithmCfg,
    RslRl_Projection_PPOPolicyCfg,
    RslRl_Projection_PPOAlgorithmCfg,
    RslRl_Dual_AE_PPOPolicyCfg,
    RslRl_Dual_AE_PPOAlgorithmCfg,
)


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
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


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class MultiG1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "multi_g1_flat"


    
###################
# SONIC
###################

@configclass
class SONIC_G1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "sonic_g1_flat"

    
@configclass
class SONIC_G1Flat_FSQVAE_PPORunnerCfg(SONIC_G1FlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "sonic_g1_flat_fsqvae"
    policy = RslRl_FSQVAE_PpoPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # FSQVAE specific configs
        actor_sg_dim=580,
        fsqvae_latent_dim=64,
        fsq_levels=[8,8,8,5,5,5],
        num_codebooks=12,
        robot_encoder_hidden_dims=[512, 256],
        recover_decoder_hidden_dims=[256, 512],
    )
    algorithm = RslRl_FSQVAE_PpoAlgorithmCfg(
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
        # specific to autoencoder ppo
        reconstruction_loss_coef=1e-2,
    )

    
# FSQVAE Multi Tracking Env Runner Config
@configclass
class SONIC_Multi_G1Flat_FSQVAE_PPORunnerCfg(SONIC_G1Flat_FSQVAE_PPORunnerCfg):
    max_iterations = 50000
    experiment_name = "sonic_multi_g1_flat_fsqvae"


# Only Support Multi Tracking Env for SONIC + VAE/FSQVAE
# SONIC Multi Tracking Env Runner Config
@configclass
class SONIC_Multi_G1Flat_VQVAE_Scratch_PPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vqvae_scratch"
    empirical_normalization = False # disable empirical normalization for high-dim input
    policy = RslRl_SONIC_PpoPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # SONIC specific configs
        actor_sg_dim=580,
        actor_sh_dim=1260, # human state dimension 126x10
        fsqvae_latent_dim=64,
        activate_signals="robot", # Use robot signals for zero-shot training
        fsq_levels=[8,8,8,5,5,5],
        num_codebooks=32,
        robot_encoder_hidden_dims=[1024, 512, 256],
        human_encoder_hidden_dims=[1024, 512, 256],
        recover_decoder_hidden_dims=[256, 512, 1024],
    )
    algorithm = RslRl_SONIC_PpoAlgorithmCfg(
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
        # specific to autoencoder ppo
        reconstruction_loss_coef_sg=1e-1,
        reconstruction_loss_coef_sh=1e-1,
        token_loss_coef=1.0,
        cycle_loss_coef=1.0,
        pretrain_vae=False, # NOTE scratch training
        finetune_human_encoder=False, # not finetune in scratch training
    )
    
@configclass
class SONIC_Multi_G1Flat_VQVAE_Finetune_PPORunnerCfg(SONIC_Multi_G1Flat_VQVAE_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vqvae_finetune"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_encoder = True
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training

@configclass
class SONIC_Multi_G1Flat_VQVAE_Scratch_SMPLX_PPORunnerCfg(SONIC_Multi_G1Flat_VQVAE_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vqvae_scratch_smplx"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_encoder = False # train all networks from scratch
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training

@configclass
class SONIC_Multi_G1Flat_Projection_Scratch_PPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_projection_scratch"
    empirical_normalization = False # disable empirical normalization for high-dim input
    policy = RslRl_Projection_PPOPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Projection specific configs
        actor_sg_dim=580,
        actor_sh_dim=1260, # human state dimension 126x10
        projection_hidden_dims=64,
        activate_signals="robot", # Use robot signals for zero-shot training
        robot_projection_hidden_dims=[1024, 512, 256],
        human_projection_hidden_dims=[1024, 512, 256],
    )
    algorithm = RslRl_Projection_PPOAlgorithmCfg(
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
        # specific to projection ppo
        projection_alignment_coef=1.0,
        finetune_human_projection=False, # not finetune in scratch training
    )
    
@configclass
class SONIC_Multi_G1Flat_Projection_Finetune_PPORunnerCfg(SONIC_Multi_G1Flat_Projection_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_projection_finetune"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_projection = True
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training
        
        
@configclass
class SONIC_Multi_G1Flat_Projection_Scratch_SMPLX_PPORunnerCfg(SONIC_Multi_G1Flat_Projection_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_projection_scratch_smplx"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_projection = False # train all networks from scratch
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training
        
@configclass
class SONIC_Multi_G1Flat_VAE_Scratch_PPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vae_scratch"
    empirical_normalization = False # disable empirical normalization for high-dim input
    policy = RslRl_VAE_PPOPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # VAE specific configs
        actor_sg_dim=580,
        actor_sh_dim=1260, # human state dimension 126x10
        vae_latent_dim=64,
        activate_signals="robot", # Use robot signals for zero-shot training
        robot_encoder_hidden_dims=[1024, 512, 256],
        human_encoder_hidden_dims=[1024, 512, 256],
        recover_decoder_hidden_dims=[256, 512, 1024],
    )
    algorithm = RslRl_VAE_PPOAlgorithmCfg(
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
        # specific to VAE-PPO
        reconstruction_loss_coef_sg=1.0,
        reconstruction_loss_coef_sh=1e-1,
        gaussian_alignment_loss_coef=1e-1,
        kl_loss_coef=1e-3,
        finetune_human_encoder=False, # not finetune in scratch training
    )
    
@configclass 
class SONIC_Multi_G1Flat_VAE_Finetune_PPORunnerCfg(SONIC_Multi_G1Flat_VAE_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vae_finetune"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_encoder = True
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training        
        
        
@configclass
class SONIC_Multi_G1Flat_VAE_Scratch_SMPLX_PPORunnerCfg(SONIC_Multi_G1Flat_VAE_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_vae_scratch_smplx"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_encoder = False # train all networks from scratch
        self.policy.activate_signals = "smplx" # Use smplx signals for finetune training
        
        
@configclass
class SONIC_Multi_G1Flat_DualAE_Scratch_PPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_dualae_scratch"
    empirical_normalization = False # disable empirical normalization for high-dim input
    policy = RslRl_Dual_AE_PPOPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Dual_AE specific configs
        actor_sg_dim=580,
        actor_sh_dim=1260, # human state dimension 126x10
        latent_dim=64,
        activate_signals="robot", # Use robot signals for zero-shot training
        robot_encoder_hidden_dims=[1024, 512, 256],
        human_encoder_hidden_dims=[1024, 512, 256],
        robot_decoder_hidden_dims=[256, 512, 1024],
        human_decoder_hidden_dims=[256, 512, 1024],
    )
    algorithm = RslRl_Dual_AE_PPOAlgorithmCfg(
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
        # specific to Dual_AE_PPO
        reconstruction_loss_coef_sg=1.0,
        reconstruction_loss_coef_sh=1e-1,
        alignment_loss_coef=1e-1,
        consistency_loss_coef=0.0,
        finetune_human_encoder=False, # not finetune in scratch training
        finetune_robot_encoder=False,
    )


@configclass
class SONIC_Multi_G1Flat_DualAE_Scratch_SMPLX_PPORunnerCfg(SONIC_Multi_G1Flat_DualAE_Scratch_PPORunnerCfg):
    max_iterations = 30_000
    experiment_name = "sonic_multi_g1_flat_dualae_scratch_smplx"
    
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.finetune_human_encoder = False # train all networks from scratch
        self.policy.activate_signals = "smplx" # Use smplx signals for training