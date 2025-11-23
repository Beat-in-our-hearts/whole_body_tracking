from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from whole_body_tracking.rsl_rl import (
    RslRl_VAE_PPOPolicyCfg, 
    RslRl_VAE_PPOAlgorithmCfg,
    RslRl_FSQVAE_PpoPolicyCfg,
    RslRl_FSQVAE_PpoAlgorithmCfg,
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
class SONIC_G1Flat_VAE_PPORunnerCfg(SONIC_G1FlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "sonic_g1_flat_vae"
    policy = RslRl_VAE_PPOPolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # VAE specific configs
        actor_sg_dim=580,
        vae_latent_dim=64,
        robot_encoder_hidden_dims=[512, 256],
        recover_decoder_hidden_dims=[256, 512],
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
        # specific to autoencoder ppo
        reconstruction_loss_coef=1e-2,
    )

    
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
        num_codebooks=6,
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
    
# VAE Multi Tracking Env Runner Config
@configclass
class SONIC_Multi_G1Flat_VAE_PPORunnerCfg(SONIC_G1Flat_VAE_PPORunnerCfg):
    max_iterations = 50000
    experiment_name = "sonic_multi_g1_flat_vae"
    
# FSQVAE Multi Tracking Env Runner Config
@configclass
class SONIC_Multi_G1Flat_FSQVAE_PPORunnerCfg(SONIC_G1Flat_FSQVAE_PPORunnerCfg):
    max_iterations = 50000
    experiment_name = "sonic_multi_g1_flat_fsqvae"