from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from whole_body_tracking.saferl.rsl_rl import RslSafeRlOnPolicyRunnerCfg, RslSafeRlPpoActorCriticCfg, RslSafeRlPpoAlgorithmCfg

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
class G1FlatBaselinePPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 10000
    experiment_name = "g1_flat_baseline"
    
@configclass
class G1FlatContactPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 10000
    experiment_name = "g1_flat_contact"


########################
#    Safe RL Configs   #
########################
    
@configclass
class SafeRL_G1FlatPPORunnerCfg(RslSafeRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_saferl"
    empirical_normalization = True
    policy = RslSafeRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        cost_critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslSafeRlPpoAlgorithmCfg(
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
        # safe RL specific
        cost_value_loss_coef=1.0,
        constraint_threshold=0.01, #0.05
        lagrangian_multiplier_init=0.1,
        lagrangian_multiplier_lr=4e-4, # armswing, legswing, 1e-4 
        lagrangian_multiplier_max=10.0,
        lagrangian_multiplier_min=0.0,
    )
    
@configclass
class G1FlatSafeRLPPORunnerCfg(SafeRL_G1FlatPPORunnerCfg):
    max_iterations = 10000
    experiment_name = "g1_flat_saferl"
    
@configclass
class G1FlatSafeRLPPORunnerCfg_Lafan1(SafeRL_G1FlatPPORunnerCfg):
    max_iterations = 30000
    experiment_name = "g1_flat_saferl_lafan1"
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.constraint_threshold = 0.15
        self.algorithm.lagrangian_multiplier_lr = 1e-4
        self.algorithm.lagrangian_multiplier_init = 1e-1
        self.algorithm.lagrangian_multiplier_min = self.algorithm.lagrangian_multiplier_init
    
@configclass
class G1FlatSafeRLPPORunnerCfg_AMASS(SafeRL_G1FlatPPORunnerCfg):
    max_iterations = 10000
    experiment_name = "g1_flat_saferl_amass"
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.constraint_threshold = 0.01
        self.algorithm.lagrangian_multiplier_lr = 2e-4
        self.algorithm.lagrangian_multiplier_init = 0.1
        self.algorithm.lagrangian_multiplier_min = self.algorithm.lagrangian_multiplier_init
    


#########################
#     Deploy Configs    #
#########################


@configclass
class Deploy_G1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "deploy_g1_flat"

@configclass
class Deploy_G1FlatSafeRLPPORunnerCfg(G1FlatSafeRLPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "deploy_g1_flat_saferl"