from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from typing import Literal

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

############################################
# config for smooth loc metrics experiment #
############################################

@configclass
class RslRlPpo_Smooth_AlgorithmCfg(RslRlPpoAlgorithmCfg):
    
    smooth_alg: Literal["CAPS", "L2C2", "LipsNet++"] | None = None
    """The smoothing algorithm to use. Either "CAPS", "L2C2", "LipsNet++", or None. Default is None."""
    
    caps_lambda_t: float = 0.0
    """The temporal smoothness coefficient for CAPS. Used in the loss term: 𝓛_T = ||π_θ(s_t) - π_θ(s_{t+1})|| """
    
    caps_lambda_s: float = 0.0
    """The spatial smoothness coefficient for CAPS. Used in the loss term: 𝓛_S = ||π_θ(s_t) - π_θ(s'_t)||, where s'_t is a perturbed state."""
    
    caps_sigma: float = 0.0
    """The standard deviation of the Gaussian noise used to perturb the state for spatial smoothness in CAPS."""
    
    l2c2_lambda_pi: float = 0.0
    """The policy smoothness coefficient for L2C2. Used in the loss term: 𝓛_π = ||π_θ(s_t) - π_θ(\bar s_t)||, where 
    \bar s_t = s_t + (s_{t+1} - s_t)·u, where u ~ 𝒰(.)
    """
    
    l2c2_lambda_v: float = 0.0
    """The value function smoothness coefficient for L2C2. Used in the loss term: 𝓛_V = ||V_θ(s_t) - V_θ(\bar s_t)||, where
    \bar s_t = s_t + (s_{t+1} - s_t)·u, where u ~ 𝒰(.)
    """
    
    lips_lambda_pi: float = 0.0
    """The policy Lipschitz continuity coefficient for LipsNet++. Used in the loss term: 𝓛_{Lips} = ||\nabla_{s_t} \pi_\theta(s_t)|| """
    
    smooth_warmup: int = 0
    """The number of iterations to warm up the smoothing regularization. 𝓛_{smooth} *= min(1, step / smooth_warmup)"""

@configclass
class G1FlatPPORunnerBaselineCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "g1_flat_smoothloc"
        
@configclass
class G1FlatPPORunnerCapsCfg(G1FlatPPORunnerBaselineCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm = RslRlPpo_Smooth_AlgorithmCfg(
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
            # smooth loc
            smooth_alg="CAPS",
            caps_lambda_t=0.01,
            caps_lambda_s=0.01,
            caps_sigma=0.05,
        )

@configclass
class G1FlatPPORunnerL2C2Cfg(G1FlatPPORunnerBaselineCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm = RslRlPpo_Smooth_AlgorithmCfg(
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
            # smooth loc
            smooth_alg="L2C2",
            l2c2_lambda_pi=0.005,
            l2c2_lambda_v=0.0025,
        )
        
@configclass
class G1FlatPPORunnerLipsNetCfg(G1FlatPPORunnerBaselineCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm = RslRlPpo_Smooth_AlgorithmCfg(
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
            # smooth loc
            smooth_alg="LipsNet++",
            lips_lambda_pi=0.005,
        )