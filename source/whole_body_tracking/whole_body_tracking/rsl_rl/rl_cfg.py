from __future__ import annotations

from typing import Literal
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rnd_cfg import RslRlRndCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRl_VAE_PPOPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the VAE_PPO policy."""

    class_name: str = "ActorCriticVAE"
    """The policy class name. Default is ActorCriticVAE."""
    
    actor_sg_dim: int = MISSING
    """The state-goal dimension for the actor."""

    vae_latent_dim: int = MISSING
    """The latent dimension for the Autoencoder."""

    robot_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""

    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""
    

@configclass
class RslRl_VAE_PPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the VAE_PPO algorithm."""

    class_name: str = "VAE_PPO"
    """The algorithm class name. Default is VAE_PPO."""
    
    reconstruction_loss_coef: float = MISSING
    """The coefficient for the reconstruction loss."""


@configclass
class RslRl_FSQVAE_PpoPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the FSQVAEPPO policy."""

    class_name: str = "ActorCriticFSQVAE"
    """The policy class name. Default is ActorCriticFSQVAE."""
    
    actor_sg_dim: int = MISSING
    """The state-goal dimension for the actor."""
    
    fsqvae_latent_dim: int = MISSING
    """The latent dimension for FSQVAE."""
    
    fsq_levels: list[int] = MISSING
    """The FSQ levels for quantization."""
    
    num_codebooks: int = MISSING
    """The number of codebooks for FSQ quantization."""

    robot_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""
    
    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""


@configclass
class RslRl_FSQVAE_PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the FSQVAEPPO algorithm."""

    class_name: str = "FSQVAE_PPO"
    """The algorithm class name. Default is FSQVAEPPO."""
    
    reconstruction_loss_coef: float = MISSING
    """The coefficient for the reconstruction loss."""


@configclass
class RslRl_SONIC_PpoPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the SONIC_PPO policy."""

    class_name: str = "ActorCriticSONIC"
    """The policy class name. Default is ActorCriticSONIC."""
    
    actor_sg_dim: int = MISSING
    """The state-goal dimension for the actor."""
    
    actor_sh_dim: int = MISSING
    """The human state dimension for the actor."""
    
    fsqvae_latent_dim: int = MISSING
    """The latent dimension for FSQVAE."""
    
    fsq_levels: list[int] = MISSING
    """The FSQ levels for quantization."""
    
    num_codebooks: int = MISSING
    """The number of codebooks for FSQ quantization."""

    robot_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot encoder network."""
    
    human_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the human encoder network."""
    
    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""
    
@configclass
class RslRl_SONIC_PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the SONIC_PPO algorithm."""

    class_name: str = "SONIC_PPO"
    """The algorithm class name. Default is SONIC_PPO."""
    
    reconstruction_loss_coef_sg: float = MISSING
    """The coefficient for the robot goal state reconstruction loss."""
    
    reconstruction_loss_coef_sh: float = MISSING
    """The coefficient for the human state reconstruction loss."""
    
    token_loss_coef: float = MISSING
    """The coefficient for the latent token alignment loss."""
    
    cycle_loss_coef: float = MISSING
    """The coefficient for the cycle consistency loss."""
    
    pretrain_vae: bool = False
    """Whether to pretrain the VAE before policy training."""