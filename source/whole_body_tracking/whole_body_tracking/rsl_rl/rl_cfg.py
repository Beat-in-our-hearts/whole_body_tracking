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
    
    actor_sp_dim: int = MISSING
    """The state-proprioception dimension for the actor."""

    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""
    
    latent_dim: int = MISSING
    """The latent dimension for the Autoencoder."""

    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""

    robot_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot decoder network."""
    

@configclass
class RslRl_VAE_PPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the VAE_PPO algorithm."""

    class_name: str = "VAE_PPO"
    """The algorithm class name. Default is VAE_PPO."""
    
    reconstruction_loss_coef: float = MISSING
    """The coefficient for the reconstruction loss."""


@configclass
class RslRlFSQVAEPpoPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the FSQVAEPPO policy."""

    class_name: str = "ActorCriticFSQVAE"
    """The policy class name. Default is ActorCriticFSQVAE."""
    
    actor_sg_dim: int = MISSING
    """The state-goal dimension for the actor."""
    
    actor_sp_dim: int = MISSING
    """The state-proprioception dimension for the actor."""

    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""
    
    latent_dim: int = MISSING
    """The latent dimension for FSQVAE."""
    
    fsq_levels: list[int] = MISSING
    """The FSQ levels for quantization."""

    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""

    robot_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot decoder network."""


@configclass
class RslRlFSQVAEPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the FSQVAEPPO algorithm."""

    class_name: str = "FSQVAE_PPO"
    """The algorithm class name. Default is FSQVAEPPO."""
    
    reconstruction_loss_coef: float = MISSING
    """The coefficient for the reconstruction loss."""


