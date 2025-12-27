from __future__ import annotations

from typing import Literal
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rnd_cfg import RslRlRndCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRl_Triple_AE_PPOPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the Triple_AE_PPO policy."""

    class_name: str = "ActorCritic_Triple_AE"
    """The policy class name. Default is ActorCritic_Triple_AE."""
    
    actor_sg_dim: int = MISSING
    """The robot state dimension for the actor."""
    
    actor_sh_dim: int = MISSING
    """The human state dimension for the actor."""

    actor_sk_dim: int = MISSING
    """The keypoints SE3 state dimension for the actor."""

    latent_dim: int = MISSING
    """The latent dimension for the Triple Autoencoder."""
    
    activate_signals: Literal["robot", "smplx", "keypoints"] = "robot"
    """Which signals to activate: 'robot', 'smplx', or 'keypoints'. Default is 'robot'."""

    robot_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot encoder network."""
    
    human_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the human encoder network."""

    keypoints_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the keypoints encoder network."""

    robot_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot decoder network."""
    
    human_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the human decoder network."""

    keypoints_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the keypoints decoder network."""


@configclass
class RslRl_Triple_AE_PPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Triple_AE_PPO algorithm."""

    class_name: str = "Triple_AE_PPO"
    """The algorithm class name. Default is Triple_AE_PPO."""
    
    reconstruction_loss_coef_sg: float = MISSING
    """The coefficient for the robot goal state reconstruction loss."""
    
    reconstruction_loss_coef_sh: float = MISSING
    """The coefficient for the human state reconstruction loss."""

    reconstruction_loss_coef_sk: float = MISSING
    """The coefficient for the keypoints state reconstruction loss."""
    
    alignment_loss_coef: float = MISSING
    """The coefficient for the three-way latent space alignment loss (MSE)."""
    
    consistency_loss_coef: float = MISSING
    """The coefficient for the cross-modal consistency loss."""
    
    finetune_human_encoder: bool = False
    """Whether to finetune the human encoder and decoder. Default is False."""
    
    finetune_robot_encoder: bool = False
    """Whether to finetune the robot encoder and decoder. Default is False."""

    finetune_keypoints_encoder: bool = False
    """Whether to finetune the keypoints encoder and decoder. Default is False."""