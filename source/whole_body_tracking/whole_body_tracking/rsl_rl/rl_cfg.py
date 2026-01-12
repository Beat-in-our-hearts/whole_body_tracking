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
    
@configclass
class RslRl_Triple_AE_PPO_Single_Finetune_PolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the Triple_AE_PPO_Single_Finetune policy.
    
    This policy is designed for single modality finetuning with frozen encoder/decoder.
    The cmd encoder and decoder are frozen by default, only training the action decoder.
    """

    class_name: str = "ActorCritic_Triple_AE_Single_Finetune"
    """The policy class name. Default is ActorCritic_Triple_AE_Single_Finetune."""
    
    actor_cmd_dim: int = MISSING
    """The cmd state dimension for the actor (input to the encoder)."""

    latent_dim: int = 32
    """The latent dimension for the Autoencoder. Default is 32."""
    
    cmd_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the cmd encoder network."""
    
    cmd_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the cmd decoder network."""
    
    freeze: bool = True
    """Whether to freeze the cmd encoder and decoder. Default is True."""


@configclass
class RslRl_Triple_AE_PPO_Single_Finetune_AlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Triple_AE_PPO_Single_Finetune algorithm.
    
    This algorithm is designed for finetuning a pre-trained Triple_AE model on a single modality.
    Only the reconstruction loss for cmd state is used (no alignment or consistency losses).
    """

    class_name: str = "Triple_AE_PPO_Single_Finetune"
    """The algorithm class name. Default is Triple_AE_PPO_Single_Finetune."""
    
    reconstruction_loss_coef_cmd: float = 0.0
    """The coefficient for the cmd state reconstruction loss. Default is 0.0."""