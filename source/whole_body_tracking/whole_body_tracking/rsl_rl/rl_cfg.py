from __future__ import annotations

from typing import Literal
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rnd_cfg import RslRlRndCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAutoencoderPpoPolicyCfg(RslRlPpoActorCriticCfg):
    """Configuration for the AutoencoderPPO policy."""

    class_name: str = "ActorCriticAutoencoder"
    """The policy class name. Default is ActorCriticAutoencoder."""

    latent_dim: int = MISSING
    """The latent dimension for FSQVAE."""

    fsq_levels: list[int] = MISSING
    """The FSQ levels for quantization."""

    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""

    robot_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the robot decoder network."""

    recover_decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the recovery decoder network."""



@configclass
class RslRlAutoencoderPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AutoencoderPPO algorithm."""

    class_name: str = "AutoencoderPPO"
    """The algorithm class name. Default is AutoencoderPPO."""

    reconstruction_loss_coef: float = 1.0
    """The coefficient for the reconstruction loss."""


