"""Cost manager for computing cost signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from ..envs import ManagerBasedSafeRLEnv

@configclass
class CostTermCfg(ManagerTermBaseCfg):
    """Configuration for a cost term."""
    
    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the cost signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the cost term.

    This is multiplied with the cost term's value to compute the final
    cost.
    Note:
        If the weight is zero, the cost term is ignored.
    """
    
    def __post_init__(self):
        super().__post_init__()
        # validate parameters
        if self.weight is MISSING:
            raise ValueError("[CostTerm] Weight for the cost term cannot be MISSING.")
        if self.func is MISSING:
            raise ValueError("[CostTerm] Function for the cost term cannot be MISSING.")
        if self.weight < 0.0:
            raise ValueError("[CostTerm] Weight for the cost term cannot be negative(< 0.0).")

class CostManager(ManagerBase):
    """Manager for computing cost signals for a given world.

    The cost manager computes the total cost as a sum of the weighted cost terms. The cost
    terms are parsed from a nested config class containing the cost manager's settings and cost
    terms configuration.

    The cost terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each cost term should instantiate the :class:`CostTermCfg` class.

    .. note::

        The cost manager multiplies the cost term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed cost terms are balanced with
        respect to the chosen time-step interval in the environment.

    """
    
    _env: ManagerBasedSafeRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedSafeRLEnv):
        """Initialize the cost manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CostTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[CostTermCfg] = list()
        self._class_term_cfgs: list[CostTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual cost term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # create buffer for managing cost per environment
        self._cost_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Buffer which stores the current step cost for each term for each environment
        self._step_cost = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for cost manager."""
        msg = f"<CostManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active cost terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual cost terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual cost terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual cost terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Cost/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the cost terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the cost signal as a weighted sum of individual terms.

        This function calls each cost term managed by the class and adds them to compute the net
        cost signal. It also updates the episodic sums corresponding to individual cost terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net cost signal of shape (num_envs,).
        """
        # reset computation
        self._cost_buf[:] = 0.0
        # iterate over all the cost terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_cost[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total cost
            self._cost_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current cost for this step.
            self._step_cost[:, term_idx] = value / dt

        return self._cost_buf

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: CostTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the cost term.
            cfg: The configuration for the cost term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Cost term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> CostTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the cost term.

        Returns:
            The configuration of the cost term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Cost term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_cost[env_idx, idx].cpu().item()]))
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, CostTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CostTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid weight type
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
