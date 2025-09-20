from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg


@configclass
class ManagerBasedSafeRLEnvCfg(ManagerBasedRLEnvCfg):
    costs: object = MISSING
    """Cost settings.
        
    Please refer to the :class:`saferl.managers.CostManager` class for more details.
    """