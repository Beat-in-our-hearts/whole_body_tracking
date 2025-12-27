"""Package containing task implementations for various robotic environments."""

from pathlib import Path
from isaaclab_tasks.utils import import_packages

# Define DATASETS_DIR before importing packages to avoid circular import
REPLAY_DATASETS_DIR: str = str(Path(__file__).parent.parent.parent.parent.parent / "datasets" / "npz_datasets")
EXTEMDED_DATASETS_DIR: str = str(Path(__file__).parent.parent.parent.parent.parent / "datasets" / "extended_datasets")

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)