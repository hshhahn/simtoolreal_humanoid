"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.
"""

from . import simtoolreal  # side effect: gym.register("Isaacsimenvs-SimToolReal-Direct-v0", ...)
from . import g1_wuji_sonic
from . import kuka_parallel_gripper

__all__ = ["simtoolreal", "g1_wuji_sonic", "kuka_parallel_gripper"]
