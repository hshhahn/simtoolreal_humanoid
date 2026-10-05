"""isaacsimenvs: Isaac Sim / Isaac Lab environment package.

Importing ``isaacsimenvs`` (transitively) imports every task subpackage under
``isaacsimenvs.tasks``, firing their ``gym.register`` side effects. After this,
any task id registered by isaacsimenvs is resolvable via ``gym.spec`` /
``gym.make`` / ``isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry``.
"""

from . import tasks  # noqa: F401  side effect: gym.register(...) for each task

__all__ = ["tasks"]
