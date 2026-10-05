"""OmegaConf <-> plain-dict helpers.

Lifted verbatim from `isaacgymenvs/utils/reformat.py` so isaacsimenvs has no
cross-package import from isaacgymenvs.
"""

from __future__ import annotations

from typing import Dict

from omegaconf import DictConfig


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Convert an OmegaConf DictConfig to a plain dict, resolving interpolations."""
    ret: Dict = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Pretty-print a nested dict for debugging."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)
