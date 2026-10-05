"""Per-handle-head-type procedural-asset size distributions for SimToolReal.

Clean port of isaacgymenvs/tasks/simtoolreal/object_size_distributions.py — no
imports from isaacgymenvs. Each ``ObjectSizeDistribution`` defines min/max
handle and head scales (cuboid xyz or cylinder height+diameter) plus density
ranges. Methods sample N instances via np.random.uniform.

Twelve pre-built distributions cover six handle-head types (hammer,
screwdriver, marker, spatula, eraser, brush) with cuboid and cylinder
variants where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

Scale2 = tuple[float, float]
Scale3 = tuple[float, float, float]
Scale = Union[Scale2, Scale3]


@dataclass
class ObjectSizeDistribution:
    type: Literal["hammer", "screwdriver", "marker", "spatula", "eraser", "brush"]
    handle_min_lengths: Scale
    handle_max_lengths: Scale
    head_min_lengths: Optional[Scale]
    head_max_lengths: Optional[Scale]
    handle_min_density: float
    handle_max_density: float
    head_min_density: Optional[float]
    head_max_density: Optional[float]

    def __post_init__(self) -> None:
        assert len(self.handle_min_lengths) == len(self.handle_max_lengths)
        assert (self.head_min_lengths is None) == (self.head_max_lengths is None)
        assert self.handle_min_density <= self.handle_max_density
        assert (self.head_min_density is None) == (self.head_max_density is None)
        if self.head_min_lengths is not None:
            assert len(self.head_min_lengths) == len(self.head_max_lengths)
            assert self.head_min_density is not None
            assert self.head_max_density is not None
            assert self.head_min_density <= self.head_max_density

    @property
    def shape(self) -> Literal["cuboid", "cylinder"]:
        return "cuboid" if len(self.handle_min_lengths) == 3 else "cylinder"

    def sample_handle_scales(self, num_objects: int) -> np.ndarray:
        return np.random.uniform(
            self.handle_min_lengths,
            self.handle_max_lengths,
            size=(num_objects, len(self.handle_min_lengths)),
        )

    def sample_head_scales(self, num_objects: int) -> Optional[np.ndarray]:
        if self.head_min_lengths is None or self.head_max_lengths is None:
            return None
        return np.random.uniform(
            self.head_min_lengths,
            self.head_max_lengths,
            size=(num_objects, len(self.head_min_lengths)),
        )

    def sample_handle_densities(self, num_objects: int) -> np.ndarray:
        return np.random.uniform(
            self.handle_min_density, self.handle_max_density, size=num_objects
        )

    def sample_head_densities(self, num_objects: int) -> Optional[np.ndarray]:
        if self.head_min_density is None or self.head_max_density is None:
            return None
        return np.random.uniform(
            self.head_min_density, self.head_max_density, size=num_objects
        )


# 3D-printed objects are ~300-600 kg/m^3; hammer heads / mallets are ~800-2000 kg/m^3.
LOW_DENSITY_MIN, LOW_DENSITY_MAX = 300.0, 600.0
HIGH_DENSITY_MIN, HIGH_DENSITY_MAX = 800.0, 2000.0


OBJECT_SIZE_DISTRIBUTIONS: list[ObjectSizeDistribution] = [
    # Hammer — cuboid handle.
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.02, 0.015),
        handle_max_lengths=(0.30, 0.04, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Hammer — cylinder handle.
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.015),
        handle_max_lengths=(0.30, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Screwdriver — cuboid
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025, 0.025),
        handle_max_lengths=(0.12, 0.04, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Screwdriver — cylinder
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025),
        handle_max_lengths=(0.12, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Marker — cylinder only
    ObjectSizeDistribution(
        type="marker",
        handle_min_lengths=(0.075, 0.015),
        handle_max_lengths=(0.15, 0.03),
        head_min_lengths=(0.01, 0.005, 0.005),
        head_max_lengths=(0.03, 0.01, 0.01),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Spatula — cuboid
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125, 0.006),
        handle_max_lengths=(0.2, 0.025, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Spatula — cylinder
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125),
        handle_max_lengths=(0.2, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Eraser — cuboid, no head
    ObjectSizeDistribution(
        type="eraser",
        handle_min_lengths=(0.07, 0.02, 0.02),
        handle_max_lengths=(0.15, 0.07, 0.07),
        head_min_lengths=None,
        head_max_lengths=None,
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=None,
        head_max_density=None,
    ),
    # Brush — cuboid + head v1
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),
        head_max_lengths=(0.12, 0.05, 0.08),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cylinder + head v1
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),
        head_max_lengths=(0.12, 0.05, 0.08),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cuboid + head v2
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),
        head_max_lengths=(0.12, 0.12, 0.04),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cylinder + head v2
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),
        head_max_lengths=(0.12, 0.12, 0.04),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
]


__all__ = ["ObjectSizeDistribution", "OBJECT_SIZE_DISTRIBUTIONS"]
