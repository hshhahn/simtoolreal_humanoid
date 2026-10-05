from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class ObjectSizeDistribution:
    type: Literal[
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    ]
    handle_min_lengths: Union[Tuple[float, float, float], Tuple[float, float]]
    handle_max_lengths: Union[Tuple[float, float, float], Tuple[float, float]]
    head_min_lengths: Optional[Union[Tuple[float, float, float], Tuple[float, float]]]
    head_max_lengths: Optional[Union[Tuple[float, float, float], Tuple[float, float]]]
    handle_min_density: float
    handle_max_density: float
    head_min_density: Optional[float]
    head_max_density: Optional[float]

    def __post_init__(self):
        assert len(self.handle_min_lengths) == len(self.handle_max_lengths), (
            f"handle_min_lengths and handle_max_lengths must have the same length: {self.handle_min_lengths} and {self.handle_max_lengths}"
        )
        assert (self.head_min_lengths is None) == (self.head_max_lengths is None), (
            f"head_min_lengths and head_max_lengths must both be None or both be not None: {self.head_min_lengths} and {self.head_max_lengths}"
        )

        assert self.handle_min_density <= self.handle_max_density, (
            f"handle_min_density must be less than or equal to handle_max_density: {self.handle_min_density} and {self.handle_max_density}"
        )
        assert (self.head_min_density is None) == (self.head_max_density is None), (
            f"head_min_density and head_max_density must both be None or both be not None: {self.head_min_density} and {self.head_max_density}"
        )

        if self.head_min_lengths is not None and self.head_max_lengths is not None:
            assert len(self.head_min_lengths) == len(self.head_max_lengths), (
                f"head_min_lengths and head_max_lengths must have the same length: {self.head_min_lengths} and {self.head_max_lengths}"
            )
            assert (
                self.head_min_density is not None and self.head_max_density is not None
            ), (
                f"head_min_density and head_max_density must both be not None: {self.head_min_density} and {self.head_max_density}"
            )
            assert self.head_min_density <= self.head_max_density, (
                f"head_min_density must be less than or equal to head_max_density: {self.head_min_density} and {self.head_max_density}"
            )

    @property
    def shape(self) -> Literal["cuboid", "cylinder"]:
        if len(self.handle_min_lengths) == 3:
            return "cuboid"
        elif len(self.handle_min_lengths) == 2:
            return "cylinder"
        else:
            raise ValueError(f"Invalid handle min lengths: {self.handle_min_lengths}")

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


# 3D printed objects are about 300-400 kg/m^3
LOW_DENSITY_MIN, LOW_DENSITY_MAX = 300, 600

# Hammer head and mallet are 800-1500 kg/m^3
HIGH_DENSITY_MIN, HIGH_DENSITY_MAX = 800, 2000


OBJECT_SIZE_DISTRIBUTIONS: List[ObjectSizeDistribution] = [
    # Hammer
    # Handle: (x) Lengths are [15cm, 30cm]
    #         (y) Widths are [2cm, 4cm]
    #         (z) Height is [1.5cm, 3cm]
    #         (yz diameter) are [1.5cm, 3cm]
    #         (shape) box or cylinder
    # Head:   (x) [2cm, 6cm]
    #         (y) [5cm, 12cm]
    #         (z) [2cm, 6cm]
    #         (shape) box
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.02, 0.015),  # Box
        handle_max_lengths=(0.3, 0.04, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.015),  # Cylinder
        handle_max_lengths=(0.3, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Screwdriver
    # Handle: (x) Lengths are [7cm, 12cm]
    #         (y) Widths are [2.5cm, 4cm]
    #         (z) Height is [2.5cm, 4cm]
    #         (yz diameter) are [2.5cm, 4cm]
    #         (shape) box or cylinder
    # Head:   (x) [7cm, 15cm]
    #         (y) [1cm, 1.5cm]
    #         (z) [1cm, 1.5cm]
    #         (shape) box
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025, 0.025),  # Box
        handle_max_lengths=(0.12, 0.04, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025),  # Cylinder
        handle_max_lengths=(0.12, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Marker
    # Handle: (x) Lengths are [7.5cm, 15cm]
    #         (yz diameter) are [1.5cm, 3cm]
    #         (shape) cylinder
    # Head:   (x) [1cm, 3cm]
    #         (y) [0.5cm, 1cm]
    #         (z) [0.5cm, 1cm]
    #         (shape) box
    ObjectSizeDistribution(
        type="marker",
        handle_min_lengths=(0.075, 0.015),  # Cylinder
        handle_max_lengths=(0.15, 0.03),
        head_min_lengths=(0.01, 0.005, 0.005),
        head_max_lengths=(0.03, 0.01, 0.01),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Spatula
    # Handle: (x) Lengths are [10cm, 20cm]
    #         (y) Widths are [1.25cm, 2.5cm]
    #         (z) Heights are [0.6cm, 2.5cm]
    #         (yz diameter) are [1.25cm, 2.5cm]
    #         (shape) box or cylinder
    # Head:  (x) [5cm, 15cm]
    #        (y) [3cm, 7cm]
    #        (z) [1cm, 3cm]
    #        (shape) box
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125, 0.006),  # Box
        handle_max_lengths=(0.2, 0.025, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125),  # Cylinder
        handle_max_lengths=(0.2, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Eraser
    # Handle: (x) Lengths are [7cm, 15cm]
    #         (y) Widths are [2cm, 7cm]
    #         (z) Heights are [2cm, 7cm]
    #         (shape) box
    # Head:   None
    ObjectSizeDistribution(
        type="eraser",
        handle_min_lengths=(0.07, 0.02, 0.02),  # Box
        handle_max_lengths=(0.15, 0.07, 0.07),
        head_min_lengths=None,
        head_max_lengths=None,
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=None,
        head_max_density=None,
    ),
    # Brush
    # Handle: (x) Lengths are [5cm, 20cm]
    #         (y) Widths are [1cm, 4cm]
    #         (z) Heights are [1cm, 3cm]
    #         (yz diameter) are [1cm, 3cm]
    #         (shape) box or cylinder
    # Head v1:  (x) [5cm, 12cm]
    #           (y) [3cm, 5cm]
    #           (z) [3cm, 8cm]
    #           (shape) box
    # Head v2:  (x) [5cm, 12cm]
    #           (y) [5cm, 12cm]
    #           (z) [2cm, 4cm]
    #           (shape) box
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),  # Box
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),  # v1
        head_max_lengths=(0.12, 0.05, 0.08),  # v1
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),  # Cylinder
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),  # v1
        head_max_lengths=(0.12, 0.05, 0.08),  # v1
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),  # Box
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),  # v2
        head_max_lengths=(0.12, 0.12, 0.04),  # v2
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),  # Cylinder
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),  # v2
        head_max_lengths=(0.12, 0.12, 0.04),  # v2
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
]
