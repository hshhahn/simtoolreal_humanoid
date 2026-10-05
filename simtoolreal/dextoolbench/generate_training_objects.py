"""Generate training tools based on object size distributions.

This script generates multiple tool variants for training, sampling sizes from
the distributions defined in object_size_distributions.py.
"""

import shutil
from typing import List, Optional, Tuple

import numpy as np

from dextoolbench.create_training_object_utils import (
    Cuboid,
    Cylinder,
    ToolConfig,
    create_tool,
)

# Import the training size distributions
from isaacgymenvs.tasks.simtoolreal.object_size_distributions import (
    OBJECT_SIZE_DISTRIBUTIONS,
    ObjectSizeDistribution,
)
from isaacgymenvs.utils.utils import get_repo_root_dir

# Base output directory for training tools
# BASE_OUTPUT_DIR = Path("/share/portal/kk837/sapg/assets/urdf/dextoolbench_training")
BASE_OUTPUT_DIR = get_repo_root_dir() / "assets/urdf/dextoolbench_training"

# Total number of tools to generate
TOTAL_NUM_TOOLS = 200

# Random seed for reproducibility
RANDOM_SEED = 42


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def generate_random_samples(
    num_dims: int, num_samples: int, rng: np.random.Generator
) -> List[Tuple[float, ...]]:
    """Generate random sample points in [0, 1]^num_dims space."""
    points = rng.uniform(0, 1, size=(num_samples, num_dims))
    return [tuple(p) for p in points]


def sample_dimensions(
    min_lengths: Tuple[float, ...],
    max_lengths: Tuple[float, ...],
    t_values: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Sample dimensions by interpolating between min and max using t_values."""
    return tuple(
        lerp(min_l, max_l, t)
        for min_l, max_l, t in zip(min_lengths, max_lengths, t_values)
    )


def create_tool_config_from_distribution(
    dist: ObjectSizeDistribution,
    handle_dims: Tuple[float, ...],
    head_dims: Optional[Tuple[float, ...]],
    variant_idx: int,
) -> ToolConfig:
    """Create a ToolConfig from sampled dimensions."""
    # Map distribution type to tool_type
    type_mapping = {
        "hammer": "hammer",
        "knife": "knife",
        "screwdriver": "screwdriver",
        "marker": "marker",
        "spatula": "spatula",
        "whiteboard_eraser": "eraser",
        "phone": "phone",
    }
    tool_type = type_mapping.get(dist.type, dist.type)

    # Determine shape prefix
    shape_prefix = "cuboid" if dist.shape == "cuboid" else "cylinder"

    # Create handle primitive
    if dist.shape == "cuboid":
        handle = Cuboid(
            length=handle_dims[0], width=handle_dims[1], height=handle_dims[2]
        )
    else:
        handle = Cylinder(length=handle_dims[0], radius=handle_dims[1])

    # Create head primitive (or minimal head if none)
    if head_dims is not None and len(head_dims) == 3:
        head = Cuboid(length=head_dims[0], width=head_dims[1], height=head_dims[2])
    elif head_dims is not None and len(head_dims) == 2:
        head = Cylinder(length=head_dims[0], radius=head_dims[1])
    else:
        # No head - create minimal cuboid
        head = Cuboid(length=0.001, width=0.001, height=0.001)

    # Generate unique name
    name = f"{shape_prefix}_{dist.type}_v{variant_idx:03d}"

    return ToolConfig(
        name=name,
        handle=handle,
        head=head,
        tool_type=tool_type,
    )


def generate_tools_from_distribution(
    dist: ObjectSizeDistribution,
    dist_idx: int,
    num_samples: int,
    rng: np.random.Generator,
) -> List[ToolConfig]:
    """Generate multiple tool configs from a single distribution using random sampling."""
    configs = []

    # Determine number of varying dimensions
    handle_dims = len(dist.handle_min_lengths)
    head_dims = len(dist.head_min_lengths) if dist.head_min_lengths else 0
    total_dims = handle_dims + head_dims

    # Generate random samples
    samples = generate_random_samples(total_dims, num_samples, rng)

    for sample_idx, t_values in enumerate(samples):
        # Split t_values for handle and head
        handle_t = t_values[:handle_dims]
        head_t = t_values[handle_dims:] if head_dims > 0 else None

        # Sample dimensions
        handle_sampled = sample_dimensions(
            dist.handle_min_lengths,
            dist.handle_max_lengths,
            handle_t,
        )

        head_sampled = None
        if dist.head_min_lengths and dist.head_max_lengths and head_t:
            head_sampled = sample_dimensions(
                dist.head_min_lengths,
                dist.head_max_lengths,
                head_t,
            )

        # Create variant index that's unique across all distributions
        variant_idx = dist_idx * 1000 + sample_idx

        config = create_tool_config_from_distribution(
            dist, handle_sampled, head_sampled, variant_idx
        )
        configs.append(config)

    return configs


def distribute_samples(total: int, num_distributions: int) -> List[int]:
    """Distribute total samples across distributions as evenly as possible."""
    base = total // num_distributions
    remainder = total % num_distributions
    # Give one extra sample to the first 'remainder' distributions
    return [base + (1 if i < remainder else 0) for i in range(num_distributions)]


def generate_all_training_tools() -> None:
    """Generate all training tool configurations from size distributions."""
    print("=" * 60)
    print("Generating training tools from size distributions")
    print(f"Target: {TOTAL_NUM_TOOLS} tools total")
    print("=" * 60)

    # Initialize random number generator with seed for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    # Distribute samples across distributions
    num_distributions = len(OBJECT_SIZE_DISTRIBUTIONS)
    samples_per_dist = distribute_samples(TOTAL_NUM_TOOLS, num_distributions)

    # Collect all configs
    all_configs: List[ToolConfig] = []

    for dist_idx, dist in enumerate(OBJECT_SIZE_DISTRIBUTIONS):
        num_samples = samples_per_dist[dist_idx]
        print(f"\n--- Distribution {dist_idx}: {dist.type} ({dist.shape}) ---")
        configs = generate_tools_from_distribution(dist, dist_idx, num_samples, rng)
        all_configs.extend(configs)
        print(f"  Generated {len(configs)} variants")

    print(f"\n{'=' * 60}")
    print(f"Total configurations: {len(all_configs)}")
    print(f"{'=' * 60}")

    # Clear output directory if it exists
    if BASE_OUTPUT_DIR.exists():
        print(f"\nClearing existing output directory: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)

    # Generate all tools
    print("\nGenerating tools...")
    for config in all_configs:
        output_dir = BASE_OUTPUT_DIR / config.tool_type / config.name
        create_tool(output_dir, config)

    print("\n" + "=" * 60)
    print(f"Generated {len(all_configs)} training tools")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print("=" * 60)

    # Print summary by tool type
    print("\nSummary by tool type:")
    type_counts = {}
    for config in all_configs:
        type_counts[config.tool_type] = type_counts.get(config.tool_type, 0) + 1
    for tool_type, count in sorted(type_counts.items()):
        print(f"  {tool_type}: {count} variants")


if __name__ == "__main__":
    generate_all_training_tools()
