"""Create primitive tools (hammer, mallet, etc.) using cuboids and cylinders.

Each tool has a handle and a head. This script generates both:
1. A URDF file with primitive shapes (for simulation)
2. A corresponding OBJ file with colors (for visualization)
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import trimesh
from PIL import Image

# =============================================================================
# Primitive Shape Dataclasses
# =============================================================================


@dataclass
class Cuboid:
    """A cuboid (box) shape.

    The shape is aligned along X axis (length is along X).
    """

    length: float  # X dimension (along tool axis)
    width: float  # Y dimension
    height: float  # Z dimension

    def to_urdf_geometry(self) -> str:
        """Return URDF geometry XML string."""
        return f'<box size="{self.length} {self.width} {self.height}"/>'

    def to_trimesh(self, as_head: bool = False) -> trimesh.Trimesh:
        """Create a trimesh mesh for this shape."""
        return trimesh.creation.box(extents=(self.length, self.width, self.height))

    def get_length(self, as_head: bool = False) -> float:
        """Return the length along the tool axis (X)."""
        return self.length

    def get_scale(self) -> Tuple[float, float, float]:
        """Return the scale of the cuboid."""
        return (self.length, self.width, self.height)


@dataclass
class Cylinder:
    """A cylinder shape.

    When used as a handle: aligned along X axis (length is along X).
    When used as a head: aligned along Y axis (length perpendicular to handle).
    """

    length: float  # Length of cylinder
    radius: float  # Radius of cylinder

    def to_urdf_geometry(self) -> str:
        """Return URDF geometry XML string."""
        return f'<cylinder length="{self.length}" radius="{self.radius}"/>'

    def to_trimesh(self, as_head: bool = False) -> trimesh.Trimesh:
        """Create a trimesh mesh for this shape."""
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.length)
        if as_head:
            # Rotate 90 degrees around X axis to align with Y axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        else:
            # Rotate 90 degrees around Y axis to align with X axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
        mesh.apply_transform(rotation)
        return mesh

    def get_length(self, as_head: bool = False) -> float:
        """Return the length along the tool axis (X)."""
        if as_head:
            return 2 * self.radius  # Head extends by diameter in X direction
        return self.length

    def get_scale(self) -> Tuple[float, float, float]:
        """Return the scale of the cylinder."""
        return (self.length, self.radius * 2, self.radius * 2)


# Type alias for any shape
Shape = Union[Cuboid, Cylinder]


def get_urdf_rpy(shape: Shape, as_head: bool = False) -> str:
    """Get the rotation (roll, pitch, yaw) for URDF geometry."""
    if isinstance(shape, Cylinder):
        if as_head:
            return f"{math.pi / 2} {0} {0}"
        else:
            return f"{0} {math.pi / 2} {0}"
    return "0 0 0"


# =============================================================================
# Tool Configuration
# =============================================================================

# Default color: BAMBU PLA Matte Ice Blue (#A3D8E1)
DEFAULT_COLOR = (0.639, 0.847, 0.882, 1.0)


@dataclass
class ToolConfig:
    """Configuration for a tool with handle and head."""

    name: str
    handle: Shape
    head: Shape
    tool_type: str = "hammer"
    color: Tuple[float, float, float, float] = DEFAULT_COLOR
    density: float = 400.0

    def get_head_offset(self) -> float:
        """Calculate head offset (places head at end of handle)."""
        handle_length = self.handle.get_length(as_head=False)
        head_length = self.head.get_length(as_head=True)
        return handle_length / 2 + head_length / 2


# =============================================================================
# URDF Generation
# =============================================================================


def create_tool_urdf(output_path: Path, config: ToolConfig) -> Path:
    """Create a tool URDF with a handle and head."""
    head_offset = config.get_head_offset()
    handle_rpy = get_urdf_rpy(config.handle, as_head=False)
    head_rpy = get_urdf_rpy(config.head, as_head=True)
    color = config.color

    urdf = f"""<?xml version="1.0"?>
<robot name="{config.name}">

  <link name="{config.name}">
    <!-- Handle -->
    <visual>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      <geometry>
        {config.handle.to_urdf_geometry()}
      </geometry>
      <material name="material">
        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      <geometry>
        {config.handle.to_urdf_geometry()}
      </geometry>
    </collision>

    <!-- Head -->
    <visual>
      <origin xyz="{head_offset} 0 0" rpy="{head_rpy}"/>
      <geometry>
        {config.head.to_urdf_geometry()}
      </geometry>
      <material name="material">
        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{head_offset} 0 0" rpy="{head_rpy}"/>
      <geometry>
        {config.head.to_urdf_geometry()}
      </geometry>
    </collision>

    <inertial>
      <density value="{config.density}"/>
    </inertial>
  </link>

</robot>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(urdf)
    print(f"Created URDF: {output_path}")
    return output_path


# =============================================================================
# OBJ Generation
# =============================================================================


def create_tool_obj(output_path: Path, config: ToolConfig) -> Path:
    """Create a tool OBJ mesh with uniform color, MTL material file, and texture image."""
    head_offset = config.get_head_offset()
    color = config.color

    # Create handle mesh
    handle_mesh = config.handle.to_trimesh(as_head=False)

    # Create head mesh and translate it
    head_mesh = config.head.to_trimesh(as_head=True)
    head_mesh.apply_translation([head_offset, 0, 0])

    # Combine meshes
    combined_mesh = trimesh.util.concatenate([handle_mesh, head_mesh])

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a solid-color texture image (for FoundationPose compatibility)
    texture_path = output_path.with_suffix(".png")
    texture_size = 64  # Small texture is sufficient for solid color
    color_rgb = tuple(int(c * 255) for c in color[:3])
    texture_img = Image.new("RGB", (texture_size, texture_size), color_rgb)
    texture_img.save(texture_path)
    print(f"Created texture: {texture_path}")

    # Create UV coordinates for the mesh (simple planar mapping)
    # Map all vertices to the center of the texture (since it's a solid color)
    uv = np.zeros((len(combined_mesh.vertices), 2))
    uv[:, 0] = 0.5  # u = 0.5
    uv[:, 1] = 0.5  # v = 0.5

    # Create material with texture
    material = trimesh.visual.material.SimpleMaterial(
        image=texture_img,
        diffuse=color_rgb,
    )

    # Apply texture visuals with UV coordinates
    combined_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=material,
        image=texture_img,
    )

    # Save as OBJ
    combined_mesh.export(str(output_path), file_type="obj")
    print(f"Created OBJ: {output_path}")

    # Create MTL file with texture reference
    mtl_path = output_path.with_suffix(".mtl")
    mtl_content = f"""# Material file for {output_path.name}

newmtl material_0
Ka 0.0 0.0 0.0
Kd {color[0]} {color[1]} {color[2]}
Ks 0.4 0.4 0.4
Ns 10.0
d {color[3]}
map_Kd {texture_path.name}
"""
    with open(mtl_path, "w") as f:
        f.write(mtl_content)
    print(f"Created MTL: {mtl_path}")

    # Update OBJ to reference the MTL file
    with open(output_path, "r") as f:
        obj_content = f.read()

    if "mtllib" not in obj_content:
        obj_content = f"mtllib {mtl_path.name}\nusemtl material_0\n" + obj_content
        with open(output_path, "w") as f:
            f.write(obj_content)

    # Also store a .stl file
    stl_path = output_path.with_suffix(".stl")
    combined_mesh.export(str(stl_path), file_type="stl")
    print(f"Created STL: {stl_path}")

    return output_path


# =============================================================================
# Convenience function
# =============================================================================


def create_tool(output_dir: Path, config: ToolConfig) -> Tuple[Path, Path]:
    """Create both URDF and OBJ files for a tool."""
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = output_dir / f"{config.name}.urdf"
    obj_path = output_dir / f"{config.name}.obj"

    create_tool_urdf(urdf_path, config)
    create_tool_obj(obj_path, config)

    return urdf_path, obj_path


# =============================================================================
# Test Case
# =============================================================================

if __name__ == "__main__":
    base_dir = Path("/share/portal/kk837/sapg/assets/urdf/dextoolbench/test_hammer")

    # Test 1: Cuboid handle + Cuboid head (classic hammer)
    cuboid_hammer = ToolConfig(
        name="cuboid_hammer",
        handle=Cuboid(length=0.20, width=0.025, height=0.02),
        head=Cuboid(length=0.02, width=0.08, height=0.02),
        tool_type="hammer",
    )
    create_tool(base_dir / cuboid_hammer.name, cuboid_hammer)

    # Test 2: Cuboid handle + Cylinder head (mallet)
    cylinder_mallet = ToolConfig(
        name="cylinder_mallet",
        handle=Cuboid(length=0.20, width=0.025, height=0.02),
        head=Cylinder(length=0.06, radius=0.025),
        tool_type="hammer",
    )
    create_tool(base_dir / cylinder_mallet.name, cylinder_mallet)

    print(f"\nGenerated test tools in: {base_dir}")
