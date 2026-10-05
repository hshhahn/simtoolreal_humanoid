import math
from pathlib import Path
from typing import Optional, Tuple, Union

import trimesh


def generate_cuboid_urdf_constant_density(
    filepath: Path, scale: Tuple[float, float, float], density: float = 400
) -> Path:
    """
    Generate a URDF file for a cuboid with uniform density.

    Parameters
    ----------
    filepath : Path
        Path where the URDF file will be saved.
    scale : tuple of float (length, width, height)
        Dimensions of the cuboid along x, y, z axes.
    density : float, default=400
        Material density in kg/m^3.

    Returns
    -------
    Path
        Path to the written URDF file.

    Notes
    -----
    The cuboid's origin is at its center. Visual and collision geometries are identical.
    """
    urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">

  <link name="cuboid">
    <!-- Handle -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    # print(f"✅ URDF written to {filepath}")
    return filepath


def generate_cylinder_urdf_constant_density(
    filepath: Path, height: float, diameter: float, density: float = 400
) -> Path:
    """
    Generate a URDF file for a cylinder with uniform density.

    Parameters
    ----------
    filepath : Path
        Path where the URDF file will be saved.
    height : float
        Cylinder height along the +x axis (after rotation).
    diameter : float
        Cylinder diameter.
    density : float, default=400
        Material density in kg/m^3.

    Returns
    -------
    Path
        Path to the written URDF file.

    Notes
    -----
    The cylinder is rotated so that its main axis aligns with +x in the URDF.
    Visual and collision geometries are identical.
    """
    # In URDFs, cylinders are along z axis
    # But we rotate them to be along +x
    # Height is along +x
    # Radius is along +y and +z
    radius = diameter / 2
    urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">

  <link name="cylinder">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder length="{height}" radius="{radius}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder length="{height}" radius="{radius}"/>
      </geometry>
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    # print(f"✅ URDF written to {filepath}")
    return filepath


def generate_handle_head_urdf_constant_density(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    density: float = 400,
):
    """
    Generate a URDF file for a composite handle-head object with uniform density.

    Parameters
    ----------
    filepath : Path
        Path where the URDF file will be saved.
    handle_scale : tuple
        Dimensions of the handle. Can be:
        - 3D tuple (length_x, length_y, length_z) for a cuboid
        - 2D tuple (height, diameter) for a cylinder
    head_scale : tuple
        Dimensions of the head. Same format as handle_scale.
    density : float, default=400
        Material density in kg/m^3 for both handle and head.

    Returns
    -------
    Path
        Path to the written URDF file.

    Notes
    -----
    - The handle is placed at the origin.
    - The head is offset along +x based on handle and head dimensions.
    - Visual and collision geometries are identical.
    """
    if len(handle_scale) == 3:
        handle_len_x, handle_len_y, handle_len_z = handle_scale
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{handle_len_x} {handle_len_y} {handle_len_z}"/>
        </geometry>
        """
    elif len(handle_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +x
        handle_height, handle_diameter = handle_scale
        handle_radius = handle_diameter / 2
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
        <geometry>
          <cylinder length="{handle_height}" radius="{handle_radius}"/>
        </geometry>
        """
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")

    if len(head_scale) == 3:
        head_len_x, head_len_y, head_len_z = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">

  <link name="handle_head">
    <!-- Handle -->
    <visual>
      {handle_text}
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      {handle_text}
    </collision>

    <!-- Head -->
    <visual>
      {head_text}
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      {head_text}
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    # print(f"✅ URDF written to {filepath}")
    return filepath


def compute_mass_and_inertia(
    scale: Union[Tuple[float, float, float], Tuple[float, float]], density: float
):
    """
    Compute the mass and principal moments of inertia for a cuboid, cylinder, or capsule.

    Parameters
    ----------
    scale : tuple
        Shape dimensions.
        - Cuboid: (lx, ly, lz)
        - Cylinder or capsule: (height, diameter)
    density : float
        Material density in kg/m^3.

    Returns
    -------
    tuple
        (mass, ixx, iyy, izz) in kg and kg·m².

    Notes
    -----
    - Capsule inertia is approximated as cylinder + sphere contribution along main axis.
    - For cylinders, orientation affects which axis is considered main.
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly**2 + lz**2)
        iyy = (1 / 12) * m * (lx**2 + lz**2)
        izz = (1 / 12) * m * (lx**2 + ly**2)
    elif len(scale) == 2:
        from typing import Literal

        MODE: Literal["cylinder", "capsule"] = "capsule"
        # MODE: Literal["cylinder", "capsule"] = "cylinder"
        if MODE == "cylinder":
            h, d = scale[0], scale[1]
            r = d / 2
            v = math.pi * (r**2) * h
            m = v * density
            izz = 0.5 * m * (r**2)
            iyy = (1 / 12) * m * (3 * r**2 + h**2)
            ixx = iyy
        elif MODE == "capsule":
            h, d = (
                scale[0],
                scale[1],
            )  # h = cylindrical height (excluding hemispheres), d = diameter
            r = d / 2

            # masses
            m_c = density * math.pi * r**2 * h
            m_h = density * (2 / 3) * math.pi * r**3  # one hemisphere
            m = m_c + 2 * m_h

            # cylinder inertias about its centroid (axis = z)
            I_c_axis = 0.5 * m_c * r**2
            I_c_perp = (1 / 12) * m_c * (3 * r**2 + h**2)

            # hemisphere inertias about its own centroid
            I_h_axis = (2 / 5) * m_h * r**2
            I_h_perp = (83 / 320) * m_h * r**2

            # hemisphere COM offset from capsule COM
            d_com = (h / 2) + (3 * r / 8)

            # combine
            izz = I_c_axis + 2 * I_h_axis
            ixx = iyy = I_c_perp + 2 * (I_h_perp + m_h * d_com**2)

        else:
            raise ValueError(f"Invalid mode: {MODE}")
    else:
        raise ValueError(f"Invalid scale: {scale}")
    return m, ixx, iyy, izz


def generate_handle_head_urdf_variable_density(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_density: float = 400,
    head_density: float = 800,
):
    """
    Generate a URDF for a handle-head object with independent densities for handle and head.

    Parameters
    ----------
    filepath : Path
        Path to save the URDF file.
    handle_scale : tuple
        Dimensions of the handle. Cuboid (lx, ly, lz) or cylinder (height, diameter).
    head_scale : tuple
        Dimensions of the head. Cuboid or cylinder.
    handle_density : float, default=400
        Material density of the handle.
    head_density : float, default=800
        Material density of the head.

    Returns
    -------
    Path
        Path to the written URDF file.

    Notes
    -----
    - Computes the center of mass of the combined object.
    - Adjusts inertial matrix to reflect parallel axis theorem.
    """
    if len(handle_scale) == 3:
        handle_len_x, handle_len_y, handle_len_z = handle_scale
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{handle_len_x} {handle_len_y} {handle_len_z}"/>
        </geometry>
        """
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(
            scale=handle_scale, density=handle_density
        )

    elif len(handle_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +x
        handle_height, handle_diameter = handle_scale
        handle_radius = handle_diameter / 2
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
        <geometry>
          <cylinder length="{handle_height}" radius="{handle_radius}"/>
        </geometry>
        """
        # Note we flip ixx to the end because we rotate so it is along +x
        handle_mass, handle_izz, handle_iyy, handle_ixx = compute_mass_and_inertia(
            scale=handle_scale, density=handle_density
        )
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")

    if len(head_scale) == 3:
        head_len_x, head_len_y, head_len_z = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(
            scale=head_scale, density=head_density
        )

    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
        # Note we flip iyy to the end because we rotate so it is along +y
        head_mass, head_ixx, head_izz, head_iyy = compute_mass_and_inertia(
            scale=head_scale, density=head_density
        )
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    # Compute mass and inertia
    total_mass = handle_mass + head_mass

    # x_offset is the distance from handle center to head center
    com_x = (handle_mass * 0 + head_mass * x_offset) / total_mass

    d_handle = 0 - com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle**2) + (head_iyy + head_mass * d_head**2)
    izz = (handle_izz + handle_mass * d_handle**2) + (head_izz + head_mass * d_head**2)

    DEBUG_PRINT = False
    if DEBUG_PRINT:
        print(f"handle_scale: {handle_scale}")
        print(f"handle_density: {handle_density}")
        print(f"handle_mass: {handle_mass}")
        print(f"handle_ixx: {handle_ixx}")
        print(f"handle_iyy: {handle_iyy}")
        print(f"handle_izz: {handle_izz}")
        print(f"head_scale: {head_scale}")
        print(f"head_density: {head_density}")
        print(f"head_mass: {head_mass}")
        print(f"head_ixx: {head_ixx}")
        print(f"head_iyy: {head_iyy}")
        print(f"head_izz: {head_izz}")
        print(f"total_mass: {total_mass}")
        print(f"com_x: {com_x}")
        print(f"d_handle: {d_handle}")
        print(f"d_head: {d_head}")
        print(f"ixx: {ixx}")
        print(f"iyy: {iyy}")
        print(f"izz: {izz}")
        breakpoint()

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">

  <link name="handle_head">
    <!-- Handle -->
    <visual>
      {handle_text}
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      {handle_text}
    </collision>

    <!-- Head -->
    <visual>
      {head_text}
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      {head_text}
    </collision>

    <inertial>
      <origin xyz="{com_x} 0 0" rpy="0 0 0"/>
      <mass value="{handle_mass + head_mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    # print(f"✅ URDF written to {filepath}")
    return filepath


def generate_handle_head_urdf_variable_density_2_links(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_density: float = 400,
    head_density: float = 800,
):
    """
    Generate a URDF with separate links for handle and head with independent densities.

    Parameters
    ----------
    filepath : Path
        Path to save the URDF.
    handle_scale : tuple
        Handle dimensions (cuboid or cylinder).
    head_scale : tuple
        Head dimensions (cuboid or cylinder).
    handle_density : float, default=400
        Material density of the handle.
    head_density : float, default=800
        Material density of the head.

    Returns
    -------
    Path
        Path to the written URDF file.

    Notes
    -----
    - The head is attached via a fixed joint to the handle.
    - Useful when separate link dynamics are desired.
    """
    if len(handle_scale) == 3:
        handle_len_x, handle_len_y, handle_len_z = handle_scale
        handle_text = f"""\
        <geometry>
          <box size="{handle_len_x} {handle_len_y} {handle_len_z}"/>
        </geometry>
        """
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(
            scale=handle_scale, density=handle_density
        )
        handle_rpy = "0 0 0"

    elif len(handle_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +x
        handle_height, handle_diameter = handle_scale
        handle_radius = handle_diameter / 2
        handle_text = f"""\
        <geometry>
          <cylinder length="{handle_height}" radius="{handle_radius}"/>
        </geometry>
        """
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(
            scale=handle_scale, density=handle_density
        )
        handle_rpy = "0 -1.5707963267948966 0"
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")

    if len(head_scale) == 3:
        head_len_x, head_len_y, head_len_z = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
        head_text = f"""\
        <geometry>
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(
            scale=head_scale, density=head_density
        )
        head_rpy = "0 0 0"
    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(
            scale=head_scale, density=head_density
        )
        head_rpy = "-1.5707963267948966 0 0"
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    # Setting two densities doesn't work

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">

  <link name="handle">
    <!-- Handle -->
    <visual>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      {handle_text}
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      {handle_text}
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      <mass value="{handle_mass}"/>
      <inertia ixx="{handle_ixx}" iyy="{handle_iyy}" izz="{handle_izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <link name="head">
    <!-- Head -->
    <visual>
      <origin xyz="0 0 0" rpy="{head_rpy}"/>
      {head_text}
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{head_rpy}"/>
      {head_text}
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="{head_rpy}"/>
      <mass value="{head_mass}"/>
      <inertia ixx="{head_ixx}" iyy="{head_iyy}" izz="{head_izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="handle_head_joint" type="fixed">
    <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
    <parent link="handle"/>
    <child link="head"/>
  </joint>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    # print(f"✅ URDF written to {filepath}")
    return filepath


def generate_handle_urdf(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_density: float = 400,
):
    """
    Generate a URDF for a single handle (cuboid or cylinder) with specified density.

    Parameters
    ----------
    filepath : Path
        Path to save the URDF file.
    handle_scale : tuple
        Dimensions of the handle. Cuboid (lx, ly, lz) or cylinder (height, diameter).
    handle_density : float, default=400
        Material density of the handle.

    Returns
    -------
    Path
        Path to the written URDF file.
    """
    if len(handle_scale) == 3:
        return generate_cuboid_urdf_constant_density(
            filepath=filepath, scale=handle_scale, density=handle_density
        )
    elif len(handle_scale) == 2:
        return generate_cylinder_urdf_constant_density(
            filepath=filepath,
            height=handle_scale[0],
            diameter=handle_scale[1],
            density=handle_density,
        )
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")


def generate_handle_head_urdf(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float], None],
    handle_density: float = 400,
    head_density: Optional[float] = 800,
):
    """
    Generate a URDF for a handle or a composite handle-head object.

    Parameters
    ----------
    filepath : Path
        Path to save the URDF file.
    handle_scale : tuple
        Dimensions of the handle. Cuboid (lx, ly, lz) or cylinder (height, diameter).
    head_scale : tuple or None
        Dimensions of the head. If None, only the handle is created.
    handle_density : float, default=400
        Density of the handle.
    head_density : float or None, default=800
        Density of the head. Must be provided if head_scale is not None.

    Returns
    -------
    Path
        Path to the written URDF file.

    Raises
    ------
    ValueError
        If head_scale and head_density are inconsistent.

    Notes
    -----
    - Automatically chooses between single-link or two-link URDF generation.
    """
    if head_scale is None and head_density is None:
        return generate_handle_urdf(
            filepath=filepath, handle_scale=handle_scale, handle_density=handle_density
        )
    elif head_scale is not None and head_density is not None:
        # For some reason, the 2-link approach is not working well, causing physics instability
        # return generate_handle_head_urdf_variable_density_2_links(
        return generate_handle_head_urdf_variable_density(
            filepath=filepath,
            handle_scale=handle_scale,
            head_scale=head_scale,
            handle_density=handle_density,
            head_density=head_density,
        )
    else:
        raise ValueError(
            f"Invalid head scale: {head_scale} and head density: {head_density}"
        )


def generate_handle_head_trimesh(
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float], None],
) -> trimesh.Trimesh:
    """
    Generate a trimesh for a handle-head object.
    """
    if head_scale is None:
        return _generate_handle_trimesh(
            handle_scale=handle_scale,
        )
    else:
        return _generate_handle_head_trimesh(
            handle_scale=handle_scale,
            head_scale=head_scale,
        )


def _generate_handle_trimesh(
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
):
    """
    Generate a trimesh for a handle object.
    """
    import trimesh

    if len(handle_scale) == 3:
        return trimesh.creation.box(
            extents=handle_scale,
        )
    elif len(handle_scale) == 2:
        # Ensure height is along +x
        # Default cylinder axis is along +z
        rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
        mesh = trimesh.creation.cylinder(
            height=handle_scale[0],
            radius=handle_scale[1] / 2,
        )
        mesh.apply_transform(rotation)
        return mesh
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")


def _generate_handle_head_trimesh(
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
):
    """
    Generate a trimesh for a handle-head object.
    """
    handle_mesh = _generate_handle_trimesh(handle_scale=handle_scale)
    head_mesh = _generate_head_trimesh(head_scale=head_scale, handle_scale=handle_scale)
    merged_mesh = trimesh.util.concatenate([handle_mesh, head_mesh])
    return merged_mesh


def _generate_head_trimesh(
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
):
    """
    Generate a trimesh for a head object.
    """
    if len(head_scale) == 3:
        mesh = trimesh.creation.box(
            extents=head_scale,
        )
        head_len_x, _, _ = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
    elif len(head_scale) == 2:
        rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        mesh = trimesh.creation.cylinder(
            height=head_scale[0],
            radius=head_scale[1] / 2,
        )
        mesh.apply_transform(rotation)
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    mesh.apply_translation([x_offset, 0, 0])
    return mesh
