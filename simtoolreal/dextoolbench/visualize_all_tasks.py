"""Visualize all trajectory data from FoundationPose human videos using viser.

Select an object category (e.g., spatula, brush) from the dropdown.
Each task within that category is shown in a separate section with all
objects performing that task displayed side by side.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf

from isaacgymenvs.utils.utils import get_repo_root_dir

# Mesh scale
MESH_SCALE = 1.0

# Colors for different tool types (RGB 0-255)
TOOL_COLORS = {
    "brush": (50, 200, 50),
    "eraser": (200, 50, 50),
    "hammer": (200, 150, 50),
    "marker": (50, 50, 200),
    "screwdriver": (200, 50, 200),
    "spatula": (50, 200, 200),
}
DEFAULT_COLOR = (150, 150, 150)


@dataclass
class VisualizeAllTasksArgs:
    data_dir: Path = Path("dextoolbench/data")
    """Root directory containing trajectory data (object_category/object_name/task_name)."""

    assets_dir: Path = Path("assets/urdf/dextoolbench")
    """Assets directory for tool meshes."""

    trajectory_line_width: float = 3.0
    """Width of the trajectory spline lines."""

    trajectory_point_size: float = 0.008
    """Radius of start/end marker spheres."""

    frame_axes_length: float = 0.03
    """Length of the animated frame axes."""

    frame_axes_radius: float = 0.002
    """Radius of the animated frame axes."""

    animation_fps: float = 30.0
    """Target animation frames per second."""

    object_spacing_x: float = 1.5
    """Horizontal spacing between objects within a task."""

    task_spacing_y: float = 1.5
    """Vertical spacing between task sections."""

    show_goal_volume: bool = False
    """Whether to show the goal volume boxes."""

    port: int = 8080
    """Viser server port."""


@dataclass
class TrajectoryInfo:
    """Information about a single trajectory."""

    tool_type: str
    object_name: str
    task_name: str
    json_path: Path


@dataclass
class TrajectoryAnimation:
    """Pre-computed trajectory data for animation."""

    frame_handle: viser.SceneNodeHandle
    positions: np.ndarray
    wxyz_quats: np.ndarray
    num_frames: int


@dataclass
class CategoryView:
    """All scene elements for a category view."""

    category: str
    scene_handles: List[viser.SceneNodeHandle] = field(default_factory=list)
    animations: List[TrajectoryAnimation] = field(default_factory=list)


# Cache for mesh paths
_mesh_path_cache: Dict[Tuple[str, str], Optional[Path]] = {}


def load_trajectory_as_world_frame(json_path: Path) -> np.ndarray:
    """Load trajectory as (N, 7) numpy array in world frame.

    If the JSON has a "goals" key, the poses are assumed to already be in world frame.
    Otherwise, the poses are assumed to be in robot frame and are converted to world
    frame by adding (0, 0.8, 0) to positions (the robot base offset in world frame).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if "goals" in data:
        # Already in world frame
        return np.array(data["goals"], dtype=np.float32)
    else:
        # In robot frame, convert to world frame (robot is at y=0.8 in world)
        poses = np.array(data, dtype=np.float32)
        poses[:, 1] += 0.8
        return poses


def find_obj_mesh(assets_dir: Path, tool_type: str, object_name: str) -> Optional[Path]:
    """Find the OBJ mesh file for a tool."""
    obj_path = assets_dir / tool_type / object_name / f"{object_name}.obj"
    if obj_path.exists():
        return obj_path

    folder_path = assets_dir / tool_type / object_name
    if folder_path.exists():
        obj_files = list(folder_path.glob("*.obj"))
        if obj_files:
            return obj_files[0]
    return None


def get_mesh_path(assets_dir: Path, tool_type: str, object_name: str) -> Optional[Path]:
    """Get cached mesh path."""
    cache_key = (tool_type, object_name)
    if cache_key not in _mesh_path_cache:
        _mesh_path_cache[cache_key] = find_obj_mesh(assets_dir, tool_type, object_name)
    return _mesh_path_cache[cache_key]


def load_mesh_fresh(obj_path: Path) -> Optional[trimesh.Trimesh]:
    """Load a fresh mesh from disk."""
    try:
        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            else:
                return None
        if MESH_SCALE != 1.0:
            mesh.apply_scale(MESH_SCALE)
        return mesh
    except Exception as e:
        print(f"    Warning: Failed to load mesh {obj_path}: {e}")
        return None


def find_all_trajectories(base_dir: Path) -> List[TrajectoryInfo]:
    """Find all trajectories and return as flat list."""
    trajectories = []

    for json_path in sorted(base_dir.rglob("poses.json")):
        print(f"Found trajectory: {json_path}")
        relative_path = json_path.relative_to(base_dir)
        parts = relative_path.parts

        if len(parts) >= 4:
            tool_type = parts[0]
            object_name = parts[1]
            task_name = parts[2]
            trajectories.append(
                TrajectoryInfo(
                    tool_type=tool_type,
                    object_name=object_name,
                    task_name=task_name,
                    json_path=json_path,
                )
            )
        elif len(parts) == 3:
            tool_type = parts[0]
            object_name = parts[1]
            task_name = Path(parts[2]).stem
            trajectories.append(
                TrajectoryInfo(
                    tool_type=tool_type,
                    object_name=object_name,
                    task_name=task_name,
                    json_path=json_path,
                )
            )

    return trajectories


def get_trajectories_by_category(
    trajectories: List[TrajectoryInfo],
) -> Dict[str, Dict[str, List[TrajectoryInfo]]]:
    """Group trajectories by category (tool_type) -> task_name -> list of trajectories."""
    by_category: Dict[str, Dict[str, List[TrajectoryInfo]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for traj in trajectories:
        by_category[traj.tool_type][traj.task_name].append(traj)

    # Convert to regular dicts
    return {cat: dict(tasks) for cat, tasks in by_category.items()}


def normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Normalize trajectory to start at origin."""
    if len(trajectory) == 0:
        return trajectory
    normalized = trajectory.copy()
    start_pos = trajectory[0, :3].copy()
    normalized[:, :3] = trajectory[:, :3] - start_pos
    return normalized


def convert_xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions from xyzw to wxyz format."""
    return quaternions[:, [3, 0, 1, 2]]


def create_category_view(
    server: viser.ViserServer,
    category: str,
    tasks: Dict[str, List[TrajectoryInfo]],
    args: VisualizeAllTasksArgs,
    table_urdf_path: Path,
) -> CategoryView:
    """Create visualization for a category with sections for each task.

    Each object gets its own table. Trajectory poses are in world frame
    (after load_trajectory_as_world_frame conversion), then shifted by a
    layout offset (x_pos, current_y, 0) so scenes sit side by side.
    """

    view = CategoryView(category=category)
    color = TOOL_COLORS.get(category, DEFAULT_COLOR)

    # Calculate layout
    task_names = sorted(tasks.keys())
    current_y = 0.0
    max_width = 0.0

    for task_idx, task_name in enumerate(task_names):
        task_trajectories = tasks[task_name]
        num_objects = len(task_trajectories)

        # Calculate x positions for objects in this task
        task_width = (num_objects - 1) * args.object_spacing_x
        max_width = max(max_width, task_width)
        start_x = -task_width / 2

        # Add task section label (positioned near the tables)
        task_label = server.scene.add_label(
            f"/view/task_{task_idx}/label",
            text=task_name.replace("_", " "),
            position=(start_x - 0.5, current_y, 0.8),
        )
        view.scene_handles.append(task_label)

        # Add each object in this task
        for obj_idx, traj_info in enumerate(
            sorted(task_trajectories, key=lambda t: t.object_name)
        ):
            x_pos = start_x + obj_idx * args.object_spacing_x
            base_path = f"/view/task_{task_idx}/obj_{obj_idx}"

            # Layout offset for this scene
            layout_offset = np.array([x_pos, current_y, 0.0], dtype=np.float32)

            # Add table for this scene
            table_frame = server.scene.add_frame(
                f"{base_path}/table",
                position=(x_pos, current_y, 0.38),
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            _table_viser = ViserUrdf(
                server, table_urdf_path, root_node_name=f"{base_path}/table"
            )
            view.scene_handles.append(table_frame)

            # Load trajectory (already in world frame)
            trajectory = load_trajectory_as_world_frame(traj_info.json_path)

            if len(trajectory) < 2:
                continue

            # Apply layout offset to world-frame positions
            positions = trajectory[:, :3] + layout_offset
            wxyz_quats = convert_xyzw_to_wxyz(trajectory[:, 3:7])

            # Add trajectory line
            line_handle = server.scene.add_spline_catmull_rom(
                f"{base_path}/line",
                positions=positions,
                color=color,
                line_width=args.trajectory_line_width,
            )
            view.scene_handles.append(line_handle)

            # Add start/end markers
            start_handle = server.scene.add_icosphere(
                f"{base_path}/start",
                radius=args.trajectory_point_size * 1.5,
                color=(50, 255, 50),
                position=tuple(positions[0]),
            )
            view.scene_handles.append(start_handle)

            end_handle = server.scene.add_icosphere(
                f"{base_path}/end",
                radius=args.trajectory_point_size * 1.5,
                color=(255, 50, 50),
                position=tuple(positions[-1]),
            )
            view.scene_handles.append(end_handle)

            # Add animated frame
            frame_handle = server.scene.add_frame(
                f"{base_path}/pose",
                position=tuple(positions[0]),
                wxyz=tuple(wxyz_quats[0]),
                axes_length=args.frame_axes_length,
                axes_radius=args.frame_axes_radius,
            )
            view.scene_handles.append(frame_handle)

            # Add tool mesh
            mesh_path = get_mesh_path(
                args.assets_dir, traj_info.tool_type, traj_info.object_name
            )
            if mesh_path is not None:
                tool_mesh = load_mesh_fresh(mesh_path)
                if tool_mesh is not None:
                    mesh_handle = server.scene.add_mesh_trimesh(
                        f"{base_path}/pose/mesh",
                        mesh=tool_mesh,
                    )
                    view.scene_handles.append(mesh_handle)
            else:
                fallback = trimesh.creation.box(extents=(0.02, 0.01, 0.01))
                fallback.visual.face_colors = [*color, 255]
                mesh_handle = server.scene.add_mesh_trimesh(
                    f"{base_path}/pose/mesh",
                    mesh=fallback,
                )
                view.scene_handles.append(mesh_handle)

            # Store animation
            view.animations.append(
                TrajectoryAnimation(
                    frame_handle=frame_handle,
                    positions=positions,
                    wxyz_quats=wxyz_quats,
                    num_frames=len(positions),
                )
            )

            # Add object label near the table
            obj_label = server.scene.add_label(
                f"{base_path}/label",
                text=traj_info.object_name.replace("_", " "),
                position=(x_pos, current_y - 0.35, 0.35),
            )
            view.scene_handles.append(obj_label)

            # Add goal volume (optional)
            if args.show_goal_volume:
                goal_volume_min = np.array([-0.35, -0.2, 0.6]) + layout_offset
                goal_volume_max = np.array([0.35, 0.2, 0.95]) + layout_offset
                goal_volume_position = (goal_volume_min + goal_volume_max) / 2
                goal_volume_size = goal_volume_max - goal_volume_min
                goal_handle = server.scene.add_box(
                    f"{base_path}/goal_volume",
                    position=goal_volume_position,
                    dimensions=goal_volume_size,
                    color=(0, 255, 0),
                    opacity=0.5,
                )
                view.scene_handles.append(goal_handle)

        # Move to next task section
        current_y += args.task_spacing_y

    # Add category title at top center
    title_handle = server.scene.add_label(
        "/view/title",
        text=f"{category.upper()}",
        position=(0, -0.6, 1.0),
    )
    view.scene_handles.append(title_handle)

    # Add grid
    total_height = current_y + 0.5
    grid_handle = server.scene.add_grid(
        "/view/grid",
        width=max(max_width + 2.0, 3.0),
        height=max(total_height, 2.0),
        position=(0, total_height / 2 - 0.5, -0.01),
        cell_size=0.1,
    )
    view.scene_handles.append(grid_handle)

    return view


def clear_view(view: CategoryView) -> None:
    """Remove all scene elements."""
    for handle in view.scene_handles:
        handle.remove()
    view.scene_handles.clear()
    view.animations.clear()


def main() -> None:
    """Main visualization with category dropdown."""
    args: VisualizeAllTasksArgs = tyro.cli(VisualizeAllTasksArgs)

    print("=" * 80)
    print(args)
    print("=" * 80)

    # Find all trajectories
    all_trajectories = find_all_trajectories(args.data_dir)

    if not all_trajectories:
        print(f"No trajectories found in {args.data_dir}")
        return

    # Group by category then task
    by_category = get_trajectories_by_category(all_trajectories)
    categories = sorted(by_category.keys())

    print(
        f"Found {len(all_trajectories)} trajectories across {len(categories)} categories:"
    )
    for cat in categories:
        tasks = by_category[cat]
        total_trajs = sum(len(t) for t in tasks.values())
        print(f"  {cat.upper()}: {len(tasks)} tasks, {total_trajs} trajectories")
        for task_name, trajs in sorted(tasks.items()):
            objects = [t.object_name for t in trajs]
            print(f"    - {task_name}: {', '.join(objects)}")

    # Load table URDF
    TABLE_URDF_PATH = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    # Pre-cache mesh paths
    print("\nLocating meshes...")
    for traj in all_trajectories:
        get_mesh_path(args.assets_dir, traj.tool_type, traj.object_name)

    # Start viser server
    server = viser.ViserServer(port=args.port)
    print(f"\nViser server running at http://localhost:{args.port}")

    # Current view state
    current_view: Optional[CategoryView] = None
    current_category: Optional[str] = None

    # Create GUI
    with server.gui.add_folder("Controls"):
        category_dropdown = server.gui.add_dropdown(
            "Category",
            options=categories,
            initial_value=categories[0] if categories else "",
        )

        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=3.0,
            step=0.1,
            initial_value=1.0,
        )

    def switch_category(category: str) -> None:
        """Switch to a different category."""
        nonlocal current_view, current_category

        if category == current_category:
            return

        # Clear old view
        if current_view is not None:
            clear_view(current_view)

        # Create new view
        print(f"\nLoading category: {category}")
        current_view = create_category_view(
            server, category, by_category[category], args, TABLE_URDF_PATH
        )
        current_category = category
        print(f"  Loaded {len(current_view.animations)} trajectories")

    @category_dropdown.on_update
    def on_category_change(event: viser.GuiEvent) -> None:
        switch_category(category_dropdown.value)

    # Load initial category
    if categories:
        switch_category(categories[0])

    print("\nUse the dropdown to select different object categories.")
    print("Each task within the category is shown in a separate row.")
    print("Press Ctrl+C to exit.")

    # Animation loop
    frame_idx = 0
    base_dt = 1.0 / args.animation_fps

    while True:
        start_time = time.time()

        if current_view is not None:
            for anim in current_view.animations:
                pose_idx = frame_idx % anim.num_frames
                anim.frame_handle.position = anim.positions[pose_idx]
                anim.frame_handle.wxyz = anim.wxyz_quats[pose_idx]

        frame_idx += 1

        dt = base_dt / speed_slider.value
        elapsed = time.time() - start_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
