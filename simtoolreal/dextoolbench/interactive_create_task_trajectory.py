"""Interactive goal trajectory creator using viser.

Place an object on a table, set a start pose and goal waypoints via
sliders / transform controls, then save to a JSON trajectory file.

Usage:
    python interactive_create_task_trajectory.py --object-category hammer --object-name claw_hammer --task-name swing_down
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir


@dataclass
class InteractiveCreateTaskTrajectoryArgs:
    object_category: str = "hammer"
    """Object category (e.g. hammer, marker, spatula)."""

    object_name: str = "claw_hammer"
    """Object name within the category."""

    task_name: str = "my_new_task"
    """Task / trajectory name (used as the output filename stem)."""

    table_urdf: str = "urdf/table_narrow.urdf"
    """Table URDF path relative to assets/."""

    output_dir: Optional[Path] = None
    """Output directory. Defaults to dextoolbench/trajectories/<object_category>/<object_name>/."""

    port: int = 8080
    """Viser server port."""


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


class GoalTrajectoryCreator:
    def __init__(self, args: InteractiveCreateTaskTrajectoryArgs):
        self.args = args
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.start_pose = None
        self.goals = []
        # Euler angles (degrees) are the source of truth for orientation
        self._euler_deg = np.array([0.0, 0.0, 0.0])
        self._position = np.array([0.0, 0.0, 0.85])

        self.output_dir = (
            args.output_dir
            if args.output_dir is not None
            else (
                get_repo_root_dir()
                / "dextoolbench/trajectories"
                / args.object_category
                / args.object_name
            )
        )

        self._setup_scene()

    def _setup_scene(self):
        args = self.args

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -0.8, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.7)

        goal_volume_min = np.array([-0.35, -0.2, 0.6])
        goal_volume_max = np.array([0.35, 0.2, 0.95])
        goal_volume_position = (goal_volume_min + goal_volume_max) / 2
        goal_volume_size = goal_volume_max - goal_volume_min
        self.server.scene.add_box(
            "/goal_volume",
            position=goal_volume_position,
            dimensions=goal_volume_size,
            color=(0, 255, 0),
            opacity=0.5,
        )
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        table_urdf = get_repo_root_dir() / "assets" / args.table_urdf
        self.server.scene.add_frame(
            "/table", position=(0, 0, 0.38), wxyz=(1, 0, 0, 0), show_axes=False
        )
        ViserUrdf(
            self.server,
            table_urdf,
            root_node_name="/table",
            mesh_color_override=(0, 0, 0, 0.5),
        )

        object_urdf = NAME_TO_OBJECT[args.object_name].urdf_path
        self.object_control = self.server.scene.add_transform_controls(
            "/object_control",
            position=(0.0, 0.0, 0.85),
            wxyz=(1, 0, 0, 0),
            scale=0.15,
        )
        ViserUrdf(self.server, object_urdf, root_node_name="/object_control")

        robot_urdf = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        self.robot_control = self.server.scene.add_transform_controls(
            "/robot_control",
            position=(0.0, 0.8, 0),
            wxyz=(1, 0, 0, 0),
            scale=0.15,
        )
        HOME_JOINT_POS_IIWA = np.array(
            [
                -1.571,
                1.571 - np.deg2rad(10),
                -0.000,
                1.376 + np.deg2rad(10),
                -0.000,
                1.485,
                1.308,
            ]
        )
        HOME_JOINT_POS_SHARPA = np.zeros(22)
        HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])
        robot_viser = ViserUrdf(
            self.server, robot_urdf, root_node_name="/robot_control"
        )
        robot_viser.update_cfg(HOME_JOINT_POS)

        self.server.gui.add_markdown(f"**Category:** {args.object_category}")
        self.server.gui.add_markdown(f"**Object:** {args.object_name}")
        self.server.gui.add_markdown(f"**Task:** {args.task_name}")
        self.server.gui.add_markdown("*Goal min Z >= 0.63 (table 0.38 + offset 0.25)*")
        self.current_pose_text = self.server.gui.add_markdown("**Current:** --")
        self.server.gui.add_markdown("---")

        # Position sliders
        self.slider_x = self.server.gui.add_slider(
            "X", min=-0.3, max=0.3, step=0.01, initial_value=self._position[0]
        )
        self.slider_y = self.server.gui.add_slider(
            "Y", min=-0.3, max=0.3, step=0.01, initial_value=self._position[1]
        )
        self.slider_z = self.server.gui.add_slider(
            "Z", min=0.5, max=1.2, step=0.01, initial_value=self._position[2]
        )
        # Rotation sliders (euler angles in degrees) - these are the source of truth
        self.slider_roll = self.server.gui.add_slider(
            "Roll", min=-180, max=180, step=1, initial_value=self._euler_deg[0]
        )
        self.slider_pitch = self.server.gui.add_slider(
            "Pitch", min=-180, max=180, step=1, initial_value=self._euler_deg[1]
        )
        self.slider_yaw = self.server.gui.add_slider(
            "Yaw", min=-180, max=180, step=1, initial_value=self._euler_deg[2]
        )

        self.slider_x.on_update(lambda _: self._on_slider_update())
        self.slider_y.on_update(lambda _: self._on_slider_update())
        self.slider_z.on_update(lambda _: self._on_slider_update())
        self.slider_roll.on_update(lambda _: self._on_slider_update())
        self.slider_pitch.on_update(lambda _: self._on_slider_update())
        self.slider_yaw.on_update(lambda _: self._on_slider_update())

        self.server.gui.add_markdown("---")
        self.start_pose_text = self.server.gui.add_markdown("**Start:** --")
        self.last_goal_text = self.server.gui.add_markdown("**Last goal:** --")
        self.status = self.server.gui.add_markdown("**Goals count:** 0")
        self.server.gui.add_markdown("---")
        self.server.gui.add_button("Set Start Pose").on_click(
            lambda _: self._set_start()
        )
        self.server.gui.add_button("Add Goal").on_click(lambda _: self._add_goal())
        self.server.gui.add_button("Undo Last Goal").on_click(
            lambda _: self._undo_goal()
        )
        self.server.gui.add_button("Save Trajectory").on_click(lambda _: self._save())

    def _euler_to_quat_xyzw(self):
        """Convert internal Euler angles (degrees) to quaternion (xyzw)."""
        euler_rad = np.deg2rad(self._euler_deg)
        return R.from_euler("xyz", euler_rad).as_quat()

    def _get_pose(self):
        """Get current pose as [x, y, z, qx, qy, qz, qw] from internal state."""
        quat_xyzw = self._euler_to_quat_xyzw()
        return np.concatenate([self._position, quat_xyzw])

    def _on_slider_update(self):
        """Update internal state and object control from slider values."""
        self._position = np.array(
            [self.slider_x.value, self.slider_y.value, self.slider_z.value]
        )
        self._euler_deg = np.array(
            [self.slider_roll.value, self.slider_pitch.value, self.slider_yaw.value]
        )
        # Update visual - quaternion computed post-hoc from Euler angles
        self.object_control.position = tuple(self._position)
        self.object_control.wxyz = quat_xyzw_to_wxyz(self._euler_to_quat_xyzw())

    def _set_start(self):
        self.start_pose = self._get_pose()
        self.start_pose_text.content = (
            f"**Start:** {self._format_pose(self.start_pose)}"
        )
        self._update_status()

    def _add_goal(self):
        self.goals.append(self._get_pose())
        self._update_status()

    def _undo_goal(self):
        if self.goals:
            self.goals.pop()
            self._update_status()

    def _format_pose(self, pose):
        xyz = pose[:3].round(3).tolist()
        quat = pose[3:7].round(3).tolist()
        return f"xyz={xyz} quat={quat}"

    def _update_status(self):
        self.status.content = f"**Goals count:** {len(self.goals)}"
        if self.goals:
            self.last_goal_text.content = (
                f"**Last goal:** {self._format_pose(self.goals[-1])}"
            )
        else:
            self.last_goal_text.content = "**Last goal:** --"

    def _save(self):
        if self.start_pose is None or not self.goals:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.args.task_name}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "start_pose": self.start_pose.tolist(),
                    "goals": [g.tolist() for g in self.goals],
                },
                f,
                indent=2,
            )
        print(f"Saved: {output_path}")

    def run(self):
        print(f"Viser: http://localhost:{self.args.port}")
        # Initialize the visual from internal state
        self._on_slider_update()
        while True:
            self.current_pose_text.content = (
                f"**Current:** {self._format_pose(self._get_pose())}"
            )
            time.sleep(0.1)


def main():
    args: InteractiveCreateTaskTrajectoryArgs = tyro.cli(
        InteractiveCreateTaskTrajectoryArgs
    )
    GoalTrajectoryCreator(args).run()


if __name__ == "__main__":
    main()
