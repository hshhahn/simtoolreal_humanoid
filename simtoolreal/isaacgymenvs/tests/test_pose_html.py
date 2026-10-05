import json
import re
from pathlib import Path

import pytest

from simtoolreal_shared.pose_html import portable_visual_urdf, render_pose_html


def _scene_payload(html: str) -> dict:
    match = re.search(
        r'<script id="scene-json" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_render_pose_html_serializes_all_frames():
    html = render_pose_html(
        robots=[{"name": "robot", "urdf_path": "https://example.test/robot.urdf"}],
        joint_names=["joint"],
        joint_positions=[[0.0], [0.5]],
        object_poses={"object": [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]},
        robot_base_poses=[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
        dt=1 / 60,
    )

    payload = _scene_payload(html)
    assert payload["trajectory"]["positions"] == [[0.0], [0.5]]
    assert payload["trajectory"]["timestamps"] == [0.0, 0.016667]
    assert "__SCENE_JSON__" not in html


def test_render_pose_html_rejects_mismatched_trajectories():
    with pytest.raises(ValueError, match="object_poses"):
        render_pose_html(
            robots=[],
            joint_names=["joint"],
            joint_positions=[[0.0], [0.5]],
            object_poses={"object": [[0, 0, 0, 0, 0, 0, 1]]},
            robot_base_poses=[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
            dt=1 / 60,
        )


def test_portable_visual_urdf_embeds_external_mesh(tmp_path: Path):
    mesh_path = tmp_path / "shape.stl"
    mesh_path.write_bytes(b"solid shape\nendsolid shape\n")
    urdf_path = tmp_path / "object.urdf"
    urdf = (
        '<robot name="object"><link name="object">'
        '<visual><geometry><mesh filename="shape.stl"/></geometry></visual>'
        '<collision><geometry><mesh filename="shape.stl"/></geometry></collision>'
        "</link></robot>"
    )

    portable = portable_visual_urdf(urdf, urdf_path, "https://example.test/repo")

    assert "data:model/stl;base64," in portable
    assert "<collision>" not in portable
