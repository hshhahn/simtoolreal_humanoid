"""Render the physical wrist-fixture grasp witnesses from recorded PhysX poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.render_g1_policy_comparison import (
    Image, ImageDraw, VisualRobot, camera_pose, font, imageio, np, pyrender,
)


def xyzw(pose):
    return list(pose[:3]) + list(pose[4:7]) + [pose[3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = json.loads((directory / "report.json").read_text())
    witnesses = [(key, value) for key, value in report["best_traces"].items()
                 if value["success"] and len(value["commands"]) == 3]
    if len(witnesses) != 2:
        raise RuntimeError("Expected verified three-command witnesses for both markers")
    scenes = []
    traces = []
    table = Path(report["hand_urdf"]).parent / "manipulation_table.urdf"
    for key, witness in witnesses:
        scene = pyrender.Scene(bg_color=[.96, .97, .98, 1.], ambient_light=[.4, .4, .4])
        robot = VisualRobot(scene, directory / "wuji_wrist_fixture.urdf")
        obj = VisualRobot(scene, directory / f"marker{witness['marker']}.urdf", [.97, .32, .04, 1.])
        tabletop = VisualRobot(scene, table, [.54, .37, .24, 1.])
        tabletop.update([0., 0., .475, 0., 0., 0., 1.])
        for eye, intensity in (([.8, -.5, 1.8], 2.8), ([-.5, .5, 1.2], 1.5)):
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                      pose=camera_pose(eye, [0., 0., .6]))
        camera = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4., znear=.005, zfar=5.),
                           pose=camera_pose([.24, -.36, .78], [0., 0., .59]))
        scenes.append((scene, robot, obj))
        traces.append(json.loads((directory / f"{key}_trace.json").read_text()))
    renderer = pyrender.OffscreenRenderer(640, 540)
    snapshots = {}
    writer = imageio.get_writer(str(directory / "three_dof_grasp.mp4"), fps=10,
                               codec="libx264", quality=8, macro_block_size=2)
    try:
        for index in range(min(map(len, traces))):
            canvas = Image.new("RGB", (1280, 650), (245, 247, 250))
            draw = ImageDraw.Draw(canvas)
            for marker, ((scene, robot, obj), trace) in enumerate(zip(scenes, traces)):
                frame = trace[index]
                robot.update(xyzw(frame["robot_base_pose"]),
                             dict(zip(frame["robot_joint_names"], frame["robot_joint_pos"])))
                obj.update(xyzw(frame["object_state"][:7]))
                color, _ = renderer.render(scene)
                canvas.paste(Image.fromarray(color), (640 * marker, 70))
                diameter = 17 if marker == 0 else 28
                draw.text((640 * marker + 18, 14), f"3 commands | {diameter} mm marker", font=font(24, True), fill=(25, 38, 55))
                draw.text((640 * marker + 18, 45), "Controlled wrist fixture; recorded physical simulation", font=font(17), fill=(65, 75, 90))
                draw.text((640 * marker + 18, 615),
                          f"{frame['phase']} | t={frame['time_s']:.1f}s | object z={frame['object_state'][2]:.3f}m",
                          font=font(19), fill=(25, 38, 55))
            writer.append_data(np.asarray(canvas))
            phase = traces[0][index]["phase"]
            if phase in ("close", "hold", "release"):
                snapshots[phase] = canvas.copy()
    finally:
        writer.close()
        renderer.delete()
    for phase, canvas in snapshots.items():
        canvas.save(directory / f"{phase}.png")
    print(directory / "three_dof_grasp.mp4")


if __name__ == "__main__":
    main()
