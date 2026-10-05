"""Render recorded simulation poses with isolated OpenGL/EGL, without Isaac RTX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".render_deps"))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
from scipy.spatial.transform import Rotation
import trimesh


def origin(node):
    matrix = np.eye(4)
    if node is not None:
        matrix[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return matrix


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return matrix


def camera_pose(eye, target):
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, up, back))
    matrix[:3, 3] = eye
    return matrix


class VisualRobot:
    def __init__(self, scene, path, color=None):
        root = ET.parse(path).getroot()
        links = {node.get("name"): node for node in root.findall("link")}
        joints = root.findall("joint")
        children = {node.find("child").get("link") for node in joints}
        self.scene = scene
        self.root_name = (set(links) - children).pop()
        self.nodes = {self.root_name: pyrender.Node(matrix=np.eye(4))}
        scene.add_node(self.nodes[self.root_name])
        self.joints = {}
        pending = list(joints)
        while pending:
            progressed = False
            for joint in list(pending):
                parent = joint.find("parent").get("link")
                if parent not in self.nodes:
                    continue
                child = joint.find("child").get("link")
                transform = origin(joint.find("origin"))
                node = pyrender.Node(matrix=transform)
                self.nodes[child] = node
                scene.add_node(node, parent_node=self.nodes[parent])
                axis = joint.find("axis")
                vector = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
                vector /= np.linalg.norm(vector)
                self.joints[joint.get("name")] = (node, transform, vector, joint.get("type"))
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError(f"Disconnected URDF: {path}")
        materials = {node.get("name"): node for node in root.findall("material")}
        mesh_cache = {}
        self.triangles = 0
        for link_name, link in links.items():
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                mesh_node = geometry.find("mesh")
                if mesh_node is not None:
                    mesh_path = Path(mesh_node.get("filename"))
                    if not mesh_path.is_absolute():
                        mesh_path = path.parent / mesh_path
                    if mesh_path not in mesh_cache:
                        mesh_cache[mesh_path] = trimesh.load_mesh(mesh_path, process=False)
                    mesh = mesh_cache[mesh_path].copy()
                    mesh.apply_scale(np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" "))
                elif geometry.find("box") is not None:
                    mesh = trimesh.creation.box(extents=np.fromstring(geometry.find("box").get("size"), sep=" "))
                elif geometry.find("cylinder") is not None:
                    cylinder = geometry.find("cylinder")
                    mesh = trimesh.creation.cylinder(radius=float(cylinder.get("radius")),
                                                    height=float(cylinder.get("length")), sections=32)
                elif geometry.find("sphere") is not None:
                    mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(geometry.find("sphere").get("radius")))
                else:
                    raise ValueError(f"Unsupported visual in {path}")
                rgba = list(color) if color is not None else [.65, .68, .72, 1.]
                material_node = visual.find("material")
                if color is None and material_node is not None:
                    color_node = material_node.find("color")
                    if color_node is None and material_node.get("name") in materials:
                        color_node = materials[material_node.get("name")].find("color")
                    if color_node is not None:
                        rgba = np.fromstring(color_node.get("rgba"), sep=" ").tolist()
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba,
                    metallicFactor=.1, roughnessFactor=.65, doubleSided=True,
                    alphaMode="BLEND" if rgba[3] < 1 else "OPAQUE")
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
                scene.add(render_mesh, pose=origin(visual.find("origin")), parent_node=self.nodes[link_name])
                self.triangles += len(mesh.faces)

    def update(self, pose, positions=None):
        self.scene.set_pose(self.nodes[self.root_name], pose_matrix(pose))
        for name, value in (positions or {}).items():
            if name not in self.joints:
                continue
            node, base, axis, kind = self.joints[name]
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif kind == "prismatic":
                motion[:3, 3] = axis * value
            self.scene.set_pose(node, base @ motion)


class PolicyScene:
    def __init__(self, directory, report):
        self.scene = pyrender.Scene(bg_color=[.96, .97, .985, 1.], ambient_light=[.35, .35, .35])
        self.robot = VisualRobot(self.scene, Path(report["robot_urdf"]))
        self.table = VisualRobot(self.scene, directory / "table.urdf", [.56, .36, .20, 1.])
        self.tool = VisualRobot(self.scene, directory / "object.urdf", [.98, .36, .045, 1.])
        self.goal = VisualRobot(self.scene, directory / "object.urdf", [.15, .72, .32, .65])
        floor = trimesh.creation.box(extents=[4., 4., .015])
        floor_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.84, .88, .93, 1.], roughnessFactor=1.)
        floor_pose = np.eye(4)
        floor_pose[2, 3] = -.01
        self.scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_material), pose=floor_pose)
        for eye, intensity in [([1., -2., 3.], 2.8), ([-2., .5, 2.], 1.2)]:
            self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                           pose=camera_pose(eye, [0., .3, .5]))
        self.camera = self.scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4, znear=.01, zfar=20.))

    def update(self, frame):
        self.robot.update(frame["robot_base_pose"], dict(zip(frame["robot_joint_names"], frame["robot_joint_pos"])))
        self.table.update(frame["table_pose"])
        self.tool.update(frame["object_pose"])
        self.goal.update(frame["goal_pose"])

    def render(self, renderer, eye, target):
        self.scene.set_pose(self.camera, camera_pose(eye, target))
        color, depth = renderer.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        if not (depth > 0).any():
            raise RuntimeError("No geometry visible in rendered scene")
        return Image.fromarray(color)


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def policy_canvas(label, report, frame, full, close, trace):
    canvas = Image.new("RGB", (960, 720), "#f4f7fb")
    canvas.paste(full, (10, 94))
    canvas.paste(close, (560, 94))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), f"{label.upper()}  |  best checkpoint {report['checkpoint_epoch']:,}", fill="#15273c", font=font(27, True))
    draw.text((24, 57), f"Best training return: {report['best_training_mean_reward']:.1f}  |  deterministic rollout", fill="#526276", font=font(17))
    draw.text((27, 100), "FULL ROBOT", fill="#53647a", font=font(14, True))
    draw.text((575, 100), "HAND / TOOL VIEW", fill="#53647a", font=font(14, True))
    elapsed = frame["time_s"]
    visible_trace = [item for item in trace if item["time_s"] <= elapsed + 1e-6]
    lifts = sum(item["lift_event"] for item in visible_trace)
    goals = sum(item["success"] for item in visible_trace)
    draw.rounded_rectangle((560, 467, 950, 659), radius=12, fill="white", outline="#d8e1ec")
    rows = [("Time", f"{elapsed:.2f} s"), ("Tool height", f"{frame['object_pose'][2] * 100:.1f} cm"),
            ("Goal pose error", f"{frame['keypoint_error_m'] * 100:.1f} cm"),
            ("Lift bonuses / goal events", f"{lifts} / {goals}")]
    for index, (key, value) in enumerate(rows):
        y = 483 + index * 40
        draw.text((578, y), key, fill="#526276", font=font(16))
        draw.text((935, y), value, fill="#17283b", font=font(17, True), anchor="ra")
    draw.ellipse((28, 682, 42, 696), fill="#f77913")
    draw.text((51, 678), "Physical tool", fill="#334359", font=font(16))
    draw.ellipse((210, 682, 224, 696), fill="#2ebd5a")
    draw.text((233, 678), "Target pose", fill="#334359", font=font(16))
    draw.text((940, 681), "Seed 42 · env 0 · resets included", fill="#607188", font=font(14), anchor="ra")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    reports = json.loads((out / "evaluation.json").read_text())
    labels = ["mlp", "lstm"]
    frames = {label: json.loads((out / label / "frames.json").read_text()) for label in labels}
    traces = {label: json.loads((out / label / "trace.json").read_text()) for label in labels}
    scenes = {label: PolicyScene(out / label, reports[label]) for label in labels}
    renderer = pyrender.OffscreenRenderer(540, 560)
    fps = 1 / reports["mlp"]["frame_timestep"]
    writer_options = dict(fps=fps, codec="libx264", quality=8, macro_block_size=None,
                          output_params=["-movflags", "+faststart", "-threads", "2"])
    writers = {label: imageio.get_writer(str(out / f"{label}_best.mp4"), **writer_options) for label in labels}
    comparison = imageio.get_writer(str(out / "comparison.mp4"), **writer_options)
    gif_frames = []
    count = min(len(frames[label]) for label in labels)
    # Show the lift interaction rather than an initial reset pose in the still preview.
    preview_indices = {}
    for label in labels:
        crossings = [item["time_s"] for item in traces[label] if item["lift_event"]]
        preview_time = crossings[0] + .15 if crossings else 2.
        preview_indices[label] = min(range(count), key=lambda index: abs(frames[label][index]["time_s"] - preview_time))
    stills = {}
    for index in range(count):
        canvases = []
        for label in labels:
            scene, frame = scenes[label], frames[label][index]
            scene.update(frame)
            renderer.viewport_width, renderer.viewport_height = 540, 560
            full = scene.render(renderer, [-1.1, -1.7, 1.25], [0., .32, .70])
            renderer.viewport_width, renderer.viewport_height = 390, 360
            close = scene.render(renderer, [-.67, -.53, .94], [-.18, .31, .64])
            canvas = policy_canvas(label, reports[label], frame, full, close, traces[label])
            writers[label].append_data(np.asarray(canvas))
            canvases.append(canvas)
            if index == preview_indices[label]:
                stills[label] = canvas.copy()
                canvas.save(out / f"{label}_preview.png")
        combined = Image.new("RGB", (1920, 720))
        combined.paste(canvases[0], (0, 0))
        combined.paste(canvases[1], (960, 0))
        comparison.append_data(np.asarray(combined))
        gif_frames.append(combined.resize((1120, 420), Image.Resampling.LANCZOS).quantize(colors=128))
        if index % 20 == 0:
            print(f"RENDER {index}/{count}", flush=True)
    for writer in writers.values():
        writer.close()
    comparison.close()
    renderer.delete()
    gif_frames[0].save(out / "comparison.gif", save_all=True, append_images=gif_frames[1:],
                       duration=round(1000 / fps), loop=0, optimize=True)
    overview = Image.new("RGB", (1920, 720))
    overview.paste(stills["mlp"], (0, 0))
    overview.paste(stills["lstm"], (960, 0))
    overview.save(out / "comparison.png")
    metadata = {"renderer": "URDF visual meshes in isolated OpenGL/EGL at recorded simulation poses",
                "frames": count, "fps": fps, "duration_s": count / fps,
                "preview_frame_indices": preview_indices,
                "robot_visual_triangles": {label: scenes[label].robot.triangles for label in labels},
                "videos": {label: str(out / f"{label}_best.mp4") for label in labels},
                "comparison_video": str(out / "comparison.mp4"), "comparison_gif": str(out / "comparison.gif")}
    (out / "rendering.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("RENDER_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
