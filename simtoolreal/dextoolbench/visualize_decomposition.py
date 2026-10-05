"""Viser viewer for DexToolBench CoACD collision decompositions.

Shows each tool's original visual mesh (translucent) with its convex
collision hulls overlaid in distinct colors, so you can confirm the
decomposition produced by ``generate_collision_meshes.py`` hugs the tool.

    python dextoolbench/visualize_decomposition.py
    python dextoolbench/visualize_decomposition.py --object_name claw_hammer --port 8082

Use the dropdown to switch tools; toggles/slider control the overlay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser

from dextoolbench.objects import NAME_TO_OBJECT

HULL_COLORS = [
    (0.90, 0.20, 0.20), (0.20, 0.70, 0.20), (0.20, 0.30, 0.90),
    (0.90, 0.70, 0.10), (0.80, 0.30, 0.80), (0.10, 0.80, 0.80),
    (0.90, 0.50, 0.10), (0.50, 0.50, 0.50), (0.60, 0.20, 0.60),
    (0.30, 0.90, 0.50), (0.90, 0.30, 0.50), (0.40, 0.60, 0.90),
    (0.70, 0.90, 0.20), (0.90, 0.40, 0.70), (0.20, 0.50, 0.70),
    (0.80, 0.60, 0.30), (0.50, 0.30, 0.90), (0.30, 0.80, 0.30),
]


def _paths(object_name: str) -> tuple[Path, Path]:
    """Return (visual mesh, collision-decomp dir) for an object."""
    urdf = NAME_TO_OBJECT[object_name].urdf_path
    visual = urdf.parent / f"{urdf.stem}.obj"
    decomp_dir = urdf.parent / f"{urdf.stem}_collision"
    return visual, decomp_dir


@dataclass
class Args:
    object_name: str | None = None
    """Initial tool to show (default: first object)."""
    port: int = 8082
    """Viser server port."""


def main() -> None:
    args = tyro.cli(Args)

    # Only tools that have been decomposed.
    names = [n for n in NAME_TO_OBJECT if _paths(n)[1].exists() and
             list(_paths(n)[1].glob("decomp_*.obj"))]
    if not names:
        print("No CoACD decompositions found. Run generate_collision_meshes.py first.")
        return
    initial = args.object_name if args.object_name in names else names[0]

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    handles: list = []

    dropdown = server.gui.add_dropdown("Tool", options=names, initial_value=initial)
    show_original = server.gui.add_checkbox("Show original mesh", initial_value=True)
    show_hulls = server.gui.add_checkbox("Show CoACD hulls", initial_value=True)
    original_opacity = server.gui.add_slider(
        "Original opacity", min=0.05, max=1.0, step=0.05, initial_value=0.35
    )
    stats = server.gui.add_markdown("**Stats:** loading...")

    def load(object_name: str) -> None:
        for h in handles:
            h.remove()
        handles.clear()
        visual_path, decomp_dir = _paths(object_name)
        decomp_files = sorted(decomp_dir.glob("decomp_*.obj"))

        if show_original.value and visual_path.exists():
            m = trimesh.load(visual_path, force="mesh")
            handles.append(server.scene.add_mesh_simple(
                f"/original/{object_name}",
                vertices=np.asarray(m.vertices, np.float32),
                faces=np.asarray(m.faces, np.int32),
                color=(0.7, 0.7, 0.7),
                opacity=original_opacity.value,
            ))

        verts = faces = 0
        if show_hulls.value:
            for i, f in enumerate(decomp_files):
                h = trimesh.load(f, force="mesh")
                verts += len(h.vertices); faces += len(h.faces)
                handles.append(server.scene.add_mesh_simple(
                    f"/hulls/{object_name}/decomp_{i}",
                    vertices=np.asarray(h.vertices, np.float32),
                    faces=np.asarray(h.faces, np.int32),
                    color=HULL_COLORS[i % len(HULL_COLORS)],
                    opacity=0.85,
                ))
        stats.content = (
            f"**{object_name}**\n\n"
            f"- Convex hulls: **{len(decomp_files)}**\n"
            f"- Hull totals: {verts} verts, {faces} faces"
        )

    for ctrl in (dropdown, show_original, show_hulls, original_opacity):
        ctrl.on_update(lambda _evt: load(dropdown.value))
    load(initial)

    print(f"Viser decomposition viewer at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer.")


if __name__ == "__main__":
    main()
