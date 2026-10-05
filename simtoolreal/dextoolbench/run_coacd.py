"""CoACD convex decomposition of a mesh (offline, outside any simulator).

Parameters and QA mirror the Fabrica benchmark recipe
(threshold=0.03 for thin tools, fixed seed for reproducibility, Hausdorff
overshoot check). Used by ``generate_collision_meshes.py`` to pre-decompose
DexToolBench tool meshes into convex parts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import trimesh
import tyro


def run_coacd(
    mesh_path: Path,
    output_dir: Path,
    max_convex_hull: int = -1,
    threshold: float = 0.03,
    preprocess_resolution: int = 50,
    preprocess_mode: str = "auto",
    pca: bool = False,
    merge: bool = True,
    resolution: int = 3000,
    mcts_nodes: int = 25,
    mcts_iterations: int = 250,
    mcts_max_depth: int = 4,
    seed: int = 0,
    mode: Literal["subprocess", "python"] = "python",
) -> List[trimesh.Trimesh]:
    """Run CoACD on a mesh; write decomp_*.obj parts + a colored combined
    scene into output_dir; return the convex parts."""
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"
    assert mesh_path.suffix == ".obj", f"Mesh file {mesh_path} is not an OBJ file"

    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "subprocess":
        from subprocess import run as run_cmd

        output_mesh_path = output_dir / mesh_path.name
        cmd = f"coacd -i {mesh_path} -o {output_mesh_path} -c {max_convex_hull} -t {threshold}"
        print(f"Running command: {cmd}")
        run_cmd(cmd, shell=True, check=True)
        output_mesh = trimesh.load(output_mesh_path)
        parts = list(output_mesh.split())
    else:
        import coacd

        input_mesh = trimesh.load(mesh_path, force="mesh")
        coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
        convex_vs_fs_parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=preprocess_resolution,
            pca=pca,
            merge=merge,
            resolution=resolution,
            mcts_nodes=mcts_nodes,
            mcts_iterations=mcts_iterations,
            mcts_max_depth=mcts_max_depth,
            seed=seed,
        )
        parts = [trimesh.Trimesh(vs, fs) for vs, fs in convex_vs_fs_parts]

    # Colored combined scene for quick visual inspection.
    import numpy as np

    np.random.seed(0)
    scene = trimesh.Scene()
    for part in parts:
        part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(part)
    scene.export(output_dir / mesh_path.name)

    for i, part in enumerate(parts):
        part.export(output_dir / f"decomp_{i}.obj")
    print(f"Decomposition complete. {len(parts)} parts -> {output_dir}")
    return parts


def compute_overshoot(mesh_path: Path, output_dir: Path, n_samples: int = 5000) -> float:
    """Max distance (meters) the convex hulls extend beyond the original mesh —
    a QA metric for decomposition tightness."""
    import numpy as np

    orig = trimesh.load_mesh(str(mesh_path), process=False)
    decomp_files = sorted(output_dir.glob("decomp_*.obj"))
    if not decomp_files:
        return float("inf")
    samples_per = max(1, n_samples // len(decomp_files))
    hull_pts = np.concatenate([
        trimesh.sample.sample_surface(
            trimesh.load_mesh(str(f), process=False), samples_per
        )[0]
        for f in decomp_files
    ])
    _, dists, _ = trimesh.proximity.closest_point(orig, hull_pts)
    return float(dists.max())


@dataclass
class RunCoacdArgs:
    mesh_path: Path
    """Path to the input OBJ mesh file."""
    output_dir: Path
    """Directory to save the decomposed convex hull parts."""
    max_convex_hull: int = -1
    """Maximum number of convex hulls (-1 for unlimited)."""
    threshold: float = 0.03
    """Concavity threshold (lower = tighter fit, more parts)."""
    mode: Literal["subprocess", "python"] = "python"
    """'subprocess' calls the coacd CLI, 'python' uses the Python API."""


def main():
    args: RunCoacdArgs = tyro.cli(RunCoacdArgs)
    run_coacd(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        max_convex_hull=args.max_convex_hull,
        threshold=args.threshold,
        mode=args.mode,
    )
    overshoot = compute_overshoot(args.mesh_path, args.output_dir)
    print(f"Max hull overshoot: {overshoot * 1000:.2f}mm")


if __name__ == "__main__":
    main()
