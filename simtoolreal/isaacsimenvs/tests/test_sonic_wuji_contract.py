"""Compare batched torch SONIC to the official ONNX and check hand semantics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from simtoolreal_shared.sonic import (
    FSQ_LEVELS,
    FrozenSonicDecoder,
    quantize_sonic_latent,
    standing_encoder_input,
)
from simtoolreal_shared.wuji_synergy import synergy_targets, wuji_joint_names


def check_fsq_grid(model_dir: Path, options: ort.SessionOptions) -> None:
    """Use the published encoder to independently establish the native grid."""
    encoder_path = model_dir / "model_encoder.onnx"
    encoder = onnx.load(str(encoder_path))
    constants = {
        node.name: numpy_helper.to_array(node.attribute[0].t)
        for node in encoder.graph.node
        if node.op_type == "Constant" and node.name.startswith("/quantizer/")
    }
    bound = constants["/quantizer/Constant_2"]
    offset = constants["/quantizer/Constant_3"]
    divisor = constants["/quantizer/Constant_4"]
    assert np.unique(divisor).size == 1
    half_width = float(divisor[0])
    lower_code = int(np.rint(-bound - offset).min())
    upper_code = int(np.rint(bound - offset).max())
    grid = np.arange(lower_code, upper_code + 1, dtype=np.float32) / half_width
    assert len(grid) == FSQ_LEVELS == 32 and half_width == 16.0
    assert grid[0] == -1.0 and grid[-1] == 0.9375

    # Include saturating values and midpoint ties as well as every native code.
    samples = np.linspace(-2.0, 2.0, 4097, dtype=np.float32)
    normalized = torch.from_numpy(samples[:, None].repeat(64, axis=1))
    quantized = quantize_sonic_latent(normalized)
    torch.testing.assert_close(
        quantized.unique(), torch.from_numpy(grid), rtol=0, atol=0
    )
    actual = quantized[:, 0].double().numpy()
    distances = (samples.astype(np.float64)[:, None] - grid.astype(np.float64)) ** 2
    np.testing.assert_allclose(
        (actual - samples.astype(np.float64)) ** 2,
        distances.min(axis=1),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        quantize_sonic_latent(quantized), quantized, rtol=0, atol=0
    )
    torch.testing.assert_close(
        normalized[:, 0], torch.from_numpy(samples), rtol=0, atol=0
    )
    ties = torch.tensor([[-0.09375, -0.03125, 0.03125, 0.09375] * 16])
    expected_ties = torch.tensor([[-0.125, 0.0, 0.0, 0.125] * 16])
    torch.testing.assert_close(
        quantize_sonic_latent(ties), expected_ties, rtol=0, atol=0
    )

    # Actual exported encoder tokens must survive the projection unchanged.
    session = ort.InferenceSession(
        str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    native_tokens = torch.from_numpy(
        session.run(None, {"obs_dict": standing_encoder_input(np.zeros(29))})[0]
    )
    torch.testing.assert_close(
        quantize_sonic_latent(native_tokens), native_tokens, rtol=0, atol=0
    )
    print("[contract] native SONIC FSQ grid: 32 levels, step=1/16, range=[-1, 0.9375]")


def main():
    repo = Path(__file__).resolve().parents[2]
    model = repo.parent / "models/GEAR-SONIC/sonic_v1_1/model_decoder.onnx"
    torch.set_num_threads(2)
    decoder = FrozenSonicDecoder(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    check_fsq_grid(model.parent, options)
    official = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 0.2, (5, 994)).astype(np.float32)
    values[:, :64] = quantize_sonic_latent(torch.from_numpy(values[:, :64])).numpy()
    reference = np.concatenate(
        [official.run(None, {"obs_dict": sample[None]})[0] for sample in values]
    )
    with torch.no_grad():
        actual = decoder(
            torch.from_numpy(values[:, :64]), torch.from_numpy(values[:, 64:])
        ).numpy()
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-5)
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    print(
        f"[contract] torch batch vs official SONIC ONNX: max error {np.max(np.abs(actual - reference)):.3g}"
    )

    root = ET.parse(repo / "assets/urdf/g1_wuji/g1_wuji.urdf").getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    names = wuji_joint_names("right")
    lower = torch.tensor([float(joints[n].find("limit").get("lower")) for n in names])
    upper = torch.tensor([float(joints[n].find("limit").get("upper")) for n in names])
    commands = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    targets = synergy_targets(commands, lower, upper)
    assert targets.shape == (5, 20)
    assert torch.all(targets >= lower) and torch.all(targets <= upper)
    changed = [
        (targets[i] - targets[0]).abs().gt(1e-6).nonzero().flatten().tolist()
        for i in range(1, 4)
    ]
    assert changed == [[0], [1, 2, 3], [4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]], (
        changed
    )
    assert torch.all(targets[:, [5, 9, 13, 17]] == 0.0), (
        "non-thumb abduction must remain neutral"
    )
    print(
        "[contract] thumb rotation, thumb bend and shared finger bend affect exactly the intended joints"
    )


if __name__ == "__main__":
    main()
