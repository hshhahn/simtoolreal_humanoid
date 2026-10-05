"""Frozen, batched SONIC 1.1 inference from NVIDIA's published ONNX weights.

The decoder is an MLP with SiLU activations. Loading its existing weights
into torch avoids a separate CPU inference call for every simulated robot.
No SONIC weights are trained by the task policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


LATENT_DIM = 64
FSQ_LEVELS = 32
FSQ_HALF_WIDTH = FSQ_LEVELS // 2
FSQ_TOKEN_MIN = -1.0
FSQ_TOKEN_MAX = (FSQ_HALF_WIDTH - 1) / FSQ_HALF_WIDTH
FSQ_TOKEN_STEP = 1.0 / FSQ_HALF_WIDTH
BODY_DIM = 29
HISTORY_LENGTH = 10
PROPRIO_DIM = 930
DECODER_DIM = LATENT_DIM + PROPRIO_DIM
ENCODER_DIM = 1751
HF_REPO = "nvidia/GEAR-SONIC"
HF_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"


def quantize_sonic_latent(latent: torch.Tensor) -> torch.Tensor:
    """Snap normalized decoder-space predictions to SONIC 1.1's FSQ grid.

    The released encoder uses 32 levels per coordinate: integer codes
    -16..15 divided by 16. Policy outputs remain continuous in [-1, 1];
    the decoder receives only these native levels, including in residual mode.
    This projects token coordinates directly, without applying the encoder's
    tanh bounding transform again. PPO keeps the continuous sampled actions.
    """
    if latent.shape[-1] != LATENT_DIM:
        raise ValueError("SONIC quantization requires 64 latent coordinates")
    codes = (latent.clamp(FSQ_TOKEN_MIN, FSQ_TOKEN_MAX) * FSQ_HALF_WIDTH).round()
    return codes / FSQ_HALF_WIDTH


class FrozenSonicDecoder(nn.Module):
    def __init__(self, onnx_path: str | Path, device: str | torch.device = "cpu"):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SONIC 1.1 decoder is missing: {path}. Run setup_g1_wuji_sonic.sh."
            )
        model = onnx.load(str(path))
        input_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        output_dim = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        if (input_dim, output_dim) != (DECODER_DIM, BODY_DIM):
            raise ValueError(
                f"Expected SONIC 1.1 dimensions 994 -> 29, got {input_dim} -> {output_dim}"
            )
        initializers = {
            x.name: numpy_helper.to_array(x) for x in model.graph.initializer
        }
        matmuls = [x for x in model.graph.node if x.op_type == "MatMul"]
        if (
            len(matmuls) != 9
            or sum(x.op_type == "Sigmoid" for x in model.graph.node) != 8
        ):
            raise ValueError("Unexpected SONIC 1.1 decoder architecture")
        layers = []
        for index, node in enumerate(matmuls):
            weight = initializers[node.input[1]]
            bias_name = f"module.decoders.g1_dyn.module.{2 * index}.bias"
            bias = initializers[bias_name]
            layer = nn.Linear(weight.shape[0], weight.shape[1])
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight.copy().T))
                layer.bias.copy_(torch.from_numpy(bias.copy()))
            layers.append(layer)
            if index < len(matmuls) - 1:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers).to(device).eval()
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self, latent: torch.Tensor, proprioception: torch.Tensor
    ) -> torch.Tensor:
        if latent.shape[-1] != LATENT_DIM or proprioception.shape[-1] != PROPRIO_DIM:
            raise ValueError(
                "SONIC decoder requires 64 latent and 930 proprioception values"
            )
        return self.network(torch.cat((latent, proprioception), dim=-1))


def standing_encoder_input(joint_pos_sonic_order: np.ndarray) -> np.ndarray:
    """Build the released deployment encoder's G1-mode reference input.

    One mode index followed by three one-hot values, ten future joint
    positions, ten velocities, and ten heading-normalized 6D orientations.
    Unused teleop/SMPL fields are zero. The reference is a standing pose.
    """
    q = np.asarray(joint_pos_sonic_order, dtype=np.float32)
    if q.shape != (BODY_DIM,):
        raise ValueError("Standing reference must have 29 joints in SONIC order")
    obs = np.zeros((1, ENCODER_DIM), dtype=np.float32)
    obs[0, :4] = (0.0, 1.0, 0.0, 0.0)
    obs[0, 4:294] = np.tile(q, HISTORY_LENGTH)
    # 6D orientation is the first two columns, row-major: [1, 0, 0, 1, 0, 0].
    identity_6d = np.array((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    obs[0, 584:644] = np.tile(identity_6d, HISTORY_LENGTH)
    obs[0, 644:650] = identity_6d
    return obs


def encode_standing(
    onnx_path: str | Path, joint_pos_sonic_order: np.ndarray
) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokens = session.run(
        None, {"obs_dict": standing_encoder_input(joint_pos_sonic_order)}
    )[0]
    if tokens.shape != (1, LATENT_DIM) or not np.isfinite(tokens).all():
        raise ValueError("Invalid SONIC 1.1 standing latent")
    return tokens[0]
