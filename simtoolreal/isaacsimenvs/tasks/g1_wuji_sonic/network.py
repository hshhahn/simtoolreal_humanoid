"""Task PPO head initialization; the frozen SONIC decoder stays in the env."""

import math

import torch
from rl_games.algos_torch.network_builder import A2CBuilder


class SonicPolicyBuilder(A2CBuilder):
    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.bounded_hand_mean = params.get("bounded_hand_mean", False)
            if self.bounded_hand_mean and self.mu.out_features != 67:
                raise ValueError("Bounded Wuji means require 64 body + 3 hand actions")

        def forward(self, obs_dict):
            mean, log_std, value, states = super().forward(obs_dict)
            if self.bounded_hand_mean:
                # Bound the Gaussian mean, not PPO's sampled actions or log
                # probabilities. The env still clips samples and applies FSQ
                # only to the 64 body commands, as before.
                mean = torch.cat((mean[..., :64], mean[..., 64:].tanh()), dim=-1)
            return mean, log_std, value, states

    def build(self, name, **kwargs):
        network = self.Network(self.params, **kwargs)
        initial_mean = self.params.get("initial_mean")
        if initial_mean is not None:
            if len(initial_mean) != kwargs["actions_num"]:
                raise ValueError("SONIC policy initial mean must have 67 outputs")
            with torch.no_grad():
                network.mu.weight.normal_(mean=0.0, std=0.001)
                bias = torch.tensor(initial_mean, dtype=network.mu.bias.dtype)
                if network.bounded_hand_mean:
                    bias[64:] = torch.atanh(bias[64:].clamp(-0.999999, 0.999999))
                network.mu.bias.copy_(bias)
        hand_std = self.params.get("hand_std_init")
        if hand_std is not None:
            if not math.isfinite(hand_std) or hand_std <= 0:
                raise ValueError("hand_std_init must be finite and positive")
            if kwargs["actions_num"] != 67 or network.sigma.shape != (67,):
                raise ValueError("Wuji exploration requires 67 state-independent log-stds")
            with torch.no_grad():
                network.sigma[64:] = math.log(hand_std)
        return network
