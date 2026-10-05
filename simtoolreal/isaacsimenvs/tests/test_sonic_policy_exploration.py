"""Check actual PPO distributions, normalized hand means and decoder independence."""

import copy
import importlib.util
import math
from pathlib import Path
import unittest

import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder, register_network


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "sonic_policy_network", REPO / "isaacsimenvs/tasks/g1_wuji_sonic/network.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
register_network("sonic_actor_critic", lambda **kwargs: module.SonicPolicyBuilder(**kwargs))


class SonicPolicyExplorationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.params = yaml.safe_load(
            (REPO / "isaacsimenvs/cfg/train/G1WujiSonicPPO.yaml").read_text()
        )["params"]
        # Includes the FSQ endpoints; body initialization must remain literal.
        self.initial_mean = torch.cat((torch.linspace(-1, 0.9375, 64), torch.full((3,), -0.5)))

    def build(self, improved=True):
        params = copy.deepcopy(self.params)
        params["network"]["initial_mean"] = self.initial_mean.tolist()
        if improved:
            params["network"].update(hand_std_init=0.5, bounded_hand_mean=True)
        return ModelBuilder().load(params).build({
            "actions_num": 67, "input_shape": (432,), "num_seqs": 1,
            "value_size": 1, "normalize_value": False, "normalize_input": False,
        })

    def test_independent_noise_and_initial_mean(self):
        model = self.build().eval()
        # Remove the small observation-dependent head variation for exact bias checks.
        with torch.no_grad():
            model.a2c_network.mu.weight.zero_()
            result = model({"obs": torch.zeros(8192, 432), "is_train": False})
        torch.testing.assert_close(result["mus"][0], self.initial_mean)
        torch.testing.assert_close(result["sigmas"][:, :64], torch.full((8192, 64), math.exp(-2)))
        torch.testing.assert_close(result["sigmas"][:, 64:], torch.full((8192, 3), 0.5))
        hand = result["actions"][:, 64:].clamp(-1, 1)
        # Explore both open and substantial closure, rather than a clipped open constant.
        assert (hand < -0.9).any(dim=0).all()
        assert (hand > 0.2).any(dim=0).all()

    def test_hand_means_stay_bounded_and_ppo_gradients_are_finite(self):
        model = self.build()
        with torch.no_grad():
            model.a2c_network.mu.weight.zero_()
            model.a2c_network.mu.bias[64:] = torch.tensor([-5.0, 5.0, -2.0])
        obs = torch.randn(16, 432)
        sampled = model({"obs": obs, "is_train": False})
        assert torch.all(sampled["mus"][:, 64:].abs() <= 1)
        torch.testing.assert_close(sampled["mus"][0, :64], self.initial_mean[:64])
        trained = model({"obs": obs, "is_train": True, "prev_actions": sampled["actions"].detach()})
        torch.testing.assert_close(trained["prev_neglogp"], sampled["neglogpacs"])
        loss = trained["prev_neglogp"].mean() - 0.01 * trained["entropy"].mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    def test_old_configuration_keeps_original_distribution(self):
        legacy = self.build(improved=False)
        with torch.no_grad():
            legacy.a2c_network.mu.weight.zero_()
            legacy.a2c_network.mu.bias[64:] = -2.0
            output = legacy({"obs": torch.zeros(1, 432), "is_train": False})
        torch.testing.assert_close(output["mus"][0, 64:], torch.full((3,), -2.0))
        torch.testing.assert_close(output["sigmas"][0], torch.full((67,), math.exp(-2)))
        assert legacy.state_dict().keys() == self.build().state_dict().keys()


if __name__ == "__main__":
    unittest.main()
