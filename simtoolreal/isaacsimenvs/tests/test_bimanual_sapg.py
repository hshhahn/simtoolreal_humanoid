"""CPU checks for both Wuji action heads in the actual SAPG distribution."""

import importlib.util
from pathlib import Path
import unittest

import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder, register_network

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "bimanual_network", REPO / "isaacsimenvs/tasks/g1_wuji_sonic/network.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
register_network("sonic_actor_critic", lambda **kwargs: module.SonicPolicyBuilder(**kwargs))


class BimanualSAPGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        params = yaml.safe_load(
            (REPO / "isaacsimenvs/cfg/train/G1WujiSonicSAPG.yaml").read_text()
        )["params"]
        self.mean = torch.cat((
            torch.linspace(-1.0, 0.9375, 64),
            torch.tensor([-0.5, -0.3, -0.1, 0.2, 0.4, 0.6]),
        ))
        params["network"].update(
            initial_mean=self.mean.tolist(), hand_std_init=0.5, bounded_hand_mean=True,
        )
        self.ids = torch.linspace(50.0, 0.0, 6)
        self.model = ModelBuilder().load(params).build({
            "actions_num": 70, "input_shape": (458,), "num_seqs": 6,
            "value_size": 1, "normalize_value": True, "normalize_input": True,
            "type": "extra_param", "coef_ids": self.ids, "coef_id_idx": 457,
        })
        # Compare rollout/training likelihoods with the same normalization
        # statistics. eval() freezes these buffers and still permits gradients.
        self.model.eval()
        self.obs = torch.randn(96, 458)
        self.obs[:, 457] = self.ids.repeat_interleave(16)
        self.inputs = {
            "obs": self.obs, "rnn_states": self.model.get_default_rnn_state(),
            "seq_length": 16,
        }

    def test_initial_means_and_hand_noise_cover_all_groups(self):
        self.assertEqual(tuple(self.model.a2c_network.sigma.shape), (6, 70))
        with torch.no_grad():
            self.model.a2c_network.mu.weight.zero_()
            result = self.model({**self.inputs, "is_train": False})
        torch.testing.assert_close(result["mus"], self.mean.expand(96, -1))
        torch.testing.assert_close(result["sigmas"][:, 64:], torch.full((96, 6), 0.5))
        self.assertEqual(result["actions"].shape, (96, 70))

    def test_both_hands_contribute_finite_ppo_gradients(self):
        with torch.no_grad():
            sampled = self.model({**self.inputs, "is_train": False})
        trained = self.model({
            **self.inputs, "is_train": True, "prev_actions": sampled["actions"],
        })
        torch.testing.assert_close(trained["prev_neglogp"], sampled["neglogpacs"])
        (trained["prev_neglogp"].mean() - 0.01 * trained["entropy"].mean()).backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        sigma_gradient = self.model.a2c_network.sigma.grad[:, 64:]
        self.assertTrue((sigma_gradient.abs().sum(dim=0) > 0).all())
        self.assertTrue((sigma_gradient.abs().sum(dim=1) > 0).all())
        self.assertTrue((self.model.a2c_network.mu.bias.grad[64:].abs() > 0).all())

    def test_both_hand_means_are_bounded_without_changing_body(self):
        with torch.no_grad():
            self.model.a2c_network.mu.weight.zero_()
            self.model.a2c_network.mu.bias[64:] = torch.tensor([-5., 5., -3., 3., -8., 8.])
            result = self.model({**self.inputs, "is_train": False})
        self.assertTrue((result["mus"][:, 64:].abs() <= 1).all())
        torch.testing.assert_close(result["mus"][:, :64], self.mean[:64].expand(96, -1))
        self.assertFalse(torch.allclose(result["mus"][:, 64:67], result["mus"][:, 67:70]))


if __name__ == "__main__":
    unittest.main()
