"""Verify recurrent SONIC policy gradients and episode-boundary memory resets."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder, register_network


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "sonic_lstm_network", REPO / "isaacsimenvs/tasks/g1_wuji_sonic/network.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
register_network("sonic_actor_critic", lambda **kwargs: module.SonicPolicyBuilder(**kwargs))


class SonicLSTMPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.params = yaml.safe_load(
            (REPO / "isaacsimenvs/cfg/train/G1WujiSonicLSTMPPO.yaml").read_text()
        )["params"]
        base = yaml.safe_load(
            (REPO / "isaacsimenvs/cfg/train/G1WujiSonicPPO.yaml").read_text()
        )["params"]
        comparison = copy.deepcopy(self.params)
        comparison["network"].pop("rnn")
        comparison["config"]["name"] = base["config"]["name"]
        comparison["config"]["seq_length"] = base["config"]["seq_length"]
        comparison["config"].pop("zero_rnn_on_done")
        self.assertEqual(comparison, base)
        self.params["network"].update(
            initial_mean=[0.0] * 64 + [-0.5] * 3,
            hand_std_init=0.5,
            bounded_hand_mean=True,
        )
        self.model = ModelBuilder().load(self.params).build({
            "actions_num": 67, "input_shape": (432,), "num_seqs": 2,
            "value_size": 1, "normalize_value": False, "normalize_input": False,
        })

    def test_sequence_training_keeps_action_contract_and_finite_gradients(self):
        self.assertTrue(self.model.is_rnn())
        states = self.model.get_default_rnn_state()
        self.assertEqual([tuple(s.shape) for s in states], [(1, 2, 1024)] * 2)
        obs = torch.randn(32, 432)
        dones = torch.zeros(32)
        dones[7] = 1
        inputs = {"obs": obs, "rnn_states": states, "seq_length": 16, "dones": dones}
        with torch.no_grad():
            sampled = self.model({**inputs, "is_train": False})
        self.assertEqual(sampled["actions"].shape, (32, 67))
        self.assertTrue(torch.all(sampled["mus"][:, 64:].abs() <= 1))
        torch.testing.assert_close(sampled["sigmas"][:, 64:], torch.full((32, 3), 0.5))
        trained = self.model({
            **inputs, "is_train": True, "prev_actions": sampled["actions"],
        })
        torch.testing.assert_close(trained["prev_neglogp"], sampled["neglogpacs"])
        trained["prev_neglogp"].mean().backward()
        rnn_gradients = [p.grad for p in self.model.a2c_network.rnn.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in rnn_gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in rnn_gradients), 0)

    def test_done_resets_only_the_terminated_environment_memory(self):
        obs = torch.randn(2, 432)
        states = tuple(torch.randn_like(s) for s in self.model.get_default_rnn_state())
        reset_states = tuple(s.clone() for s in states)
        for s in reset_states:
            s[:, 0] = 0
        common = {"obs": obs, "is_train": False, "seq_length": 1}
        with torch.no_grad():
            reset = self.model({**common, "rnn_states": states, "dones": torch.tensor([1., 0.])})
            expected = self.model({**common, "rnn_states": reset_states})
            continued = self.model({**common, "rnn_states": states})
        torch.testing.assert_close(reset["mus"], expected["mus"])
        for actual, manual, previous in zip(
            reset["rnn_states"], expected["rnn_states"], continued["rnn_states"]
        ):
            torch.testing.assert_close(actual, manual)
            torch.testing.assert_close(actual[:, 1], previous[:, 1])
            self.assertFalse(torch.allclose(actual[:, 0], previous[:, 0]))


if __name__ == "__main__":
    unittest.main()
