"""CPU regression tests for catastrophic velocity and nonfinite transitions."""

import unittest

import torch

from simtoolreal_shared.physics_guard import invalid_physics_mask, mask_invalid_reward_terms


class PhysicsGuardTests(unittest.TestCase):
    def test_finite_velocity_outlier_is_rejected_per_environment(self):
        velocity = torch.tensor([[20.0, -8.0], [1e11, 0.0], [0.0, 136.0]])
        limits = torch.tensor([37.0, 13.578])
        roots = torch.zeros(3, 13)
        mask = invalid_physics_mask(velocity, limits, (roots,), 10.0)
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_nonfinite_velocity_or_body_state_is_rejected(self):
        velocity = torch.zeros(4, 2)
        velocity[0, 0] = float('nan')
        root = torch.zeros(4, 13)
        root[1, 0] = float('inf')
        body = torch.zeros(4, 3, 13)
        body[2, 2, 6] = float('-inf')
        mask = invalid_physics_mask(velocity, torch.ones(2), (root, body), 10.0)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_invalid_rewards_are_finite_and_valid_values_are_unchanged(self):
        terms = {
            'velocity_penalty': torch.tensor([-1.59e9, float('nan'), -1.25]),
            'tool_task_reward': torch.tensor([-1.59e9, float('nan'), 12.5]),
            'total_reward': torch.tensor([-1.59e9, float('nan'), 12.25]),
        }
        original = {name: value.clone() for name, value in terms.items()}
        result = mask_invalid_reward_terms(terms, torch.tensor([True, True, False]), 5.0)
        torch.testing.assert_close(result['total_reward'], torch.tensor([-5., -5., 12.25]))
        torch.testing.assert_close(result['velocity_penalty'], torch.tensor([0., 0., -1.25]))
        for name, value in result.items():
            self.assertTrue(torch.isfinite(value).all())
            if name in original:
                torch.testing.assert_close(value[2], original[name][2], rtol=0, atol=0)
        self.assertTrue(torch.isnan(terms['total_reward'][1]))
        self.assertEqual(terms['total_reward'][0], original['total_reward'][0])


if __name__ == '__main__':
    unittest.main()
