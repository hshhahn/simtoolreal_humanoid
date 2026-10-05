from omegaconf import OmegaConf

from isaacgymenvs.utils.training_results import is_pbt_restart_result


class RestartableEnv:
    def change_on_restart(self, task_config):
        del task_config


def test_pbt_config_and_restartable_env_is_restart_result():
    result = (OmegaConf.create({"task": {}}), RestartableEnv())

    assert is_pbt_restart_result(result)


def test_finite_training_result_is_not_restart_result():
    assert not is_pbt_restart_result((123.5, 100))


def test_arbitrary_tuple_is_not_restart_result():
    assert not is_pbt_restart_result((OmegaConf.create({}), object()))
