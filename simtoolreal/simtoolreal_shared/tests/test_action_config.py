import pytest

from simtoolreal_shared.action_config import validate_privileged_actions


def test_disabled_privileged_actions_remain_supported():
    validate_privileged_actions(False)


def test_enabled_privileged_actions_fail_fast():
    with pytest.raises(ValueError, match="mixed the three privileged torque channels"):
        validate_privileged_actions(True)
