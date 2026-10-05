"""Shared validation for SimToolReal policy-action configuration."""


def validate_privileged_actions(enabled: bool) -> None:
    """Reject the historically invalid privileged-action implementation.

    The legacy configuration keys remain accepted so existing configurations
    with ``privilegedActions: false`` continue to work unchanged.
    """

    if enabled:
        raise ValueError(
            "privilegedActions=true is unsupported: the historical Isaac Gym "
            "implementation mixed the three privileged torque channels into "
            "the seven arm-control channels"
        )


__all__ = ["validate_privileged_actions"]
