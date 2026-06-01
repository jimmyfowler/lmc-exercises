from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def linearize_dynamics(
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    state: jnp.ndarray,
    control: jnp.ndarray,
    time: float = 0.0,
):
    """Linearize dynamics around the given state and control.

    Args:
        dynamics: Function(state, control, timestep) -> next_state
        state: (n,) array of state around which to linearize
        control: (m,) array of control around which to linearize
        time: float, time at which to linearize, in case dynamics are time-varying

    Returns:
        A: State transition matrix (n, n)
        B: Control input matrix (n, m)
        C: Constant term (n,)
    """
    A, B = jax.jacobian(dynamics, argnums=(0, 1))(state, control, time)
    C = dynamics(state, control, time) - (
        A @ state + B @ control
    )  # evaluate dynamics to get constant term
    return A, B, C


def linearize_trajectory(rocket_dt, states_ref, controls_ref):
    As, Bs, Cs = jax.vmap(linearize_dynamics, in_axes=(None, 0, 0))(
        rocket_dt, states_ref[:-1], controls_ref
    )
    return As, Bs, Cs


def SCP