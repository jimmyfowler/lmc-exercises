import dynamaxsys
import jax.numpy as jnp
from typing import Dict

class RocketDynamics(dynamaxsys.Dynamics):
    """Rocket dynamics class.
    Refer to 05a-rocket-derivation.md for the derivation of the dynamics.
    """

    def __init__(self, params: Dict[str, float]):
        """
        Initialize the rocket dynamics class.
        Args:
            params: Dictionary containing the rocket parameters.
        """
        state_dim = 6
        control_dim = 3

        def rocket_ode(
            states: jnp.ndarray,
            controls: jnp.ndarray,
            disturbance: jnp.ndarray = jnp.zeros(state_dim),
            time: float = 0.0,  # need to provide time for the dynamics class, but not used in this implementation
        ):
            mass_thruster = params["mass_thruster"]
            mass_body = params["mass_body"]
            length = params["length"]
            gravity = params["gravity"]
            _, _, theta, v_x, v_z, omega = states

            # precompute sin and cos for efficiency
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)

            M = jnp.array(
                [
                    [
                        mass_thruster + mass_body,
                        0,
                        0.5 * mass_body * length * cos_theta,
                    ],
                    [
                        0,
                        mass_thruster + mass_body,
                        -0.5 * mass_body * length * sin_theta,
                    ],
                    [
                        0.5 * mass_body * length * cos_theta,
                        -0.5 * mass_body * length * sin_theta,
                        1 / 3 * mass_body * length**2,
                    ],
                ]
            )

            C = jnp.array(
                [
                    [0, 0, -0.5 * mass_body * length * omega * sin_theta],
                    [0, 0, -0.5 * mass_body * length * omega * cos_theta],
                    [0, 0, 0],
                ]
            )

            g_vec = jnp.array(
                [
                    0,
                    (mass_thruster + mass_body) * gravity,
                    -0.5 * mass_body * gravity * length * sin_theta,
                ]
            )

            B = jnp.array([[1, 0, 1], [0, 1, 0], [0, 0, length * cos_theta]])

            Bu = B @ controls

            q_dot = jnp.array(
                [
                    v_x,
                    v_z,
                    omega,
                ]
            )

            xdot = jnp.concat([q_dot, jnp.linalg.inv(M) @ (Bu - C @ q_dot - g_vec)]) + disturbance

            return xdot
            ###### end of add your code here

        super().__init__(rocket_ode, state_dim, control_dim)
