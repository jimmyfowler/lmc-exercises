import functools
import os
from typing import Callable, List

import dynamaxsys
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display


def PD_controller(
    state: jnp.ndarray, target: jnp.ndarray, gains: jnp.ndarray, limits: jnp.ndarray
):
    """PD controller for the rocket landing problem
    Args:
        state: Current state of the rocket
        target: Target state of the rocket
        gains: Gains for the PD controller
        limits: Limits for the control inputs
    Returns:
        Control input to the rocket
    """
    p_x, p_z, theta, v_x, v_z, omega = state
    p_x_target, p_z_target, theta_target, v_x_target, v_z_target, omega_target = target
    Kp_x, Kd_x, Kp_z, Kd_z, Kp_theta, Kd_theta = gains
    lower_bounds, upper_bounds = limits
    T_x_min, T_z_min, T_theta_min = lower_bounds
    T_x_max, T_z_max, T_theta_max = upper_bounds

    # TODO: Implement the PD controller here
    ###### add your code here
    T_x = -Kp_x * (p_x - p_x_target) - Kd_x * (v_x - v_x_target)

    T_z = -Kp_z * (p_z - p_z_target) - Kd_z * (v_z - v_z_target)
    T_theta = -Kp_theta * (theta - theta_target) - Kd_theta * (omega - omega_target)

    # Control saturation (thrust limits)
    T_x = jnp.clip(T_x, T_x_min, T_x_max)
    T_z = jnp.clip(T_z, T_z_min, T_z_max)
    T_theta = jnp.clip(T_theta, T_theta_min, T_theta_max)

    ###### end of add your code here

    return jnp.array([T_x, T_z, T_theta])


def initial_guess_closed_loop(
    rocket_dt: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    n_steps: int,
    controller: Callable,
):
    """
    Generate an initial guess for the trajectory using a closed-loop controller.
    Args:
        initial_state: Initial state of the rocket
        target: Target state of the rocket
        controller: Closed-loop controller function
        limits: Limits for the control inputs
    Returns:

    """

    # set up the PD controller. Use functools.partial to fix the target, gains, and limits
    controller = functools.partial(
        PD_controller, target=target_state, gains=gains, limits=limits
    )
    return closed_loop_sim(rocket_dt, initial_state, controller, n_steps)


def initial_guess_straight_line(
    x0: jnp.ndarray,
    xf: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """
    Generate a straight-line initial guess for the rocket trajectory
    between the initial and final states.
    Args:
        x0: Initial state of the rocket (shape: [state_dim,]).
        xf: Final state of the rocket (shape: [state_dim,]).
        n_steps: Number of time steps in the trajectory.

    Returns:
        A straight-line trajectory guess (shape: [n_steps, state_dim]).
    """
    return jnp.linspace(x0, xf, n_steps)


# helper functions for simulation
@eqx.filter_jit
def closed_loop_sim(
    dynamics_discrete: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    controller: Callable[[jnp.ndarray], jnp.ndarray],
    sim_steps: int,
):
    """Simulate the closed-loop system given a closed-loop controller using jax.lax.scan"""

    def step_fn(state, _):
        control = controller(state)
        next_state = dynamics_discrete(state, control)
        return next_state, (next_state, control)

    # Dummy scan inputs as we need sim_steps steps
    dummy_inputs = jnp.arange(sim_steps)
    final_state, (all_states, all_controls) = jax.lax.scan(
        step_fn, initial_state, dummy_inputs
    )

    # Include the initial state
    states = jnp.concatenate([initial_state[None], all_states], axis=0)
    return states, all_controls


@eqx.filter_jit
def open_loop_sim(
    dynamics_discrete: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    controls: jnp.ndarray,
):
    """Simulate the open-loop system given a dynamics model using jax.lax.scan"""

    def step_fn(state, control):
        next_state = dynamics_discrete(state, control)
        return next_state, next_state

    # Run scan
    final_state, all_next_states = jax.lax.scan(step_fn, initial_state, controls)

    # Concatenate initial state at the front
    states = jnp.concatenate([initial_state[None], all_next_states], axis=0)
    return states


# helper functions for plotting
def plot_rocket_trajectory(states, controls, lower, upper):
    """
    Plots rocket state and control trajectories.

    Args:
        states: Array of shape (n_steps+1, state_dim)
        controls: Array of shape (n_steps, control_dim)
        lower: np.array of shape (control_dim,) — lower limits for controls
        upper: np.array of shape (control_dim,) — upper limits for controls
    """
    n_steps = controls.shape[-2]
    if len(states.shape) == 2:
        alpha = 1.0
    else:
        alpha = 0.4
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    # Position
    axs[0, 0].plot(states[..., :, 0].T, color="C0", alpha=alpha)
    axs[0, 0].plot(states[..., :, 1].T, color="C1", alpha=alpha)
    axs[0, 0].set_ylabel("Position (m)")
    leg = axs[0, 0].legend(["p_x (horizontal)", "p_z (altitude)"])
    # Set legend line color to match line color (C0, C1)
    for i, line in enumerate(leg.get_lines()):
        line.set_color(f"C{i}")
    axs[0, 0].grid(True)

    # Angle
    axs[1, 0].plot(states[..., :, 2].T, color="C0", alpha=alpha)
    axs[1, 0].set_ylabel("Angle (rad)")
    axs[1, 0].legend(["theta"])
    axs[1, 0].grid(True)

    # Velocities
    axs[0, 1].plot(states[..., :, 3].T, color="C0", alpha=alpha)
    axs[0, 1].plot(states[..., :, 4].T, color="C1", alpha=alpha)
    axs[0, 1].set_ylabel("Velocity (m/s)")
    leg = axs[0, 1].legend(["v_x", "v_z"])
    for i, line in enumerate(leg.get_lines()):
        line.set_color(f"C{i}")
    axs[0, 1].grid(True)

    # Controls
    axs[1, 1].plot(controls[..., :, 0].T, color="C0", alpha=alpha)
    axs[1, 1].plot(controls[..., :, 1].T, color="C1", alpha=alpha)
    axs[1, 1].plot(controls[..., :, 2].T, color="C2", alpha=alpha)
    leg = axs[1, 1].legend(["T_x", "T_z", "T_nose"])
    for i, line in enumerate(leg.get_lines()):
        line.set_color(f"C{i}")

    axs[1, 1].hlines(lower[0], 0, n_steps, color="C0", linestyle="--")
    axs[1, 1].hlines(upper[0], 0, n_steps, color="C0", linestyle="--")

    axs[1, 1].hlines(lower[1], 0, n_steps, color="C1", linestyle="--")
    axs[1, 1].hlines(upper[1], 0, n_steps, color="C1", linestyle="--")

    axs[1, 1].hlines(lower[2], 0, n_steps, color="C2", linestyle="--")
    axs[1, 1].hlines(upper[2], 0, n_steps, color="C2", linestyle="--")

    axs[1, 1].set_ylabel("Thrust (N)")
    axs[1, 1].set_xlabel("Time step")

    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_rocket(
    ax: plt.Axes,
    state: jnp.ndarray,
    control: jnp.ndarray | None = None,
    color: str = "C0",
    label: str | None = None,
    rocket_length: float = 2.0,
    thrust_limit: List[float] = [200.0, 300.0, 100.0],
):
    """
    Draw a simple rocket at the specified state, with optional thrust vectors at base and nose.

    Args:
      ax: matplotlib axis
      state: array-like of shape (6,) [x, z, theta, vx, vz, omega]
      control: array-like of shape (,), assumed [thrust, gimbal_angle] or [Tx, Tz], etc.
      color: color for rocket body
      label: optional label for the rocket
      rocket_length: length of the rocket
    """
    x, z, theta = float(state[0]), float(state[1]), float(state[2])

    # Compute end points of rocket body in world frame
    # Rocket body from (-L/2, 0) to (+L/2, 0) in body frame
    dx = rocket_length * np.sin(theta)
    dz = rocket_length * np.cos(theta)

    x_base = x
    z_base = z
    x_nose = x + dx
    z_nose = z + dz

    # Draw main body
    ax.plot(
        [x_base, x_nose],
        [z_base, z_nose],
        color=color,
        linewidth=6,
        solid_capstyle="round",
        label=label,
    )

    # Draw thrust vectors at base and nose if control is provided
    if control is not None:
        T_base_x, T_base_z, T_nose_x = control
        thrust_scale = 5.0  # scaling factor for vector length

        ax.arrow(
            x_base,
            z_base,
            thrust_scale * T_base_x / thrust_limit[0],
            thrust_scale * T_base_z / thrust_limit[1],
            head_width=0.06,
            head_length=0.13,
            fc="crimson",
            ec="crimson",
            alpha=0.6,
            length_includes_head=True,
            zorder=5,
        )
        ax.arrow(
            x_nose,
            z_nose,
            thrust_scale * T_nose_x / thrust_limit[2],
            0,
            head_width=0.06,
            head_length=0.13,
            fc="crimson",
            ec="crimson",
            alpha=0.6,
            length_includes_head=True,
            zorder=5,
        )

    # Draw ground
    ax.hlines(0, -3, 3, color="k", linewidth=2, zorder=0)


def animate_rocket_trajectory(
    states,
    controls=None,
    interval=50,
    ax=None,
    save_path=None,
    color="C0",
    label=None,
    rocket_length=2.0,
    thrust_limit: List[float] = [200.0, 300.0, 100.0],
    obstacle_center: tuple = None,
    obstacle_radius: float = None,
    obstacle_color: str = "gray",
    obstacle_alpha: float = 0.4,
):
    """
    Animate rocket trajectory and pose over time, output as HTML if possible.
    Args:
      states: array-like, shape [N,6]
      controls: array-like, shape [N,...], optional
      interval: ms between frames (default 50)
      ax: optional matplotlib axis
      save_path: optional, if provided save as .mp4, .gif, or .html
      obstacle_center: (x, z) tuple for obstacle center (optional)
      obstacle_radius: float, radius of the obstacle (optional)
      obstacle_color: color spec for obstacle (default 'gray')
      obstacle_alpha: alpha blending value for obstacle (default 0.4)
    """

    states = np.asarray(states)
    if controls is not None:
        controls = np.asarray(controls)
    N = len(states)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    xs = states[:, 0]
    zs = states[:, 1]

    # Draw ground
    ground_x = [xs.min() - 1, xs.max() + 1]
    ground_z = [0, 0]
    ax.plot(ground_x, ground_z, "k-", linewidth=2)

    # Draw circular obstacle if specified
    obstacle_artist = None
    if obstacle_center is not None and obstacle_radius is not None:
        from matplotlib.patches import Circle

        ox, oz = obstacle_center
        obstacle_artist = Circle(
            (ox, oz),
            obstacle_radius,
            color=obstacle_color,
            alpha=obstacle_alpha,
            zorder=1,
        )
        ax.add_patch(obstacle_artist)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Rocket landing trajectory")

    # Main trajectory (background)
    (traj_line,) = ax.plot([], [], "k--", alpha=0.4)

    rocket_artists = []

    def init():
        traj_line.set_data([], [])
        for artist in rocket_artists:
            artist.remove()
        rocket_artists.clear()
        return [traj_line] + ([obstacle_artist] if obstacle_artist is not None else [])

    def animate(i):
        # Update trajectory line
        traj_line.set_data(xs[: i + 1], zs[: i + 1])
        # Clear previous rockets
        for artist in rocket_artists:
            artist.remove()
        rocket_artists.clear()
        # Draw rocket at current state
        try:
            plot_rocket(
                ax,
                states[i],
                controls[i],
                color=color,
                label=label,
                rocket_length=rocket_length,
                thrust_limit=thrust_limit,
            )
        except Exception:
            plot_rocket(
                ax,
                states[i],
                color=color,
                label=label,
                rocket_length=rocket_length,
                thrust_limit=thrust_limit,
            )
        # Collect all new artists
        for child in ax.get_children():
            if isinstance(child, (plt.Line2D, plt.Polygon)) and child not in [
                traj_line
            ]:
                rocket_artists.append(child)
        output = [traj_line] + rocket_artists
        if obstacle_artist is not None:
            output.append(obstacle_artist)
        return output

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=N,
        init_func=init,
        blit=False,
        interval=interval,
        repeat=False,
    )

    # HTML output by default if not saving as a file
    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            ani.save(save_path, writer="pillow")
        display(HTML(ani.to_jshtml()))
        plt.close(fig)
    else:
        display(HTML(ani.to_jshtml()))
        plt.close(fig)
