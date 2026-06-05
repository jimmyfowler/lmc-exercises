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


    T_x = -Kp_x * (p_x - p_x_target) - Kd_x * (v_x - v_x_target)

    T_z = -Kp_z * (p_z - p_z_target) - Kd_z * (v_z - v_z_target)
    T_theta = -Kp_theta * (theta - theta_target) - Kd_theta * (omega - omega_target)

    # Control saturation (thrust limits)
    T_x = jnp.clip(T_x, T_x_min, T_x_max)
    T_z = jnp.clip(T_z, T_z_min, T_z_max)
    T_theta = jnp.clip(T_theta, T_theta_min, T_theta_max)

    return jnp.array([T_x, T_z, T_theta])


def initial_guess_closed_loop(
    rocket_dt: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    target_state: jnp.ndarray,
    controller: Callable,
    gains: jnp.ndarray,
    limits: jnp.ndarray,
    n_steps: int,
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


@eqx.filter_jit
def closed_loop_sim(
    dynamics_discrete: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    controller: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int,
    disturbances: jnp.ndarray | None = None,  # size (sim_steps, state_dim)
):
    """Simulate the closed-loop system given a closed-loop controller using jax.lax.scan"""

    if disturbances is None:
        disturbances = jnp.zeros((n_steps, initial_state.shape[0]))
        
    def step_fn(state, disturbance):
        control = controller(state)
        next_state = dynamics_discrete(state, control, disturbance)
        return next_state, (next_state, control)

    final_state, (all_states, all_controls) = jax.lax.scan(
        step_fn, initial_state, disturbances
    )

    # Include the initial state
    states = jnp.concatenate([initial_state[None], all_states], axis=0)
    return states, all_controls


def simulate_closed_loop(
    dynamics_func,
    initial_state: np.ndarray,
    controller,
    n_steps: int,
    disturbances: np.ndarray | None = None,
    has_numpy=False,
) -> np.ndarray:
    """
    Simulate trajectory given initial state and control sequence.

    Args:
        initial_state: (n,) array of initial state
        policy: Function(state) -> control
        n_steps: Number of time steps to simulate
        dynamics_func: Function(state, control, dt) -> next_state
        dt: Time step
    Returns:
        trajectory: (N+1, n) array of states over time
    """

    if disturbances is None:
        if has_numpy:
            disturbances = np.zeros((n_steps, initial_state.shape[0]))
        else:
            disturbances = jnp.zeros((n_steps, initial_state.shape[0]))
        
    def _scan_func(state, disturbance):
        control = controller(state)
        next_state = dynamics_func(state, control, disturbance)
        return next_state, (next_state, control)

    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

    if has_numpy:
        _, trajectory = scan(_scan_func, initial_state, disturbances, length=n_steps)
    else:
        _, trajectory = jax.lax.scan(_scan_func, initial_state, disturbances, length=n_steps)
    trajectory = jnp.vstack([initial_state, trajectory])
    return trajectory


@eqx.filter_jit
def open_loop_sim(
    dynamics_discrete: dynamaxsys.Dynamics,
    initial_state: jnp.ndarray,
    controls: jnp.ndarray,
    disturbances: jnp.ndarray | None = None,
):
    """Simulate the open-loop system given a dynamics model using jax.lax.scan"""

    if disturbances is None:
        disturbances = jnp.zeros((controls.shape[0], initial_state.shape[0]))

    def step_fn(state, ctrl_disturb):
        control, disturbance = ctrl_disturb
        next_state = dynamics_discrete(state, control, disturbance)
        return next_state, next_state

    # Run scan
    final_state, all_next_states = jax.lax.scan(
        step_fn, initial_state, (controls, disturbances)
    )

    # Concatenate initial state at the front
    states = jnp.concatenate([initial_state[None], all_next_states], axis=0)
    return states


# helper functions for plotting
def plot_rocket_trajectory(states, controls, lower, upper, title = None, theta_limit = None):
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
    
    if title is not None:
        fig.suptitle(title)

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
    axs[1, 0].plot(states[..., :, 2].T, color="C0", alpha=alpha, label="theta")
    axs[1, 0].set_ylabel("Angle (rad)")
    if theta_limit is not None:
        axs[1, 0].hlines([theta_limit, -theta_limit], 0, n_steps, color="r", linestyle="--", label="theta limit")
        handles, labels = axs[1, 0].get_legend_handles_labels()
        axs[1, 0].legend([handles[0], handles[-1]], [labels[0], labels[-1]])
    else:
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
    colors: list = ["white", "gold"],
    label: str | None = None,
    rocket_length: float = 2.0,
    thrust_limit: List[float] = [200.0, 300.0, 100.0],
):
    """
    Draw a simplified Apollo-era lunar lander at the specified state, with optional thrust vectors.

    Args:
      ax: matplotlib axis
      state: array-like of shape (6,) [x, z, theta, vx, vz, omega]
      control: array-like of shape (3,), [Tx, Tz, T_nose]
      color: color for lander body
      label: optional label for the lander
      rocket_length: characteristic length of the lander
    """
    from matplotlib.patches import Rectangle, Polygon
    
    x, z, theta = float(state[0]), float(state[1]), float(state[2])
    c, s = np.cos(theta), np.sin(theta)
    
    # Lander dimensions
    descent_stage_width = rocket_length * 0.6
    descent_stage_height = rocket_length * 0.4
    ascent_stage_width = rocket_length * 0.45
    ascent_stage_height = rocket_length * 0.3
    leg_length = rocket_length * 0.5
    
    # Define lander geometry in body frame (center at origin)
    # Descent stage: rectangular platform
    descent_x = descent_stage_width / 2
    descent_z = descent_stage_height / 2
    
    # Descent stage corners (in body frame, relative to center)
    descent_corners = np.array([
        [-descent_x, -descent_z],
        [descent_x, -descent_z],
        [descent_x, descent_z],
        [-descent_x, descent_z],
    ])
    
    # Ascent stage (cabin) on top
    ascent_x = ascent_stage_width / 2
    ascent_z = descent_z + ascent_stage_height
    ascent_corners = np.array([
        [-ascent_x, descent_z],
        [ascent_x, descent_z],
        [ascent_x, ascent_z],
        [-ascent_x, ascent_z],
    ])
    
    # Landing legs (4 legs extending outward and downward)
    leg_angle = 70 * np.pi / 180
    legs = [
        [descent_x * 0.8, descent_z],  # base
        [descent_x * 0.8 + leg_length * np.cos(leg_angle), 
         descent_z - leg_length * np.sin(leg_angle)],  # tip
    ]
    legs_neg = [
        [-descent_x * 0.8, descent_z],
        [-descent_x * 0.8 - leg_length * np.cos(leg_angle),
         descent_z - leg_length * np.sin(leg_angle)],
    ]
    
    # Rotate to world frame
    def rotate(points):
        return np.array([[c, -s], [s, c]]) @ points.T
    
    descent_world = rotate(descent_corners).T + np.array([x, z])
    ascent_world = rotate(ascent_corners).T + np.array([x, z])
    legs_world = rotate(np.array(legs)).T + np.array([x, z])
    legs_neg_world = rotate(np.array(legs_neg)).T + np.array([x, z])
    
    # Draw descent stage
    descent_poly = Polygon(descent_world, closed=True, edgecolor="black", 
                          facecolor="gold", linewidth=2, zorder=10, label=label)
    ax.add_patch(descent_poly)
    
    # Draw ascent stage (cabin)
    ascent_poly = Polygon(ascent_world, closed=True, edgecolor="black",
                         facecolor="white", linewidth=2, zorder=10)
    ax.add_patch(ascent_poly)
    
    # Draw landing legs
    ax.plot(legs_world[:, 0], legs_world[:, 1], color="gold", linewidth=2.5, zorder=9)
    ax.plot(legs_neg_world[:, 0], legs_neg_world[:, 1], color="gold", linewidth=2.5, zorder=9)
    
    # Draw thrust vectors if control is provided
    if control is not None:
        T_base_x, T_base_z, T_nose_x = control
        thrust_scale = 5.0
        
        # Thrust from descent stage (base)
        thrust_base_world = rotate(np.array([[0, -descent_z]]).T).T[0] + np.array([x, z])
        ax.arrow(
            thrust_base_world[0],
            thrust_base_world[1],
            thrust_scale * T_base_x / thrust_limit[0],
            thrust_scale * T_base_z / thrust_limit[1],
            head_width=0.06,
            head_length=0.13,
            fc="orange",
            ec="orange",
            alpha=0.7,
            length_includes_head=True,
            zorder=5,
        )
        
        # Thrust from RCS thrusters (nose/top)
        thrust_nose_world = rotate(np.array([[0, ascent_z - z]]).T).T[0] + np.array([x, z])
        ax.arrow(
            thrust_nose_world[0],
            thrust_nose_world[1],
            thrust_scale * T_nose_x / thrust_limit[2],
            0,
            head_width=0.06,
            head_length=0.13,
            fc="orange",
            ec="orange",
            alpha=0.7,
            length_includes_head=True,
            zorder=5,
        )


def animate_rocket_trajectory(
    states,
    controls=None,
    interval=50,
    ax=None,
    save_path=None,
    color="white",
    label=None,
    rocket_length=2.0,
    thrust_limit: List[float] = [200.0, 300.0, 100.0],
    obstacle_center: tuple = None,
    obstacle_radius: float = None,
    obstacle_color: str = "gray",
    obstacle_alpha: float = 0.4,
):
    """
    Animate lunar lander trajectory and pose over time, output as HTML if possible.
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
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    xs = states[:, 0]
    zs = states[:, 1]
    
    z_min = zs.min() - 1
    z_max = zs.max() + 1

    # Set dark background (space)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Draw lunar regolith (grey ground below z=0)
    ground_x = [xs.min() - 2, xs.max() + 2]
    ax.fill_between(ground_x, z_min, 0, color='gray', zorder=1)
    
    # Draw ground line
    ax.plot(ground_x, [0, 0], color="lightgray", linewidth=3, zorder=2)

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
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("z [m]", color="white")
    ax.set_title("Lunar Lander Trajectory", color="white", fontsize=14)
    
    # Style the axes for dark theme
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')

    # Main trajectory (background)
    (traj_line,) = ax.plot([], [], color="cyan", linestyle="--", alpha=0.4, linewidth=1.5)

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
                label=label,
                rocket_length=rocket_length,
                thrust_limit=thrust_limit,
            )
        except Exception:
            plot_rocket(
                ax,
                states[i],
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


def plot_mpc_sqp_predictions(
    mpc_states_trajectory: np.ndarray,
    mpc_controls_trajectory: np.ndarray,
    sqp_states_predictions: list,
    sqp_controls_predictions: list,
    lower_control_limit: np.ndarray,
    upper_control_limit: np.ndarray,
    theta_limit: float = 0.25,
    title: str = "MPC with SQP Predictions",
):
    """
    Plot MPC trajectory with overlaid SQP predictions from each MPC iteration.
    
    Args:
        mpc_states_trajectory: (N+1, 6) actual MPC executed states
        mpc_controls_trajectory: (N, 3) actual MPC executed controls
        sqp_states_predictions: list of length N, each element shape (H+1, 6) where H is horizon
        sqp_controls_predictions: list of length N, each element shape (H, 3)
        lower_control_limit: (3,) lower control bounds
        upper_control_limit: (3,) upper control bounds
        theta_limit: angle constraint magnitude
        title: plot title
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    n_steps = len(mpc_states_trajectory) - 1
    
    # Color map for predictions (gradient from early to late MPC steps)
    colors_prediction = plt.cm.viridis(np.linspace(0, 1, len(sqp_states_predictions)))
    
    # ===== Position =====
    ax = axs[0, 0]
    
    # Plot all SQP predictions
    for i, (pred_states, color) in enumerate(zip(sqp_states_predictions, colors_prediction)):
        ax.plot(pred_states[:, 0], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
        ax.plot(pred_states[:, 1], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
    
    # Plot actual MPC trajectory (thicker, on top)
    ax.plot(mpc_states_trajectory[:, 0], color='C0', linewidth=2.5, label='p_x (actual)', zorder=10)
    ax.plot(mpc_states_trajectory[:, 1], color='C1', linewidth=2.5, label='p_z (actual)', zorder=10)
    
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ground (z=0)')
    ax.set_ylabel("Position (m)", fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Angle =====
    ax = axs[1, 0]
    
    # Plot all SQP predictions
    for i, (pred_states, color) in enumerate(zip(sqp_states_predictions, colors_prediction)):
        ax.plot(pred_states[:, 2], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
    
    # Plot actual MPC trajectory
    ax.plot(mpc_states_trajectory[:, 2], color='C2', linewidth=2.5, label='theta (actual)', zorder=10)
    
    # Angle limits
    ax.hlines([theta_limit, -theta_limit], 0, n_steps, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='theta limit')
    ax.set_ylabel("Angle (rad)", fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Velocities =====
    ax = axs[0, 1]
    
    # Plot all SQP predictions
    for i, (pred_states, color) in enumerate(zip(sqp_states_predictions, colors_prediction)):
        ax.plot(pred_states[:, 3], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
        ax.plot(pred_states[:, 4], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
    
    # Plot actual MPC trajectory
    ax.plot(mpc_states_trajectory[:, 3], color='C0', linewidth=2.5, label='v_x (actual)', zorder=10)
    ax.plot(mpc_states_trajectory[:, 4], color='C1', linewidth=2.5, label='v_z (actual)', zorder=10)
    
    ax.set_ylabel("Velocity (m/s)", fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Controls =====
    ax = axs[1, 1]
    
    # Plot all SQP predictions
    for i, (pred_controls, color) in enumerate(zip(sqp_controls_predictions, colors_prediction)):
        ax.plot(pred_controls[:, 0], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
        ax.plot(pred_controls[:, 1], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
        ax.plot(pred_controls[:, 2], color=color, linewidth=1.0, alpha=0.5, linestyle='--')
    
    # Plot actual MPC trajectory
    ax.plot(mpc_controls_trajectory[:, 0], color='C0', linewidth=2.5, label='T_x (actual)', zorder=10)
    ax.plot(mpc_controls_trajectory[:, 1], color='C1', linewidth=2.5, label='T_z (actual)', zorder=10)
    ax.plot(mpc_controls_trajectory[:, 2], color='C2', linewidth=2.5, label='T_nose (actual)', zorder=10)
    
    # Control limits
    ax.hlines(lower_control_limit[0], 0, n_steps, color='C0', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.hlines(upper_control_limit[0], 0, n_steps, color='C0', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.hlines(lower_control_limit[1], 0, n_steps, color='C1', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.hlines(upper_control_limit[1], 0, n_steps, color='C1', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.hlines(lower_control_limit[2], 0, n_steps, color='C2', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.hlines(upper_control_limit[2], 0, n_steps, color='C2', linestyle='--', linewidth=1.0, alpha=0.5)
    
    ax.set_ylabel("Thrust (N)", fontweight='bold')
    ax.set_xlabel("Time step", fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
