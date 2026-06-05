import functools
import time
from typing import Callable

import cvxpy as cp
import dynamaxsys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from RocketDynamics import RocketDynamics
from utils import closed_loop_sim, open_loop_sim


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


def SQP(
    problem: cp.Problem,
    rocket_dt: dynamaxsys.Dynamics,
    initial_guess: dict,
    convergence_tol: float = 1e-2,
    max_sqp_iterations: int = 100,
    verbose: bool = False,
):
    """
    Run Sequential Quadratic Programming (SQP) to solve the given problem.

    Args:
        rocket_dt: dynamaxsys rocket dynamics object
        problem: cvxpy Problem object representing the optimization problem to solve
        initial_guess: Initial guess trajectory and control sequence to initialize the SQP algorithm
        n_sqp: number of SQP iterations to run
    Returns:
        controls_list: list of control trajectories at each SQP iteration
        solver_states_list: list of state trajectories at each SQP iteration
        costs: list of cost values at each SQP iteration
        status_list: list of solver status at each SQP iteration

    Assumes the following parameters and variables are defined in the cvxpy problem:
        - previous_states: cvxpy Parameter of shape (N, n) representing the trajectory around which to linearize
        - previous_controls: cvxpy Parameter of shape (N-1, m) representing the control trajectory around which to linearize
        - As: cvxpy Parameter of shape (N-1, n, n) representing the linearized state transition matrices
        - Bs: cvxpy Parameter of shape (N-1, n, m) representing the linearized control input matrices
        - Cs: cvxpy Parameter of shape (N-1, n) representing the linearized constant terms

    """

    # extract parameters and variables from the problem
    previous_states = problem.param_dict["previous_states"]
    previous_controls = problem.param_dict["previous_controls"]
    As = problem.param_dict["As"]
    Bs = problem.param_dict["Bs"]
    Cs = problem.param_dict["Cs"]
    initial_state = problem.param_dict["initial_state"]

    # extract optimization variables
    states = problem.var_dict["states"]
    controls = problem.var_dict["controls"]

    # extract initial guess states and controls
    previous_states_values = initial_guess["states"]
    previous_controls_values = initial_guess["controls"]

    # initialize lists to store results
    controls_list = [
        previous_controls_values
    ]  # list of controls to collect for plotting
    solver_states_list = [previous_states_values]  # list of solver states for plotting
    costs = [None]  # list of costs to collect for plotting,
    status_list = [None]  # list of solver statuses to collect for plotting
    # initialized with None so indexing lines up with states and controls lists

    # loop through SQP iterations
    for i in range(max_sqp_iterations):
        # initialize/update previous trajectory & controls
        # convert to numpy array bc jax arrays don't work as cvxpy variables
        previous_states.value = np.array(previous_states_values)
        previous_controls.value = np.array(previous_controls_values)

        # linearize about previous trajectory & controls
        As_values, Bs_values, Cs_values = linearize_trajectory(
            rocket_dt, previous_states_values, previous_controls_values
        )
        # update cvxpy parmamters with new linearized system matrices
        As.value = np.array(As_values)
        Bs.value = np.array(Bs_values)
        Cs.value = np.array(Cs_values)

        # solve for the optimal trajectory & controls
        problem.solve(solver=cp.CLARABEL)

        if verbose:
            print(
                f"SQP iteration {i + 1}/{max_sqp_iterations}\t Status: {problem.status}\t Cost: {problem.value}"
            )

        # update previous states and controls
        previous_states_values = open_loop_sim(
            rocket_dt, initial_state.value, controls.value
        )
        previous_controls_values = jnp.array(controls.value)

        # collect cost and control sequence
        costs.append(problem.value)
        controls_list.append(controls.value)
        solver_states_list.append(states.value)
        status_list.append(problem.status)

        # convergence
        if i > 1:  # check after at least 2 iterations to have 2 cost values to compare
            cost_change = abs(costs[-1] - costs[-2])
            if cost_change < convergence_tol:
                if verbose:
                    print(f"Convergenced at iteration {i + 1}.")
                break

    return controls_list, solver_states_list, costs, status_list


def build_mpc_problem(
    params: dict,
    rocket_ct: RocketDynamics,
    lower_control_limit: jnp.ndarray,
    upper_control_limit: jnp.ndarray,
) -> cp.Problem:
    """
    Build the CVXPY tracking MPC problem from a parameter dict.

    Required params keys:
        mpc_horizon, state_costs_mpc, control_costs_mpc, terminal_state_costs_mpc,
        state_trust_region_penalty_mpc, control_trust_region_penalty_mpc,
        theta_limit_slack_penalty, above_ground_slack_penalty, theta_limit
    """
    mpc_horizon = params["mpc_horizon"]
    state_costs_mpc = params["state_costs_mpc"]
    control_costs_mpc = params["control_costs_mpc"]
    terminal_state_costs_mpc = params["terminal_state_costs_mpc"]
    state_trp = params["state_trust_region_penalty_mpc"]
    control_trp = params["control_trust_region_penalty_mpc"]
    theta_slack_penalty = params["theta_limit_slack_penalty"]
    altitude_slack_penalty = params["above_ground_slack_penalty"]
    theta_limit = params["theta_limit"]

    state_dim = rocket_ct.state_dim
    control_dim = rocket_ct.control_dim

    # --- cost matrices ---
    Q_diag = np.array(state_costs_mpc)
    R_diag = np.array(control_costs_mpc)
    QN_diag = np.array(terminal_state_costs_mpc)
    QTRP_diag = np.ones(state_dim) * state_trp
    RTRP_diag = np.ones(control_dim) * control_trp

    # --- variables ---
    states = cp.Variable((mpc_horizon + 1, state_dim), name="states")
    controls = cp.Variable((mpc_horizon, control_dim), name="controls")
    theta_limit_slack = cp.Variable(
        mpc_horizon + 1, nonneg=True, name="theta_limit_slack"
    )
    above_ground_slack = cp.Variable(
        mpc_horizon + 1, nonneg=True, name="above_ground_slack"
    )

    # --- parameters ---
    reference_states = cp.Parameter(
        (mpc_horizon + 1, state_dim), name="reference_states"
    )
    reference_controls = cp.Parameter(
        (mpc_horizon, control_dim), name="reference_controls"
    )
    As = cp.Parameter((mpc_horizon, state_dim, state_dim), name="As")
    Bs = cp.Parameter((mpc_horizon, state_dim, control_dim), name="Bs")
    Cs = cp.Parameter((mpc_horizon, state_dim), name="Cs")
    previous_states = cp.Parameter((mpc_horizon + 1, state_dim), name="previous_states")
    previous_controls = cp.Parameter(
        (mpc_horizon, control_dim), name="previous_controls"
    )
    initial_state = cp.Parameter(state_dim, name="initial_state")

    u_min = cp.Constant(lower_control_limit, name="u_min")
    u_max = cp.Constant(upper_control_limit, name="u_max")

    # --- cost terms ---
    state_deviation = states - previous_states
    control_deviation = controls - previous_controls

    state_tracking_cost = cp.sum(
        cp.multiply(
            Q_diag, cp.square(states[:mpc_horizon] - reference_states[:mpc_horizon])
        )
    )
    control_tracking_cost = cp.sum(
        cp.multiply(R_diag, cp.square(controls - reference_controls))
    )
    QTRP_cost = cp.sum(cp.multiply(QTRP_diag, cp.square(state_deviation)))
    RTRP_cost = cp.sum(cp.multiply(RTRP_diag, cp.square(control_deviation)))
    terminal_cost = cp.sum(
        cp.multiply(QN_diag, cp.square(states[-1] - reference_states[-1]))
    )
    slack_cost = cp.sum(theta_slack_penalty * theta_limit_slack) + cp.sum(
        altitude_slack_penalty * above_ground_slack
    )

    objective = (
        state_tracking_cost
        + control_tracking_cost
        + QTRP_cost
        + RTRP_cost
        + terminal_cost
        + slack_cost
    )

    # --- constraints ---
    dynamics_constraints = [
        states[k + 1] == As[k] @ states[k] + Bs[k] @ controls[k] + Cs[k]
        for k in range(mpc_horizon)
    ]
    constraints = [
        *dynamics_constraints,
        states[0, :] == initial_state,
        states[:, 1] >= -above_ground_slack,
        states[:, 2] <= theta_limit + theta_limit_slack,
        states[:, 2] >= -theta_limit - theta_limit_slack,
        controls >= u_min,
        controls <= u_max,
    ]

    return cp.Problem(cp.Minimize(objective), constraints)



def run_mpc(
    problem: cp.Problem,
    rocket_dt: dynamaxsys.Dynamics,
    nominal_states: np.ndarray,
    nominal_controls: np.ndarray,
    params: dict,
    initial_state_value: np.ndarray,
    disturbance_fn: Callable = None,  # disturbance_fn(step, state) -> (state_dim,) array
) -> dict:
    """
    Run a full MPC simulation and return performance metrics.

    Args:
        problem:           CVXPY tracking problem built by build_mpc_problem
        rocket_dt:         discrete-time dynamics
        nominal_states:    (N+1, state_dim) nominal trajectory from offline SQP
        nominal_controls:  (N, control_dim) nominal controls from offline SQP
        params:            same dict passed to build_mpc_problem, plus:
                               n_steps_total, max_sqp_iterations_tracking,
                               sqp_convergence_tol
        initial_state_value: (state_dim,) starting state
        disturbance_fn:    optional callable(step, state) -> disturbance array.
                           None means no disturbance.

    Returns:
        dict of metrics and full trajectories for plotting
    """
    mpc_horizon      = params["mpc_horizon"]
    n_steps          = params["n_steps_total"]
    max_sqp_iters    = params["max_sqp_iterations_tracking"]
    conv_tol         = params["sqp_convergence_tol"]
    state_dim        = nominal_states.shape[1]
    control_dim      = nominal_controls.shape[1]

    # pull parameter/variable handles out of problem once
    ref_states_param   = problem.param_dict["reference_states"]
    ref_controls_param = problem.param_dict["reference_controls"]
    initial_state_param = problem.param_dict["initial_state"]

    mpc_states   = [initial_state_value]
    mpc_controls = []
    sqp_iter_counts = []
    solve_times     = []
    solver_statuses = []
    crashed = False

    current_state = np.array(initial_state_value)

    # warm start: first window of nominal trajectory
    initial_guess = {
        "states":   np.array(nominal_states[:mpc_horizon + 1]),
        "controls": np.array(nominal_controls[:mpc_horizon]),
    }

    for i in range(n_steps):
        # --- slide reference window ---
        end_idx = min(i + mpc_horizon + 1, len(nominal_states))
        ref_s = nominal_states[i:end_idx]
        ref_c = nominal_controls[i:end_idx - 1]

        # pad near end of nominal trajectory
        if ref_s.shape[0] < mpc_horizon + 1:
            pad_s = mpc_horizon + 1 - ref_s.shape[0]
            pad_c = mpc_horizon     - ref_c.shape[0]
            ref_s = np.vstack([ref_s, np.tile(nominal_states[-1], (pad_s, 1))])
            ref_c = np.vstack([ref_c, np.zeros((pad_c, control_dim))])

        ref_states_param.value   = ref_s
        ref_controls_param.value = ref_c
        initial_state_param.value = current_state

        # --- SQP solve ---
        t0 = time.perf_counter()
        controls_list, solver_states_list, costs, status_list = SQP(
            problem, rocket_dt, initial_guess,
            max_sqp_iterations=max_sqp_iters,
            convergence_tol=conv_tol,
            verbose=False,
        )
        solve_times.append(time.perf_counter() - t0)
        sqp_iter_counts.append(len(controls_list) - 1)  # minus 1 for initial guess sentinel
        solver_statuses.append(problem.status)

        # --- apply first control ---
        control_input = controls_list[-1][0]
        mpc_controls.append(control_input)

        # --- optional disturbance ---
        disturbance = (
            np.array(disturbance_fn(i, current_state))
            if disturbance_fn is not None
            else np.zeros(state_dim)
        )

        current_state = np.array(
            rocket_dt(current_state, control_input, jnp.array(disturbance))
        )
        mpc_states.append(current_state)

        # --- warm start for next step ---
        shifted_controls = np.vstack([controls_list[-1][1:], controls_list[-1][-1:]])
        initial_guess = {
            "states": np.array(open_loop_sim(
                rocket_dt, jnp.array(current_state), jnp.array(shifted_controls)
            )),
            "controls": shifted_controls,
        }

        # --- early termination ---
        if current_state[1] <= 0:
            crashed = current_state[1] < -0.05  # below ground by more than numerical noise
            break

    mpc_states   = np.array(mpc_states)
    mpc_controls = np.array(mpc_controls)
    final_state  = mpc_states[-1]

    return {
        # trajectories (for plotting)
        "states":   mpc_states,
        "controls": mpc_controls,

        # landing quality
        "landing_position_error": float(np.linalg.norm(final_state[:2])),
        "landing_velocity":       float(np.linalg.norm(final_state[3:5])),
        "final_angle":            float(abs(final_state[2])),

        # constraint violations
        "total_control_effort":   float(np.sum(mpc_controls ** 2)),
        "crashed":                crashed,
        "steps_taken":            len(mpc_controls),

        # solver diagnostics
        "mean_sqp_iterations": float(np.mean(sqp_iter_counts)),
        "max_sqp_iterations":  float(np.max(sqp_iter_counts)),
        "mean_solve_time_s":   float(np.mean(solve_times)),
        "total_solve_time_s":  float(np.sum(solve_times)),
        "n_solver_failures":   sum(s not in ("optimal", "optimal_inaccurate") for s in solver_statuses),
    }