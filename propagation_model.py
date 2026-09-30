import networkx as nx
import numpy as np
from typing import Optional, Union
import math

def SIModel(G: nx.Graph, num_timesteps: int, beta: float = 0.05, mode: str = "global", prob_matrix: Optional[np.ndarray] = None, initial_infected: Optional[Union[float, int, list, str]] = None,
    infection_scale: Optional[float] = None, seed: Optional[int] = None, use_prod: Optional[bool] = True, degree_bias: Optional[float] = None):
    """
    Simulate SI propagation.
    Return an array of shape (num_timesteps, n_nodes) with binary infection states.

    Parameters:
      - G: NetworkX graph.
      - num_timesteps: Number of simulation steps.
      - beta: Global infection rate.
      - mode: "local" or "global".
      - prob_matrix: Local infection probability matrix (n x n).
      - use_prod: Whether to aggregate infection probabilities using a product.
      - degree_bias: Degree-based transmission scaling; None disables scaling.
      - initial_infected: Initially infected nodes.
      - seed: Random seed.
    """
    random_seed = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    snapshots = np.zeros((num_timesteps, n))
    expected_nodes = set(range(n))
    actual_nodes = set(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError(
            "Input graph G is required to have integer node IDs from 0 to n-1 "
            f"for direct indexing into NumPy arrays. Found node IDs: {list(G.nodes())[:5]}..."
        )
    neighbors = {u: list(G.neighbors(u)) for u in nodes}
    degrees = np.array([G.degree(u) for u in nodes], dtype=float)

    if mode == "global":
        P = nx.to_numpy_array(G, weight=None) * beta
    elif mode == "local":
        if prob_matrix is None:
            raise ValueError(
                "When 'mode' is set to 'local', a custom 'prob_matrix' (N x N NumPy array) "
                "must be provided to define pairwise local probabilities."
            )
        else:
            P = np.array(prob_matrix, dtype=np.float64)
            if P.shape != (n, n):
                raise ValueError(
                    f"Provided 'prob_matrix' shape {P.shape} is incorrect. "
                    f"It must match the number of nodes ({n}, {n})."
                )
    else:
        raise ValueError(
            f"Invalid value for parameter 'mode': '{mode}'. "
            "Must be either 'global' or 'local'."
        )

    if initial_infected is None:
        source = {random_seed.integers(0, n)}

    elif isinstance(initial_infected, float):
        if not 0.0 < initial_infected <= 1.0:
            raise ValueError(
                f"If 'initial_infected' is a float, it must be a percentage between 0.0 and 1.0. "
                f"Received value: {initial_infected}"
            )
        num_sources = math.ceil(n * initial_infected)
        # Randomly select num_sources source nodes
        source_indices = random_seed.choice(n, size=num_sources, replace=False)
        source = set(source_indices)

    elif isinstance(initial_infected, int):
        if not 0 < initial_infected <= n:
            raise ValueError(
                f"If 'initial_infected' is an integer, it must be a count between 1 and the total number of nodes ({n}). "
                f"Received value: {initial_infected}"
            )
        # Randomly select initial_infected source nodes
        source_indices = random_seed.choice(n, size=initial_infected, replace=False)
        source = set(source_indices)

    elif isinstance(initial_infected, list):
        for idx in initial_infected:
            if not isinstance(idx, int) or not (0 <= idx < n):
                raise ValueError(
                    f"Initial infected list contains invalid index/ID '{idx}'. "
                    f"Since the graph is indexed 0 to n-1, all elements in the list must be integers in this range.")
        source = set(initial_infected)

    else:
        raise ValueError(
            f"Invalid type for 'initial_infected'. Expected types are 'None', 'int' (count), "
            f"'float' (percentage), or 'list' (explicit indices/IDs). "
            f"Received type: {type(initial_infected).__name__}"
        )
    for idx in source:
        snapshots[0, idx] = 1

    # Degree-based scaling
    if degree_bias is not None and degree_bias != 0.0:
        deg_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-8)
        deg_scale = 1.0 + degree_bias * (deg_norm - 0.5)
        # Scale transmission probabilities
        for u in range(n):
            P[u, :] *= deg_scale[u]
        # Clip probabilities to [0, 1]
        P = np.clip(P, 0.0, 1.0)

    # Simulate propagation
    for t in range(1, num_timesteps):
        prev_state = snapshots[t - 1].copy()

        # Compute the current infected fraction
        current_infected_ratio = prev_state.sum() / n
        if current_infected_ratio >= infection_scale:
            snapshots[t:] = prev_state
            break
        new_state = prev_state.copy()
        susceptible_nodes = np.where(prev_state == 0)[0]

        for v in susceptible_nodes:
            inf_neighbors = [u for u in neighbors[v] if prev_state[u] == 1]
            if not inf_neighbors:
                continue

            if use_prod:
                # Aggregate infection probabilities using a product
                prod_term = 1.0
                for u in inf_neighbors:
                    prod_term *= (1.0 - P[u, v])
                p = 1.0 - prod_term
                if random_seed.random() < p:
                    new_state[v] = 1
            else:
                for u in inf_neighbors:
                    if random_seed.random() < P[u, v]:
                        new_state[v] = 1
                        break

        snapshots[t] = new_state
    adj_matrix = nx.to_numpy_array(G, weight=None)

    return adj_matrix, P, snapshots


def SIRModel(G: nx.Graph, num_timesteps: int, beta: float = 0.05, gamma: float = 0.02, mode: str = "global", prob_matrix: Optional[np.ndarray] = None,
    infection_scale: Optional[float] = None, initial_infected: Optional[Union[float, int, list, str]] = None, seed: Optional[int] = None, use_prod: Optional[bool] = True, degree_bias: Optional[float] = None, reset_recovered: bool = False):
    """
    Simulate SIR propagation.
    Return an array of shape (num_timesteps, n_nodes) with node states:
        0 = susceptible (S)
        1 = infected (I)
        2 = recovered (R)
    """
    random_seed = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    snapshots = np.zeros((num_timesteps, n))
    expected_nodes = set(range(n))
    actual_nodes = set(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError("Graph node IDs must be 0...(n-1).")

    neighbors = {u: list(G.neighbors(u)) for u in nodes}
    degrees = np.array([G.degree(u) for u in nodes], dtype=float)

    # Infection probability matrix
    if mode == "global":
        P = nx.to_numpy_array(G, weight=None) * beta
    elif mode == "local":
        if prob_matrix is None:
            raise ValueError("Local mode requires prob_matrix.")
        P = np.array(prob_matrix, dtype=np.float64)
        if P.shape != (n, n):
            raise ValueError("Invalid prob_matrix shape.")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Initially infected nodes
    if initial_infected is None:
        source = {random_seed.integers(0, n)}
    elif isinstance(initial_infected, float):
        num_sources = math.ceil(n * initial_infected)
        source = set(random_seed.choice(n, size=num_sources, replace=False))
    elif isinstance(initial_infected, int):
        source = set(random_seed.choice(n, size=initial_infected, replace=False))
    elif isinstance(initial_infected, list):
        source = set(initial_infected)
    else:
        raise ValueError("Invalid initial_infected type.")

    for idx in source:
        snapshots[0, idx] = 1  # Initial infection state

    # Degree-based scaling
    if degree_bias is not None and degree_bias != 0.0:
        deg_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-8)
        deg_scale = 1.0 + degree_bias * (deg_norm - 0.5)
        for u in range(n):
            P[u, :] *= deg_scale[u]
        P = np.clip(P, 0.0, 1.0)

    # Simulate propagation
    for t in range(1, num_timesteps):
        prev_state = snapshots[t - 1].copy()
        new_state = prev_state.copy()

        if infection_scale is not None:
            total_infected_ratio = np.sum(prev_state > 0) / n
            if total_infected_ratio >= infection_scale:
                # Freeze the state once the size threshold is reached
                snapshots[t:] = prev_state
                break

        # Recovery phase: infected nodes may recover
        infected_nodes = np.where(prev_state == 1)[0]
        for u in infected_nodes:
            if random_seed.random() < gamma:
                new_state[u] = 2  # Transition to recovered

        # Infection phase: susceptible nodes may become infected
        susceptible_nodes = np.where(prev_state == 0)[0]
        for v in susceptible_nodes:
            inf_neighbors = [u for u in neighbors[v] if prev_state[u] == 1]
            if not inf_neighbors:
                continue
            if use_prod:
                prod_term = 1.0
                for u in inf_neighbors:
                    prod_term *= (1.0 - P[u, v])
                p = 1.0 - prod_term
                if random_seed.random() < p:
                    new_state[v] = 1
            else:
                for u in inf_neighbors:
                    if random_seed.random() < P[u, v]:
                        new_state[v] = 1
                        break

        snapshots[t] = new_state
    if reset_recovered:
        snapshots[snapshots == 2] = 0
    adj_matrix = nx.to_numpy_array(G, weight=None)

    return adj_matrix, P, snapshots


def ICModel(G: nx.Graph, max_timesteps: int, infection_prob: float = 0.1, initial_infected: Optional[Union[float, int, list]] = 0.1,
    stop_ratio: float = 0.3, seed: Optional[int] = None, degree_bias: Optional[float] = None,):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = G.number_of_nodes()

    expected_nodes = set(range(n))
    actual_nodes = set(nodes)
    if actual_nodes != expected_nodes:
        raise ValueError(
            "Graph node IDs must be 0...(n-1) for direct indexing. "
            f"Found: {list(G.nodes())[:5]}..."
        )

    neighbors = {u: list(G.neighbors(u)) for u in nodes}
    degrees = np.array([G.degree(u) for u in nodes], dtype=float)

    # Base infection probability matrix: infection_prob on every edge
    adj_matrix = nx.to_numpy_array(G, weight=None)
    P = adj_matrix * float(infection_prob)

    # Degree-based scaling
    if degree_bias is not None and degree_bias != 0.0:
        # Normalize to [0, 1]
        deg_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-8)
        deg_scale = 1.0 + degree_bias * (deg_norm - 0.5)
        for u in range(n):
            P[u, :] *= deg_scale[u]
        P = np.clip(P, 0.0, 1.0)

    # Initially infected nodes
    if initial_infected is None:
        # Select one random source if none is specified
        source = {rng.integers(0, n)}
    elif isinstance(initial_infected, float):
        # Select by proportion
        num_sources = max(1, math.ceil(n * initial_infected))
        source = set(rng.choice(n, size=num_sources, replace=False))
    elif isinstance(initial_infected, int):
        # Select by count
        num_sources = max(1, initial_infected)
        source = set(rng.choice(n, size=num_sources, replace=False))
    elif isinstance(initial_infected, list):
        source = set(initial_infected)
    else:
        raise ValueError("Invalid initial_infected type.")

    # Initialize IC states
    snapshots = []
    current_infected = set(source)      # All nodes infected so far
    new_infected = set(source)          # Nodes newly infected at the previous step; only these nodes propagate in IC

    # Infection snapshot at t = 0
    state = np.zeros(n, dtype=int)
    for idx in current_infected:
        state[idx] = 1
    snapshots.append(state.copy())

    t = 1
    while new_infected and t < max_timesteps:
        if stop_ratio is not None:
            infection_ratio = len(current_infected) / n
            if infection_ratio >= stop_ratio:
                break

        newly_infected_this_step = set()

        # IC step: newly infected nodes try to infect susceptible neighbors
        for u in new_infected:
            for v in neighbors[u]:
                if v in current_infected:
                    continue  # Previously infected nodes cannot be infected again
                p = P[u, v]
                if p <= 0.0:
                    continue
                if rng.random() < p:
                    newly_infected_this_step.add(v)

        if not newly_infected_this_step:
            # No new infections; terminate propagation
            break

        current_infected.update(newly_infected_this_step)
        new_infected = newly_infected_this_step

        # Record the cumulative infection state
        state = np.zeros(n, dtype=int)
        for idx in current_infected:
            state[idx] = 1
        snapshots.append(state.copy())
        t += 1

    return adj_matrix, P, snapshots
