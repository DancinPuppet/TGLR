import os
from pathlib import Path
import ast
import networkx as nx
import args
import node_feature
import pickle
from propagation_model import SIModel, SIRModel, ICModel
import numpy as np
import scipy.sparse as sp


def twitter25(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    # Used to construct snapshot sequences
    T = int(getattr(args_para, "num_timesteps", 50))
    for subfolder in root_dir.iterdir():
        if subfolder.is_dir():
            print(f'loading {subfolder.name}\n')
            graph_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_single_graph_pro.json'
            review_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_review.json'
            quote_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_quote.json'
            G_graph = nx.Graph()
            # Read the graph
            node_first_infected_time = {}
            with open(graph_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_strip = line.strip()
                    left_str, right_str = line_strip.split("->")
                    try:
                        left_list = ast.literal_eval(left_str)
                        right_list = ast.literal_eval(right_str)
                    except Exception as e:
                        print(f"Parse error: {line}\n{e}")
                        continue

                    left_type, left_user, left_event, left_time = left_list
                    right_type, right_user, right_event, right_time = right_list

                    left_user = str(left_user)
                    right_user = str(right_user)
                    # Record timestamps for snapshot construction
                    left_time = int(left_time)
                    right_time = int(right_time)

                    # if left_user not in G_graph:
                    #     if left_type == 0:
                    #         G_graph.add_node(left_user, source=1)
                    #     else:
                    #         G_graph.add_node(left_user, source=0)
                    if left_user not in G_graph:
                        G_graph.add_node(left_user, source=0)
                    if right_user not in G_graph:
                        G_graph.add_node(right_user, source=0)

                    node_first_infected_time[left_user] = min(node_first_infected_time.get(left_user, left_time), left_time)
                    node_first_infected_time[right_user] = min(node_first_infected_time.get(right_user, right_time), right_time)
                    G_graph.add_edge(left_user, right_user)

            # Sort by infection time and select the earliest 10% as sources
            if node_first_infected_time:
                sorted_nodes = sorted(node_first_infected_time.items(), key=lambda x: x[1])
                num_sources = max(1, int(len(sorted_nodes) * 0.1))
                source_nodes = {node for node, _ in sorted_nodes[:num_sources]}

                # Mark source nodes
                for node in source_nodes:
                    G_graph.nodes[node]['source'] = 1

                print(f'{subfolder.name}: total nodes={len(sorted_nodes)}, source nodes={num_sources}')
            else:
                source_nodes = set()

            # Construct snapshots with shape (T, N)
            node_list = list(G_graph.nodes())
            N = len(node_list)
            print(f"Node count before snapshot construction: {N}")
            idx = {u: i for i, u in enumerate(node_list)}
            if len(node_first_infected_time) < N:
                t_fallback = max(node_first_infected_time.values()) if node_first_infected_time else 0
                for u in node_list:
                    node_first_infected_time.setdefault(u, t_fallback)

            times = np.array([node_first_infected_time[u] for u in node_list], dtype=np.int64)
            t_min, t_max = int(times.min()), int(times.max())
            if t_max == t_min:
                thr = np.full((T,), t_min, dtype=np.int64)
            else:
                thr = np.quantile(times, np.linspace(0.0, 1.0, T)).astype(np.int64)

            snapshots = np.zeros((T, N), dtype=np.float32)
            for k in range(T):
                snapshots[k, :] = (times <= thr[k]).astype(np.float32)
            snapshots[-1, :] = 1.0  # All nodes are infected at the final time step

            # Set the observation time step
            T_keep = min(T, 30)
            # graph-level attributes
            G_graph.graph['snapshots'] = snapshots[:T_keep,:]
            G_graph.graph['node_list'] = node_list
            G_graph.graph['source_nodes'] = source_nodes

            # node-level attributes: state
            obs_t = T_keep - 1
            for u in node_list:
                i = idx[u]
                # Set the infection state at the selected observation time step
                G_graph.nodes[u]['state'] = float(snapshots[obs_t, i])  # Typically 1.0

            print(f"Node count before compute_node_features: {G_graph.number_of_nodes()}")
            # Compute structural features on the graph and attach them to nodes
            if not args_para.features_with_profiles:
                G_new_graph = node_feature.compute_node_features(G_graph)
            else:
                G_new_graph = node_feature.compute_node_features_with_profiles(G_graph, review_file_path, quote_file_path)

            print(f"Node count after compute_node_features: {G_new_graph.number_of_nodes()}")
            G_new_graph.id = subfolder.name
            graphs.append(G_new_graph)

    # Compute the maximum number of nodes
    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Maximum graph node count: {max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs


def twitter15(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    T = int(getattr(args_para, "num_timesteps", 50))
    for file in root_dir.glob('*.txt'):
        G_graph = nx.Graph()
        node_first_infected_time = {}
        print(f'processing {file}')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                left_str, right_str = line_strip.split("->")
                try:
                    left_list = ast.literal_eval(left_str)
                    right_list = ast.literal_eval(right_str)
                except Exception as e:
                    print(f"Parse error: {line}\n{e}")
                    continue

                left_user, left_event, left_time = left_list
                right_user, right_event, right_time = right_list

                if 'ROOT' in left_user:
                    continue

                if left_user == right_user:
                    continue

                # if int(left_user) not in G_graph:
                #     if left_time == '0.0':
                #         G_graph.add_node(int(left_user), source=1)
                #     else:
                #         G_graph.add_node(int(left_user), source=0)
                if int(left_user) not in G_graph:
                    G_graph.add_node(int(left_user), source=0)
                if int(right_user) not in G_graph:
                    G_graph.add_node(int(right_user), source=0)

                lt = float(left_time)
                rt = float(right_time)
                node_first_infected_time[int(left_user)] = min(node_first_infected_time.get(int(left_user), lt), lt)
                node_first_infected_time[int(right_user)] = min(node_first_infected_time.get(int(right_user), rt), rt)

                G_graph.add_edge(int(left_user), int(right_user))
            num_components = nx.number_connected_components(G_graph)
            assert num_components == 1, 'Error: graph is disconnected'

        # Sort by infection time and select the earliest 10% as sources
        if node_first_infected_time:
            sorted_nodes = sorted(node_first_infected_time.items(), key=lambda x: x[1])
            num_sources = max(1, int(len(sorted_nodes) * 0.1))
            source_nodes = {node for node, _ in sorted_nodes[:num_sources]}

            # Mark source nodes
            for node in source_nodes:
                G_graph.nodes[node]['source'] = 1

            print(f'{file.stem}: total nodes={len(sorted_nodes)}, source nodes={num_sources}')
        else:
            source_nodes = set()

        node_list = list(G_graph.nodes())
        N = len(node_list)
        idx = {u: i for i, u in enumerate(node_list)}

        if len(node_first_infected_time) < N:
            t_fallback = max(node_first_infected_time.values()) if node_first_infected_time else 0.0
            for u in node_list:
                node_first_infected_time.setdefault(u, t_fallback)

        times = np.array([node_first_infected_time[u] for u in node_list], dtype=np.float64)
        t_min, t_max = float(times.min()), float(times.max())

        if t_max == t_min:
            thr = np.full((T,), t_min, dtype=np.float64)
        else:
            thr = np.quantile(times, np.linspace(0.0, 1.0, T))  # Floating-point thresholds

        snapshots = np.zeros((T, N), dtype=np.float32)
        for k in range(T):
            snapshots[k, :] = (times <= thr[k]).astype(np.float32)
        snapshots[-1, :] = 1.0

        # Set the observation time step
        T_keep = min(T, 30)
        # graph-level attributes
        G_graph.graph['snapshots'] = snapshots[:T_keep,:]
        G_graph.graph['node_list'] = node_list
        G_graph.graph['source_nodes'] = source_nodes

        # Node states at observation step obs_t
        obs_t = T_keep - 1
        for u in node_list:
            i = idx[u]
            G_graph.nodes[u]['state'] = float(snapshots[obs_t, i])

        # Compute node features
        G_new_graph = node_feature.compute_node_features(G_graph)
        G_new_graph.id = file.stem
        graphs.append(G_new_graph)

    # Compute the maximum number of nodes
    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Maximum graph node count: {max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs


def twitter16(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    T = int(getattr(args_para, "num_timesteps", 50))
    for file in root_dir.glob('*.txt'):
        G_graph = nx.Graph()
        node_first_infected_time = {}
        print(f'processing {file}')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                left_str, right_str = line_strip.split("->")
                try:
                    left_list = ast.literal_eval(left_str)
                    right_list = ast.literal_eval(right_str)
                except Exception as e:
                    print(f"Parse error: {line}\n{e}")
                    continue

                left_user, left_event, left_time = left_list
                right_user, right_event, right_time = right_list

                if 'ROOT' in left_user:
                    continue

                if left_user == right_user:
                    continue

                # if int(left_user) not in G_graph:
                #     if left_time == '0.0':
                #         G_graph.add_node(int(left_user), source=1)
                #     else:
                #         G_graph.add_node(int(left_user), source=0)
                if int(left_user) not in G_graph:
                    G_graph.add_node(int(left_user), source=0)
                if int(right_user) not in G_graph:
                    G_graph.add_node(int(right_user), source=0)

                lt = float(left_time)
                rt = float(right_time)
                node_first_infected_time[int(left_user)] = min(node_first_infected_time.get(int(left_user), lt), lt)
                node_first_infected_time[int(right_user)] = min(node_first_infected_time.get(int(right_user), rt), rt)

                G_graph.add_edge(int(left_user), int(right_user))
            num_components = nx.number_connected_components(G_graph)
            assert num_components == 1, 'Error: graph is disconnected'

        # Sort by infection time and select the earliest 10% as sources
        if node_first_infected_time:
            sorted_nodes = sorted(node_first_infected_time.items(), key=lambda x: x[1])
            num_sources = max(1, int(len(sorted_nodes) * 0.1))
            source_nodes = {node for node, _ in sorted_nodes[:num_sources]}

            # Mark source nodes
            for node in source_nodes:
                G_graph.nodes[node]['source'] = 1

            print(f'{file.stem}: total nodes={len(sorted_nodes)}, source nodes={num_sources}')
        else:
            source_nodes = set()

        node_list = list(G_graph.nodes())
        N = len(node_list)
        idx = {u: i for i, u in enumerate(node_list)}

        if len(node_first_infected_time) < N:
            t_fallback = max(node_first_infected_time.values()) if node_first_infected_time else 0.0
            for u in node_list:
                node_first_infected_time.setdefault(u, t_fallback)

        times = np.array([node_first_infected_time[u] for u in node_list], dtype=np.float64)
        t_min, t_max = float(times.min()), float(times.max())

        if t_max == t_min:
            thr = np.full((T,), t_min, dtype=np.float64)
        else:
            thr = np.quantile(times, np.linspace(0.0, 1.0, T))  # Floating-point thresholds

        snapshots = np.zeros((T, N), dtype=np.float32)
        for k in range(T):
            snapshots[k, :] = (times <= thr[k]).astype(np.float32)
        snapshots[-1, :] = 1.0

        # Set the observation time step
        T_keep = min(T, 30)
        # graph-level attributes
        G_graph.graph['snapshots'] = snapshots[:T_keep, :]
        G_graph.graph['node_list'] = node_list
        G_graph.graph['source_nodes'] = source_nodes

        # Node states at observation step obs_t
        obs_t = T_keep - 1
        for u in node_list:
            i = idx[u]
            G_graph.nodes[u]['state'] = float(snapshots[obs_t, i])
        # Compute structural features on the graph and attach them to nodes
        G_new_graph = node_feature.compute_node_features(G_graph)
        G_new_graph.id = file.stem
        graphs.append(G_new_graph)

    # Compute the maximum number of nodes
    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Maximum graph node count: {max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs


def karate(args_para: args.Args, spread_model: str, num_graphs: int = 100, num_timesteps: int = 30) -> dict:
    """
    Generate propagation data on the Karate network.

    Args:
        args_para: Configuration object.
        spread_model: Propagation model ("SI" or "SIR").
        num_graphs: Number of samples to generate.
        num_timesteps: Number of time steps per sample.

    Returns:
        dict: Dictionary containing adjacency, infection probability, and influence matrix lists.
    """
    graph_data_path = args_para.data_path
    graph_data_path = graph_data_path.replace(f"_{spread_model}/", "/") + "karate_graph.mtx"

    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    print(f"Karate club graph has {n_nodes} nodes and {G.number_of_edges()} edges.")
    # Set propagation parameters
    seed = 1
    influ_mat_list = []
    if spread_model == "SIR":
        beta = 0.05
        gamma = 0.02
    else:
        beta = 0.05

    print(f"Generating {num_graphs} samples for {spread_model} model...")

    for i in range(num_graphs):
        # initial_infected = random.uniform(0.1, 0.3)
        initial_infected = 0.1
        infection_scale = 0.5
        if spread_model == "SI":
            adj_matrix, prob_matrix, snapshots = SIModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                mode="global",
                initial_infected=initial_infected,
                infection_scale = infection_scale,
                seed=seed  + i,
                use_prod=False,
                degree_bias=1.0
            )
        elif spread_model == "SIR":
            adj_matrix, prob_matrix, snapshots = SIRModel(
            G=G,
            num_timesteps=num_timesteps,
            beta=beta,
            gamma=gamma,
            mode="global",
            initial_infected=initial_infected,
            infection_scale = infection_scale,
            seed=seed  + i,
            use_prod=False,
            degree_bias=1.0,
            reset_recovered=True
        )
        elif spread_model == "IC":
            adj_matrix, prob_matrix, snapshots = ICModel(
            G=G,
            max_timesteps=num_timesteps,
            infection_prob=beta,
            initial_infected=initial_infected,
            stop_ratio=infection_scale,
            seed=seed  + i,
            degree_bias=1.0,
        )
        else:
            raise ValueError("Error, no effective model!")
        # Pad the time dimension to num_timesteps
        snapshots = np.asarray(snapshots)          # Convert to a NumPy array first

        # Expand a single snapshot from (n_nodes,) to (1, n_nodes)
        if snapshots.ndim == 1:
            snapshots = snapshots[None, :]        # (1, n_nodes)

        T_actual = snapshots.shape[0]             # Actual number of time steps
        n_nodes = snapshots.shape[1]

        if T_actual < num_timesteps:
            # Repeat the final snapshot until num_timesteps is reached
            last = snapshots[-1:, :]              # (1, n_nodes)
            pad = np.repeat(last, num_timesteps - T_actual, axis=0)  # (num_timesteps - T_actual, n_nodes)
            snapshots = np.concatenate([snapshots, pad], axis=0)     # (num_timesteps, n_nodes)
        elif T_actual > num_timesteps:
            # Truncate sequences longer than num_timesteps
            snapshots = snapshots[:num_timesteps]

        influ_mat_list.append(snapshots)

    n_edges = np.count_nonzero(adj_matrix) // 2
    data_dict = {
    'adj_matrix': adj_matrix,
    'prob_matrix': prob_matrix,
    'influ_mat_list': influ_mat_list,
    'n_nodes': n_nodes,
    'n_edges': n_edges,
    'num_samples': num_graphs,
    'num_timesteps': num_timesteps
    }

    print(f"Generated karate data: {n_nodes} nodes, {n_edges} edges, {num_graphs} samples, {num_timesteps} timesteps")

    adj_matrix = data_dict['adj_matrix']
    influ_mat_list = data_dict['influ_mat_list']
    n_nodes = data_dict['n_nodes']
    prob_matrix = data_dict['prob_matrix']

    graphs = []
    for sample_idx, snapshot in enumerate(influ_mat_list):
        # snapshot: (num_timesteps, n_nodes)
        initial_state = snapshot[0]
        final_state = snapshot[-1]
        assert len(final_state) == n_nodes

        # Construct the graph
        G_graph = nx.from_numpy_array(adj_matrix)
        node_list = list(G_graph.nodes())
        G_graph.graph['node_list'] = node_list

        G_graph.graph['snapshots'] = snapshot
        source_nodes = set(np.where(initial_state == 1)[0])
        G_graph.graph['source_nodes'] = source_nodes

        # Add node features
        for node in G_graph.nodes():
            G_graph.nodes[node]['source'] = float(initial_state[node])
            G_graph.nodes[node]['state']  = float(final_state[node])
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    print(f"Converted {len(graphs)} graphs, each with {n_nodes} nodes and 'source' feature.")
    args_para.max_num_node = n_nodes
    return graphs



def jazz(args_para: args.Args, spread_model: str, num_graphs: int = 100, num_timesteps: int = 30) -> dict:
    """
    Generate propagation data on the Jazz network.

    Args:
        args_para: Configuration object.
        spread_model: Propagation model ("SI" or "SIR").
        num_graphs: Number of samples to generate.
        num_timesteps: Number of time steps per sample.

    Returns:
        dict: Dictionary containing adjacency, infection probability, and influence matrix lists.
    """
    graph_data_path = args_para.data_path
    graph_data_path = graph_data_path.replace(f"_{spread_model}/", "/") + "jazz_graph.edges"
    G = nx.read_edgelist(graph_data_path, delimiter=",", nodetype=int, data=False)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    n_nodes = G.number_of_nodes()
    print(f"Jazz graph has {n_nodes} nodes and {G.number_of_edges()} edges.")

    seed = 1
    influ_mat_list = []
    if spread_model == "SIR":
        beta = 0.05
        gamma = 0.02
    else:
        beta = 0.05

    print(f"Generating {num_graphs} samples for {spread_model} model...")

    for i in range(num_graphs):
        initial_infected = 0.1
        infection_scale = 0.3

        if spread_model == "SI":
            adj_matrix, prob_matrix, snapshots = SIModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0
            )
        elif spread_model == "SIR":
            adj_matrix, prob_matrix, snapshots = SIRModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                gamma=gamma,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0,
                reset_recovered=True
            )
        elif spread_model == "IC":
            adj_matrix, prob_matrix, snapshots = ICModel(
            G=G,
            max_timesteps=num_timesteps,
            infection_prob=beta,
            initial_infected=initial_infected,
            stop_ratio=infection_scale,
            seed=seed  + i,
            degree_bias=1.0
        )
        else:
            raise ValueError("Error, no effective model!")

        snapshots = np.asarray(snapshots)
        # Expand a single snapshot from (n_nodes,) to (1, n_nodes)
        if snapshots.ndim == 1:
            snapshots = snapshots[None, :]        # (1, n_nodes)

        T_actual = snapshots.shape[0]
        n_nodes = snapshots.shape[1]
        if T_actual < num_timesteps:
            last = snapshots[-1:, :]
            pad = np.repeat(last, num_timesteps - T_actual, axis=0)
            snapshots = np.concatenate([snapshots, pad], axis=0)
        elif T_actual > num_timesteps:
            snapshots = snapshots[:num_timesteps]

        influ_mat_list.append(snapshots)

    n_edges = np.count_nonzero(adj_matrix) // 2
    data_dict = {
        'adj_matrix': adj_matrix,
        'prob_matrix': prob_matrix,
        'influ_mat_list': influ_mat_list,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'num_samples': num_graphs,
        'num_timesteps': num_timesteps
    }

    print(f"Generated jazz data: {n_nodes} nodes, {n_edges} edges, {num_graphs} samples, {num_timesteps} timesteps")

    adj_matrix = data_dict['adj_matrix']
    influ_mat_list = data_dict['influ_mat_list']
    n_nodes = data_dict['n_nodes']
    prob_matrix = data_dict['prob_matrix']

    graphs = []
    for sample_idx, snapshot in enumerate(influ_mat_list):
        # snapshot: (num_timesteps, n_nodes)
        initial_state = snapshot[0]
        final_state = snapshot[-1]
        assert len(final_state) == n_nodes

        # Construct the graph
        G_graph = nx.from_numpy_array(adj_matrix)
        node_list = list(G_graph.nodes())
        G_graph.graph['node_list'] = node_list

        G_graph.graph['snapshots'] = snapshot
        source_nodes = set(np.where(initial_state == 1)[0])
        G_graph.graph['source_nodes'] = source_nodes

        # Add node features
        for node in G_graph.nodes():
            G_graph.nodes[node]['source'] = float(initial_state[node])
            G_graph.nodes[node]['state'] = float(final_state[node])
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    print(f"Converted {len(graphs)} graphs, each with {n_nodes} nodes and 'source' feature.")
    args_para.max_num_node = n_nodes
    return graphs

def cora_ml(args_para: args.Args, spread_model: str, num_graphs: int = 1000, num_timesteps: int = 30) -> dict:
    """
    Generate propagation data on the Cora-ML network.

    Args:
        args_para: Configuration object.
        spread_model: Propagation model ("SI" or "SIR").
        num_graphs: Number of samples to generate.
        num_timesteps: Number of time steps per sample.

    Returns:
        dict: Dictionary containing adjacency, infection probability, and influence matrix lists.
    """
    graph_data_path = args_para.data_path
    graph_data_path = graph_data_path.replace(f"_{spread_model}/", "/") + "cora_ml.npz"
    z = np.load(graph_data_path, allow_pickle=True)

    # --- rebuild adjacency (CSR) ---
    adj = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]),
        shape=tuple(z["adj_shape"])
    )

    G = nx.from_scipy_sparse_array(adj)
    print(
        f"Cora-ML graph has {G.number_of_nodes()} nodes "
        f"and {G.number_of_edges()} edges."
    )
    # 1. Check connectivity
    if nx.is_connected(G):
        print("✅ The graph is already connected.")
    else:
        print("❌ The graph is NOT connected.")

        # 2. Find the largest connected component (LCC)
        # nx.connected_components yields sets of nodes
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        num_nodes_lcc = len(largest_cc_nodes)

        print(f"   -> Size of Largest Connected Component (LCC): {num_nodes_lcc} nodes")
        print(f"   -> Removed {G.number_of_nodes() - num_nodes_lcc} isolated/disconnected nodes.")

        # 3. Extract the subgraph
        # Copy the subgraph to obtain an independently editable graph
        G_lcc = G.subgraph(largest_cc_nodes).copy()

        # 4. Relabel nodes consecutively
        # Extracted node IDs may be nonconsecutive, e.g., [0, 1, 5, 8, ...]
        # Reset IDs to [0, 1, 2, 3, ...] to preserve matrix indexing
        G_lcc = nx.convert_node_labels_to_integers(G_lcc, first_label=0, ordering='default')

        # 5. Update G
        G = G_lcc

        print(f"✅ Graph updated to LCC. Current nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    seed = 1
    influ_mat_list = []
    if spread_model == "SIR":
        beta = 0.05
        gamma = 0.02
    else:
        beta = 0.05

    print(f"Generating {num_graphs} samples for {spread_model} model...")

    for i in range(num_graphs):
        initial_infected = 0.1
        infection_scale = 0.3

        if spread_model == "SI":
            adj_matrix, prob_matrix, snapshots = SIModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0
            )
        elif spread_model == "SIR":
            adj_matrix, prob_matrix, snapshots = SIRModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                gamma=gamma,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0,
                reset_recovered=True
            )
        elif spread_model == "IC":
            adj_matrix, prob_matrix, snapshots = ICModel(
            G=G,
            max_timesteps=num_timesteps,
            infection_prob=beta,
            initial_infected=initial_infected,
            stop_ratio=infection_scale,
            seed=seed  + i,
            degree_bias=1.0
        )
        else:
            raise ValueError("Error, no effective model!")

        snapshots = np.asarray(snapshots)
        # Expand a single snapshot from (n_nodes,) to (1, n_nodes)
        if snapshots.ndim == 1:
            snapshots = snapshots[None, :]        # (1, n_nodes)

        T_actual = snapshots.shape[0]
        n_nodes = snapshots.shape[1]
        if T_actual < num_timesteps:
            last = snapshots[-1:, :]
            pad = np.repeat(last, num_timesteps - T_actual, axis=0)
            snapshots = np.concatenate([snapshots, pad], axis=0)
        elif T_actual > num_timesteps:
            snapshots = snapshots[:num_timesteps]

        influ_mat_list.append(snapshots)

    n_edges = np.count_nonzero(adj_matrix) // 2
    data_dict = {
        'adj_matrix': adj_matrix,
        'prob_matrix': prob_matrix,
        'influ_mat_list': influ_mat_list,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'num_samples': num_graphs,
        'num_timesteps': num_timesteps
    }

    print(f"Generated jazz data: {n_nodes} nodes, {n_edges} edges, {num_graphs} samples, {num_timesteps} timesteps")

    adj_matrix = data_dict['adj_matrix']
    influ_mat_list = data_dict['influ_mat_list']
    n_nodes = data_dict['n_nodes']
    prob_matrix = data_dict['prob_matrix']

    graphs = []
    for sample_idx, snapshot in enumerate(influ_mat_list):
        # snapshot: (num_timesteps, n_nodes)
        initial_state = snapshot[0]
        final_state = snapshot[-1]
        assert len(final_state) == n_nodes

        # Construct the graph
        G_graph = nx.from_numpy_array(adj_matrix)
        node_list = list(G_graph.nodes())
        G_graph.graph['node_list'] = node_list

        G_graph.graph['snapshots'] = snapshot
        source_nodes = set(np.where(initial_state == 1)[0])
        G_graph.graph['source_nodes'] = source_nodes

        # Add node features
        for node in G_graph.nodes():
            G_graph.nodes[node]['source'] = float(initial_state[node])
            G_graph.nodes[node]['state'] = float(final_state[node])
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    print(f"Converted {len(graphs)} graphs, each with {n_nodes} nodes and 'source' feature.")
    args_para.max_num_node = n_nodes
    return graphs


def facebook(args_para: args.Args, spread_model: str, num_graphs: int = 100, num_timesteps: int = 30) -> dict:
    """
    Generate propagation data on the Facebook network.

    Args:
        args_para: Configuration object.
        spread_model: Propagation model ("SI" or "SIR").
        num_graphs: Number of samples to generate.
        num_timesteps: Number of time steps per sample.

    Returns:
        dict: Dictionary containing adjacency, infection probability, and influence matrix lists.
    """
    graph_data_path = args_para.data_path
    graph_data_path = graph_data_path.replace(f"_{spread_model}/", "/") + "facebook_edge.txt"

    G = nx.read_edgelist(graph_data_path, delimiter=" ", nodetype=int, data=False)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")

    print(
        f"Facebook graph has {G.number_of_nodes()} nodes "
        f"and {G.number_of_edges()} edges."
    )

    seed = 1
    influ_mat_list = []
    if spread_model == "SIR":
        beta = 0.05
        gamma = 0.02
    else:
        beta = 0.05

    print(f"Generating {num_graphs} samples for {spread_model} model...")

    for i in range(num_graphs):
        initial_infected = 0.1
        infection_scale = 0.3

        if spread_model == "SI":
            adj_matrix, prob_matrix, snapshots = SIModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0
            )
        elif spread_model == "SIR":
            adj_matrix, prob_matrix, snapshots = SIRModel(
                G=G,
                num_timesteps=num_timesteps,
                beta=beta,
                gamma=gamma,
                mode="global",
                initial_infected=initial_infected,
                infection_scale=infection_scale,
                seed=seed + i,
                use_prod=False,
                degree_bias=1.0,
                reset_recovered=True
            )
        elif spread_model == "IC":
            adj_matrix, prob_matrix, snapshots = ICModel(
            G=G,
            max_timesteps=num_timesteps,
            infection_prob=beta,
            initial_infected=initial_infected,
            stop_ratio=infection_scale,
            seed=seed  + i,
            degree_bias=1.0
        )
        else:
            raise ValueError("Error, no effective model!")

        snapshots = np.asarray(snapshots)
        # Expand a single snapshot from (n_nodes,) to (1, n_nodes)
        if snapshots.ndim == 1:
            snapshots = snapshots[None, :]        # (1, n_nodes)

        T_actual = snapshots.shape[0]
        n_nodes = snapshots.shape[1]
        if T_actual < num_timesteps:
            last = snapshots[-1:, :]
            pad = np.repeat(last, num_timesteps - T_actual, axis=0)
            snapshots = np.concatenate([snapshots, pad], axis=0)
        elif T_actual > num_timesteps:
            snapshots = snapshots[:num_timesteps]

        influ_mat_list.append(snapshots)

    n_edges = np.count_nonzero(adj_matrix) // 2
    data_dict = {
        'adj_matrix': adj_matrix,
        'prob_matrix': prob_matrix,
        'influ_mat_list': influ_mat_list,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'num_samples': num_graphs,
        'num_timesteps': num_timesteps
    }

    print(f"Generated jazz data: {n_nodes} nodes, {n_edges} edges, {num_graphs} samples, {num_timesteps} timesteps")

    adj_matrix = data_dict['adj_matrix']
    influ_mat_list = data_dict['influ_mat_list']
    n_nodes = data_dict['n_nodes']
    prob_matrix = data_dict['prob_matrix']

    graphs = []
    for sample_idx, snapshot in enumerate(influ_mat_list):
        # snapshot: (num_timesteps, n_nodes)
        initial_state = snapshot[0]
        final_state = snapshot[-1]
        assert len(final_state) == n_nodes

        # Construct the graph
        G_graph = nx.from_numpy_array(adj_matrix)
        node_list = list(G_graph.nodes())
        G_graph.graph['node_list'] = node_list

        G_graph.graph['snapshots'] = snapshot
        source_nodes = set(np.where(initial_state == 1)[0])
        G_graph.graph['source_nodes'] = source_nodes

        # Add node features
        for node in G_graph.nodes():
            G_graph.nodes[node]['source'] = float(initial_state[node])
            G_graph.nodes[node]['state'] = float(final_state[node])
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    print(f"Converted {len(graphs)} graphs, each with {n_nodes} nodes and 'source' feature.")
    args_para.max_num_node = n_nodes
    return graphs

def digg(args_para: args.Args) -> list:
    graphs = []
    graphs_path = args_para.data_path + 'digg_graph.mtx'
    G = nx.Graph()
    with open(graphs_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue  # Skip comments and empty lines
            parts = line.split()
            if len(parts) == 3:
                n_rows, n_cols, n_edges = map(int, parts)
            elif len(parts) == 2:
                i, j = map(int, parts)
                G.add_edge(i - 1, j - 1)

    return graphs

def weibo(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    graphs_path = root_dir / 'WeiboGraphs'
    T = int(getattr(args_para, "num_timesteps", 50))

    for file in graphs_path.glob('*_graph.json'):
        G_graph = nx.Graph()
        node_first_infected_time = {}
        print(f'processing {file}')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                if "->" not in line_strip:
                    continue

                left_str, right_str = line_strip.split("->", 1)
                try:
                    left_list = ast.literal_eval(left_str)
                    right_list = ast.literal_eval(right_str)
                except Exception as e:
                    print(f"Parse error: {line_strip}\n{e}")
                    continue

                if len(left_list) != 4 or len(right_list) != 4:
                    continue

                left_flag, left_user, left_mid, left_time = left_list
                right_flag, right_user, right_mid, right_time = right_list

                if int(left_user) == int(right_user):
                    continue

                # if int(left_user) not in G_graph:
                #     G_graph.add_node(int(left_user), source=1 if int(left_flag) == 0 else 0)
                # else:
                #     if int(left_flag) == 0:
                #         G_graph.nodes[int(left_user)]['source'] = 1

                # if int(right_user) not in G_graph:
                #     G_graph.add_node(int(right_user), source=1 if int(right_flag) == 0 else 0)
                # else:
                #     if int(right_flag) == 0:
                #         G_graph.nodes[int(right_user)]['source'] = 1

                if int(left_user) not in G_graph:
                    G_graph.add_node(int(left_user), source=0)
                if int(right_user) not in G_graph:
                    G_graph.add_node(int(right_user), source=0)

                lt = float(left_time)
                rt = float(right_time)
                node_first_infected_time[int(left_user)] = min(node_first_infected_time.get(int(left_user), lt), lt)
                node_first_infected_time[int(right_user)] = min(node_first_infected_time.get(int(right_user), rt), rt)

                G_graph.add_edge(int(left_user), int(right_user))

        num_components = nx.number_connected_components(G_graph)
        assert num_components == 1, 'Error: graph is disconnected'

        # Sort by infection time and select the earliest 10% as sources
        if node_first_infected_time:
            sorted_nodes = sorted(node_first_infected_time.items(), key=lambda x: x[1])
            num_sources = max(1, int(len(sorted_nodes) * 0.1))
            source_nodes = {node for node, _ in sorted_nodes[:num_sources]}

            # Mark source nodes
            for node in source_nodes:
                G_graph.nodes[node]['source'] = 1

            print(f'{file.stem}: total nodes={len(sorted_nodes)}, source nodes={num_sources}')
        else:
            source_nodes = set()

        # Construct snapshots
        node_list = list(G_graph.nodes())
        N = len(node_list)
        idx = {u: i for i, u in enumerate(node_list)}

        if len(node_first_infected_time) < N:
            t_fallback = max(node_first_infected_time.values()) if node_first_infected_time else 0.0
            for u in node_list:
                node_first_infected_time.setdefault(u, t_fallback)

        times = np.array([node_first_infected_time[u] for u in node_list], dtype=np.float64)
        t_min, t_max = float(times.min()), float(times.max())

        if t_max == t_min:
            thr = np.full((T,), t_min, dtype=np.float64)
        else:
            thr = np.quantile(times, np.linspace(0.0, 1.0, T))

        snapshots = np.zeros((T, N), dtype=np.float32)
        for k in range(T):
            snapshots[k, :] = (times <= thr[k]).astype(np.float32)
        snapshots[-1, :] = 1.0

        # Set the observation time step
        T_keep = min(T, 30)

        # graph-level attributes
        G_graph.graph['snapshots'] = snapshots[:T_keep, :]
        G_graph.graph['node_list'] = node_list
        G_graph.graph['source_nodes'] = source_nodes

        # node-level state
        obs_t = T_keep - 1
        for u in node_list:
            i = idx[u]
            G_graph.nodes[u]['state'] = float(snapshots[obs_t, i])

        G_new_graph = node_feature.compute_node_features(G_graph)
        stem = file.stem
        event_id = stem.split("_graph")[0]
        G_new_graph.id = event_id

        graphs.append(G_new_graph)

    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Maximum graph node count: {max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('Weibo graphs loaded, connected (or largest-CC), features computed !')

    return graphs



def data(args_para: args.Args, load_exited_graphs=False):
    if load_exited_graphs and not Path(args_para.graphs_saved_path).is_file():
        raise FileNotFoundError(
            f"Missing dataset: {args_para.graphs_saved_path}. "
            "See README.md for dataset availability and placement instructions."
        )
    if not load_exited_graphs:
        Path(args_para.graphs_saved_path).parent.mkdir(parents=True, exist_ok=True)
    graphs = []
    print(f"{args_para.graph_type} data loading......")
    if args_para.graph_type == 'twitter25':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter25(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'twitter15':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter15(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'twitter16':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter16(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'weibo':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = weibo(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if 'karate' in args_para.graph_type:
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited karate data from {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            if 'SIR' in args_para.graph_type:
                graphs = karate(args_para, 'SIR')
            elif 'IC' in args_para.graph_type:
                graphs = karate(args_para, 'IC')
            elif 'LT' in args_para.graph_type:
                graphs = karate(args_para, 'LT')
            else:
                graphs = karate(args_para, 'SI')
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if 'jazz' in args_para.graph_type:
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            if 'SIR' in args_para.graph_type:
                graphs = jazz(args_para, 'SIR')
            elif 'IC' in args_para.graph_type:
                graphs = jazz(args_para, 'IC')
            elif 'LT' in args_para.graph_type:
                graphs = jazz(args_para, 'LT')
            else:
                graphs = jazz(args_para, 'SI')
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if 'cora_ml' in args_para.graph_type:
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            if 'SIR' in args_para.graph_type:
                graphs = cora_ml(args_para, 'SIR')
            elif 'IC' in args_para.graph_type:
                graphs = cora_ml(args_para, 'IC')
            elif 'LT' in args_para.graph_type:
                graphs = cora_ml(args_para, 'LT')
            else:
                graphs = cora_ml(args_para, 'SI')
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if 'facebook' in args_para.graph_type:
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            if 'SIR' in args_para.graph_type:
                graphs = facebook(args_para, 'SIR')
            elif 'IC' in args_para.graph_type:
                graphs = facebook(args_para, 'IC')
            elif 'LT' in args_para.graph_type:
                graphs = facebook(args_para, 'LT')
            else:
                graphs = facebook(args_para, 'SI')
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'digg':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = digg(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")
    for i, graph in enumerate(graphs):
        graph.graph['id'] = i
    return graphs
