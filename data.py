import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp

def compute_composite_score(features: dict[str, float]) -> float:
    # Weights for the composite node score
    weights = {
        'degree_centrality': 0.50,
        'clustering': 0.05,
        'betweenness_centrality': 0.20,
        'pagerank': 0.05,
        'avg_neighbor_degree': 0.20
    }
    score = 0
    for feature, weight in weights.items():
        if feature in features:
            score += weight * features[feature]

    assert score <= 1.0, "Error: the computed score exceeds the expected range"
    return score


def generate_node_mapping(G: nx.Graph):
    node_scores = {}
    for node in G.nodes():
        feature = G.nodes[node]
        score = compute_composite_score(feature)
        node_scores[node] = score

    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=False)
    node_mapping = {node: i for i, (node, _) in enumerate(sorted_nodes)}

    return node_mapping


def nx_to_pyg(graph, args, return_node_mapping=False):
    """
    nx.Graph -> pyg.Data
    """

    if len(graph.nodes()) == 0:
        data = Data(
            x=torch.empty((0, 0)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.empty((0,), dtype=torch.long),
        )
        return (data, {}) if return_node_mapping else data

    snapshots = graph.graph.get('snapshots', None)
    node_list = graph.graph.get('node_list', None)

    num_nodes = graph.number_of_nodes()

    snapshots_full = None
    T_full = 0

    if snapshots is not None:
        if node_list is not None:
            if args.scored_based_mapping:
                node_mapping = generate_node_mapping(graph)
            else:
                nodes = list(graph.nodes())
                # random.shuffle(nodes)
                node_mapping = {node: i for i, node in enumerate(nodes)}

            forward_perm = np.array([node_mapping[u] for u in node_list])
            inverse_perm = np.argsort(forward_perm)
            col_perm_tensor = torch.from_numpy(inverse_perm).long()
            snapshots_reordered = snapshots[:, col_perm_tensor]
            snapshots_full = torch.tensor(snapshots_reordered, dtype=torch.float)
            T_full = snapshots_full.size(0)
        else:
            nodes = list(graph.nodes())
            node_mapping = {node: i for i, node in enumerate(nodes)}
            snapshots_full = torch.tensor(snapshots, dtype=torch.float)
            T_full = snapshots_full.size(0)

    Y_last_full = None
    if snapshots_full is not None and T_full > 0:
        Y_last_full = snapshots_full[-1]   # [N]

    Y = None
    T = 0
    obs_start = 0
    obs_len = 0

    if snapshots_full is not None:
        T1 = int(args.obs_len)
        if T1 > T_full:
            raise ValueError(f"args.obs_len={T1} > T_full={T_full}.")

        obs_start = args.ob_start_idx

        if obs_start < 0 or obs_start + T1 > T_full:
            raise ValueError(f"Invalid start_idx={obs_start} for T1={T1}, T_full={T_full}")

        Y = snapshots_full[obs_start: obs_start + T1]  # [T1, N]
        T = T1
        obs_len = T1


    edge_index = []
    for src, dst, _attr in graph.edges(data=True):
        edge_index.append([node_mapping[src], node_mapping[dst]])
        edge_index.append([node_mapping[dst], node_mapping[src]])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    node_states = torch.zeros(num_nodes, dtype=torch.long)
    for node, features in graph.nodes(data=True):
        idx = node_mapping[node]
        node_labels[idx] = int(features.get("source", 0))
        node_states[idx] = int(features.get("state", 0))

    node_states[node_states == 0] = -1

    struct_names = ["degree_centrality", "avg_neighbor_degree", "clustering", "core_number", "pagerank"]
    struct_feat = torch.zeros((num_nodes, 5), dtype=torch.float)
    for node, mapped_idx in node_mapping.items():
        nd = graph.nodes[node]
        for j, name in enumerate(struct_names):
            struct_feat[mapped_idx, j] = float(nd.get(name, 0.0))

    if not args.approximate:
        nodes_in_order = [u for u, _ in sorted(node_mapping.items(), key=lambda kv: kv[1])]
        W_matrix = nx.to_numpy_array(graph, nodelist=nodes_in_order).astype(float)
        I_matrix = np.eye(num_nodes)
        degrees = np.array([graph.degree(node) for node in nodes_in_order], dtype=float)
        degrees[degrees == 0] = 1.0
        D_invert_sqrt = np.diag(1.0 / np.sqrt(degrees))
        S = D_invert_sqrt @ W_matrix @ D_invert_sqrt

        Y_vector = node_states.cpu().numpy().astype(float)  # [-1,1]
        a = 0.5
        try:
            inv_matrix = np.linalg.inv(I_matrix - a * S)
            d = (1 - a) * (inv_matrix @ Y_vector)
            lpsi_feat = torch.tensor(d, dtype=torch.float).view(-1, 1)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed; using the original features")
            lpsi_feat = node_states.float().view(-1, 1)
    else:
        nodes_in_order = [u for u, _ in sorted(node_mapping.items(), key=lambda kv: kv[1])]
        W = nx.to_scipy_sparse_array(graph, nodelist=nodes_in_order, format="csr", dtype=float)
        degrees = np.array([graph.degree(node) for node in nodes_in_order], dtype=float)
        degrees[degrees == 0] = 1.0
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ W @ D_inv_sqrt

        Y_vec = node_states.cpu().numpy().astype(float)
        alpha = 0.5
        iters = 20
        G_vec = Y_vec.copy()
        for _ in range(iters):
            G_vec = alpha * (S @ G_vec) + (1 - alpha) * Y_vec

        lpsi_feat = torch.tensor(G_vec, dtype=torch.float).view(-1, 1)

    if Y is None or T < 2:
        rise_flag = torch.zeros((num_nodes, 1), dtype=torch.float)
        t_first_inf_norm = torch.ones((num_nodes, 1), dtype=torch.float)

    else:
        up = ((Y[1:] == 1.0) & (Y[:-1] == 0.0))      # [T-1, N]
        rise_any = up.any(dim=0)                     # [N]
        rise_flag = rise_any.float().view(-1, 1)     # [N, 1]
        first_t = torch.full((num_nodes,), fill_value=T, dtype=torch.long)
        idxs = up.float().argmax(dim=0)
        first_t[rise_any] = idxs[rise_any] + 1
        t_first_inf_norm = (first_t.float() / float(T)).view(-1, 1)

    self_time_feat = torch.cat([rise_flag, t_first_inf_norm], dim=1)  # [N, 2]

    if Y is None or T == 0:
        ctx_flat = torch.zeros((num_nodes, 0), dtype=torch.float)
        rho_last_feat = torch.zeros((num_nodes, 2), dtype=torch.float)
    else:
        nodes_in_order = [u for u, _ in sorted(node_mapping.items(), key=lambda kv: kv[1])]
        A = nx.to_scipy_sparse_array(graph, nodelist=nodes_in_order, format="csr", dtype=np.float32)
        deg = np.asarray(A.sum(axis=1)).reshape(-1)  # [N]
        deg_safe = np.where(deg > 0, deg, 1.0)
        Y_np = Y.cpu().numpy().astype(np.float32)  # [T, N], OBS WINDOW ONLY

        # Infected neighbor counts: I_nb[t] = A @ Y[t]
        I_nb = np.stack([A @ Y_np[t] for t in range(T)], axis=0)  # [T, N]
        # Add a dimension with deg[None, :]
        S_nb = deg[None, :] - I_nb                                # [T, N]

        rho_I = I_nb / deg_safe[None, :]
        rho_S = S_nb / deg_safe[None, :]

        Y_last = Y_last_full.cpu().numpy().astype(np.float32)   # [N]

        # Compute infected and susceptible neighbor ratios at the final step
        I_last = (A @ Y_last)                                      # [N]
        rho_I_last = I_last / deg_safe                             # [N]
        rho_S_last = (deg - I_last) / deg_safe                     # [N]
        rho_last_feat = torch.tensor(
            np.stack([rho_I_last, rho_S_last], axis=1), dtype=torch.float
        )

        # ΔI and ΔR among neighbors between t-1 -> t
        dI = []
        dR = []
        for t in range(1, T):
            new_inf = ((Y_np[t] == 1.0) & (Y_np[t-1] == 0.0)).astype(np.float32)
            new_rec = ((Y_np[t] == 0.0) & (Y_np[t-1] == 1.0)).astype(np.float32)
            dI.append(A @ new_inf)
            dR.append(A @ new_rec)
        dI = np.stack(dI, axis=0)  # [T-1, N]
        dR = np.stack(dR, axis=0)  # [T-1, N]

        eps = 1e-8
        S_prev = S_nb[:-1]  # [T-1, N]
        I_prev = I_nb[:-1]  # [T-1, N]

        lambda_hat = np.zeros_like(dI, dtype=np.float32)
        gamma_hat = np.zeros_like(dR, dtype=np.float32)

        # Keep lambda_hat at zero for nodes with zero degree
        maskS = S_prev > 0
        maskI = I_prev > 0
        lambda_hat[maskS] = dI[maskS] / (S_prev[maskS] + eps)
        gamma_hat[maskI] = dR[maskI] / (I_prev[maskI] + eps)

        # flatten order: lambda(T-1), gamma(T-1), rhoI(T), rhoS(T)
        lam_flat = torch.tensor(lambda_hat.transpose(1, 0), dtype=torch.float)  # [N, T-1]
        gam_flat = torch.tensor(gamma_hat.transpose(1, 0), dtype=torch.float)  # [N, T-1]
        rhoI_flat = torch.tensor(rho_I.transpose(1, 0), dtype=torch.float)     # [N, T]
        rhoS_flat = torch.tensor(rho_S.transpose(1, 0), dtype=torch.float)     # [N, T]

        ctx_flat = torch.cat([lam_flat, gam_flat, rhoI_flat, rhoS_flat], dim=1)  # [N, 2*(T-1)+2*T]

    # ---------- (E) assemble x ----------
    # struct(5) | lpsi(1) | self_time(2) | ctx_flat(2*(T-1)+2*T) | rho_last_feat(2)
    x = torch.cat([struct_feat, lpsi_feat, self_time_feat, ctx_flat, rho_last_feat], dim=1)
    assert x.size(1) == args.node_features_dim, f"Dimension mismatch: expected {args.node_features_dim}, got {x.size(1)}"

    data = Data(x=x, edge_index=edge_index, y=node_labels)

    if Y is not None:
        data.snapshots_obs = Y               # [T1, N] in pyg order
        data.obs_start = int(obs_start)
        data.obs_len = int(obs_len)
        data.T1 = int(T)
        data.T_full = int(T_full)
    # data.snapshots_full = snapshots_full

    if not return_node_mapping:
        return data
    else:
        return data, node_mapping



class GraphGroupDataset(Dataset):
    def __init__(self, graph_groups, args):
        """
        graph_groups: List[nx.Graph]
        """
        self.graph_groups = graph_groups
        self.args = args

    def __len__(self):
        return len(self.graph_groups)

    def __getitem__(self, idx):
        return nx_to_pyg(self.graph_groups[idx], self.args)
