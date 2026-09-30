import pickle
import networkx as nx
import numpy as np

# Helper functions
def gini(x):
    x = np.array(list(x), dtype=float)
    if len(x) == 0:
        return 0.0
    x = x[x >= 0]
    if x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    return 1 + 1/n - 2 * (cumx / cumx[-1]).sum() / n

def proxy_thread_depth_via_eccentricity(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return 0
    giant = max(nx.connected_components(G), key=len)
    H = G.subgraph(giant).copy()
    ecc = nx.eccentricity(H)  # {node: max shortest-path length}
    return max(ecc.values()) if ecc else 0

def avg_shortest_path_on_giant_component(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return 0.0
    giant = max(nx.connected_components(G), key=len)
    H = G.subgraph(giant).copy()
    if H.number_of_nodes() <= 1:
        return 0.0
    try:
        return nx.average_shortest_path_length(H)
    except nx.NetworkXError:
        return 0.0

def interaction_metrics_undirected(G: nx.Graph):
    n = G.number_of_nodes()
    if n == 0:
        return None
    return dict(
        giant_component_ratio = (len(max(nx.connected_components(G), key=len)) / n) if n > 0 else 0.0,
        avg_clustering = nx.average_clustering(G) if n > 1 else 0.0,
        proxy_thread_depth = proxy_thread_depth_via_eccentricity(G),
        gini_degree = gini([d for _, d in G.degree()]),
        avg_spl_gcc = avg_shortest_path_on_giant_component(G),
        nodes = n  # Weighting factor
    )

def weighted_average(metrics_list):
    if not metrics_list:
        return {}
    total_nodes = sum(m['nodes'] for m in metrics_list)
    avg_metrics = {}
    for key in metrics_list[0]:
        if key == 'nodes':
            continue
        avg_metrics[key] = sum(m[key] * m['nodes'] for m in metrics_list) / total_nodes
    return avg_metrics

def main():
    # Batch processing
    datasets = ["twitter25", "twitter15", "twitter16", "weibo"]
    base_path = "./data/saved_graphs/"

    results = {}

    for name in datasets:
        pkl_file = f"{base_path}{name}_graph.pkl"
        with open(pkl_file, "rb") as f:
            graphs = pickle.load(f)
        print(f'{name} processing...')
        metrics_list = []
        for G in graphs:
            m = interaction_metrics_undirected(G)
            if m is not None:
                metrics_list.append(m)

        results[name] = weighted_average(metrics_list)

    # Print the comparison table
    print(f"{'Dataset':<12} {'GiantRatio':<12} {'Clustering':<12} {'Depth':<8} {'GiniDeg':<10} {'Avg SPL':<10}")
    for name, m in results.items():
        print(f"{name:<12} {m['giant_component_ratio']:<12.4f} {m['avg_clustering']:<12.4f} {m['proxy_thread_depth']:<8.2f} {m['gini_degree']:<10.4f} {m['avg_spl_gcc']:<10.4f}")


if __name__ == "__main__":
    main()
