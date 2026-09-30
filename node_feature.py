import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime, timezone

def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0

def compute_node_features(G: nx.Graph) -> nx.Graph:
    """
    Compute structural features for all nodes.
    """
    G_graph = G

    # H1: Degree centrality
    degree_centrality = nx.degree_centrality(G_graph)
    # H3: Local clustering coefficient
    clustering = nx.clustering(G_graph)
    # H4: PageRank score
    pagerank = nx.pagerank(G_graph)
    # H5: Core number
    core_number = nx.core_number(G_graph)

    # Normalize average neighbor degree by the maximum degree
    max_degree = max(dict(G_graph.degree()).values()) if G_graph.number_of_nodes() > 0 else 1
    # Normalize core numbers by the maximum core number
    max_core_number = max(core_number.values()) if core_number else 1
    # Normalize PageRank by its maximum value
    pagerank_values = list(pagerank.values())
    max_pagerank = max(pagerank_values) if pagerank_values else 1
    for node in G_graph.nodes():
        node_features = {
            'degree_centrality': degree_centrality[node],
            'clustering': clustering[node],
            'pagerank': pagerank[node] / max_pagerank,
            'avg_neighbor_degree': (np.mean([G_graph.degree(n) for n in G_graph.neighbors(node)])) / max_degree if G_graph.degree(node) > 0 else 0,  # H2: Average neighbor degree
            'core_number':  core_number[node] / max_core_number if max_core_number > 0 else 0
        }

        # Attach features to the nodes of G_tree
        G_graph.nodes[node]['degree_centrality'] = node_features['degree_centrality']
        G_graph.nodes[node]['avg_neighbor_degree'] = node_features['avg_neighbor_degree']
        G_graph.nodes[node]['clustering'] = node_features['clustering']
        G_graph.nodes[node]['pagerank'] = node_features['pagerank']
        G_graph.nodes[node]['core_number'] = node_features['core_number']

    # # Normalize relative timestamps
    # timestamps = [G_tree.nodes[n]['timestamp'] for n in G_tree.nodes]
    # min_t, max_t = min(timestamps), max(timestamps)
    # normal = max_t - min_t
    # for node in G_tree.nodes:
    #     t = G_tree.nodes[node]['timestamp']
    #     if not normal:
    #         G_tree.nodes[node]['timestamp'] = 0.0
    #     else:
    #         G_tree.nodes[node]['timestamp'] = (t - min_t) / normal
    return G_graph

def extract_user_features_with_profiles(review_file_path, quote_file_path):
    user_stats = defaultdict(lambda: {
        "likes": [], "retweets": [], "replies": [], "views": [],
        "reply_count": 0, "quote_count": 0,
        "is_blue_verified": 0, "join": None,
        "followers_count": 0, "following_count": 0,
        "posts_count": 0,
        "account_age_days": 0,
    })

    def update_user_profile(user_stats, uid, profile):
        user_stats[uid]["is_blue_verified"] = int(profile.get("is_blue_verified", False))
        join_str = profile.get("join", "")
        account_age_days = 0
        if join_str:
            try:
                join_time = datetime.strptime(join_str, "%a %b %d %H:%M:%S %z %Y")
                account_age_days = (datetime.now(timezone.utc) - join_time).days
            except Exception:
                account_age_days = 0
        user_stats[uid]["account_age_days"] = account_age_days
        user_stats[uid]["followers_count"] = _to_int(profile.get("followers_count", 0))
        user_stats[uid]["following_count"] = _to_int(profile.get("following_count", 0))
        user_stats[uid]["posts_count"] = _to_int(profile.get("posts_count", 0))

    # Iterate over review and quote files
    for file_path, key in [(review_file_path, "reply_count"), (quote_file_path, "quote_count")]:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                profile = data.get("user_profile", {})
                uid = str(profile.get("id", ""))
                if not uid:
                    continue
                update_user_profile(user_stats, uid, profile)
                # Update interaction data
                user_stats[uid]["likes"].append(_to_int(data.get("likes", 0)))
                user_stats[uid]["retweets"].append(_to_int(data.get("retweet_count", 0)))
                user_stats[uid]["replies"].append(_to_int(data.get("reply_count", 0)))
                user_stats[uid]["views"].append(_to_int(data.get("views", 0)))
                user_stats[uid][key] += 1

    # Aggregate metrics
    user_features = {}
    for uid, s in user_stats.items():
        rates = [(l + r + rp) / (v + 1) for l, r, rp, v in zip(s["likes"], s["retweets"], s["replies"], s["views"])]
        max_rate = max(rates) if rates else 0
        mean_rate = np.mean(rates) if rates else 0
        ratio = s["followers_count"] / (s["following_count"] + 1)
        max_views = max(s["views"]) if s["views"] else 0

        user_features[uid] = {
            # Propagation interaction features
            "max_engagement_rate": max_rate,
            "mean_engagement_rate": mean_rate,
            "interaction_count": s["reply_count"] + s["quote_count"],
            "reply_count": s["reply_count"],
            "quote_count": s["quote_count"],
            "post_count": len(s["likes"]),
            "max_views": np.log1p(max_views),
            # User profile features
            "is_blue_verified": s["is_blue_verified"],
            "followers_count": np.log1p(s["followers_count"]),
            "following_count": np.log1p(s["following_count"]),
            "posts_count": np.log1p(s["posts_count"]),
            "followers_to_following_ratio": ratio,
            "account_age_days": s["account_age_days"]
        }
    return user_features

def compute_node_features_with_profiles(G: nx.Graph, review_file_path, quote_file_path) -> nx.Graph:
    """
    Compute structural, user profile, and propagation features for all nodes.
    """

    G_graph = G
    # Extract user propagation features
    user_features = extract_user_features_with_profiles(review_file_path, quote_file_path)

    # Attach features to graph nodes
    for uid, feats in user_features.items():
        if uid in G_graph.nodes:
            for k, v in feats.items():
                G_graph.nodes[uid][k] = v
        # else:
        #     G_graph.add_node(uid, **feats)
    # H1: Degree centrality
    degree_centrality = nx.degree_centrality(G_graph)
    # H3: Local clustering coefficient
    clustering = nx.clustering(G_graph)
    # H4: PageRank score
    pagerank = nx.pagerank(G_graph)
    # H5: Core number
    core_number = nx.core_number(G_graph)

    # Normalize average neighbor degree by the maximum degree
    max_degree = max(dict(G_graph.degree()).values()) if G_graph.number_of_nodes() > 0 else 1
    # Normalize core numbers by the maximum core number
    max_core_number = max(core_number.values()) if core_number else 1
    # Normalize PageRank by its maximum value
    pagerank_values = list(pagerank.values())
    max_pagerank = max(pagerank_values) if pagerank_values else 1
    for node in G_graph.nodes():
        node_features = {
            'degree_centrality': degree_centrality[node],
            'clustering': clustering[node],
            'pagerank': pagerank[node] / max_pagerank,
            'avg_neighbor_degree': (np.mean(
                [G_graph.degree(n) for n in G_graph.neighbors(node)])) / max_degree if G_graph.degree(node) > 0 else 0,
            # H2: Average neighbor degree
            'core_number': core_number[node] / max_core_number if max_core_number > 0 else 0
        }

        # Attach features to the nodes of G_tree
        G_graph.nodes[node]['degree_centrality'] = node_features['degree_centrality']
        G_graph.nodes[node]['avg_neighbor_degree'] = node_features['avg_neighbor_degree']
        G_graph.nodes[node]['clustering'] = node_features['clustering']
        G_graph.nodes[node]['pagerank'] = node_features['pagerank']
        G_graph.nodes[node]['core_number'] = node_features['core_number']

    # # Normalize relative timestamps
    # timestamps = [G_tree.nodes[n]['timestamp'] for n in G_tree.nodes]
    # min_t, max_t = min(timestamps), max(timestamps)
    # normal = max_t - min_t
    # for node in G_tree.nodes:
    #     t = G_tree.nodes[node]['timestamp']
    #     if not normal:
    #         G_tree.nodes[node]['timestamp'] = 0.0
    #     else:
    #         G_tree.nodes[node]['timestamp'] = (t - min_t) / normal

    # 3. Min-max normalization of user profile features
    profile_keys = [
        "followers_count", "following_count", "posts_count",
        "account_age_days", "max_engagement_rate",
        "mean_engagement_rate", "interaction_count",
        "reply_count", "quote_count", "post_count", "followers_to_following_ratio", "max_views"
    ]
    for key in profile_keys:
        values = np.array([G_graph.nodes[n].get(key, 0.0) for n in G_graph.nodes()])
        if len(values) == 0 or np.std(values) == 0:
            continue
        min_v, max_v = np.min(values), np.max(values)
        if max_v > min_v:
            normed = (values - min_v) / (max_v - min_v)
        else:
            normed = np.zeros_like(values)
        for idx, node in enumerate(G_graph.nodes()):
            G_graph.nodes[node][key] = float(normed[idx])

    # Inspect a subset of node features
    sample_nodes = list(G_graph.nodes())[:2]  # Print the first five nodes
    print("\n=== Example node features ===")
    for node in sample_nodes:
        print(f"\nNode ID: {node}")
        attrs = G_graph.nodes[node]
        for k, v in attrs.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, bool):
                print(f"  {k}: {v}")
    print("====================\n")

    return G_graph

# G = nx.Graph()
# G.add_edges_from([(0,1), (1,2)])  # Connected component
# G.add_node(3)                     # Isolated node
# G = compute_node_features(G)
#
# for n, data in G.nodes(data=True):
#     print(n, data)
