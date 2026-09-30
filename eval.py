# import torch
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
# from fscore_caculate import F_score_computation_multi

import torch
import time

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score
)


def evaluate_model_with_metrics(model, dataloader, device):
    """
    Evaluate the model and return multiple metrics.
    """
    model.eval()
    y_true_all = []
    y_pred_all = []
    correct_graphs = 0
    total_graphs = 0
    wrong_graph_indices = []
    graph_index = 0


    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
            y = batch_data.y

            # Model inference
            pred_y = model.inference(x, edge_index, batch)

            # Split labels by graph
            true_y = []
            for i in torch.unique(batch):
                mask = (batch == i)
                true_y.append(y[mask].float())
            # Collect predictions and ground-truth labels
            for pred, true in zip(pred_y, true_y):
                # Record graph-level accuracy
                if torch.argmax(pred) == torch.argmax(true):
                    correct_graphs += 1
                else:
                    wrong_graph_indices.append(graph_index)

                total_graphs += 1
                graph_index += 1
                y_pred_all.append(pred.cpu())
                y_true_all.append(true.cpu())

    # Convert to NumPy arrays
    y_pred_all = [torch.argmax(y) for y in y_pred_all]
    y_true_all = [torch.argmax(y) for y in y_true_all]
    y_pred_all = torch.tensor(y_pred_all).numpy()
    y_true_all = torch.tensor(y_true_all).numpy()
    # y_pred_all = torch.cat(y_pred_all, dim=0).cpu().numpy()
    # y_true_all = torch.cat(y_true_all, dim=0).cpu().numpy()

    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"[Eval] Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    print(f"[Eval] Total wrong predictions: {len(wrong_graph_indices)}")
    print(f"[Eval] Wrongly predicted graph indices: {wrong_graph_indices}")
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

def evaluate_model_with_metrics_with_multi_source(model, dataloader, device, threshold=0.5):
    """
    Graph-level evaluation: compute metrics from sigmoid probabilities per graph, then average.
    """

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    graph_metrics = {
        'accuracy': [],
        'f1': [],
        'recall': [],
        'precision': [],
        'auc': []
    }
    k = 1.0
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
            y = batch_data.y  # (total_nodes, )

            # Probabilities after sigmoid
            pred_prob = model.inference(x, edge_index, batch)

            unique_graphs = torch.unique(batch).tolist()
            assert len(pred_prob) == len(unique_graphs), \
                f"Mismatch: got {len(pred_prob)} preds but {len(unique_graphs)} graphs in batch"

            for graph_id, pred_prob_g in zip(unique_graphs, pred_prob):
                mask = (batch == graph_id)
                y_true_g = y[mask].cpu().numpy()
                y_prob_g = pred_prob_g.detach().cpu().numpy()
                # print(y_prob_g)
                # print(threshold)
                y_pred_g = (y_prob_g >= threshold).astype(int)

                acc_g = accuracy_score(y_true_g, y_pred_g)
                f1_g = f1_score(y_true_g, y_pred_g, average='binary', zero_division=0)
                recall_g = recall_score(y_true_g, y_pred_g, average='binary', zero_division=0)
                precision_g = precision_score(y_true_g, y_pred_g, average='binary', zero_division=0)

                try:
                    auc_g = roc_auc_score(y_true_g, y_prob_g)
                except ValueError:
                    auc_g = float('nan')  # AUC is undefined when the graph contains only one label class

                graph_metrics['accuracy'].append(acc_g)
                graph_metrics['f1'].append(f1_g)
                graph_metrics['recall'].append(recall_g)
                graph_metrics['precision'].append(precision_g)
                graph_metrics['auc'].append(auc_g)

    avg_metrics = {
        k: torch.tensor([v for v in vals if not torch.isnan(torch.tensor(v))]).mean().item()
        for k, vals in graph_metrics.items() if len(vals) > 0
    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time

    print(f"[Eval] Mean per-graph Accuracy:   {avg_metrics['accuracy']:.4f}")
    print(f"[Eval] Mean per-graph F1:         {avg_metrics['f1']:.4f}")
    print(f"[Eval] Mean per-graph Recall:     {avg_metrics['recall']:.4f}")
    print(f"[Eval] Mean per-graph Precision:  {avg_metrics['precision']:.4f}")
    print(f"[Eval] Mean per-graph AUC:        {avg_metrics['auc']:.4f}")
    print(f"[Eval] Total evaluated graphs:    {len(graph_metrics['accuracy'])}")
    print(f"\n[Eval] Total inference time: {total_time:.4f} seconds")

    return avg_metrics


# def _build_adj_list(num_nodes: int, edge_index_local: np.ndarray, undirected: bool = True):
#     """
#     edge_index_local: shape (2, E), node ids in [0, num_nodes-1]
#     """
#     adj = [[] for _ in range(num_nodes)]
#     src = edge_index_local[0]
#     dst = edge_index_local[1]
#     for u, v in zip(src, dst):
#         adj[u].append(v)
#         if undirected and u != v:
#             adj[v].append(u)
#     return adj


# def _multi_source_aed_topk(adj, true_sources, topk_nodes, unreachable_val=None):
#     """
#     AED@K for multi-source:
#       For each true source s, compute min_{v in topk_nodes} dist(s, v)
#       then average over sources.

#     unreachable_val:
#       - None: use num_nodes as penalty
#       - or a number you choose
#     """
#     num_nodes = len(adj)
#     if unreachable_val is None:
#         unreachable_val = num_nodes  # penalty if disconnected

#     topk_set = set(topk_nodes.tolist() if hasattr(topk_nodes, "tolist") else list(topk_nodes))
#     true_sources = list(true_sources)

#     if len(true_sources) == 0:
#         return float('nan')  # no positives in this graph

#     # BFS from each source until we hit any topk node (early stop)
#     dists = []
#     for s in true_sources:
#         if s in topk_set:
#             dists.append(0)
#             continue

#         visited = np.zeros(num_nodes, dtype=bool)
#         q = deque()
#         q.append((s, 0))
#         visited[s] = True

#         found = False
#         best = unreachable_val
#         while q:
#             u, du = q.popleft()
#             # early stop: if current distance already >= best, can stop
#             if du >= best:
#                 continue
#             for v in adj[u]:
#                 if not visited[v]:
#                     visited[v] = True
#                     nd = du + 1
#                     if v in topk_set:
#                         best = nd
#                         found = True
#                         # we can stop BFS early because BFS explores by increasing distance
#                         q.clear()
#                         break
#                     q.append((v, nd))
#         dists.append(best if found else unreachable_val)

#     return float(np.mean(dists))


# def evaluate_model_with_metrics_with_multi_source(
#     model, dataloader, device,
#     threshold=0.5,
#     hit_ks=(1, 3, 5, 10),
#     aed_k=10,
#     undirected_for_aed=True
# ):
#     """
#     Graph-level evaluation: compute metrics from sigmoid probabilities per graph, then average.
#     Additional metrics:
#       - PR-AUC (average precision)
#       - Hit@K (multi-source hit ratio)
#       - AED@K (average error distance, hop-based)
#     """

#     model.eval()

#     graph_metrics = {
#         'accuracy': [],
#         'f1': [],
#         'recall': [],
#         'precision': [],
#         'auc': [],        # ROC-AUC
#         'pr_auc': [],     # Average Precision (AP)
#         'aed@k': [],      # AED at aed_k
#     }
#     # dynamic keys for hit@k
#     for k in hit_ks:
#         graph_metrics[f'hit@{k}'] = []

#     with torch.no_grad():
#         for batch_data in dataloader:
#             batch_data = batch_data.to(device)
#             x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
#             y = batch_data.y  # (total_nodes, )

#             # Probabilities after sigmoid
#             pred_prob = model.inference(x, edge_index, batch)

#             unique_graphs = torch.unique(batch).tolist()
#             assert len(pred_prob) == len(unique_graphs), \
#                 f"Mismatch: got {len(pred_prob)} preds but {len(unique_graphs)} graphs in batch"

#             for graph_id, pred_prob_g in zip(unique_graphs, pred_prob):
#                 node_mask = (batch == graph_id)

#                 # --- labels/probs for this graph
#                 y_true_g = y[node_mask].detach().cpu().numpy().astype(int)
#                 y_prob_g = pred_prob_g.detach().cpu().numpy()
#                 y_pred_g = (y_prob_g >= threshold).astype(int)

#                 # --- basic metrics (thresholded)
#                 acc_g = accuracy_score(y_true_g, y_pred_g)
#                 f1_g = f1_score(y_true_g, y_pred_g, average='binary', zero_division=0)
#                 recall_g = recall_score(y_true_g, y_pred_g, average='binary', zero_division=0)
#                 precision_g = precision_score(y_true_g, y_pred_g, average='binary', zero_division=0)

#                 # --- ROC-AUC
#                 try:
#                     auc_g = roc_auc_score(y_true_g, y_prob_g)
#                 except ValueError:
#                     auc_g = float('nan')  # all same class

#                 # --- PR-AUC (Average Precision)
#                 # Guard AP against errors or instability with a single label class
#                 try:
#                     pr_auc_g = average_precision_score(y_true_g, y_prob_g)
#                 except ValueError:
#                     pr_auc_g = float('nan')

#                 # collect
#                 graph_metrics['accuracy'].append(acc_g)
#                 graph_metrics['f1'].append(f1_g)
#                 graph_metrics['recall'].append(recall_g)
#                 graph_metrics['precision'].append(precision_g)
#                 graph_metrics['auc'].append(auc_g)
#                 graph_metrics['pr_auc'].append(pr_auc_g)

#                 # --- Hit@K and AED@K need node indices inside this graph
#                 # map global node ids -> local [0..n-1]
#                 global_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
#                 num_nodes_g = global_idx.numel()
#                 global_to_local = {int(g.item()): i for i, g in enumerate(global_idx)}

#                 # local edge_index for this graph
#                 # keep edges where both ends are in this graph
#                 src = edge_index[0].detach().cpu().numpy()
#                 dst = edge_index[1].detach().cpu().numpy()
#                 in_g = np.isin(src, global_idx.cpu().numpy()) & np.isin(dst, global_idx.cpu().numpy())
#                 src_g = src[in_g]
#                 dst_g = dst[in_g]
#                 # remap to local ids
#                 src_local = np.array([global_to_local[int(u)] for u in src_g], dtype=np.int64)
#                 dst_local = np.array([global_to_local[int(v)] for v in dst_g], dtype=np.int64)
#                 edge_local = np.stack([src_local, dst_local], axis=0) if src_local.size > 0 else np.zeros((2, 0), dtype=np.int64)

#                 # true sources local ids
#                 true_sources_local = np.where(y_true_g == 1)[0].tolist()

#                 # predicted ranking (top indices by prob)
#                 order = np.argsort(-y_prob_g)  # descending
#                 # Hit@K (multi-source hit ratio): (# true sources in topK) / (# true sources)
#                 # Alternatively, use any-hit: 1 if any source is in topK, else 0
#                 for k in hit_ks:
#                     topk = order[:min(k, len(order))]
#                     if len(true_sources_local) == 0:
#                         hitk = float('nan')
#                     else:
#                         hitk = float(len(set(topk).intersection(true_sources_local))) / float(len(true_sources_local))
#                     graph_metrics[f'hit@{k}'].append(hitk)

#                 # AED@K
#                 if len(true_sources_local) == 0:
#                     aed = float('nan')
#                 else:
#                     topk_for_aed = order[:min(aed_k, len(order))]
#                     adj = _build_adj_list(num_nodes_g, edge_local, undirected=undirected_for_aed)
#                     aed = _multi_source_aed_topk(adj, true_sources_local, topk_for_aed, unreachable_val=num_nodes_g)
#                 graph_metrics['aed@k'].append(aed)

#     # average (ignore nan)
#     avg_metrics = {}
#     for name, vals in graph_metrics.items():
#         if len(vals) == 0:
#             continue
#         t = torch.tensor([v for v in vals], dtype=torch.float32)
#         t = t[~torch.isnan(t)]
#         avg_metrics[name] = t.mean().item() if t.numel() > 0 else float('nan')

#     # print
#     print(f"[Eval] Mean per-graph Accuracy:   {avg_metrics.get('accuracy', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph F1:         {avg_metrics.get('f1', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph Recall:     {avg_metrics.get('recall', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph Precision:  {avg_metrics.get('precision', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph ROC-AUC:    {avg_metrics.get('auc', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph PR-AUC(AP): {avg_metrics.get('pr_auc', float('nan')):.4f}")
#     for k in hit_ks:
#         print(f"[Eval] Mean per-graph Hit@{k}:     {avg_metrics.get(f'hit@{k}', float('nan')):.4f}")
#     print(f"[Eval] Mean per-graph AED@{aed_k}:    {avg_metrics.get('aed@k', float('nan')):.4f}")
#     print(f"[Eval] Total evaluated graphs:    {len(graph_metrics['accuracy'])}")

#     return avg_metrics
