import os
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
from eval import evaluate_model_with_metrics_with_multi_source
from visual import visualize_manifold_with_ttt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch.optim as optim

def collect_train_representations(
    model, dataloader, device,
    current_epoch, total_epochs,
    max_samples=1000
):
    """
    Collect z_pred from the training set.
    """
    model.eval()
    z_list = []
    y_list = []

    use_diffusion = hasattr(model, 'use_diffusion') and model.use_diffusion

    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= max_samples:
                break

            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx = batch.batch.to(device)
            y = batch.y.to(device).view(-1)

            rise_flag = x[:, 6]
            if model.args.pruning:
                cand_mask = (rise_flag == 0)
            else:
                cand_mask = (rise_flag <= 1)
            cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

            if cand_idx.numel() == 0:
                continue

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            h_all = model.aggregation(x_main, edge_index)
            h_cand = h_all[cand_idx]
            h_enc = model.enc(h_cand)
            mu = model.fc_mu(h_enc)
            logvar = model.fc_logvar(h_enc)

            # Process context features
            ctx_flat = x[:, model.main_dim : -2]
            if ctx_flat.numel() > 0 and model.obs_len > 1:
                try:
                    T1 = model.obs_len
                    ctx_cand = ctx_flat[cand_idx]
                    tokens = model._build_ctx_tokens(ctx_cand, T1)
                    _, h_last = model.ctx_gru(tokens)
                    c = h_last.squeeze(0)

                    q = mu.unsqueeze(1)
                    k = c.unsqueeze(1)
                    v = c.unsqueeze(1)
                    attn_out, _ = model.cross_attn(q, k, v)
                    delta_mu = model.delta_ffn(attn_out.squeeze(1))
                    mu_prime = mu + delta_mu
                except:
                    mu_prime = mu
            else:
                mu_prime = mu

            # Generate z_pred
            eps = torch.randn_like(mu_prime)
            z0 = mu_prime + torch.exp(0.5 * logvar) * eps
            z_pred = z0

            z_list.append(z_pred.cpu())

            y_cand = y[cand_idx]
            y_list.append(y_cand.cpu())

            total_samples += len(y_cand)

    if len(z_list) == 0:
        return None, None, None

    z_all = torch.cat(z_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    info = {
        'epoch': current_epoch,
        'num_samples': len(z_all)
    }

    return z_all, y_all, info

def collect_test_representations_with_ttt(
    model, dataloader, device,
    current_epoch, total_epochs,
    max_samples=5000
):
    """
    Collect raw test-set mu and adapted mu_opt after TTT.
    Compute metrics consistently with evaluate_model_with_metrics_with_multi_source.
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    model.eval()
    z_raw_list = []
    z_opt_list = []
    y_list = []

    # Store complete information for each graph
    graph_data = []  # [(y_true_g, prob_raw_g, prob_opt_g), ...]

    total_samples = 0
    processed_graphs = 0

    for batch in dataloader:
        if total_samples >= max_samples:
            break

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device).view(-1)

        rise_flag = x[:, 6]
        cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            continue

        x_main_base = x[:, :8-2]
        rho_last = x[:, -2:]
        x_main = torch.cat([x_main_base, rho_last], dim=1)

        # Match inference(): predict for all nodes
        with torch.no_grad():
            h_all = model.aggregation(x_main, edge_index)
            h_enc = model.enc(h_all)  # All nodes
            mu = model.fc_mu(h_enc)   # [N_total, latent]

            # Predictions from mu for all nodes
            logits_raw = model.pred_head(mu)
            pred_prob_raw_all = torch.softmax(logits_raw, dim=1)[:, 1]  # [N_total]

        # Keep candidate-node mu values for visualization
        mu_cand = mu[cand_idx]
        z_raw_list.append(mu_cand.cpu())

        # Optimize all nodes with TTT
        with torch.enable_grad():
            mu_opt = mu.clone().detach().requires_grad_(True)
            optimizer = optim.SGD([mu_opt], lr=model.ttt_lr, momentum=0.9)

            for step in range(model.ttt_steps):
                t = torch.randint(0, model.ttt_t_max, (mu_opt.size(0),), device=device).long()
                z_0_hat = model.diffusion(mu_opt, t)
                loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)

                optimizer.zero_grad()
                loss_ttt.backward()
                optimizer.step()

            z_final = mu_opt.detach()

            # Keep candidate-node mu_opt values for visualization
            z_final_cand = z_final[cand_idx]
            z_opt_list.append(z_final_cand.cpu())

            # Predictions from mu_opt for all nodes
            with torch.no_grad():
                logits_opt = model.pred_head(z_final)
                pred_prob_opt_all = torch.softmax(logits_opt, dim=1)[:, 1]  # [N_total]

        # Compute metrics by graph, consistently with the evaluation function
        unique_graphs = torch.unique(batch_idx).tolist()

        for graph_id in unique_graphs:
            mask_g = (batch_idx == graph_id)
            y_true_g = y[mask_g].cpu().numpy()
            prob_raw_g = pred_prob_raw_all[mask_g].cpu().numpy()
            prob_opt_g = pred_prob_opt_all[mask_g].cpu().numpy()

            graph_data.append((y_true_g, prob_raw_g, prob_opt_g))

            # Inspect the first three graphs
            if processed_graphs < 3:
                num_cand = cand_mask[mask_g].sum().item()
                print(f"\n  [DEBUG] Graph {processed_graphs}:")
                print(f"    Total nodes: {len(y_true_g)}, Candidates: {num_cand}")
                print(f"    Positive samples: {y_true_g.sum()}")
                print(f"    y_true:   {y_true_g[:5]}...")
                print(f"    prob_raw: {prob_raw_g[:5]}...")
                print(f"    prob_opt: {prob_opt_g[:5]}...")

            processed_graphs += 1

        # Keep candidate-node labels for visualization
        y_cand = y[cand_idx]
        y_list.append(y_cand.cpu())
        total_samples += len(y_cand)

    if len(z_raw_list) == 0:
        return None, None, None, None

    z_raw = torch.cat(z_raw_list, dim=0)
    z_opt = torch.cat(z_opt_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    # Match the evaluation function's metric computation
    threshold = 0.5

    graph_metrics_raw = {'f1': [], 'auc': [], 'accuracy': []}
    graph_metrics_opt = {'f1': [], 'auc': [], 'accuracy': []}

    for y_true_g, prob_raw_g, prob_opt_g in graph_data:
        # Raw
        y_pred_raw = (prob_raw_g >= threshold).astype(int)

        f1_raw_g = f1_score(y_true_g, y_pred_raw, average='binary', zero_division=0)
        acc_raw_g = accuracy_score(y_true_g, y_pred_raw)
        graph_metrics_raw['f1'].append(f1_raw_g)
        graph_metrics_raw['accuracy'].append(acc_raw_g)

        try:
            auc_raw_g = roc_auc_score(y_true_g, prob_raw_g)
            graph_metrics_raw['auc'].append(auc_raw_g)
        except ValueError:
            pass

        # Opt
        y_pred_opt = (prob_opt_g >= threshold).astype(int)

        f1_opt_g = f1_score(y_true_g, y_pred_opt, average='binary', zero_division=0)
        acc_opt_g = accuracy_score(y_true_g, y_pred_opt)
        graph_metrics_opt['f1'].append(f1_opt_g)
        graph_metrics_opt['accuracy'].append(acc_opt_g)

        try:
            auc_opt_g = roc_auc_score(y_true_g, prob_opt_g)
            graph_metrics_opt['auc'].append(auc_opt_g)
        except ValueError:
            pass

    # Compute averages
    f1_raw = np.mean(graph_metrics_raw['f1']) if len(graph_metrics_raw['f1']) > 0 else 0.0
    auc_raw = np.mean(graph_metrics_raw['auc']) if len(graph_metrics_raw['auc']) > 0 else 0.0
    acc_raw = np.mean(graph_metrics_raw['accuracy']) if len(graph_metrics_raw['accuracy']) > 0 else 0.0

    f1_opt = np.mean(graph_metrics_opt['f1']) if len(graph_metrics_opt['f1']) > 0 else 0.0
    auc_opt = np.mean(graph_metrics_opt['auc']) if len(graph_metrics_opt['auc']) > 0 else 0.0
    acc_opt = np.mean(graph_metrics_opt['accuracy']) if len(graph_metrics_opt['accuracy']) > 0 else 0.0

    f1_improvement = ((f1_opt - f1_raw) / f1_raw * 100) if f1_raw > 0 else 0
    auc_improvement = ((auc_opt - auc_raw) / auc_raw * 100) if auc_raw > 0 else 0

    info = {
        'epoch': current_epoch,
        'num_samples': len(z_raw),
        'num_graphs': processed_graphs,
        'f1_raw': f1_raw,
        'f1_opt': f1_opt,
        'f1_improvement': f1_improvement,
        'auc_raw': auc_raw,
        'auc_opt': auc_opt,
        'auc_improvement': auc_improvement,
        'acc_raw': acc_raw,
        'acc_opt': acc_opt,
    }

    print(f"\n    [Performance Stats]")
    print(f"      Evaluated: {processed_graphs} graphs, {len(z_raw)} candidate nodes")
    print(f"      Before TTT (μ):    F1={f1_raw:.4f}, AUC={auc_raw:.4f}, Acc={acc_raw:.4f}")
    print(f"      After TTT (μ_opt): F1={f1_opt:.4f}, AUC={auc_opt:.4f}, Acc={acc_opt:.4f}")
    print(f"      Improvement:       F1={f1_improvement:+.2f}%, AUC={auc_improvement:+.2f}%\n")

    return z_raw, z_opt, y_all, info


# ============================================================
# Part 1: Representation collection functions
# ============================================================

def collect_train_representations_wo_la(
    model, dataloader, device,
    current_epoch, total_epochs,
    max_samples=5000
):
    """
    Collect training-set mu_prime for the wo_LA ablation model.
    mu_prime = mu + delta_mu: deterministic cross-attention refinement without VAE sampling.
    """
    model.eval()
    z_list = []
    y_list = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= max_samples:
                break

            x          = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx  = batch.batch.to(device)
            y          = batch.y.to(device).view(-1)

            rise_flag = x[:, 6]
            if model.args.pruning:
                cand_mask = (rise_flag == 0)
            else:
                cand_mask = (rise_flag <= 1)
            cand_idx  = cand_mask.nonzero(as_tuple=False).view(-1)
            if cand_idx.numel() == 0:
                continue

            # Match the model's forward pass
            x_main_base = x[:, :8 - 2]
            rho_last    = x[:, -2:]
            x_main      = torch.cat([x_main_base, rho_last], dim=1)

            h_all  = model.aggregation(x_main, edge_index)
            h_cand = h_all[cand_idx]
            h_enc  = model.enc(h_cand)
            mu     = model.fc_mu(h_enc)

            # Cross-attention refinement produces mu_prime
            ctx_flat = x[:, model.main_dim: -2]
            if ctx_flat.numel() > 0 and model.obs_len > 1:
                try:
                    T1       = model.obs_len
                    ctx_cand = ctx_flat[cand_idx]
                    tokens   = model._build_ctx_tokens(ctx_cand, T1)
                    _, h_last = model.ctx_gru(tokens)
                    c        = h_last.squeeze(0)

                    q = mu.unsqueeze(1)
                    k = c.unsqueeze(1)
                    v = c.unsqueeze(1)
                    attn_out, _ = model.cross_attn(q, k, v)
                    delta_mu    = model.delta_ffn(attn_out.squeeze(1))
                    mu_prime    = mu + delta_mu
                except Exception:
                    mu_prime = mu
            else:
                mu_prime = mu
            # ─────────────────────────────────────────────────

            z_list.append(mu_prime.cpu())
            y_list.append(y[cand_idx].cpu())
            total_samples += cand_idx.numel()

    if len(z_list) == 0:
        return None, None, None

    z_all = torch.cat(z_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    info = {
        'epoch':       current_epoch,
        'num_samples': len(z_all),
        'repr_type':   'mu_prime (w/o VAE sampling)',
    }
    return z_all, y_all, info


def collect_test_representations_wo_la(
    model, dataloader, device,
    current_epoch, total_epochs,
    max_samples=5000
):
    """
    Collect test-set mu for the wo_LA ablation model.
    Inference uses GAT+enc mu directly, without TTT.
    Compute metrics consistently with the evaluation function for aligned reporting.
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    model.eval()
    z_list     = []
    y_list     = []
    graph_data = []          # [(y_true_g, prob_g), ...]
    total_samples     = 0
    processed_graphs  = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= max_samples:
                break

            x          = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx  = batch.batch.to(device)
            y          = batch.y.to(device).view(-1)

            rise_flag = x[:, 6]
            cand_mask = (rise_flag <= 1)
            cand_idx  = cand_mask.nonzero(as_tuple=False).view(-1)
            if cand_idx.numel() == 0:
                continue

            x_main_base = x[:, :8 - 2]
            rho_last    = x[:, -2:]
            x_main      = torch.cat([x_main_base, rho_last], dim=1)

            # Match inference(): obtain mu for all nodes
            h_all    = model.aggregation(x_main, edge_index)
            h_enc    = model.enc(h_all)
            mu       = model.fc_mu(h_enc)          # [N_total, latent]

            logits   = model.pred_head(mu)
            prob_all = torch.softmax(logits, dim=1)[:, 1]   # [N_total]

            # Keep candidate-node mu values for visualization
            z_list.append(mu[cand_idx].cpu())
            y_list.append(y[cand_idx].cpu())
            total_samples += cand_idx.numel()

            # Group by graph
            for graph_id in torch.unique(batch_idx).tolist():
                mask_g   = (batch_idx == graph_id)
                y_true_g = y[mask_g].cpu().numpy()
                prob_g   = prob_all[mask_g].cpu().numpy()
                graph_data.append((y_true_g, prob_g))
                processed_graphs += 1

    if len(z_list) == 0:
        return None, None, None

    z_all = torch.cat(z_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    # Compute metrics
    threshold = 0.5
    f1_list, auc_list, acc_list = [], [], []

    for y_true_g, prob_g in graph_data:
        y_pred_g = (prob_g >= threshold).astype(int)
        f1_list.append(f1_score(y_true_g, y_pred_g, average='binary', zero_division=0))
        acc_list.append(accuracy_score(y_true_g, y_pred_g))
        try:
            auc_list.append(roc_auc_score(y_true_g, prob_g))
        except ValueError:
            pass

    f1_mean  = float(np.mean(f1_list))  if f1_list  else 0.0
    auc_mean = float(np.mean(auc_list)) if auc_list else 0.0
    acc_mean = float(np.mean(acc_list)) if acc_list else 0.0

    info = {
        'epoch':        current_epoch,
        'num_samples':  len(z_all),
        'num_graphs':   processed_graphs,
        'f1':           f1_mean,
        'auc':          auc_mean,
        'acc':          acc_mean,
        'repr_type':    'mu (w/o TTT)',
    }

    print(f"\n    [Ablation Stats (wo_LA)]")
    print(f"      Evaluated: {processed_graphs} graphs, {len(z_all)} candidate nodes")
    print(f"      mu (no TTT): F1={f1_mean:.4f}, AUC={auc_mean:.4f}, Acc={acc_mean:.4f}\n")

    return z_all, y_all, info


# ============================================================
# Part 2: Visualization functions
# ============================================================

def visualize_manifold_ablation_wo_la(
    z_train, z_test,
    y_train, y_test,
    train_info, test_info,
    save_path='./model_saves/visual_ablation',
    epoch=None
):
    import numpy as np
    import torch
    from pathlib import Path
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    # Font settings matching the TTT visualization
    plt.rcParams['font.size']        = 15
    plt.rcParams['axes.labelsize']   = 15
    plt.rcParams['axes.titlesize']   = 19
    plt.rcParams['xtick.labelsize']  = 13
    plt.rcParams['ytick.labelsize']  = 13

    # Color constants
    COLOR_CLASS0   = '#2E338E'
    COLOR_CLASS1   = '#A60227'
    COLOR_INSIDE   = '#2ca02c'
    COLOR_BOUNDARY = '#ff7f0e'
    COLOR_OUTSIDE  = '#d62728'
    COLOR_TRAIN_BG = 'lightgray'

    # 1. Convert to NumPy
    z_train = z_train.detach().cpu().numpy() if torch.is_tensor(z_train) else np.asarray(z_train)
    z_test  = z_test.detach().cpu().numpy()  if torch.is_tensor(z_test)  else np.asarray(z_test)
    y_train = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.asarray(y_train)
    y_test  = y_test.detach().cpu().numpy()  if torch.is_tensor(y_test)  else np.asarray(y_test)

    print("-" * 50)
    z_std    = np.std(z_train, axis=0)
    dist_gap = np.linalg.norm(np.mean(z_train, 0) - np.mean(z_test, 0))
    print(f"[Ablation Diagnostic]")
    print(f"  Train std (mean): {np.mean(z_std):.4e}  (max): {np.max(z_std):.4e}")
    print(f"  Dead dims: {np.sum(z_std < 1e-6)} / {z_train.shape[1]}")
    print(f"  Distribution Gap (L2): {dist_gap:.4f}")
    print("-" * 50)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Ablation] train={len(z_train)}, test={len(z_test)}")

    # ── 2. PCA → KDE ─────────────────────────────────────────────
    pca           = PCA(n_components=2)
    z_train_pca   = pca.fit_transform(z_train)
    z_test_pca    = pca.transform(z_test)
    explained_var = pca.explained_variance_ratio_.sum()

    param_grid = {'bandwidth': np.logspace(-2, 1, 20)}
    grid       = GridSearchCV(KernelDensity(kernel='gaussian'), param_grid, cv=5, n_jobs=-1)
    grid.fit(z_train_pca)
    best_bandwidth = grid.best_params_['bandwidth']
    kde            = grid.best_estimator_

    train_density  = np.exp(kde.score_samples(z_train_pca))
    test_density   = np.exp(kde.score_samples(z_test_pca))

    threshold_high = np.percentile(train_density, 50)
    threshold_low  = np.percentile(train_density, 25)

    inside   = test_density >= threshold_high
    boundary = (test_density >= threshold_low) & (test_density < threshold_high)
    outside  = test_density < threshold_low

    print(f"  inside={inside.mean()*100:.1f}%  boundary={boundary.mean()*100:.1f}%  outside={outside.mean()*100:.1f}%")

    # ── 3. t-SNE ─────────────────────────────────────────────────
    print("  [Ablation] t-SNE...")
    all_z        = np.vstack([z_train, z_test])
    tsne         = TSNE(n_components=2, random_state=42,
                        perplexity=min(30, len(all_z) // 5))
    z_2d         = tsne.fit_transform(all_z)
    n_train      = len(z_train)
    z_train_2d   = z_2d[:n_train]
    z_test_2d    = z_2d[n_train:]
    y_train_int  = y_train.astype(int)
    y_test_int   = y_test.astype(int)

    # 4. Three-column layout matching the TTT visualization
    fig = plt.figure(figsize=(21, 8))

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        left=0.06,
        right=0.97,
        top=0.84,
        bottom=0.22,
        wspace=0.28
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax in [ax1, ax2, ax3]:
        ax.set_aspect('equal', adjustable='box')

    # Helper functions
    def set_closed_axes(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    def scatter_by_label(ax, z2d, labels, alpha=0.65, s=22,
                         edgecolors='k', linewidths=0.35, zorder=2):
        for val, color in [(0, COLOR_CLASS0), (1, COLOR_CLASS1)]:
            m = (labels == val)
            if m.sum() > 0:
                ax.scatter(z2d[m, 0], z2d[m, 1],
                           c=color, alpha=alpha, s=s, marker='o',
                           edgecolors=edgecolors, linewidths=linewidths,
                           zorder=zorder)

    def set_ax_labels(ax):
        ax.set_xlabel('t-SNE Dim 1', labelpad=3)
        ax.set_ylabel('t-SNE Dim 2', labelpad=3)

    def add_caption(ax, line1, line2=None):
        text = line1 if line2 is None else f'{line1}\n{line2}'
        ax.annotate(
            text,
            xy=(0.5, 0),
            xycoords='axes fraction',
            xytext=(0, -58),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=21,
            multialignment='center'
        )

    # ── ax1: Training Distribution ───────────────────────────────
    scatter_by_label(ax1, z_train_2d, y_train_int, alpha=0.60, s=20)
    set_ax_labels(ax1)
    add_caption(ax1, "(a) Training Distribution (\u03bc')")
    set_closed_axes(ax1)

    # ── ax2: Test Distribution ───────────────────────────────────
    ax2.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                c=COLOR_TRAIN_BG, alpha=0.15, s=5, zorder=1)
    scatter_by_label(ax2, z_test_2d, y_test_int,
                     alpha=0.85, s=50, edgecolors='black', linewidths=0.8, zorder=3)
    set_ax_labels(ax2)
    add_caption(ax2, '(b) Test Distribution (\u03bc)')
    set_closed_axes(ax2)

    # ── ax3: Coverage ────────────────────────────────────────────
    ax3.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                c=COLOR_TRAIN_BG, alpha=0.15, s=5, zorder=1)

    if inside.sum() > 0:
        ax3.scatter(z_test_2d[inside, 0], z_test_2d[inside, 1],
                    c=COLOR_INSIDE, s=55, marker='o', alpha=0.85,
                    edgecolors='darkgreen', linewidths=1.2, zorder=3,
                    label=f'Inside ({inside.sum()}, {inside.mean()*100:.1f}%)')
    if boundary.sum() > 0:
        ax3.scatter(z_test_2d[boundary, 0], z_test_2d[boundary, 1],
                    c=COLOR_BOUNDARY, s=55, marker='s', alpha=0.85,
                    edgecolors='darkorange', linewidths=1.2, zorder=3,
                    label=f'Boundary ({boundary.sum()}, {boundary.mean()*100:.1f}%)')
    if outside.sum() > 0:
        ax3.scatter(z_test_2d[outside, 0], z_test_2d[outside, 1],
                    c=COLOR_OUTSIDE, s=55, marker='x', alpha=0.90,
                    linewidths=2.2, zorder=3,
                    label=f'Outside ({outside.sum()}, {outside.mean()*100:.1f}%)')

    ax3.legend(loc='lower left', fontsize=13, framealpha=0.9, edgecolor='gray')

    # Place the L2 gap in the upper-left corner
    ax3.text(
        0.03, 0.97,
        f'L2 Gap: {dist_gap:.3f}',
        transform=ax3.transAxes,
        ha='left', va='top', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.85)
    )

    set_ax_labels(ax3)
    add_caption(
        ax3,
        '(c) Coverage: Test Distribution (\u03bc)',
        f'Inside: {inside.sum()}/{len(test_density)}'
    )
    set_closed_axes(ax3)

    # Zoomed inset
    x1, x2 = 18, 53
    y1, y2 = 33, 55

    axins = ax3.inset_axes(
        [0.1, 0.24, 0.6, 0.40],
        xlim=(x1, x2), ylim=(y1, y2)
    )

    axins.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                  c=COLOR_TRAIN_BG, alpha=0.30, s=8, zorder=1)

    if inside.sum() > 0:
        axins.scatter(z_test_2d[inside, 0], z_test_2d[inside, 1],
                      c=COLOR_INSIDE, s=55, marker='o', alpha=0.90,
                      edgecolors='darkgreen', linewidths=1.2, zorder=3)
    if boundary.sum() > 0:
        axins.scatter(z_test_2d[boundary, 0], z_test_2d[boundary, 1],
                      c=COLOR_BOUNDARY, s=55, marker='s', alpha=0.90,
                      edgecolors='darkorange', linewidths=1.2, zorder=3)
    if outside.sum() > 0:
        axins.scatter(z_test_2d[outside, 0], z_test_2d[outside, 1],
                      c=COLOR_OUTSIDE, s=55, marker='x', alpha=0.95,
                      linewidths=2.2, zorder=3)

    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor('#444444')

    ax3.indicate_inset_zoom(axins, edgecolor='#444444', linewidth=0.9, alpha=0.8)

    # 5. Shared legend at the top
    legend_handles = []

    legend_handles.append(mpatches.Patch(facecolor=COLOR_CLASS0, edgecolor='k',
                                          linewidth=0.8, label='Non-source = 0'))
    legend_handles.append(mpatches.Patch(facecolor=COLOR_CLASS1, edgecolor='k',
                                          linewidth=0.8, label='Source = 1'))
    if inside.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_INSIDE, edgecolor='darkgreen',
                                              linewidth=0.8, label='Inside'))
    if boundary.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_BOUNDARY, edgecolor='darkorange',
                                              linewidth=0.8, label='Boundary'))
    if outside.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_OUTSIDE, edgecolor='none',
                                              label='Outside'))

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=len(legend_handles),
        frameon=True,
        edgecolor='gray',
        fancybox=False,
        fontsize=18,
        handlelength=2.2,
        handleheight=1.3,
        handletextpad=0.9,
        columnspacing=2.2,
        borderpad=0.9,
        bbox_to_anchor=(0.5, 0.995)
    )

    # 6. Save the figure
    suffix       = f'epoch{epoch}' if epoch is not None else 'final'
    filepath_png = save_dir / f'ablation_wo_la_manifold_{suffix}.png'
    filepath_pdf = save_dir / f'ablation_wo_la_manifold_{suffix}.pdf'

    fig.savefig(filepath_png, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {filepath_png}")
    fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    print(f"  Saved PDF: {filepath_pdf}")
    plt.close(fig)

    # 7. Return statistics
    source_mask = y_test == 1
    stats = {
        'epoch':             epoch,
        'inside_ratio':      float(inside.mean()),
        'boundary_ratio':    float(boundary.mean()),
        'outside_ratio':     float(outside.mean()),
        'distribution_gap':  float(dist_gap),
        'kde_bandwidth':     float(best_bandwidth),
        'pca_explained_var': float(explained_var),
        'source_inside':     float(inside[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'source_outside':    float(outside[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'n_train':  int(len(z_train)),
        'n_test':   int(len(z_test)),
        'f1_test':  float(test_info.get('f1', 0.0)),
        'auc_test': float(test_info.get('auc', 0.0)),
    }

    return str(filepath_png), stats


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {checkpoint['epoch']}")
    return model, optimizer, scheduler, start_epoch


def train_GLCFGD(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss_c': f'{loss_dict["pred_loss_c"]:.4f}',
                    'pred_loss_u': f'{loss_dict["pred_loss_u"]:.4f}',
                    'kl_loss_c': f'{loss_dict["kl_loss_c"]:.4f}',
                    'kl_loss_u': f'{loss_dict["kl_loss_u"]:.4f}',
                    'diff_loss_c': f'{loss_dict["diff_loss_c"]:.4f}',
                    'diff_loss_u': f'{loss_dict["diff_loss_u"]:.4f}',
                })
        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


def train_GLAD(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}',
                    "distill_loss": f'{loss_dict["distill_loss"]:.4f}',
                    "distill_kl_loss":f'{loss_dict["distill_kl_loss"]:.4f}'
                })

        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

            # Visualization every 10 epochs
            # if epoch % 10 == 0 and hasattr(args, 'visual_save_path'):
            #     try:
            #         print(f"\n{'='*60}")
            #         print(f"[Visualization] Generating distribution plots for epoch {epoch}...")
            #         print(f"{'='*60}")

            #         z_vae_val, z_mixed_val, y_val = collect_representations_for_visualization(
            #             model, dataloader_temp, device, use_diffusion=True, max_samples=1000
            #         )

            #         if z_vae_val is not None and z_mixed_val is not None:
            #             z_vae_train, _, y_train = collect_representations_for_visualization(
            #                 model, dataloader, device, use_diffusion=False, max_samples=500
            #             )

            #             vis_results = visualize_all_distributions(
            #                 z_vae=z_vae_val,
            #                 z_mixed=z_mixed_val,
            #                 labels=y_val,
            #                 z_train=z_vae_train,
            #                 save_path=args.visual_save_path,
            #                 epoch=epoch
            #             )

            #             if 'stats' in vis_results:
            #                 stats = vis_results['stats']
            #                 print(f"\n[Visualization] Key Metrics:")
            #                 print(f"  - Expansion ratio: {stats['expansion_ratio']:.3f}x")
            #                 print(f"  - Std (VAE → Diff): {stats['vae_mean_std']:.4f} → {stats['diff_mean_std']:.4f}")

            #             if 'coverage_stats' in vis_results:
            #                 cov_stats = vis_results['coverage_stats']
            #                 print(f"  - Density improvement: {cov_stats['density_improvement']:.3f}x")
            #                 print(f"  - Low-density samples: {cov_stats['vae_low_density_ratio']*100:.1f}% → {cov_stats['mixed_low_density_ratio']*100:.1f}%")

            #             print(f"{'='*60}\n")
            #         else:
            #             print("[Visualization] Warning: No valid samples collected")

            #     except Exception as e:
            #         print(f"[Visualization] Error during visualization: {e}")
            #         import traceback
            #         traceback.print_exc()

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


def train_TGLR_w_Dy(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}'
                })

        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")

def train_TGLR_w_C(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}',
                })

        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


def train_TGLR_w_CA(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss_c': f'{loss_dict["pred_loss_c"]:.4f}',
                    'pred_loss_u': f'{loss_dict["pred_loss_u"]:.4f}',
                    'kl_loss_c': f'{loss_dict["kl_loss_c"]:.4f}',
                    'kl_loss_u': f'{loss_dict["kl_loss_u"]:.4f}',
                    'diff_loss_c': f'{loss_dict["diff_loss_c"]:.4f}',
                    'diff_loss_u': f'{loss_dict["diff_loss_u"]:.4f}',
                })
        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


def train_TGLR(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_update = False
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}',
                })

        tbar.close()
        model.log_epoch_variance("FULL", epoch)
        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.save_path,
                f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt'
            )
            tmp_path = checkpoint_path + ".tmp"
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                },
                tmp_path,
                _use_new_zipfile_serialization=False
            )
            os.replace(tmp_path, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0
                    best_update = True

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    tmp_path = best_model_path + ".tmp"

                    torch.save(
                        {
                            'epoch': epoch,
                            'best_val_f1': best_val_f1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        },
                        tmp_path,
                        _use_new_zipfile_serialization=False
                    )
                    os.replace(tmp_path, best_model_path)
                    model.log_epoch_variance("FULL_BEST", epoch, log_interval=1)
                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break
            if args.visual:
                # # Manifold visualization
                if epoch % args.visual_interval == 0 or best_update and hasattr(args, 'visual_save_path'):
                    best_update = False
                    try:
                        print(f"\n{'='*70}")
                        print(f" MANIFOLD EVOLUTION VISUALIZATION - Epoch {epoch}")
                        print(f"{'='*70}")

                        z_train_pred, y_train, train_info = collect_train_representations(
                            model=model,
                            dataloader=dataloader,
                            device=device,
                            current_epoch=epoch,
                            total_epochs=args.epochs,
                            max_samples=5000
                        )

                        z_test_raw, z_test_opt, y_test, test_info = collect_test_representations_with_ttt(
                            model=model,
                            dataloader=dataloader_temp,
                            device=device,
                            current_epoch=epoch,
                            total_epochs=args.epochs,
                            max_samples=5000
                        )

                        if z_train_pred is not None and z_test_raw is not None and z_test_opt is not None:
                            filepath, stats = visualize_manifold_with_ttt(
                                z_train_pred, z_test_raw, z_test_opt,
                                y_train, y_test,
                                train_info, test_info,
                                save_path=args.visual_save_path,
                                epoch=epoch
                            )

                            # Print key metrics
                            print(f"\n   Manifold + TTT Stats:")
                            print(f"\n     Before TTT (μ'):")
                            print(f"       Inside: {stats['raw_inside_ratio']*100:.1f}%")
                            print(f"       Outside: {stats['raw_outside_ratio']*100:.1f}%")
                            print(f"\n     After TTT (μ_opt):")
                            print(f"       Inside: {stats['opt_inside_ratio']*100:.1f}%")
                            print(f"       Outside: {stats['opt_outside_ratio']*100:.1f}%")
                            print(f"\n     TTT Improvement:")
                            print(f"       Inside ratio: +{stats['ttt_improvement']*100:.1f}%")
                            print(f"       Outside reduction: {stats['ttt_outside_reduction']*100:.1f}%")
                            print(f"{'='*70}\n")
                        else:
                            print(" Warning: No valid samples collected")

                    except Exception as e:
                        print(f" Error during visualization: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


def train_TGLR_wo_G(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}'
                })
        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")


# def train_GLAD_wo_LA(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
#     print('model loaded!')
#     start_epoch = 1

#     # Early stopping based on F1
#     early_stop_patience = 50
#     no_improve_epochs = 0
#     best_val_f1 = -1.0
#     best_epoch = -1
#     best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

#     log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

#     if args.resume_train:
#         model, optimizer, scheduler, start_epoch = load_checkpoint(
#             model, optimizer, scheduler, args.checkpoint_path, device
#         )

#     for epoch in range(start_epoch, args.epochs + 1):
#         model.train()
#         start_time = time.time()
#         loss_sum = 0
#         num_batches = 0
#         print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

#         with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
#             for idx, batch in enumerate(tbar):
#                 optimizer.zero_grad()
#                 x = batch.x.to(device)
#                 edge_index = batch.edge_index.to(device)
#                 y = batch.y.to(device)
#                 batch_data = batch.batch.to(device)

#                 loss_dict = model(x, edge_index, y, batch_data, epoch)
#                 loss = loss_dict['total_loss']
#                 loss.backward()
#                 optimizer.step()

#                 loss_sum += loss.item()
#                 num_batches += 1

#                 tbar.set_postfix({
#                     'Total': f'{loss:.4f}',
#                     'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
#                     'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
#                     'diff_loss': f'{loss_dict["diff_loss"]:.4f}'
#                 })
#         tbar.close()

#         avg_loss = loss_sum / num_batches if num_batches > 0 else 0
#         epoch_time = time.time() - start_time
#         print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
#         scheduler.step()

#         with open(log_file, "a") as f:
#             f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

#         # Keep periodic checkpoints for resuming training
#         if epoch % args.save_interval == 0 or epoch == args.epochs:
#             checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#             }, checkpoint_path)
#             print(f"Checkpoint saved to: {checkpoint_path}")

#         # Validation: retain the single best checkpoint by F1
#         if epoch % args.epoch_test == 0:
#             model.eval()
#             with torch.no_grad():
#                 result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

#                 current_f1 = float(result.get('f1', 0.0))
#                 current_acc = float(result.get('accuracy', 0.0))
#                 current_prec = float(result.get('precision', 0.0))
#                 current_rec = float(result.get('recall', 0.0))

#                 print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
#                       f"P={current_prec:.4f} | R={current_rec:.4f}")

#                 # Save the best-F1 checkpoint, replacing the previous best
#                 if current_f1 > best_val_f1:
#                     best_val_f1 = current_f1
#                     best_epoch = epoch
#                     no_improve_epochs = 0

#                     torch.save({
#                         'epoch': epoch,
#                         'best_val_f1': best_val_f1,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'scheduler_state_dict': scheduler.state_dict(),
#                     }, best_model_path)

#                     print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
#                     print(f"[BestF1] Saved to: {best_model_path}")
#                 else:
#                     no_improve_epochs += 1
#                     print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

#                 if no_improve_epochs >= early_stop_patience:
#                     print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
#                     break

#     print(f"\n=== Best Checkpoint (by F1) ===")
#     print(f"Best epoch: {best_epoch}")
#     print(f"Best val F1: {best_val_f1:.4f}")
#     print(f"Best model path: {best_model_path}")

def train_TGLR_wo_LA(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_update = False                          # Track whether validation performance has improved
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total':     f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss':   f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}'
                })
        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.save_path,
                f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt'
            )
            tmp_path = checkpoint_path + ".tmp"
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                },
                tmp_path,
                _use_new_zipfile_serialization=False
            )
            os.replace(tmp_path, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1   = float(result.get('f1', 0.0))
                current_acc  = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec  = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0
                    best_update = True           # Mark an improvement in this epoch

                    tmp_path = best_model_path + ".tmp"
                    torch.save(
                        {
                            'epoch': epoch,
                            'best_val_f1': best_val_f1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        },
                        tmp_path,
                        _use_new_zipfile_serialization=False
                    )
                    os.replace(tmp_path, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

            # Ablation visualization without VAE, diffusion, or TTA
            if args.visual:
                if epoch % args.visual_interval == 0 or (best_update and hasattr(args, 'visual_save_path')):
                    best_update = False          # Reset the improvement flag
                    try:
                        print(f"\n{'='*70}")
                        print(f" ABLATION MANIFOLD VISUALIZATION (wo_LA) - Epoch {epoch}")
                        print(f"{'='*70}")

                        # Collect training mu_prime after cross-attention, without VAE sampling
                        z_train_prime, y_train_vis, train_info = collect_train_representations_wo_la(
                            model=model,
                            dataloader=dataloader,
                            device=device,
                            current_epoch=epoch,
                            total_epochs=args.epochs,
                            max_samples=5000
                        )

                        # Collect test mu without TTT optimization
                        z_test_mu, y_test_vis, test_info = collect_test_representations_wo_la(
                            model=model,
                            dataloader=dataloader_temp,
                            device=device,
                            current_epoch=epoch,
                            total_epochs=args.epochs,
                            max_samples=5000
                        )

                        if z_train_prime is not None and z_test_mu is not None:
                            filepath, stats = visualize_manifold_ablation_wo_la(
                                z_train_prime, z_test_mu,
                                y_train_vis, y_test_vis,
                                train_info, test_info,
                                save_path=args.visual_save_path,
                                epoch=epoch
                            )

                            # Print key metrics
                            print(f"\n   Ablation Manifold Stats (wo_LA):")
                            print(f"     Training repr:  mu_prime (cross-attn, w/o VAE sampling)")
                            print(f"     Test repr:      mu       (w/o TTT)")
                            print(f"     Inside:         {stats['inside_ratio']*100:.1f}%")
                            print(f"     Boundary:       {stats['boundary_ratio']*100:.1f}%")
                            print(f"     Outside:        {stats['outside_ratio']*100:.1f}%  ← temporal mismatch gap")
                            print(f"     L2 Gap:         {stats['distribution_gap']:.4f}")
                            print(f"     Test F1:        {stats['f1_test']:.4f}")
                            print(f"{'='*70}\n")
                        else:
                            print(" Warning: No valid samples collected")

                    except Exception as e:
                        print(f" Error during ablation visualization: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")

def train_TGLR_wo_T(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                    'diff_loss': f'{loss_dict["diff_loss"]:.4f}',
                })
        tbar.close()

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")

def train_TGLR_wo_D(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1

    # Early stopping based on F1
    early_stop_patience = 50
    no_improve_epochs = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_by_f1.pt")

    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")

    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch.to(device)

                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_batches += 1

                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}',
                })
        tbar.close()
        model.log_epoch_variance("WO_DIFF ", epoch)
        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        # Periodic checkpoints for resuming training
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Validation: retain the single best checkpoint by F1
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics_with_multi_source(model, dataloader_temp, device)

                current_f1 = float(result.get('f1', 0.0))
                current_acc = float(result.get('accuracy', 0.0))
                current_prec = float(result.get('precision', 0.0))
                current_rec = float(result.get('recall', 0.0))

                print(f"[Val] Epoch {epoch} | F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                      f"P={current_prec:.4f} | R={current_rec:.4f}")

                # Save the best-F1 checkpoint, replacing the previous best
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_epoch = epoch
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)
                    model.log_epoch_variance("WO_DIFF_BEST", epoch, log_interval=1)
                    print(f"[BestF1] Updated best model at epoch {epoch}, best F1={best_val_f1:.4f}")
                    print(f"[BestF1] Saved to: {best_model_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} eval step(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    break

    print(f"\n=== Best Checkpoint (by F1) ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best model path: {best_model_path}")
