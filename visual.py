import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

def visualize_manifold_with_ttt(
    z_train_pred, z_test_raw, z_test_opt,
    y_train, y_test,
    train_info, test_info,
    save_path='./model_saves/visual',
    epoch=None
):

    # Font settings
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
    COLOR_IMPROVED = '#e41a1c'
    COLOR_TRAIN_BG = 'lightgray'

    # ===============================
    # 1. Convert to NumPy
    # ===============================
    z_train_pred = z_train_pred.detach().cpu().numpy()
    z_test_raw   = z_test_raw.detach().cpu().numpy()
    z_test_opt   = z_test_opt.detach().cpu().numpy()

    y_train = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.asarray(y_train)
    y_test  = y_test.detach().cpu().numpy()  if torch.is_tensor(y_test)  else np.asarray(y_test)

    assert len(z_test_raw) == len(z_test_opt) == len(y_test)

    print("-" * 30)
    z_std_per_dim = np.std(z_train_pred, axis=0)
    print(f"[Diagnostic] Std mean={np.mean(z_std_per_dim):.4e}  max={np.max(z_std_per_dim):.4e}")
    print(f"  Dead dims: {np.sum(z_std_per_dim < 1e-6)} / {z_train_pred.shape[1]}")
    dist_shift = np.linalg.norm(np.mean(z_train_pred, 0) - np.mean(z_test_raw, 0))
    print(f"  Distribution Shift: {dist_shift:.4f}")
    print("-" * 30)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 2  PCA → KDE
    # ===============================
    pca = PCA(n_components=2)
    z_train_pca    = pca.fit_transform(z_train_pred)
    z_test_raw_pca = pca.transform(z_test_raw)
    z_test_opt_pca = pca.transform(z_test_opt)
    explained_var  = pca.explained_variance_ratio_.sum()

    param_grid = {'bandwidth': np.logspace(-2, 1, 20)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), param_grid, cv=5, n_jobs=-1)
    grid.fit(z_train_pca)
    best_bandwidth = grid.best_params_['bandwidth']
    kde = grid.best_estimator_

    train_density    = np.exp(kde.score_samples(z_train_pca))
    test_raw_density = np.exp(kde.score_samples(z_test_raw_pca))
    test_opt_density = np.exp(kde.score_samples(z_test_opt_pca))

    threshold_high = np.percentile(train_density, 50)
    threshold_low  = np.percentile(train_density, 25)

    inside_raw   = test_raw_density >= threshold_high
    boundary_raw = (test_raw_density >= threshold_low) & (test_raw_density < threshold_high)
    outside_raw  = test_raw_density < threshold_low

    inside_opt   = test_opt_density >= threshold_high
    boundary_opt = (test_opt_density >= threshold_low) & (test_opt_density < threshold_high)
    outside_opt  = test_opt_density < threshold_low

    improved_mask = (~inside_raw) & inside_opt

    # ===============================
    # 3  t-SNE
    # ===============================
    print("  [Manifold+TTT] t-SNE...")
    all_z = np.vstack([z_train_pred, z_test_raw, z_test_opt])
    tsne  = TSNE(n_components=2, random_state=42,
                 perplexity=min(30, len(all_z) // 5))
    z_2d  = tsne.fit_transform(all_z)

    n_train = len(z_train_pred)
    n_test  = len(z_test_raw)
    z_train_2d    = z_2d[:n_train]
    z_test_raw_2d = z_2d[n_train : n_train + n_test]
    z_test_opt_2d = z_2d[n_train + n_test :]

    y_train_int = y_train.astype(int)
    y_test_int  = y_test.astype(int)

    # ===============================
    # 4. Figure and layout
    # ===============================
    fig = plt.figure(figsize=(21, 13))

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.14,
        hspace=0.42,
        wspace=0.28,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1]
    )

    ax_a = fig.add_subplot(gs[:, 0])   # Span both rows
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.set_aspect('equal', adjustable='box')

    # ===============================
    # 5. Helper functions
    # ===============================
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
        # Use normal font weight
        ax.set_xlabel('t-SNE Dim 1', labelpad=3)
        ax.set_ylabel('t-SNE Dim 2', labelpad=3)

    def add_caption(ax, line1, line2=None):
        text = line1 if line2 is None else f'{line1}\n{line2}'
        ax.annotate(
            text,
            xy=(0.5, 0),
            xycoords='axes fraction',
            xytext=(0, -62),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=21,
            # Use normal font weight
            multialignment='center'
        )

    # ===============================
    # 6. Draw subplots
    # ===============================

    # —— (a) ───────────────────────────────────────────────────────
    scatter_by_label(ax_a, z_train_2d, y_train_int, alpha=0.60, s=20)
    set_ax_labels(ax_a)
    add_caption(ax_a, '(a) Training Distribution (z)')
    set_closed_axes(ax_a)

    # —— (b) ───────────────────────────────────────────────────────
    ax_b.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                 c=COLOR_TRAIN_BG, alpha=0.15, s=5, zorder=1)
    scatter_by_label(ax_b, z_test_raw_2d, y_test_int,
                     alpha=0.85, s=50, edgecolors='black', linewidths=0.8, zorder=3)
    set_ax_labels(ax_b)
    add_caption(ax_b, '(b) Test Initial (\u03bc)')
    set_closed_axes(ax_b)

    # —— (d) After TTA ─────────────────────────────────────────────
    ax_c.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                 c=COLOR_TRAIN_BG, alpha=0.15, s=5, zorder=1)
    scatter_by_label(ax_c, z_test_opt_2d, y_test_int,
                     alpha=0.85, s=50, edgecolors='black', linewidths=0.7, zorder=3)
    set_ax_labels(ax_c)
    add_caption(ax_c, '(d) After TTA (\u03bc_opt)')
    set_closed_axes(ax_c)

    # Shared coverage plotting function
    def plot_coverage(ax, z2d, z_orig, inside, boundary, outside):
        # Use the same training background points as in panel (b)
        ax.scatter(z_train_2d[:, 0], z_train_2d[:, 1],
                   c=COLOR_TRAIN_BG, alpha=0.15, s=5, zorder=1)
        if inside.sum() > 0:
            ax.scatter(z2d[inside, 0], z2d[inside, 1],
                       c=COLOR_INSIDE, s=55, marker='o', alpha=0.85,
                       edgecolors='darkgreen', linewidths=1.2, zorder=3,
                       label=f'Inside ({inside.sum()}, {inside.mean()*100:.1f}%)')
        if boundary.sum() > 0:
            ax.scatter(z2d[boundary, 0], z2d[boundary, 1],
                       c=COLOR_BOUNDARY, s=55, marker='s', alpha=0.85,
                       edgecolors='darkorange', linewidths=1.2, zorder=3,
                       label=f'Boundary ({boundary.sum()}, {boundary.mean()*100:.1f}%)')
        if outside.sum() > 0:
            ax.scatter(z2d[outside, 0], z2d[outside, 1],
                       c=COLOR_OUTSIDE, s=55, marker='x', alpha=0.90,
                       linewidths=2.2, zorder=3,
                       label=f'Outside ({outside.sum()}, {outside.mean()*100:.1f}%)')
        ax.legend(loc='best', fontsize=13, framealpha=0.9, edgecolor='gray')
        # L2 distance between test and training centroids in the original high-dimensional space
        l2 = np.linalg.norm(np.mean(z_orig, axis=0) - np.mean(z_train_pred, axis=0))
        ax.text(
            0.97, 0.97,
            f'L2 Gap: {l2:.3f}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85)
        )
        set_closed_axes(ax)

    # —— (c) Coverage: Initial ─────────────────────────────────────
    plot_coverage(ax_d, z_test_raw_2d, z_test_raw, inside_raw, boundary_raw, outside_raw)
    set_ax_labels(ax_d)
    add_caption(
        ax_d,
        '(c) Coverage: Initial (\u03bc)',
        f'Inside: {inside_raw.sum()}/{len(test_raw_density)}'
    )

    # —— (e) Coverage: After TTA ───────────────────────────────────
    plot_coverage(ax_e, z_test_opt_2d, z_test_opt, inside_opt, boundary_opt, outside_opt)

    if improved_mask.sum() > 0:
        ax_e.scatter(
            z_test_opt_2d[improved_mask, 0],
            z_test_opt_2d[improved_mask, 1],
            facecolors='none', edgecolors=COLOR_IMPROVED,
            s=200, linewidths=2.3, marker='o', zorder=5
        )

    improvement     = inside_opt.sum() - inside_raw.sum()
    improvement_pct = (inside_opt.mean() - inside_raw.mean()) * 100
    sign_n   = '+' if improvement >= 0 else ''
    sign_pct = '+' if improvement_pct >= 0 else ''
    set_ax_labels(ax_e)
    add_caption(
        ax_e,
        '(e) Coverage: After TTA (\u03bc_opt)',
        (f'Inside: {inside_opt.sum()}/{len(test_opt_density)} '
         f'({sign_n}{improvement}, {sign_pct}{improvement_pct:.1f}%)')
    )

    # ===============================
    # 7. Shared legend at the top
    # ===============================
    legend_handles = []

    legend_handles.append(mpatches.Patch(facecolor=COLOR_CLASS0, edgecolor='k',
                                          linewidth=0.8, label='Non-source = 0'))
    legend_handles.append(mpatches.Patch(facecolor=COLOR_CLASS1, edgecolor='k',
                                          linewidth=0.8, label='Source = 1'))

    if inside_raw.sum() > 0 or inside_opt.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_INSIDE, edgecolor='darkgreen',
                                              linewidth=0.8, label='Inside'))
    if boundary_raw.sum() > 0 or boundary_opt.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_BOUNDARY, edgecolor='darkorange',
                                              linewidth=0.8, label='Boundary'))
    if outside_raw.sum() > 0 or outside_opt.sum() > 0:
        legend_handles.append(mpatches.Patch(facecolor=COLOR_OUTSIDE, edgecolor='none',
                                              label='Outside'))
    if improved_mask.sum() > 0:
        legend_handles.append(mlines.Line2D([], [], color=COLOR_IMPROVED, marker='o',
                                             markerfacecolor='none', markersize=13,
                                             linewidth=0, markeredgewidth=2.3,
                                             label='TTA: Improved \u2192 Inside'))

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

    # ===============================
    # 8. Save the figure
    # ===============================
    filepath_png = save_dir / f'manifold_with_ttt_epoch{epoch}.png'
    filepath_pdf = save_dir / f'manifold_with_ttt_epoch{epoch}.pdf'

    fig.savefig(filepath_png, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {filepath_png}")
    fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    print(f"  Saved PDF: {filepath_pdf}")
    plt.close(fig)

    # ===============================
    # 9. Return statistics
    # ===============================
    source_mask = y_test == 1
    stats = {
        'epoch': epoch,
        'stage': train_info.get('stage', 'unknown'),
        'lambda': train_info.get('lambda', 0.0),
        'raw_inside_ratio':      float(inside_raw.mean()),
        'opt_inside_ratio':      float(inside_opt.mean()),
        'raw_outside_ratio':     float(outside_raw.mean()),
        'opt_outside_ratio':     float(outside_opt.mean()),
        'raw_boundary_ratio':    float(boundary_raw.mean()),
        'opt_boundary_ratio':    float(boundary_opt.mean()),
        'ttt_improvement':       float(inside_opt.mean()  - inside_raw.mean()),
        'ttt_outside_reduction': float(outside_raw.mean() - outside_opt.mean()),
        'raw_source_inside':  float(inside_raw[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'raw_source_outside': float(outside_raw[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'opt_source_inside':  float(inside_opt[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'opt_source_outside': float(outside_opt[source_mask].mean()) if source_mask.sum() > 0 else 0.0,
        'raw_mean_density':  float(test_raw_density.mean()),
        'opt_mean_density':  float(test_opt_density.mean()),
        'density_ratio':     float(test_opt_density.mean() / test_raw_density.mean())
                             if test_raw_density.mean() > 0 else 1.0,
        'n_train':    int(len(z_train_pred)),
        'n_test':     int(len(z_test_raw)),
        'n_improved': int(improved_mask.sum()),
        'kde_bandwidth':          float(best_bandwidth),
        'pca_components':         2,
        'pca_explained_variance': float(explained_var),
    }

    return str(filepath_png), stats
