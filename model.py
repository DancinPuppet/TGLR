import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math
import torch.optim as optim

class GATModel(nn.Module):
    """
    GAT message aggregation model.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.res_proj = nn.Linear(in_channels, out_channels)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = 0.2

    def forward(self, x, edge_index):
        residual = self.res_proj(x)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x + residual


class MLPModel(nn.Module):
    """
    Return logits for source and non-source nodes (two classes).
    """
    def __init__(self, hidden_channels, out_channels, temperature, para_args, dropout=0.2):
        super(MLPModel, self).__init__()
        self.args = para_args
        self.source_probs = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)
        )
        self.temperature = temperature

    def forward(self, h, batch):
        logits = self.source_probs(h)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        graph_probs = []
        graph_samples = []
        graph_logits_split = []

        for batch_idx in torch.unique(batch):
            mask = (batch == batch_idx)
            graph_logits = logits[mask]  # [N_i, 2]
            graph_logits_split.append(graph_logits)

            if self.training:
                gumbel_probs = F.gumbel_softmax(
                    graph_logits,
                    tau=self.temperature,
                    hard=False,
                    dim=1
                )
                graph_probs.append(gumbel_probs)
                graph_samples.append(gumbel_probs)
            else:
                probs = F.softmax(graph_logits, dim=1)[:, 1]
                hard_sample = torch.zeros_like(probs)
                hard_sample[probs >= self.args.threshold] = 1.0
                graph_probs.append(probs)
                graph_samples.append(hard_sample)

        return {
            'logits': logits,
            'batch': batch,
            'graph_logits': graph_logits_split,
            'graph_probs': graph_probs,
            'graph_samples': graph_samples,
            'features': h
        }


def reparameterize(mu, logvar):
    """
    Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def compute_loss(outputs, y, batch, para_args):
    """
    Cross-entropy with increased gradient weight for positive samples.
    outputs: Dictionary containing 'graph_logits', a list of per-graph [N_i, 2] logits.
    y: Labels for all nodes, shape [N_total], with values 0 or 1.
    batch: Graph index for each node, shape [N_total].
    para_args: Additional parameters, optionally including graph_type or pos_weight.
    """
    all_pred = []
    all_true = []

    # 1. Collect logits and labels for each graph
    for i, batch_idx in enumerate(torch.unique(batch)):
        mask = (batch == batch_idx)
        true_labels = y[mask].long()
        pred_logits = outputs['graph_logits'][i]
        all_pred.append(pred_logits)
        all_true.append(true_labels)

    # 2. Concatenate across graphs
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    device = all_pred.device

    class_counts = torch.bincount(all_true, minlength=2).float().to(device)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    weights = weights / weights.sum()

    per_node_loss = F.cross_entropy(all_pred, all_true, weight=weights, reduction='none')

    pos_mask = (all_true == 1).float()
    pos_weight = min(math.log(len(all_true) + 1), 1.5)
    per_node_loss = per_node_loss * (1.0 + pos_mask * (pos_weight - 1.0))

    loss = per_node_loss.mean()
    return loss


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, time_proj):
        residual = x
        x = self.norm(x + time_proj)
        x = self.mlp(x)
        return x + residual  # Residual connection


class Diffusion(nn.Module):
    """
    Latent Diffusion model
    """

    def __init__(self, max_time_steps, beta_start, beta_end, time_embed_dim, latent_dim, hidden_dim):
        super(Diffusion, self).__init__()
        self.max_time_steps = max_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_embed_dim = time_embed_dim
        # Precompute diffusion coefficients
        self.register_buffer('betas', self.get_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_bar_prev', torch.cat([torch.tensor([1.0]), self.alphas_bar[:-1]]))
        # Reparameterization coefficients
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - self.alphas_bar))
        # Time-step positional encoding
        time_pos_enc = torch.zeros(max_time_steps, time_embed_dim)
        position = torch.arange(0, max_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, time_embed_dim, 2).float() * (-math.log(10000.0) / time_embed_dim))
        time_pos_enc[:, 0::2] = torch.sin(position * div_term)
        time_pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('time_pos_enc', time_pos_enc)  # [T, D]
        # Time-step embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        # Input projection
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        # Backbone network
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        # Output layer
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def q_sample(self, z0, t, noise=None):
        """
        Forward diffusion process.
        """
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t].unsqueeze(1)
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t].unsqueeze(1)
        # Reparameterization: zt = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * noise
        zt = sqrt_alphas_bar_t * z0 + sqrt_one_minus_alphas_bar_t * noise
        return zt

    def forward(self, z0, t):
        """
        Forward diffusion and z0 prediction.
        """
        zt = self.q_sample(z0, t)
        # Temporal positional encoding
        time_pos = self.time_pos_enc[t]
        time_emb = self.time_mlp(time_pos)
        time_proj = self.time_proj(time_emb)
        # Input projection
        x = self.x_proj(zt)
        # Apply three residual blocks
        for block in self.blocks:
            x = block(x, time_proj)
        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        z0_recon = self.output_proj(x)
        return z0_recon

    def get_beta_schedule(self):
        """
        Use a linear schedule.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.max_time_steps)

# CFG-constrained optimization
class GLCFGDModel(nn.Module):
    "GLCFGDModel: Guidance + CFG + Filtering + Diffusion"

    def __init__(self, model_args):
        super().__init__()
        self.args = model_args

        # -------- feature dims --------
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # -------- hparams --------
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)

        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # CFG-style: dual-branch training weight (uncond branch)
        self.pred_loss_uncond_weight = float(getattr(self.args, "pred_loss_uncond_weight", 0.5))
        # Set unconditional KL/diffusion weights separately; otherwise inherit pred_loss_uncond_weight
        self.kl_uncond_weight = float(getattr(self.args, "kl_uncond_weight", self.pred_loss_uncond_weight))
        self.diff_uncond_weight = float(getattr(self.args, "diff_uncond_weight", self.pred_loss_uncond_weight))

        # diffusion params (keep your original)
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        # diffusion mixing schedule (keep your style)
        self.pA = float(getattr(self.args, "diff_pA", 0.50))
        self.pB = float(getattr(self.args, "diff_pB", 0.80))
        self.diff_lam_cap = float(getattr(self.args, "diff_lam_cap", 0.10))

        # -------- modules --------
        # GAT aggregation (main static features)
        self.aggregation = GATModel(
            self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout
        )

        # VAE-style mapping
        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        # ctx encoder (GRU)
        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        # cross-attn rectification (cond only)
        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.delta_logvar_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # diffusion
        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        # predictor head (shared by both branches)
        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    # ---------- helpers ----------
    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        """
        ctx_flat layout from your code:
          lam:  (T1-1)
          gam:  (T1-1)
          rhoI: (T1)
          rhoS: (T1)
        then rhoI_ = rhoI[:,1:], rhoS_ = rhoS[:,1:] => both (T1-1)
        tokens: [n, T1-1, 4]
        """
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        return tokens

    @staticmethod
    def _pack_graph_logits(logits: torch.Tensor, batch_idx: torch.Tensor):
        graph_logits = []
        for g in torch.unique(batch_idx):
            m = (batch_idx == g)
            graph_logits.append(logits[m])
        return {"graph_logits": graph_logits}

    def _diffusion_pred(self, mu_center: torch.Tensor, logvar: torch.Tensor, progress_p: float):
        """
        Build z_pred with your diffusion schedule:
          - sample z0 around mu_center
          - denoise z0_hat
          - mix mu_center and z0_hat after pA
        """
        eps = torch.randn_like(mu_center)
        z0 = mu_center + torch.exp(0.5 * logvar) * eps

        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=z0.device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=z0.device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu_center

        return z_pred, diff_loss

    # ---------- forward ----------
    def forward(self, x, edge_index, y, batch, epoch):
        """
        Dual-branch (uncond & cond) training with shared parameters.
        - uncond: mu_u = mu
        - cond:   mu_c = mu_prime (ctx-guided rectification)
        Total loss:
          L = L_pred(cond) + w_u * L_pred(uncond)
            + kl_w * (KL(cond) + kl_u_w * KL(uncond))
            + diff_w * (Diff(cond) + diff_u_w * Diff(uncond))
        """
        device = x.device
        y = y.view(-1)

        # ----- candidate mask -----
        rise_flag = x[:, 6]  # consistent with your layout
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss_c": zero,
                "pred_loss_u": zero,
                "kl_loss_c": zero,
                "kl_loss_u": zero,
                "diff_loss_c": zero,
                "diff_loss_u": zero,
            }

        # ----- main embedding on full graph -----
        x_main_base = x[:, :8 - 2]   # [N,6]
        rho_last = x[:, -2:]         # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]
        h_all = self.aggregation(x_main, edge_index)         # [N, out_channels]
        h_cand = h_all[cand_idx]                             # [n', out_channels]

        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)               # [n', latent]
        logvar = self.fc_logvar(h_enc)       # [n', latent]

        # progress
        p = float(epoch) / float(self.args.epochs)

        # ===== uncond branch =====
        mu_u = mu
        z_pred_u, diff_loss_u = self._diffusion_pred(mu_u, logvar, p)

        # ===== cond branch (ctx-guided rectification) =====
        ctx_flat = x[:, self.main_dim : -2]  # same slice as your code
        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)  # [n', T1-1, 4]
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0)  # [n', latent]

        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        # delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
        mu_c = mu + delta_mu
        # logvar_c = logvar + delta_logvar

        z_pred_c, diff_loss_c = self._diffusion_pred(mu_c, logvar, p)

        # ----- supervised losses (both branches) -----
        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        logits_c = self.pred_head(z_pred_c)
        outputs_c = self._pack_graph_logits(logits_c, batch_cand)
        pred_loss_c = compute_loss(outputs_c, y_cand, batch_cand, self.args)

        logits_u = self.pred_head(z_pred_u)
        outputs_u = self._pack_graph_logits(logits_u, batch_cand)
        pred_loss_u = compute_loss(outputs_u, y_cand, batch_cand, self.args)

        # ----- KL (both branches) -----
        kl_loss_c = self._kl_normal_standard(mu_c, logvar).mean()
        kl_loss_u = self._kl_normal_standard(mu_u, logvar).mean()

        # ----- total -----
        w_u = float(self.pred_loss_uncond_weight)
        w_klu = float(self.kl_uncond_weight)
        w_diffu = float(self.diff_uncond_weight)

        total_loss = (
            pred_loss_c
            + w_u * pred_loss_u
            + self.kl_weight * (kl_loss_c + w_klu * kl_loss_u)
            + self.diff_weight * (diff_loss_c + w_diffu * diff_loss_u)
        )

        return {
            "total_loss": total_loss,
            "pred_loss_c": pred_loss_c,
            "pred_loss_u": pred_loss_u,
            "kl_loss_c": kl_loss_c,
            "kl_loss_u": kl_loss_u,
            "diff_loss_c": diff_loss_c,
            "diff_loss_u": diff_loss_u,
        }

    # ---------- inference ----------
    def inference(self, x, edge_index, batch, obs=None):
        """
        Inference: uncond-only (no ctx).
        """
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8 - 2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)
            h_all = self.aggregation(x_main, edge_index)
            h_enc = self.enc(h_all)
            mu = self.fc_mu(h_enc)

            logits_all = self.pred_head(mu)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# class GLCFGDModel(nn.Module):
#     "GLCFGDModel: Guidance + CFG + Filtering + Diffusion"

#     def __init__(self, model_args):
#         super().__init__()
#         self.args = model_args

#         # -------- feature dims --------
#         self.struct_dim = 5
#         self.lpsi_dim = 1
#         self.self_time_dim = 2
#         self.rho_last_feat_dim = 2
#         self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

#         # -------- hparams --------
#         self.hidden_channels = getattr(self.args, "hidden_channels", 64)
#         self.out_channels = getattr(self.args, "out_channels", 64)
#         self.heads = getattr(self.args, "heads", 4)
#         self.dropout = getattr(self.args, "dropout", 0.2)

#         self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
#         self.obs_len = int(getattr(self.args, "obs_len", 0))

#         self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
#         self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
#         self.use_diffusion = self.diff_weight > 0

#         self.ctx_token_dim = 4

#         # CFG-style: dual-branch training weight (uncond branch)
#         self.pred_loss_uncond_weight = float(getattr(self.args, "pred_loss_uncond_weight", 0.5))
#         # Set unconditional KL/diffusion weights separately; otherwise inherit pred_loss_uncond_weight
#         self.kl_uncond_weight = float(getattr(self.args, "kl_uncond_weight", self.pred_loss_uncond_weight))
#         self.diff_uncond_weight = float(getattr(self.args, "diff_uncond_weight", self.pred_loss_uncond_weight))

#         # diffusion params (keep your original)
#         self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
#         self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
#         self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
#         self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
#         self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

#         # diffusion mixing schedule (keep your style)
#         self.pA = float(getattr(self.args, "diff_pA", 0.90))
#         self.pB = float(getattr(self.args, "diff_pB", 0.95))
#         self.diff_lam_cap = float(getattr(self.args, "diff_lam_cap", 0.10))
#         self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.1))

#         # -------- modules --------
#         # GAT aggregation (main static features)
#         self.aggregation = GATModel(
#             self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout
#         )

#         # VAE-style mapping
#         self.enc = nn.Sequential(
#             nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#         )
#         self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
#         self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)


#         # ctx encoder (GRU)
#         self.ctx_gru = nn.GRU(
#             input_size=self.ctx_token_dim,
#             hidden_size=self.latent_dim,
#             num_layers=1,
#             batch_first=True,
#         )

#         # cross-attn rectification (cond only)
#         attn_heads = min(4, max(1, self.latent_dim // 16))
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.latent_dim,
#             num_heads=attn_heads,
#             dropout=self.dropout,
#             batch_first=True,
#         )
#         self.delta_ffn = nn.Sequential(
#             nn.LayerNorm(self.latent_dim),
#             nn.Linear(self.latent_dim, self.latent_dim * 2),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.latent_dim * 2, self.latent_dim),
#         )
#         self.delta_logvar_ffn = nn.Sequential(
#             nn.LayerNorm(self.latent_dim),
#             nn.Linear(self.latent_dim, self.latent_dim),
#         )

#         # diffusion
#         self.diffusion = Diffusion(
#             self.max_time_steps,
#             self.beta_start,
#             self.beta_end,
#             self.time_embed_dim,
#             self.latent_dim,
#             self.predictor_dim,
#         )

#         # predictor head (shared by both branches)
#         pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
#         self.pred_head = nn.Sequential(
#             nn.Linear(self.latent_dim, pred_hidden),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(pred_hidden, 2),
#         )

#         self.initial_parameters()

#     # ---------- helpers ----------
#     @staticmethod
#     def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

#     def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
#         """
#         ctx_flat layout from your code:
#           lam:  (T1-1)
#           gam:  (T1-1)
#           rhoI: (T1)
#           rhoS: (T1)
#         then rhoI_ = rhoI[:,1:], rhoS_ = rhoS[:,1:] => both (T1-1)
#         tokens: [n, T1-1, 4]
#         """
#         lam = ctx_cand[:, 0 : (T1 - 1)]
#         gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
#         rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
#         rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
#         rhoI_ = rhoI[:, 1:]
#         rhoS_ = rhoS[:, 1:]
#         tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
#         return tokens

#     @staticmethod
#     def _pack_graph_logits(logits: torch.Tensor, batch_idx: torch.Tensor):
#         graph_logits = []
#         for g in torch.unique(batch_idx):
#             m = (batch_idx == g)
#             graph_logits.append(logits[m])
#         return {"graph_logits": graph_logits}

#     def _diffusion_pred(self, mu_center: torch.Tensor, logvar: torch.Tensor, progress_p: float):
#         """
#         Build z_pred with your diffusion schedule:
#           - sample z0 around mu_center
#           - denoise z0_hat
#           - mix mu_center and z0_hat after pA
#         """
#         eps = torch.randn_like(mu_center)
#         z0 = mu_center + torch.exp(0.5 * logvar) * eps

#         if self.use_diffusion:
#             t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=z0.device).long()
#             z0_hat = self.diffusion(z0, t)
#             diff_loss = F.mse_loss(z0_hat, z0.detach())
#         else:
#             z0_hat = z0
#             diff_loss = torch.tensor(0.0, device=z0.device)

#         # mixing
#         if (not self.use_diffusion) or (progress_p < self.pA):
#             z_pred = z0
#         else:
#             if progress_p < self.pB:
#                 lam = (progress_p - self.pA) / (self.pB - self.pA + 1e-12)
#                 lam = float(max(0.0, min(1.0, lam)))
#                 lam = min(lam, float(self.diff_lam_cap))
#             else:
#                 lam = float(self.diff_lam_cap)
#             z_pred = (1.0 - lam) * mu_center + lam * z0_hat

#         return z_pred, diff_loss

#     # ---------- forward ----------
#     def forward(self, x, edge_index, y, batch, epoch):
#         """
#         Dual-branch (uncond & cond) training with shared parameters.
#         - uncond: mu_u = mu
#         - cond:   mu_c = mu_prime (ctx-guided rectification)
#         Total loss:
#           L = L_pred(cond) + w_u * L_pred(uncond)
#             + kl_w * (KL(cond) + kl_u_w * KL(uncond))
#             + diff_w * (Diff(cond) + diff_u_w * Diff(uncond))
#         """
#         device = x.device
#         y = y.view(-1)

#         # ----- candidate mask -----
#         rise_flag = x[:, 6]  # consistent with your layout
#         cand_mask = (rise_flag == 0)
#         cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

#         if cand_idx.numel() == 0:
#             zero = torch.tensor(0.0, device=device)
#             return {
#                 "total_loss": zero,
#                 "pred_loss_c": zero,
#                 "pred_loss_u": zero,
#                 "kl_loss_c": zero,
#                 "kl_loss_u": zero,
#                 "diff_loss_c": zero,
#                 "diff_loss_u": zero,
#             }

#         # ----- main embedding on full graph -----
#         x_main_base = x[:, :8 - 2]   # [N,6]
#         rho_last = x[:, -2:]         # [N,2]
#         x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]
#         h_all = self.aggregation(x_main, edge_index)         # [N, out_channels]
#         h_cand = h_all[cand_idx]                             # [n', out_channels]

#         h_enc = self.enc(h_cand)
#         mu = self.fc_mu(h_enc)               # [n', latent]
#         logvar = self.fc_logvar(h_enc)       # [n', latent]

#         # progress
#         p = float(epoch) / float(self.args.epochs)

#         # ===== uncond branch =====
#         mu_u = mu
#         z_pred_u, diff_loss_u = self._diffusion_pred(mu_u, logvar, p)

#         # ===== cond branch (ctx-guided rectification) =====
#         ctx_flat = x[:, self.main_dim : -2]  # same slice as your code
#         if ctx_flat.numel() == 0 or self.obs_len <= 1:
#             c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
#         else:
#             T1 = self.obs_len
#             ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
#             if ctx_flat.size(1) != ctx_dim_expected:
#                 raise ValueError(
#                     f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
#                 )
#             ctx_cand = ctx_flat[cand_idx]
#             tokens = self._build_ctx_tokens(ctx_cand, T1)  # [n', T1-1, 4]
#             _, h_last = self.ctx_gru(tokens)
#             c = h_last.squeeze(0)  # [n', latent]

#         q = mu.unsqueeze(1)
#         k = c.unsqueeze(1)
#         v = c.unsqueeze(1)
#         attn_out, _ = self.cross_attn(q, k, v)
#         delta_mu = self.delta_ffn(attn_out.squeeze(1))
#         delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
#         mu_c = mu + delta_mu
#         logvar_c = logvar + delta_logvar

#         z_pred_c, diff_loss_c = self._diffusion_pred(mu_c, logvar_c, p)

#         mu_t = mu_c.detach()
#         logvar_t = logvar_c.detach()

#         def kl_teacher_student(mu_t, logvar_t, mu_s, logvar_s):
#             var_t = torch.exp(logvar_t)
#             var_s = torch.exp(logvar_s)
#             # KL( N(mu_t, var_t) || N(mu_s, var_s) ) formula
#             kl = 0.5 * (logvar_s - logvar_t + (var_t + (mu_t - mu_s).pow(2)) / (var_s + 1e-8) - 1.0)
#             return kl.sum(dim=1)
#         distill_kl_loss = kl_teacher_student(mu_t, logvar_t, mu_u, logvar).mean()

#         # ----- supervised losses (both branches) -----
#         batch_cand = batch[cand_idx]
#         y_cand = y[cand_idx]

#         logits_c = self.pred_head(z_pred_c)
#         outputs_c = self._pack_graph_logits(logits_c, batch_cand)
#         pred_loss_c = compute_loss(outputs_c, y_cand, batch_cand, self.args)

#         logits_u = self.pred_head(z_pred_u)
#         outputs_u = self._pack_graph_logits(logits_u, batch_cand)
#         pred_loss_u = compute_loss(outputs_u, y_cand, batch_cand, self.args)

#         # ----- KL (both branches) -----
#         kl_loss_c = self._kl_normal_standard(mu_c, logvar_c).mean()
#         kl_loss_u = self._kl_normal_standard(mu_u, logvar).mean()

#         # ----- total -----
#         w_u = float(self.pred_loss_uncond_weight)
#         w_klu = float(self.kl_uncond_weight)
#         w_diffu = float(self.diff_uncond_weight)
#         w_distill = float(getattr(self.args, "distill_kl_weight", 0.1))

#         total_loss = (
#             pred_loss_c
#             + w_u * pred_loss_u
#             + self.kl_weight * (kl_loss_c + w_klu * kl_loss_u)
#             + self.diff_weight * (diff_loss_c + w_diffu * diff_loss_u)
#         )

#         return {
#             "total_loss": total_loss,
#             "pred_loss_c": pred_loss_c,
#             "pred_loss_u": pred_loss_u,
#             "kl_loss_c": kl_loss_c,
#             "kl_loss_u": kl_loss_u,
#             "diff_loss_c": diff_loss_c,
#             "diff_loss_u": diff_loss_u,
#         }

#     # ---------- inference ----------
#     def inference(self, x, edge_index, batch, obs=None):
#         """
#         Inference: uncond-only (no ctx).
#         """
#         self.eval()
#         device = x.device
#         with torch.no_grad():
#             if batch.device != device:
#                 batch = batch.to(device)

#             x_main_base = x[:, :8 - 2]
#             rho_last = x[:, -2:]
#             x_main = torch.cat([x_main_base, rho_last], dim=1)
#             h_all = self.aggregation(x_main, edge_index)
#             h_enc = self.enc(h_all)
#             mu = self.fc_mu(h_enc)

#             logits_all = self.pred_head(mu)
#             probs_all = F.softmax(logits_all, dim=1)[:, 1]

#             preds = []
#             for g in torch.unique(batch):
#                 m = (batch == g)
#                 preds.append(probs_all[m])
#             return preds

#     def initial_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

# # CFG interpolation formula
# class GLADModel(nn.Module):
#     "GLAD: Guidance + Alignment + Filtering + Diffusion"
#     def __init__(self, model_args):
#         super().__init__()

#         self.args = model_args
#         # Feature dimensions
#         self.struct_dim = 5
#         self.lpsi_dim = 1
#         self.self_time_dim = 2
#         self.rho_last_feat_dim = 2
#         self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

#         # Hyperparameter configuration
#         self.hidden_channels = getattr(self.args, "hidden_channels", 64)
#         self.out_channels = getattr(self.args, "out_channels", 64)
#         self.heads = getattr(self.args, "heads", 4)
#         self.dropout = getattr(self.args, "dropout", 0.2)
#         self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
#         self.obs_len = int(getattr(self.args, "obs_len", 0))
#         self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
#         self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
#         self.use_diffusion = self.diff_weight > 0
#         self.ctx_token_dim = 4
#         self.distill_weight = getattr(self.args, "distill_weight", 1.0)   # λ_d
#         self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
#         self.student_layers = getattr(self.args, "student_layers", 2)
#         self.use_gate = getattr(self.args, "student_use_gate", True)
#         self.student_use_gate = self.use_gate
#         self.logvar_min = getattr(self.args, "logvar_min", -6.0)
#         self.logvar_max = getattr(self.args, "logvar_max", 2.0)
#         self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
#         self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
#         self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
#         self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
#         self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))
#         self.cfg_w = float(getattr(self.args, "cfg_w", 1.0))
#         self.cfg_w_max = float(getattr(self.args, "cfg_w_max", self.cfg_w))
#         self.cfg_warmup_p = float(getattr(self.args, "cfg_warmup_p", 0.3))  # progress in [0,1]
#         self.cfg_teacher_source = str(getattr(self.args, "cfg_teacher_source", "cfg"))  # "cond" or "cfg"
#         self.pred_loss_uncond_weight = float(getattr(self.args, "pred_loss_uncond_weight", 0.0))  # optional
#         if self.cfg_w_max < 0:
#             self.cfg_w_max = 0.0

#         # Distribution distillation weight
#         self.distill_kl_weight = getattr(self.args, "distill_kl_weight", 0.1)

#         # Model configuration
#         ## Aggregate primary features with GAT
#         self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

#         ## VAE1 Encoder
#         self.enc = nn.Sequential(
#             nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#         )
#         self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
#         self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

#         ## Encode propagation dynamics with GRU
#         self.ctx_gru = nn.GRU(
#             input_size=self.ctx_token_dim,
#             hidden_size=self.latent_dim,
#             num_layers=1,
#             batch_first=True,
#         )

#         ## Cross-modal attention refinement
#         attn_heads = min(4, max(1, self.latent_dim // 16))
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.latent_dim,
#             num_heads=attn_heads,
#             dropout=self.dropout,
#             batch_first=True,
#         )
#         self.delta_ffn = nn.Sequential(
#             nn.LayerNorm(self.latent_dim),
#             nn.Linear(self.latent_dim, self.latent_dim * 2),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.latent_dim * 2, self.latent_dim),
#         )

#         # ## VAE2 Encoder
#         # layers = []
#         # in_dim = self.latent_dim
#         # for i in range(self.student_layers - 1):
#         #     layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#         #     in_dim = self.student_h
#         # layers += [nn.Linear(in_dim, self.latent_dim)]
#         # self.student_delta = nn.Sequential(*layers)   # Predict delta_mu_s

#         # layers = []
#         # in_dim = self.latent_dim
#         # for i in range(self.student_layers - 1):
#         #     layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#         #     in_dim = self.student_h
#         # layers += [nn.Linear(in_dim, self.latent_dim)]
#         # self.student_logvar = nn.Sequential(*layers)  # Predict logvar_s

#         layers = []
#         in_dim = self.latent_dim
#         for i in range(self.student_layers - 1):
#             layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#             in_dim = self.student_h
#         layers += [nn.Linear(in_dim, self.latent_dim * 2)]
#         self.student_delta_head = nn.Sequential(*layers)

#         if self.student_use_gate:
#             self.student_gate = nn.Sequential(
#                 nn.Linear(self.latent_dim, self.latent_dim),
#                 nn.Sigmoid()
#             )

#         # Diffusion
#         self.diffusion = Diffusion(
#             self.max_time_steps,
#             self.beta_start,
#             self.beta_end,
#             self.time_embed_dim,
#             self.latent_dim,
#             self.predictor_dim,
#         )

#         # Prediction probabilities
#         pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
#         self.pred_head = nn.Sequential(
#             nn.Linear(self.latent_dim, pred_hidden),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(pred_hidden, 2),
#         )

#         self.initial_parameters()

#     @staticmethod
#     def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """KL divergence from the standard normal distribution."""
#         return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

#     def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
#         """Context features."""
#         n = ctx_cand.size(0)
#         lam = ctx_cand[:, 0 : (T1 - 1)]
#         gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
#         rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
#         rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]

#         # Align dimensions
#         rhoI_ = rhoI[:, 1:]
#         rhoS_ = rhoS[:, 1:]

#         # tokens: [n, T1-1, 4]
#         tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
#         assert tokens.shape == (n, T1 - 1, 4)
#         return tokens

#     def forward(self, x, edge_index, y, batch, epoch):
#         """forward training with CFG interpolation"""
#         device = x.device
#         y = y.view(-1)

#         # ----- candidate mask (unchanged) -----
#         rise_flag = x[:, 6]
#         cand_mask = (rise_flag == 0)
#         cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

#         if cand_idx.numel() == 0:
#             zero = torch.tensor(0.0, device=device)
#             return {
#                 "total_loss": zero,
#                 "pred_loss": zero,
#                 "pred_loss_u": zero,
#                 "kl_loss": zero,
#                 "diff_loss": zero,
#                 "distill_loss": zero,
#                 "distill_kl_loss": zero,
#                 "cfg_w": torch.tensor(0.0, device=device),
#             }

#         # ----- main embedding (unchanged) -----
#         x_main_base = x[:, :8-2]     # [N,6]
#         rho_last = x[:, -2:]         # [N,2]
#         x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]
#         h_all = self.aggregation(x_main, edge_index)         # [N, out_channels]
#         h_cand = h_all[cand_idx]                             # [n', out_channels]

#         h_enc = self.enc(h_cand)
#         mu = self.fc_mu(h_enc)               # [n', latent]
#         logvar = self.fc_logvar(h_enc)       # [n', latent]

#         # ====== build temporal context c (unchanged) ======
#         ctx_flat = x[:, self.main_dim : -2]
#         if ctx_flat.numel() == 0 or self.obs_len <= 1:
#             c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
#         else:
#             T1 = self.obs_len
#             ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
#             if ctx_flat.size(1) != ctx_dim_expected:
#                 raise ValueError(
#                     f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
#                 )
#             ctx_cand = ctx_flat[cand_idx]
#             tokens = self._build_ctx_tokens(ctx_cand, T1)  # [n', T1-1, 4]
#             _, h_last = self.ctx_gru(tokens)
#             c = h_last.squeeze(0)

#         # ====== conditional correction (cond branch) ======
#         q = mu.unsqueeze(1)
#         k = c.unsqueeze(1)
#         v = c.unsqueeze(1)
#         attn_out, _ = self.cross_attn(q, k, v)
#         delta_mu = self.delta_ffn(attn_out.squeeze(1))
#         mu_prime = mu + delta_mu     # cond mean

#         # ====== CFG interpolation (NEW) ======
#         # w schedule: warmup then clamp
#         p = epoch / float(self.args.epochs)  # progress in [0,1]
#         if self.cfg_warmup_p > 1e-8:
#             warm = min(1.0, p / max(self.cfg_warmup_p, 1e-8))
#         else:
#             warm = 0.3
#         cfg_w = float(self.cfg_w_max) * float(warm)

#         # z_cfg = mu + w*(mu_prime - mu)
#         mu_cfg = mu + cfg_w * (mu_prime - mu)

#         # ====== diffusion regularization (keep your original) ======
#         # sample around teacher mean for diffusion branch
#         eps = torch.randn_like(mu_cfg)
#         z0 = mu_cfg + torch.exp(0.5 * logvar) * eps

#         if self.use_diffusion:
#             t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
#             z0_hat = self.diffusion(z0, t)
#             diff_loss = F.mse_loss(z0_hat, z0.detach())
#         else:
#             z0_hat = z0
#             diff_loss = torch.tensor(0.0, device=device)

#         # keep your original diffusion mixing schedule (pA/pB), but now based on mu_t (teacher mean)
#         pA, pB = 0.50, 0.80
#         lam = 0.0
#         if (not self.use_diffusion) or (p < pA):
#             z_pred = z0
#         else:
#             if p < pB and lam < 0.1:
#                 lam = (p - pA) / (pB - pA + 1e-12)
#                 lam = float(max(0.0, min(1.0, lam)))
#             else:
#                 lam = 0.1
#             z_pred = (1.0 - lam) * mu_cfg + lam * z0_hat

#         # ====== supervised losses (teacher path + optional uncond) ======
#         batch_cand = batch[cand_idx]
#         y_cand = y[cand_idx]

#         def pack_graph_logits(logits, batch_idx):
#             graph_logits = []
#             for g_ in torch.unique(batch_idx):
#                 m_ = (batch_idx == g_)
#                 graph_logits.append(logits[m_])
#             return {"graph_logits": graph_logits}

#         # teacher prediction uses z_pred (built from teacher mean + diffusion)
#         logits_t = self.pred_head(z_pred)
#         outputs_t = pack_graph_logits(logits_t, batch_cand)
#         pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)

#         # main VAE KL: keep as before (you used mu_prime; here align it with teacher mean you used)
#         # If teacher source is cfg, KL on mu_cfg makes more sense; otherwise mu_prime.
#         mu_for_kl = mu_cfg if (self.cfg_teacher_source.lower() == "cfg") else mu_prime
#         kl_loss = self._kl_normal_standard(mu_for_kl, logvar).mean()

#         w_d_target = float(getattr(self.args, "distill_weight", 1.0))
#         w_kl_target = float(getattr(self.args, "distill_kl_weight", 0.01))

#         total_loss = (
#             pred_loss
#             + self.kl_weight * kl_loss
#             + self.diff_weight * diff_loss
#         )

#         return {
#             "total_loss": total_loss,
#             "pred_loss": pred_loss,
#             "kl_loss": kl_loss,
#             "diff_loss": diff_loss,
#             "distill_loss": 0,
#             "distill_kl_loss": 0,
#         }

#     def inference(self, x, edge_index, batch, obs=None):
#         """inference: keep snapshot-only student path (unchanged)"""
#         self.eval()
#         device = x.device
#         with torch.no_grad():
#             if batch.device != device:
#                 batch = batch.to(device)

#             x_main_base = x[:, :8-2]
#             rho_last = x[:, -2:]
#             x_main = torch.cat([x_main_base, rho_last], dim=1)
#             h_all = self.aggregation(x_main, edge_index)
#             h_enc = self.enc(h_all)
#             mu = self.fc_mu(h_enc)

#             logits_all = self.pred_head(mu)
#             probs_all = F.softmax(logits_all, dim=1)[:, 1]

#             preds = []
#             for g in torch.unique(batch):
#                 m = (batch == g)
#                 preds.append(probs_all[m])
#             return preds

#     def initial_parameters(self):
#         """
#         Initialize parameters.
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

# alignment
class GLADModel(nn.Module):
    "GLAD: Guidance + Alignment + Filtering + Diffusion"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # Feature dimensions
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # Hyperparameter configuration
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))
        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0
        self.ctx_token_dim = 4
        self.distill_weight = getattr(self.args, "distill_weight", 1.0)   # λ_d
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.use_gate = getattr(self.args, "student_use_gate", True)
        self.student_use_gate = self.use_gate
        self.logvar_min = getattr(self.args, "logvar_min", -6.0)
        self.logvar_max = getattr(self.args, "logvar_max", 2.0)
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        # Distribution distillation weight
        self.distill_kl_weight = getattr(self.args, "distill_kl_weight", 0.1)

        self.ttt_steps = int(getattr(self.args, "ttt_steps", 30))       # Suggested range: 3-10 steps; previous setting: 5
        self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))        # Adaptation learning rate; previous setting: 0.05
        self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 50))      # Maximum diffusion step used for low-noise denoising

        # Model configuration
        ## Aggregate primary features with GAT
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        ## VAE1 Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        ## Encode propagation dynamics with GRU
        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        ## Cross-modal attention refinement
        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.delta_logvar_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        ## VAE2 Encoder
        layers = []
        in_dim = self.latent_dim
        for i in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)   # Predict delta_mu_s

        layers = []
        in_dim = self.latent_dim
        for i in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)  # Predict logvar_s

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        # Diffusion
        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        # Prediction probabilities
        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from the standard normal distribution."""
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        """Context features."""
        n = ctx_cand.size(0)
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]

        # Align dimensions
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]

        # tokens: [n, T1-1, 4]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (n, T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        """Training forward pass."""
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]  # struct(0..4), lpsi(5), rise(6), t_first(7)
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        # Handle the case with no infected nodes
        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "cross_loss": zero,
                "pred_loss": zero,
                "recon_loss": zero,
                "kl_loss": zero,
            }

        # Aggregate primary feature representations
        x_main_base = x[:, :8-2]          # [N,6]
        rho_last = x[:, -2:]            # [N,2]
        ## Concatenate features
        x_main = torch.cat([x_main_base, rho_last], dim=1)   # [N,8]
        h_all = self.aggregation(x_main, edge_index)    # [N, out_channels]

        # Hard clipping
        h_cand = h_all[cand_idx]  # [n', out_channels]

        # Latent representations
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        # Context features
        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
            raise ValueError(f"None ctx!!!")
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)  # [n', T1-1, 4]
            _, h_last = self.ctx_gru(tokens)  # h_last: [1, n', latent]
            c = h_last.squeeze(0)  # [n', latent]

        # Cross-attention refinement
        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        # delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
        mu_prime = mu + delta_mu
        # logvar_prime = logvar + delta_logvar

        # Distillation model
        ## Mean
        delta_s = self.student_delta(mu)   # [n', latent]
        if self.student_use_gate:
            g = self.student_gate(mu)   # [n', latent] in (0,1)
            delta_s = g * delta_s
        mu_student = mu + delta_s
        ## Variance
        logvar_s = self.student_logvar(mu)   # [n', latent]
        logvar_s = torch.clamp(logvar_s, self.logvar_min, self.logvar_max)

        # Compute MSE between student and teacher
        distill_loss = F.mse_loss(mu_student, mu_prime.detach())

        # Compute the student-teacher distribution discrepancy
        mu_t = mu_prime.detach()
        logvar_t = logvar.detach()
        def kl_teacher_student(mu_t, logvar_t, mu_s, logvar_s):
            # KL( N(mu_t, var_t) || N(mu_s, var_s) )
            var_t = torch.exp(logvar_t)
            var_s = torch.exp(logvar_s)
            kl = 0.5 * (logvar_s - logvar_t + (var_t + (mu_t - mu_s).pow(2)) / (var_s + 1e-8) - 1.0)
            return kl.sum(dim=1)

        distill_kl = kl_teacher_student(mu_t, logvar_t, mu_student, logvar_s).mean()

        # Sample in latent space
        eps = torch.randn_like(mu_prime)
        z0 = mu_prime + torch.exp(0.5 * logvar) * eps   # [n', latent]

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu_prime

        # Select the corresponding labels
        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}
        # ---- teacher path prediction loss ----
        logits_t = self.pred_head(z_pred)          # [n', 2]
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)

        w_d_target  = getattr(self.args, "distill_weight", 1.0)
        w_kl_target = getattr(self.args, "distill_kl_weight", 0.01)


        kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()
        total_loss = pred_loss+ self.kl_weight * kl_loss + self.diff_weight * diff_loss + w_d_target * distill_loss + w_kl_target *distill_kl

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss,
            "distill_loss": distill_loss,
            "distill_kl_loss":distill_kl
        }

    def inference(self, x, edge_index, batch, obs=None):
        "Inference phase"

        self.eval()
        device = x.device
        with torch.enable_grad():
            if batch.device != device:
                batch = batch.to(device)

            # -------- (1) Main branch on full graph --------
            x_main_base = x[:, :8-2]          # [N,6]
            rho_last = x[:, -2:]            # [N,2]
            x_main = torch.cat([x_main_base, rho_last], dim=1) # [N,8]
            with torch.no_grad():
                h_all = self.aggregation(x_main, edge_index)   # [N, out_channels]
                h_enc = self.enc(h_all)                        # [N, hidden]
                mu = self.fc_mu(h_enc)                         # [N, latent]

                # -------- (2) Student transfer (NO ctx) --------
                delta_s = self.student_delta(mu)   # [n', latent]
                if self.student_use_gate:
                    g = self.student_gate(mu)   # [n', latent] in (0,1)
                    delta_s = g * delta_s
                mu_student = mu + delta_s

            mu_opt = mu_student.clone().detach().requires_grad_(True)
            optimizer = optim.SGD([mu_opt], lr=self.ttt_lr, momentum=0.9)

            for step in range(self.ttt_steps):
                t = torch.randint(0, self.ttt_t_max, (mu_opt.size(0),), device=device).long()
                z_0_hat = self.diffusion(mu_opt, t)
                loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)
                # print(f"loss_ttt:{loss_ttt}")
                optimizer.zero_grad()
                loss_ttt.backward()
                optimizer.step()

            z_final = mu_opt.detach()

            # -------- (3) Prediction on full nodes --------
            logits_all = self.pred_head(z_final)        # [N, 2]
            probs_all = F.softmax(logits_all, dim=1)[:, 1] # [N]

            # -------- (4) Split per graph --------
            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        """
        Initialize parameters.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# class GLAD_wo_A_Model(nn.Module):
#     "GLAD: Guidance + Filtering + Latent + Diffusion"

#     def __init__(self, model_args):
#         super().__init__()

#         self.args = model_args
#         # feature dims
#         self.struct_dim = 5
#         self.lpsi_dim = 1
#         self.self_time_dim = 2
#         self.rho_last_feat_dim = 2
#         self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

#         # hyperparams
#         self.hidden_channels = getattr(self.args, "hidden_channels", 64)
#         self.out_channels = getattr(self.args, "out_channels", 64)
#         self.heads = getattr(self.args, "heads", 4)
#         self.dropout = getattr(self.args, "dropout", 0.2)
#         self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
#         self.obs_len = int(getattr(self.args, "obs_len", 0))

#         self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
#         self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
#         self.use_diffusion = self.diff_weight > 0

#         self.ctx_token_dim = 4

#         # distill weights (you can schedule outside if desired)
#         self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
#         self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

#         # student
#         self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
#         self.student_layers = getattr(self.args, "student_layers", 2)
#         self.student_use_gate = getattr(self.args, "student_use_gate", True)
#         self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
#         self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

#         # diffusion
#         self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
#         self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
#         self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
#         self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
#         self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

#         # models
#         self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

#         self.enc = nn.Sequential(
#             nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#         )
#         self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
#         self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

#         self.ctx_gru = nn.GRU(
#             input_size=self.ctx_token_dim,
#             hidden_size=self.latent_dim,
#             num_layers=1,
#             batch_first=True,
#         )

#         attn_heads = min(4, max(1, self.latent_dim // 16))
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.latent_dim,
#             num_heads=attn_heads,
#             dropout=self.dropout,
#             batch_first=True,
#         )
#         self.delta_ffn = nn.Sequential(
#             nn.LayerNorm(self.latent_dim),
#             nn.Linear(self.latent_dim, self.latent_dim * 2),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.latent_dim * 2, self.latent_dim),
#         )

#         # student heads
#         layers = []
#         in_dim = self.latent_dim
#         for _ in range(self.student_layers - 1):
#             layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#             in_dim = self.student_h
#         layers += [nn.Linear(in_dim, self.latent_dim)]
#         self.student_delta = nn.Sequential(*layers)

#         layers = []
#         in_dim = self.latent_dim
#         for _ in range(self.student_layers - 1):
#             layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#             in_dim = self.student_h
#         layers += [nn.Linear(in_dim, self.latent_dim)]
#         self.student_logvar = nn.Sequential(*layers)

#         if self.student_use_gate:
#             self.student_gate = nn.Sequential(
#                 nn.Linear(self.latent_dim, self.latent_dim),
#                 nn.Sigmoid()
#             )

#         self.diffusion = Diffusion(
#             self.max_time_steps,
#             self.beta_start,
#             self.beta_end,
#             self.time_embed_dim,
#             self.latent_dim,
#             self.predictor_dim,
#         )

#         pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
#         self.pred_head = nn.Sequential(
#             nn.Linear(self.latent_dim, pred_hidden),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(pred_hidden, 2),
#         )

#         self.initial_parameters()

#     @staticmethod
#     def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

#     def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
#         lam = ctx_cand[:, 0 : (T1 - 1)]
#         gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
#         rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
#         rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
#         rhoI_ = rhoI[:, 1:]
#         rhoS_ = rhoS[:, 1:]
#         tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
#         assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
#         return tokens

#     def forward(self, x, edge_index, y, batch, epoch):
#         device = x.device
#         y = y.view(-1)

#         rise_flag = x[:, 6]
#         cand_mask = (rise_flag == 0)
#         cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

#         if cand_idx.numel() == 0:
#             zero = torch.tensor(0.0, device=device)
#             return {
#                 "total_loss": zero,
#                 "pred_loss": zero,
#                 "kl_loss": zero,
#                 "diff_loss": zero,
#             }

#         x_main_base = x[:, :8-2]  # [N,6]
#         rho_last = x[:, -2:]      # [N,2]
#         x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

#         h_all = self.aggregation(x_main, edge_index)
#         h_cand = h_all[cand_idx]
#         h_enc = self.enc(h_cand)
#         mu = self.fc_mu(h_enc)
#         logvar = self.fc_logvar(h_enc)

#         ctx_flat = x[:, self.main_dim : -2]
#         if ctx_flat.numel() == 0 or self.obs_len <= 1:
#             c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
#             raise ValueError(f"None ctx!!!")
#         else:
#             T1 = self.obs_len
#             ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
#             if ctx_flat.size(1) != ctx_dim_expected:
#                 raise ValueError(
#                     f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
#                 )
#             ctx_cand = ctx_flat[cand_idx]
#             tokens = self._build_ctx_tokens(ctx_cand, T1)
#             _, h_last = self.ctx_gru(tokens)
#             c = h_last.squeeze(0)

#         q = mu.unsqueeze(1)
#         k = c.unsqueeze(1)
#         v = c.unsqueeze(1)
#         attn_out, _ = self.cross_attn(q, k, v)
#         delta_mu = self.delta_ffn(attn_out.squeeze(1))
#         mu_prime = mu + delta_mu

#         eps = torch.randn_like(mu_prime)
#         z0 = mu_prime + torch.exp(0.5 * logvar) * eps

#         # Diffusion
#         p = epoch / float(self.args.epochs)
#         pA, pB = 0.50, 0.80
#         lam = 0
#         if self.use_diffusion:
#             t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
#             z0_hat = self.diffusion(z0, t)
#             diff_loss = F.mse_loss(z0_hat, z0.detach())
#         else:
#             z0_hat = z0
#             diff_loss = torch.tensor(0.0, device=device)

#         if (not self.use_diffusion) or (p < pA):
#             # z_pred = mu_prime
#             z_pred = z0
#         else:
#             if p < pB and lam < 0.1:
#                 lam = (p - pA) / (pB - pA + 1e-12)
#                 lam = float(max(0.0, min(1.0, lam)))
#             else:
#                 lam = 0.1
#             z_pred = (1.0 - lam) * mu_prime + lam * z0_hat

#         batch_cand = batch[cand_idx]
#         y_cand = y[cand_idx]

#         def pack_graph_logits(logits, batch_idx):
#             graph_logits = []
#             for g in torch.unique(batch_idx):
#                 m = (batch_idx == g)
#                 graph_logits.append(logits[m])
#             return {"graph_logits": graph_logits}

#         logits_t = self.pred_head(z_pred)
#         outputs_t = pack_graph_logits(logits_t, batch_cand)
#         pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)

#         kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

#         total_loss = (
#             pred_loss
#             + self.kl_weight * kl_loss
#             + self.diff_weight * diff_loss
#         )

#         return {
#             "total_loss": total_loss,
#             "pred_loss": pred_loss,
#             "kl_loss": kl_loss,
#             "diff_loss": diff_loss
#         }

#     def inference(self, x, edge_index, batch, obs=None):
#         self.eval()
#         device = x.device
#         with torch.no_grad():
#             if batch.device != device:
#                 batch = batch.to(device)

#             x_main_base = x[:, :8-2]
#             rho_last = x[:, -2:]
#             x_main = torch.cat([x_main_base, rho_last], dim=1)

#             h_all = self.aggregation(x_main, edge_index)
#             h_enc = self.enc(h_all)
#             mu = self.fc_mu(h_enc)

#             logits_all = self.pred_head(mu)
#             probs_all = F.softmax(logits_all, dim=1)[:, 1]

#             preds = []
#             for g in torch.unique(batch):
#                 m = (batch == g)
#                 preds.append(probs_all[m])
#             return preds

#     def initial_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

# class GLAD_wo_A_Model(nn.Module):
#     "GLAD: Guidance + Filtering + Latent + Diffusion"

#     def __init__(self, model_args):
#         super().__init__()

#         self.args = model_args
#         # feature dims
#         self.struct_dim = 5
#         self.lpsi_dim = 1
#         self.self_time_dim = 2
#         self.rho_last_feat_dim = 2
#         self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

#         # hyperparams
#         self.hidden_channels = getattr(self.args, "hidden_channels", 64)
#         self.out_channels = getattr(self.args, "out_channels", 64)
#         self.heads = getattr(self.args, "heads", 4)
#         self.dropout = getattr(self.args, "dropout", 0.2)
#         self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
#         self.obs_len = int(getattr(self.args, "obs_len", 0))

#         self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
#         self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
#         self.use_diffusion = self.diff_weight > 0

#         self.ctx_token_dim = 4

#         # distill weights (you can schedule outside if desired)
#         self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
#         self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

#         # student
#         self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
#         self.student_layers = getattr(self.args, "student_layers", 2)
#         self.student_use_gate = getattr(self.args, "student_use_gate", True)
#         self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
#         self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

#         # diffusion
#         self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
#         self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
#         self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
#         self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
#         self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

#         self.ttt_steps = int(getattr(self.args, "ttt_steps", 100))
#         self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))
#         self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 100))

#         # models
#         self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

#         self.enc = nn.Sequential(
#             nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#         )
#         self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
#         self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

#         self.ctx_gru = nn.GRU(
#             input_size=self.ctx_token_dim,
#             hidden_size=self.latent_dim,
#             num_layers=1,
#             batch_first=True,
#         )

#         # Cross-attention block (standard Pre-LN)
#         attn_heads = 4

#         # LayerNorm for Query (before attention)
#         self.ln_q = nn.LayerNorm(self.latent_dim)

#         # LayerNorm for Key/Value (before attention)
#         self.ln_k = nn.LayerNorm(self.latent_dim)

#         # Multi-head cross-attention
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.latent_dim,
#             num_heads=attn_heads,
#             dropout=self.dropout,
#             batch_first=True,
#         )

#         # LayerNorm for FFN (before FFN)
#         self.ln_ffn = nn.LayerNorm(self.latent_dim)

#         # Feed-forward network (removed internal LayerNorm, expanded to 4x)
#         self.ffn = nn.Sequential(
#             nn.Linear(self.latent_dim, self.latent_dim * 2),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.latent_dim * 2, self.latent_dim)
#         )
#         # End of cross-attention block

#         # Replace the original self.delta_ffn and self.delta_logvar_ffn

#         # student heads
#         layers = []
#         in_dim = self.latent_dim
#         for _ in range(self.student_layers - 1):
#             layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#             in_dim = self.student_h
#         layers += [nn.Linear(in_dim, self.latent_dim)]
#         self.student_delta = nn.Sequential(*layers)

#         layers = []
#         in_dim = self.latent_dim
#         for _ in range(self.student_layers - 1):
#             layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
#             in_dim = self.student_h
#         layers += [nn.Linear(in_dim, self.latent_dim)]
#         self.student_logvar = nn.Sequential(*layers)

#         if self.student_use_gate:
#             self.student_gate = nn.Sequential(
#                 nn.Linear(self.latent_dim, self.latent_dim),
#                 nn.Sigmoid()
#             )

#         self.diffusion = Diffusion(
#             self.max_time_steps,
#             self.beta_start,
#             self.beta_end,
#             self.time_embed_dim,
#             self.latent_dim,
#             self.predictor_dim,
#         )

#         pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
#         self.pred_head = nn.Sequential(
#             nn.Linear(self.latent_dim, pred_hidden),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(pred_hidden, 2),
#         )

#         self.initial_parameters()

#     @staticmethod
#     def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

#     def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
#         lam = ctx_cand[:, 0 : (T1 - 1)]
#         gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
#         rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
#         rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
#         rhoI_ = rhoI[:, 1:]
#         rhoS_ = rhoS[:, 1:]
#         tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
#         assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
#         return tokens

#     # Additional cross-attention block method
#     def _cross_attention_block(self, mu, c):
#         """
#         Standard Pre-LN Cross-Attention Block

#         Args:
#             mu: [B, latent_dim] - static latent mean (Query)
#             c: [B, latent_dim] - temporal context (Key/Value)

#         Returns:
#             mu_refined: [B, latent_dim] - refined mean
#         """
#         # Add sequence dimension
#         q = mu.unsqueeze(0)  # [B, 1, latent_dim]
#         k = c.unsqueeze(0)   # [B, 1, latent_dim]
#         v = c.unsqueeze(0)   # [B, 1, latent_dim]

#         # ===== Sub-layer 1: Cross-Attention (Pre-LN) =====
#         # Pre-LayerNorm on Query and Key
#         q_norm = self.ln_q(q)
#         k_norm = self.ln_k(k)
#         v_norm = k_norm  # V reuses K's normalization

#         # Multi-head attention
#         attn_out, attn_weights = self.cross_attn(q_norm, k_norm, v_norm)

#         # Residual connection (on query side only)
#         q = q + attn_out

#         # ===== Sub-layer 2: Feed-Forward (Pre-LN) =====
#         # Pre-LayerNorm
#         q_norm = self.ln_ffn(q)

#         # Feed-forward network
#         ffn_out = self.ffn(q_norm)

#         # Residual connection
#         q = q + ffn_out

#         # Remove sequence dimension
#         mu_refined = q.squeeze(0)  # [B, latent_dim]

#         return mu_refined
#     # End of additional method

#     def forward(self, x, edge_index, y, batch, epoch):
#         device = x.device
#         y = y.view(-1)

#         rise_flag = x[:, 6]
#         cand_mask = (rise_flag <= 1)
#         cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

#         if cand_idx.numel() == 0:
#             zero = torch.tensor(0.0, device=device)
#             return {
#                 "total_loss": zero,
#                 "pred_loss": zero,
#                 "kl_loss": zero,
#                 "diff_loss": zero,
#             }

#         x_main_base = x[:, :8-2]
#         rho_last = x[:, -2:]
#         x_main = torch.cat([x_main_base, rho_last], dim=1)

#         h_all = self.aggregation(x_main, edge_index)
#         h_cand = h_all[cand_idx]
#         h_enc = self.enc(h_cand)
#         mu = self.fc_mu(h_enc)
#         logvar = self.fc_logvar(h_enc)

#         ctx_flat = x[:, self.main_dim : -2]
#         if ctx_flat.numel() == 0 or self.obs_len <= 1:
#             c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
#             raise ValueError(f"None ctx!!!")
#         else:
#             T1 = self.obs_len
#             ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
#             if ctx_flat.size(1) != ctx_dim_expected:
#                 raise ValueError(
#                     f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
#                 )
#             ctx_cand = ctx_flat[cand_idx]
#             tokens = self._build_ctx_tokens(ctx_cand, T1)
#             _, h_last = self.ctx_gru(tokens)
#             c = h_last.squeeze(0)

#         # Apply the new cross-attention block
#         # Original implementation:
#         # q = mu.unsqueeze(1)
#         # k = c.unsqueeze(1)
#         # v = c.unsqueeze(1)
#         # attn_out, _ = self.cross_attn(q, k, v)
#         # delta_mu = self.delta_ffn(attn_out.squeeze(1))
#         # mu_prime = mu + delta_mu

#         # Updated implementation: standard Pre-LN cross-attention block
#         mu_prime = self._cross_attention_block(mu, c)
#         # End of updated implementation

#         eps = torch.randn_like(mu_prime)
#         z0 = mu_prime + torch.exp(0.5 * logvar) * eps

#         # Diffusion
#         if self.use_diffusion:
#             t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
#             z0_hat = self.diffusion(z0, t)
#             diff_loss = F.mse_loss(z0_hat, z0.detach())
#         else:
#             z0_hat = z0
#             diff_loss = torch.tensor(0.0, device=device)

#         if self.args.is_sampling:
#             z_pred = z0
#         else:
#             z_pred = mu_prime

#         batch_cand = batch[cand_idx]
#         y_cand = y[cand_idx]

#         def pack_graph_logits(logits, batch_idx):
#             graph_logits = []
#             for g in torch.unique(batch_idx):
#                 m = (batch_idx == g)
#                 graph_logits.append(logits[m])
#             return {"graph_logits": graph_logits}

#         logits_t = self.pred_head(z_pred)
#         outputs_t = pack_graph_logits(logits_t, batch_cand)
#         pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
#         kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

#         total_loss = (
#             pred_loss
#             + self.kl_weight * kl_loss
#             + self.diff_weight * diff_loss
#         )

#         return {
#             "total_loss": total_loss,
#             "pred_loss": pred_loss,
#             "kl_loss": kl_loss,
#             "diff_loss": diff_loss
#         }

#     def inference(self, x, edge_index, batch, obs=None):
#         """
#         TTT Inference:
#         1. Obtain the initial static features mu_0.
#         2. Iteratively refine mu using diffusion loss as a self-supervised signal.
#         3. Make predictions using the optimized mu.
#         """
#         self.eval()
#         device = x.device

#         with torch.enable_grad():
#             if batch.device != device:
#                 batch = batch.to(device)

#             x_main_base = x[:, :8-2]
#             rho_last = x[:, -2:]
#             x_main = torch.cat([x_main_base, rho_last], dim=1)

#             with torch.no_grad():
#                 h_all = self.aggregation(x_main, edge_index)
#                 h_enc = self.enc(h_all)
#                 mu_init = self.fc_mu(h_enc)

#             mu_opt = mu_init.clone().detach().requires_grad_(True)
#             optimizer = optim.SGD([mu_opt], lr=self.ttt_lr, momentum=0.9)

#             for step in range(self.ttt_steps):
#                 t = torch.randint(0, self.ttt_t_max, (mu_opt.size(0),), device=device).long()
#                 z_0_hat = self.diffusion(mu_opt, t)
#                 loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)
#                 optimizer.zero_grad()
#                 loss_ttt.backward()
#                 optimizer.step()

#             z_final = mu_opt.detach()
#             logits_all = self.pred_head(z_final)
#             probs_all = F.softmax(logits_all, dim=1)[:, 1]

#             preds = []
#             for g in torch.unique(batch):
#                 m = (batch == g)
#                 preds.append(probs_all[m])
#             return preds

#     def initial_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

# # logvar refined
class TGLR_Model(nn.Module):
    "TGLR: Guidance + Filtering + Latent + Diffusion"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # feature dims
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # hyperparams
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # distill weights (you can schedule outside if desired)
        self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
        self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

        # student
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.student_use_gate = getattr(self.args, "student_use_gate", True)
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # diffusion
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.ttt_steps = int(getattr(self.args, "ttt_steps", 30))       # Suggested range: 3-10 steps; previous setting: 5
        self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))        # Adaptation learning rate; previous setting: 0.05
        self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 100))      # Maximum diffusion step used for low-noise denoising

        # models
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.delta_logvar_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        # student heads
        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)

        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)

        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)
        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero,
                "kl_loss": zero,
                "diff_loss": zero,
            }

        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)
        h_cand = h_all[cand_idx]
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)
        self._last_logvar = logvar.detach()
        if not hasattr(self, '_logvar_buf'):
            self._logvar_buf = []
        self._logvar_buf.append(logvar.detach().cpu())

        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
            raise ValueError(f"None ctx!!!")
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0)

        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        # delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
        mu_prime = mu + delta_mu
        # logvar_prime = logvar + delta_logvar

        eps = torch.randn_like(mu_prime)
        z0 = mu_prime + torch.exp(0.5 * logvar) * eps

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu_prime

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
        kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

        total_loss = (
            pred_loss
            + self.kl_weight * kl_loss
            + self.diff_weight * diff_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss
        }

    def log_epoch_variance(self, model_name: str, epoch: int, log_interval: int = 1):
        if not hasattr(self, '_logvar_buf') or len(self._logvar_buf) == 0:
            return
        if epoch % log_interval == 0:
            all_logvar = torch.cat(self._logvar_buf, dim=0)
            with torch.no_grad():
                sigma2 = all_logvar.exp()
                mean_per_dim  = sigma2.mean(dim=0)
                global_mean   = mean_per_dim.mean().item()
                global_median = mean_per_dim.median().item()
                pct_above_1   = (mean_per_dim > 1.0).float().mean().item() * 100
            print(
                f"[VAR | {model_name} | epoch {epoch:03d}] "
                f"E[σ²] mean={global_mean:.4f}  "
                f"median={global_median:.4f}  "
                f"%dims>1={pct_above_1:.1f}%  "
                f"N_cand={all_logvar.size(0)}"
            )
        self._logvar_buf = []
    # def inference(self, x, edge_index, batch, obs=None):
    #     self.eval()
    #     device = x.device
    #     with torch.no_grad():
    #         if batch.device != device:
    #             batch = batch.to(device)

    #         x_main_base = x[:, :8-2]
    #         rho_last = x[:, -2:]
    #         x_main = torch.cat([x_main_base, rho_last], dim=1)

    #         h_all = self.aggregation(x_main, edge_index)
    #         h_enc = self.enc(h_all)
    #         mu = self.fc_mu(h_enc)

    #         logits_all = self.pred_head(mu)
    #         probs_all = F.softmax(logits_all, dim=1)[:, 1]

    #         preds = []
    #         for g in torch.unique(batch):
    #             m = (batch == g)
    #             preds.append(probs_all[m])
    #         return preds

    def inference(self, x, edge_index, batch, obs=None):
        """
        TTT Inference:
        1. Obtain the initial static features mu_0.
        2. Iteratively refine mu using diffusion loss as a self-supervised signal.
        3. Make predictions using the optimized mu.
        """
        self.eval()
        device = x.device

        with torch.enable_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            with torch.no_grad():
                h_all = self.aggregation(x_main, edge_index)
                h_enc = self.enc(h_all)
                mu_init = self.fc_mu(h_enc) # [N, latent]

            mu_opt = mu_init.clone().detach().requires_grad_(True)
            optimizer = optim.SGD([mu_opt], lr=self.ttt_lr, momentum=0.9)

            for step in range(self.ttt_steps):
                # # 1. Run the full denoising chain to obtain z_0_hat
                # with torch.no_grad():
                #     z = mu_opt.detach().clone()
                #     for t_val in reversed(range(self.ttt_t_max)):
                #         t = torch.full((mu_opt.size(0),), t_val, device=device).long()
                #         z = self.diffusion(z, t)
                # z_0_hat = z
                # # 2. Obtain z_0_hat with single-step denoising
                t = torch.randint(0, self.ttt_t_max, (mu_opt.size(0),), device=device).long()
                z_0_hat = self.diffusion(mu_opt, t)
                loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)
                # print(f"loss_ttt:{loss_ttt}")
                optimizer.zero_grad()
                loss_ttt.backward()
                optimizer.step()

            z_final = mu_opt.detach()
            logits_all = self.pred_head(z_final)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TGLR_wo_G_Model(nn.Module):
    "TGLR: Filtering + Latent + Diffusion"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # feature dims
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # hyperparams
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # distill weights (you can schedule outside if desired)
        self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
        self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

        # student
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.student_use_gate = getattr(self.args, "student_use_gate", True)
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # diffusion
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.ttt_steps = int(getattr(self.args, "ttt_steps", 30))       # Suggested range: 3-10 steps; previous setting: 5
        self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))        # Adaptation learning rate; previous setting: 0.05
        self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 100))      # Maximum diffusion step used for low-noise denoising

        # models
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )

        # student heads
        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)

        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero,
                "kl_loss": zero,
                "diff_loss": zero,
            }

        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)
        h_cand = h_all[cand_idx]
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        eps = torch.randn_like(mu)
        z0 = mu + torch.exp(0.5 * logvar) * eps

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
        kl_loss = self._kl_normal_standard(mu, logvar).mean()

        total_loss = (
            pred_loss
            + self.kl_weight * kl_loss
            + self.diff_weight * diff_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss
        }

    def inference(self, x, edge_index, batch, obs=None):
        """
        TTT Inference:
        1. Obtain the initial static features mu_0.
        2. Iteratively refine mu using diffusion loss as a self-supervised signal.
        3. Make predictions using the optimized mu.
        """
        self.eval()
        device = x.device

        with torch.enable_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            with torch.no_grad():
                h_all = self.aggregation(x_main, edge_index)
                h_enc = self.enc(h_all)
                mu_init = self.fc_mu(h_enc) # [N, latent]

            mu_opt = mu_init.clone().detach().requires_grad_(True)
            optimizer = optim.SGD([mu_opt], lr=self.ttt_lr, momentum=0.9)

            for step in range(self.ttt_steps):
                # # 1. Run the full denoising chain to obtain z_0_hat
                # with torch.no_grad():
                #     z = mu_opt.detach().clone()
                #     for t_val in reversed(range(self.ttt_t_max)):
                #         t = torch.full((mu_opt.size(0),), t_val, device=device).long()
                #         z = self.diffusion(z, t)
                # z_0_hat = z
                # # 2. Obtain z_0_hat with single-step denoising
                t = torch.randint(0, self.ttt_t_max, (mu_opt.size(0),), device=device).long()
                z_0_hat = self.diffusion(mu_opt, t)
                loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)
                # print(f"loss_ttt:{loss_ttt}")
                optimizer.zero_grad()
                loss_ttt.backward()
                optimizer.step()

            z_final = mu_opt.detach()
            logits_all = self.pred_head(z_final)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TGLR_wo_LA_Model(nn.Module):
    "TGLR: Guidance + Filtering + Diffusion"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # feature dims
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # hyperparams
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # distill weights (you can schedule outside if desired)
        self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
        self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

        # student
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.student_use_gate = getattr(self.args, "student_use_gate", True)
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # diffusion
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.ttt_steps = int(getattr(self.args, "ttt_steps", 30))       # Suggested range: 3-10 steps; previous setting: 5
        self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))        # Adaptation learning rate; previous setting: 0.05
        self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 100))      # Maximum diffusion step used for low-noise denoising

        # models
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )

        # student heads
        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)

        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero,
                "kl_loss": zero,
                "diff_loss": zero,
            }

        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)
        h_cand = h_all[cand_idx]
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        ctx_flat = x[:, self.main_dim : -2]

        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
            raise ValueError(f"None ctx!!!")
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0)

        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        mu_prime = mu + delta_mu

        z0 = mu_prime

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu_prime

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)

        kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

        total_loss = (
            pred_loss
            + 0 * kl_loss
            + self.diff_weight * diff_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss
        }

    def inference(self, x, edge_index, batch, obs=None):
        """
        TTT Inference:
        1. Obtain the initial static features mu_0.
        2. Iteratively refine mu using diffusion loss as a self-supervised signal.
        3. Make predictions using the optimized mu.
        """
        self.eval()
        device = x.device

        with torch.enable_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            with torch.no_grad():
                h_all = self.aggregation(x_main, edge_index)
                h_enc = self.enc(h_all)
                mu_init = self.fc_mu(h_enc) # [N, latent]

            mu_opt = mu_init.clone().detach().requires_grad_(True)
            optimizer = optim.SGD([mu_opt], lr=self.ttt_lr, momentum=0.9)

            for step in range(self.ttt_steps):
                # # 1. Run the full denoising chain to obtain z_0_hat
                # with torch.no_grad():
                #     z = mu_opt.detach().clone()
                #     for t_val in reversed(range(self.ttt_t_max)):
                #         t = torch.full((mu_opt.size(0),), t_val, device=device).long()
                #         z = self.diffusion(z, t)
                # z_0_hat = z

                t = torch.randint(0, self.ttt_t_max, (mu_opt.size(0),), device=device).long()
                z_0_hat = self.diffusion(mu_opt, t)
                loss_ttt = F.mse_loss(z_0_hat.detach(), mu_opt)
                # print(f"loss_ttt:{loss_ttt}")
                optimizer.zero_grad()
                loss_ttt.backward()
                optimizer.step()

            z_final = mu_opt.detach()
            logits_all = self.pred_head(z_final)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TGLR_wo_T_Model(nn.Module):
    "TGLR: Guidance + Latent + Diffusion"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # feature dims
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # hyperparams
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # distill weights (you can schedule outside if desired)
        self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
        self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

        # student
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.student_use_gate = getattr(self.args, "student_use_gate", True)
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # diffusion
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.ttt_steps = int(getattr(self.args, "ttt_steps", 30))       # Suggested range: 3-10 steps; previous setting: 5
        self.ttt_lr = float(getattr(self.args, "ttt_lr", 0.01))        # Adaptation learning rate; previous setting: 0.05
        self.ttt_t_max = int(getattr(self.args, "ttt_t_max", 100))      # Maximum diffusion step used for low-noise denoising

        # models
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.delta_logvar_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        # student heads
        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)

        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)

        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)
        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero,
                "kl_loss": zero,
                "diff_loss": zero,
            }

        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)
        h_cand = h_all[cand_idx]
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
            raise ValueError(f"None ctx!!!")
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0)

        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        # delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
        mu_prime = mu + delta_mu
        # logvar_prime = logvar + delta_logvar

        eps = torch.randn_like(mu_prime)
        z0 = mu_prime + torch.exp(0.5 * logvar) * eps

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu_prime

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
        kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

        total_loss = (
            pred_loss
            + self.kl_weight * kl_loss
            + self.diff_weight * diff_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss
        }

    def inference(self, x, edge_index, batch, obs=None):
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            h_all = self.aggregation(x_main, edge_index)
            h_enc = self.enc(h_all)
            mu = self.fc_mu(h_enc)

            logits_all = self.pred_head(mu)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TGLR_wo_D_Model(nn.Module):
    "TGLR: Guidance + Filtering + Latent"

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        # feature dims
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        # hyperparams
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0

        self.ctx_token_dim = 4

        # distill weights (you can schedule outside if desired)
        self.distill_weight = float(getattr(self.args, "distill_weight", 1.0))           # w_d
        self.distill_kl_weight = float(getattr(self.args, "distill_kl_weight", 0.0))    # w_kl

        # student
        self.student_h = getattr(self.args, "student_hidden_dim", self.latent_dim)
        self.student_layers = getattr(self.args, "student_layers", 2)
        self.student_use_gate = getattr(self.args, "student_use_gate", True)
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # diffusion
        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        # models
        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fc_mu = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)
        self.fc_logvar = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.delta_logvar_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # student heads
        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_delta = nn.Sequential(*layers)

        layers = []
        in_dim = self.latent_dim
        for _ in range(self.student_layers - 1):
            layers += [nn.Linear(in_dim, self.student_h), nn.SiLU()]
            in_dim = self.student_h
        layers += [nn.Linear(in_dim, self.latent_dim)]
        self.student_logvar = nn.Sequential(*layers)

        if self.student_use_gate:
            self.student_gate = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero,
                "kl_loss": zero,
            }

        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)
        h_cand = h_all[cand_idx]
        h_enc = self.enc(h_cand)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)
        self._last_logvar = logvar.detach()
        if not hasattr(self, '_logvar_buf'):
            self._logvar_buf = []
        self._logvar_buf.append(logvar.detach().cpu())

        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() == 0 or self.obs_len <= 1:
            c = torch.zeros((cand_idx.numel(), self.latent_dim), device=device)
            raise ValueError(f"None ctx!!!")
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_flat.size(1) != ctx_dim_expected:
                raise ValueError(
                    f"ctx_flat dim mismatch: got {ctx_flat.size(1)}, expected {ctx_dim_expected} for obs_len={T1}"
                )
            ctx_cand = ctx_flat[cand_idx]
            tokens = self._build_ctx_tokens(ctx_cand, T1)
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0)

        q = mu.unsqueeze(1)
        k = c.unsqueeze(1)
        v = c.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        delta_mu = self.delta_ffn(attn_out.squeeze(1))
        # delta_logvar = self.delta_logvar_ffn(attn_out.squeeze(1))
        mu_prime = mu + delta_mu
        # logvar_prime = logvar + delta_logvar

        eps = torch.randn_like(mu_prime)
        z0 = mu_prime + torch.exp(0.5 * logvar) * eps

        z_pred = z0

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
        kl_loss = self._kl_normal_standard(mu_prime, logvar).mean()

        total_loss = (
            pred_loss
            + self.kl_weight * kl_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
        }

    def log_epoch_variance(self, model_name: str, epoch: int, log_interval: int = 1):
        if not hasattr(self, '_logvar_buf') or len(self._logvar_buf) == 0:
            return
        if epoch % log_interval == 0:
            all_logvar = torch.cat(self._logvar_buf, dim=0)
            with torch.no_grad():
                sigma2 = all_logvar.exp()
                mean_per_dim  = sigma2.mean(dim=0)
                global_mean   = mean_per_dim.mean().item()
                global_median = mean_per_dim.median().item()
                pct_above_1   = (mean_per_dim > 1.0).float().mean().item() * 100
            print(
                f"[VAR | {model_name} | epoch {epoch:03d}] "
                f"E[σ²] mean={global_mean:.4f}  "
                f"median={global_median:.4f}  "
                f"%dims>1={pct_above_1:.1f}%  "
                f"N_cand={all_logvar.size(0)}"
            )
        self._logvar_buf = []

    def inference(self, x, edge_index, batch, obs=None):
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)

            h_all = self.aggregation(x_main, edge_index)
            h_enc = self.enc(h_all)
            mu = self.fc_mu(h_enc)

            logits_all = self.pred_head(mu)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class TGLR_w_C_Model(nn.Module):
    """
    TGLR Version 2: Concat Strategy
    """

    def __init__(self, model_args):
        super().__init__()
        self.args = model_args

        # === 1. Dimensions ===
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim
        self.ctx_token_dim = 4

        # === 2. Hyperparams ===
        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        # VAE & Diffusion Params
        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        # === 3. Modules ===

        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)
        self.dynamic_embed_dim = self.hidden_channels
        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.dynamic_embed_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fusion_input_dim = self.out_channels + self.dynamic_embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.latent_dim * 2),
            nn.LayerNorm(self.latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        return tokens

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {"total_loss": zero, "pred_loss": zero, "kl_loss": zero, "diff_loss": zero}

        x_main_base = x[:, :8-2]
        rho_last = x[:, -2:]
        x_main = torch.cat([x_main_base, rho_last], dim=1)

        h_all = self.aggregation(x_main, edge_index)
        h_static = h_all[cand_idx] # [N_cand, out_channels]

        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() != 0 and self.obs_len > 1:
             ctx_cand = ctx_flat[cand_idx]
             tokens = self._build_ctx_tokens(ctx_cand, self.obs_len)
             _, h_gru_last = self.ctx_gru(tokens)
             h_dynamic = h_gru_last.squeeze(0) # [N_cand, dynamic_embed_dim]
        else:
             h_dynamic = torch.zeros(h_static.size(0), self.dynamic_embed_dim, device=device)

        h_fused = torch.cat([h_static, h_dynamic], dim=-1) # [N_cand, fusion_input_dim]

        h_enc = self.fusion_mlp(h_fused)

        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        eps = torch.randn_like(mu)
        z0 = mu + torch.exp(0.5 * logvar) * eps

        # Diffusion
        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0.size(0),), device=device).long()
            z0_hat = self.diffusion(z0, t)
            diff_loss = F.mse_loss(z0_hat, z0.detach())
        else:
            z0_hat = z0
            diff_loss = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred = z0
        else:
            z_pred = mu

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits_t = self.pred_head(z_pred)
        outputs_t = pack_graph_logits(logits_t, batch_cand)
        pred_loss = compute_loss(outputs_t, y_cand, batch_cand, self.args)
        kl_loss = self._kl_normal_standard(mu, logvar).mean()

        total_loss = pred_loss + self.kl_weight * kl_loss + self.diff_weight * diff_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "kl_loss": kl_loss,
            "diff_loss": diff_loss
        }

    def inference(self, x, edge_index, batch, obs=None):
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)
            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)
            h_all = self.aggregation(x_main, edge_index)

            h_static = h_all
            N_nodes = h_static.size(0)
            h_dynamic_zeros = torch.zeros(N_nodes, self.dynamic_embed_dim, device=device)
            h_fused = torch.cat([h_static, h_dynamic_zeros], dim=-1)
            h_enc = self.fusion_mlp(h_fused)
            mu = self.fc_mu(h_enc)
            z_pred = mu

            logits_all = self.pred_head(z_pred)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class TGLR_w_CA_Model(nn.Module):
    """
    TGLR Concat Strategy with Dual-Branch Alignment (CFG-style)
    """

    def __init__(self, model_args):
        super().__init__()
        self.args = model_args

        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim
        self.ctx_token_dim = 4

        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)
        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.pred_loss_uncond_weight = float(getattr(self.args, "pred_loss_uncond_weight", 1.0))
        self.kl_uncond_weight = float(getattr(self.args, "kl_uncond_weight", 0.5))
        self.diff_uncond_weight = float(getattr(self.args, "diff_uncond_weight", 0.5))

        self.kl_weight = float(getattr(self.args, "kl_weight", 0.01))
        self.diff_weight = float(getattr(self.args, "diff_weight", 0.1))
        self.use_diffusion = self.diff_weight > 0
        self.logvar_min = float(getattr(self.args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(self.args, "logvar_max", 2.0))

        self.max_time_steps = int(getattr(self.args, "max_time_steps", 50))
        self.pA = float(getattr(self.args, "diff_pA", 0.50))
        self.pB = float(getattr(self.args, "diff_pB", 0.80))

        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.dynamic_embed_dim = self.hidden_channels
        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.dynamic_embed_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fusion_input_dim = self.out_channels + self.dynamic_embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.latent_dim * 2),
            nn.LayerNorm(self.latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        self.beta_start = float(getattr(self.args, "beta_start", 1e-4))
        self.beta_end = float(getattr(self.args, "beta_end", 2e-2))
        self.time_embed_dim = int(getattr(self.args, "time_embed_dim", 128))
        self.predictor_dim = int(getattr(self.args, "prime_predictor_hidden_dim", max(128, self.latent_dim)))

        self.diffusion = Diffusion(
            self.max_time_steps,
            self.beta_start,
            self.beta_end,
            self.time_embed_dim,
            self.latent_dim,
            self.predictor_dim,
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    @staticmethod
    def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1)

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        return tokens

    def _pack_graph_logits(self, logits: torch.Tensor, batch_idx: torch.Tensor):
        graph_logits = []
        for g in torch.unique(batch_idx):
            m = (batch_idx == g)
            graph_logits.append(logits[m])
        return {"graph_logits": graph_logits}

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)

        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)

        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {"total_loss": zero, "pred_loss_c": zero, "pred_loss_u": zero}

        x_main_base = x[:, :8-2]
        rho_last = x[:, -2:]
        x_main = torch.cat([x_main_base, rho_last], dim=1)
        h_all = self.aggregation(x_main, edge_index)
        h_static = h_all[cand_idx]

        ctx_flat = x[:, self.main_dim : -2]
        if ctx_flat.numel() != 0 and self.obs_len > 1:
             ctx_cand = ctx_flat[cand_idx]
             tokens = self._build_ctx_tokens(ctx_cand, self.obs_len)
             _, h_gru_last = self.ctx_gru(tokens)
             h_dynamic_full = h_gru_last.squeeze(0)
        else:
             h_dynamic_full = torch.zeros(h_static.size(0), self.dynamic_embed_dim, device=device)

        h_fused_c = torch.cat([h_static, h_dynamic_full], dim=-1)

        h_dynamic_zeros = torch.zeros_like(h_dynamic_full)
        h_fused_u = torch.cat([h_static, h_dynamic_zeros], dim=-1)

        h_enc_c = self.fusion_mlp(h_fused_c)
        mu_c = self.fc_mu(h_enc_c)
        logvar_c = self.fc_logvar(h_enc_c)
        logvar_c = torch.clamp(logvar_c, self.logvar_min, self.logvar_max)

        h_enc_u = self.fusion_mlp(h_fused_u)
        mu_u = self.fc_mu(h_enc_u)
        logvar_u = self.fc_logvar(h_enc_u)
        logvar_u = torch.clamp(logvar_u, self.logvar_min, self.logvar_max)

        eps_c = torch.randn_like(mu_c)
        z0_c = mu_c + torch.exp(0.5 * logvar_c) * eps_c

        eps_u = torch.randn_like(mu_u)
        z0_u = mu_u + torch.exp(0.5 * logvar_u) * eps_u

        if self.use_diffusion:
            t = torch.randint(0, self.max_time_steps, (z0_c.size(0),), device=device).long()
            z0_hat_c = self.diffusion(z0_c, t)
            diff_loss_c = F.mse_loss(z0_hat_c, z0_c.detach())
            z0_hat_u = self.diffusion(z0_u, t)
            diff_loss_u = F.mse_loss(z0_hat_u, z0_u.detach())
        else:
            z0_hat_c = z0_c
            z0_hat_u = z0_u
            diff_loss_c = torch.tensor(0.0, device=device)
            diff_loss_u = torch.tensor(0.0, device=device)

        if self.args.is_sampling:
            z_pred_c = z0_c
            z_pred_u = z0_u
        else:
            z_pred_c = mu_c
            z_pred_u = mu_u

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]

        logits_c = self.pred_head(z_pred_c)
        outputs_c = self._pack_graph_logits(logits_c, batch_cand)
        pred_loss_c = compute_loss(outputs_c, y_cand, batch_cand, self.args)

        logits_u = self.pred_head(z_pred_u)
        outputs_u = self._pack_graph_logits(logits_u, batch_cand)
        pred_loss_u = compute_loss(outputs_u, y_cand, batch_cand, self.args)

        kl_loss_c = self._kl_normal_standard(mu_c, logvar_c).mean()
        kl_loss_u = self._kl_normal_standard(mu_u, logvar_u).mean()

        w_u = float(self.pred_loss_uncond_weight)
        w_klu = float(self.kl_uncond_weight)
        w_diffu = float(self.diff_uncond_weight)

        total_loss = (
            pred_loss_c
            + w_u * pred_loss_u
            + self.kl_weight * (kl_loss_c + w_klu * kl_loss_u)
            + self.diff_weight * (diff_loss_c + w_diffu * diff_loss_u)
        )

        return {
            "total_loss": total_loss,
            "pred_loss_c": pred_loss_c,
            "pred_loss_u": pred_loss_u,
            "kl_loss_c": kl_loss_c,
            "kl_loss_u": kl_loss_u,
            "diff_loss_c": diff_loss_c,
            "diff_loss_u": diff_loss_u
        }

    def inference(self, x, edge_index, batch, obs=None):
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)

            x_main_base = x[:, :8-2]
            rho_last = x[:, -2:]
            x_main = torch.cat([x_main_base, rho_last], dim=1)
            h_all = self.aggregation(x_main, edge_index)
            h_static = h_all

            N_nodes = h_static.size(0)
            h_dynamic_zeros = torch.zeros(N_nodes, self.dynamic_embed_dim, device=device)

            h_fused = torch.cat([h_static, h_dynamic_zeros], dim=-1)
            h_enc = self.fusion_mlp(h_fused)
            mu = self.fc_mu(h_enc)

            logits_all = self.pred_head(mu)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class TGLR_w_Dy_Model(nn.Module):
    """
    Static GAT + Dynamic Context (GRU/Attn)
    """

    def __init__(self, model_args):
        super().__init__()

        self.args = model_args
        self.struct_dim = 5
        self.lpsi_dim = 1
        self.self_time_dim = 2
        self.rho_last_feat_dim = 2
        self.main_dim = self.struct_dim + self.lpsi_dim + self.rho_last_feat_dim

        self.hidden_channels = getattr(self.args, "hidden_channels", 64)
        self.out_channels = getattr(self.args, "out_channels", 64)
        self.heads = getattr(self.args, "heads", 4)
        self.dropout = getattr(self.args, "dropout", 0.2)

        self.latent_dim = int(getattr(self.args, "hidden_dim", self.out_channels))
        self.obs_len = int(getattr(self.args, "obs_len", 0))

        self.ctx_token_dim = 4

        self.aggregation = GATModel(self.main_dim, self.hidden_channels, self.out_channels, self.heads, self.dropout)

        self.enc = nn.Sequential(
            nn.Linear(self.out_channels, max(self.out_channels, self.latent_dim)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.fc_feat = nn.Linear(max(self.out_channels, self.latent_dim), self.latent_dim)

        self.ctx_gru = nn.GRU(
            input_size=self.ctx_token_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        attn_heads = min(4, max(1, self.latent_dim // 16))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )

        pred_hidden = int(getattr(self.args, "prime_predictor_hidden_dim", 128))
        self.pred_head = nn.Sequential(
            nn.Linear(self.latent_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(pred_hidden, 2),
        )

        self.initial_parameters()

    def _build_ctx_tokens(self, ctx_cand: torch.Tensor, T1: int) -> torch.Tensor:
        lam = ctx_cand[:, 0 : (T1 - 1)]
        gam = ctx_cand[:, (T1 - 1) : 2 * (T1 - 1)]
        rhoI = ctx_cand[:, 2 * (T1 - 1) : 2 * (T1 - 1) + T1]
        rhoS = ctx_cand[:, 2 * (T1 - 1) + T1 : 2 * (T1 - 1) + 2 * T1]
        rhoI_ = rhoI[:, 1:]
        rhoS_ = rhoS[:, 1:]
        tokens = torch.stack([lam, gam, rhoI_, rhoS_], dim=-1)
        assert tokens.shape == (ctx_cand.size(0), T1 - 1, 4)
        return tokens

    def _extract_features(self, x, edge_index, indices=None):
        x_main_base = x[:, :8-2]  # [N,6]
        rho_last = x[:, -2:]      # [N,2]
        x_main = torch.cat([x_main_base, rho_last], dim=1)  # [N,8]

        h_all = self.aggregation(x_main, edge_index)

        if indices is not None:
            h_target = h_all[indices]
        else:
            h_target = h_all

        h_enc = self.enc(h_target)
        feat_static = self.fc_feat(h_enc) # [N_target, dim]
        ctx_flat = x[:, self.main_dim : -2]

        if indices is not None:
            ctx_target = ctx_flat[indices]
        else:
            ctx_target = ctx_flat

        if ctx_target.numel() == 0 or self.obs_len <= 1:
            delta_feat = torch.zeros_like(feat_static)
        else:
            T1 = self.obs_len
            ctx_dim_expected = 2 * (T1 - 1) + 2 * T1
            if ctx_target.size(1) != ctx_dim_expected:
                 raise ValueError(f"ctx dim mismatch: {ctx_target.size(1)} vs {ctx_dim_expected}")

            tokens = self._build_ctx_tokens(ctx_target, T1)
            _, h_last = self.ctx_gru(tokens)
            c = h_last.squeeze(0) # [N_target, dim]
            q = feat_static.unsqueeze(1)
            k = c.unsqueeze(1)
            v = c.unsqueeze(1)

            attn_out, _ = self.cross_attn(q, k, v)
            delta_feat = self.delta_ffn(attn_out.squeeze(1)) # [N_target, dim]

        feat_final = feat_static + delta_feat

        return feat_final

    def forward(self, x, edge_index, y, batch, epoch):
        device = x.device
        y = y.view(-1)
        rise_flag = x[:, 6]
        if self.args.pruning:
            cand_mask = (rise_flag == 0)
        else:
            cand_mask = (rise_flag <= 1)
        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)
        if cand_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "pred_loss": zero
            }

        z_fused = self._extract_features(x, edge_index, indices=cand_idx)

        batch_cand = batch[cand_idx]
        y_cand = y[cand_idx]
        def pack_graph_logits(logits, batch_idx):
            graph_logits = []
            for g in torch.unique(batch_idx):
                m = (batch_idx == g)
                graph_logits.append(logits[m])
            return {"graph_logits": graph_logits}

        logits = self.pred_head(z_fused)
        outputs = pack_graph_logits(logits, batch_cand)
        pred_loss = compute_loss(outputs, y_cand, batch_cand, self.args)
        total_loss = pred_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
        }

    def inference(self, x, edge_index, batch, obs=None):
        self.eval()
        device = x.device
        with torch.no_grad():
            if batch.device != device:
                batch = batch.to(device)

            z_fused = self._extract_features(x, edge_index, indices=None)

            logits_all = self.pred_head(z_fused)
            probs_all = F.softmax(logits_all, dim=1)[:, 1]

            preds = []
            for g in torch.unique(batch):
                m = (batch == g)
                preds.append(probs_all[m])
            return preds

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
