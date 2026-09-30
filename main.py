import argparse
from pathlib import Path
import torch
import numpy as np
import os
import networkx as nx
import random
import create_graphs
from args import Args
from data import GraphGroupDataset
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from model import GLADModel, TGLR_wo_LA_Model, TGLR_Model, TGLR_wo_T_Model, TGLR_wo_D_Model, TGLR_wo_G_Model, GLCFGDModel, TGLR_w_Dy_Model, TGLR_w_C_Model, TGLR_w_CA_Model
from train import train_GLAD, train_TGLR_wo_LA, train_TGLR, train_TGLR_wo_T, train_TGLR_wo_D, train_TGLR_wo_G, train_GLCFGD, train_TGLR_w_Dy, train_TGLR_w_C, train_TGLR_w_CA
from eval import evaluate_model_with_metrics_with_multi_source
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 127
GRAPH_SEED = 123
# GRAPH_SEED = 456
# GRAPH_SEED = 567
# GRAPH_SEED = 789


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def custom_collate_fn(batch):
    """
    Collate PyTorch Geometric Data objects into a batch.
    """
    # Combine Data objects with PyTorch Geometric Batch.from_data_list
    Batch_data = Batch.from_data_list(batch)
    return Batch_data


def custom_collate_fn1(batch):
    """
    Collate PyTorch Geometric Data objects into a batch.
    """
    data_list = [item[0] for item in batch]  # Extract Data objects
    node_mapping_list = [item[1] for item in batch]  # Extract node mappings
    Batch_data = Batch.from_data_list(data_list)
    return Batch_data, node_mapping_list


def dataloader_build(dataset, args_, shuffle=True):
    """
    Configure the data loader.
    :param dataset:
    :param args_:
    :return:
    """
    # A replacement sampler and shuffle cannot be enabled together
    dataloader_ = DataLoader(dataset, batch_size=args_.batch_size, shuffle=shuffle,
                             num_workers=args_.num_workers, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
    return dataloader_

def test_best_model(args, model, device, dataset_test):
    print("=== Start Testing Best Model (by Val F1) ===")
    best_model_path = getattr(args, "load_model_path", None) or os.path.join(
        args.save_path,
        f"{args.model_name}_{args.graph_type}_best_by_f1.pt"
    )
    # Evaluate a checkpoint saved during training
    # best_model_path =  args.load_model_path

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    print(f"[Test] Loading best model from: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loader = dataloader_build(dataset_test, args, shuffle=False)

    with torch.no_grad():
        metrics = evaluate_model_with_metrics_with_multi_source(
            model, test_loader, device
        )

    print("\n=== Test Results (Best-by-Val-F1) ===")
    print(
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"F1:        {metrics['f1']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}"
    )

    return metrics

# Temporary evaluation helper
def sample_one_batch(dataset, batch_size):
    """
    Create a data loader from a random subset of samples.
    :param dataset:
    :param batch_size:
    :return:
    """
    indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, indices)
    dataloader_temp_ = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return dataloader_temp_

def summarize_graphs(graphs):
    stats = []

    for G in graphs:
        N = G.number_of_nodes()
        E = G.number_of_edges()

        avg_degree = 2 * E / N
        density = nx.density(G)
        clustering = nx.average_clustering(G)

        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            diameter = None

        stats.append({
            "nodes": N,
            "edges": E,
            "avg_degree": avg_degree,
            "density": density,
            "clustering": clustering,
            "diameter": diameter
        })

    def describe(key):
        values = [s[key] for s in stats if s[key] is not None]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75))
        }

    summary = {
        "total_graphs": len(graphs),
        "total_nodes": sum(s["nodes"] for s in stats),
        "total_edges": sum(s["edges"] for s in stats),
        "nodes_stats": describe("nodes"),
        "edges_stats": describe("edges"),
        "degree_stats": describe("avg_degree"),
        "density_stats": describe("density"),
        "clustering_stats": describe("clustering"),
    }

    return summary

MODEL_RUNNERS = {
    "TGLR": (TGLR_Model, train_TGLR),
    "GLAD": (GLADModel, train_GLAD),
    "GLCFGD": (GLCFGDModel, train_GLCFGD),
    "TGLR_wo_G": (TGLR_wo_G_Model, train_TGLR_wo_G),
    "TGLR_wo_LA": (TGLR_wo_LA_Model, train_TGLR_wo_LA),
    "TGLR_wo_T": (TGLR_wo_T_Model, train_TGLR_wo_T),
    "TGLR_wo_D": (TGLR_wo_D_Model, train_TGLR_wo_D),
    "TGLR_w_Dy": (TGLR_w_Dy_Model, train_TGLR_w_Dy),
    "TGLR_w_C": (TGLR_w_C_Model, train_TGLR_w_C),
    "TGLR_w_CA": (TGLR_w_CA_Model, train_TGLR_w_CA),
}


def parse_args(argv=None):
    """Override experiment settings without editing the source files."""
    args = Args()
    parser = argparse.ArgumentParser(description="Train or evaluate TGLR.")
    parser.add_argument("--mode", choices=["train", "test", "stats"], default="train")
    parser.add_argument("--dataset", default=args.graph_type)
    parser.add_argument("--model", choices=sorted(MODEL_RUNNERS), default=args.model_name)
    parser.add_argument("--data-dir", default=args.root_dir)
    parser.add_argument("--save-dir", default="./runs/checkpoints")
    parser.add_argument("--output-dir", default=args.output_save_path)
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate in test mode.")
    parser.add_argument("--resume", help="Checkpoint from which to resume training.")
    parser.add_argument("--epochs", type=int, default=args.epochs)
    parser.add_argument("--batch-size", type=int, default=args.batch_size)
    parser.add_argument("--num-workers", type=int, default=args.num_workers)
    parser.add_argument("--gpu", type=int, default=args.gpu_id, help="GPU index; -1 selects CPU.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--graph-seed", type=int, default=GRAPH_SEED)
    parser.add_argument("--obs-len", type=int, default=args.obs_len)
    parser.add_argument("--obs-start", type=int, default=args.ob_start_idx)
    parser.add_argument("--diff-weight", type=float, default=args.diff_weight)
    parser.add_argument("--kl-weight", type=float, default=args.kl_weight)
    parser.add_argument("--ttt-steps", type=int, default=args.ttt_steps)
    parser.add_argument("--sampling", action=argparse.BooleanOptionalAction, default=args.is_sampling)
    parser.add_argument("--visual", action=argparse.BooleanOptionalAction, default=args.visual)
    parser.add_argument("--limit-graphs", type=int, help="Use a subset for smoke tests, not paper evaluation.")
    opts = parser.parse_args(argv)
    if opts.epochs < 1 or opts.batch_size < 1 or opts.obs_len < 2:
        parser.error("epochs and batch-size must be positive; obs-len must be at least 2")
    if opts.num_workers < 0 or opts.obs_start < 0 or opts.ttt_steps < 0:
        parser.error("num-workers, obs-start, and ttt-steps must be nonnegative")
    if opts.limit_graphs is not None and opts.limit_graphs < 10:
        parser.error("limit-graphs must be at least 10 for the 80/10/10 split")
    if opts.checkpoint and opts.mode != "test":
        parser.error("--checkpoint is for --mode test; use --resume to resume training")
    if opts.resume and opts.mode != "train":
        parser.error("--resume requires --mode train")
    args.graph_type, args.model_name = opts.dataset, opts.model
    args.root_dir = str(Path(opts.data_dir))
    args.data_path = str(Path(args.root_dir) / args.graph_type)
    args.graphs_saved_path = str(Path(args.root_dir) / "saved_graphs" / f"{args.graph_type}_graph.pkl")
    args.save_path = opts.save_dir
    args.output_save_path = opts.output_dir
    args.visual_save_path = str(Path(opts.output_dir) / "visual")
    args.load_model = opts.mode == "test"
    args.load_model_path = opts.checkpoint
    args.resume_train = bool(opts.resume)
    args.checkpoint_path = opts.resume
    args.epochs, args.batch_size, args.num_workers = opts.epochs, opts.batch_size, opts.num_workers
    args.gpu_id = opts.gpu
    args.obs_len, args.ob_start_idx = opts.obs_len, opts.obs_start
    args.node_features_dim = 8 + (4 * args.obs_len - 2) + 2
    args.in_channels = args.node_features_dim
    args.diff_weight, args.kl_weight = opts.diff_weight, opts.kl_weight
    args.ttt_steps, args.is_sampling, args.visual = opts.ttt_steps, opts.sampling, opts.visual
    return args, opts


def main(argv=None):
    args, opts = parse_args(argv)
    global SEED
    SEED = opts.seed
    set_seed(SEED)
    use_cuda = torch.cuda.is_available() and 0 <= args.gpu_id < torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    print(f"Device: {device}")
    graphs = create_graphs.data(args, load_exited_graphs=True)
    random.Random(opts.graph_seed).shuffle(graphs)
    if opts.limit_graphs is not None:
        graphs = graphs[:opts.limit_graphs]
        print("Using a reduced dataset for a smoke test; metrics are not paper results.")
    if opts.mode == "stats":
        for key, value in summarize_graphs(graphs).items():
            print(f"{key}: {value}")
        return
    graphs_len = len(graphs)
    if graphs_len < 10:
        raise ValueError("At least 10 graphs are needed for the 80/10/10 split.")
    graphs_train = graphs[:int(0.8 * graphs_len)]
    graphs_validate = graphs[int(0.8 * graphs_len):int(0.9 * graphs_len)]
    graphs_test = graphs[int(0.9 * graphs_len):]
    print(f"Graphs: {graphs_len}; train: {len(graphs_train)}; "
          f"validation: {len(graphs_validate)}; test: {len(graphs_test)}")
    dataset_test = GraphGroupDataset(graphs_test, args)
    model_class, trainer = MODEL_RUNNERS[args.model_name]
    model = model_class(args).to(device)
    if args.load_model:
        return test_best_model(args, model, device, dataset_test)
    for directory in (args.save_path, args.output_save_path, args.visual_save_path):
        Path(directory).mkdir(parents=True, exist_ok=True)
    dataset_train = GraphGroupDataset(graphs_train, args)
    dataset_validate = GraphGroupDataset(graphs_validate, args)
    dataloader = dataloader_build(dataset_train, args)
    validation_loader = dataloader_build(dataset_validate, args, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    trainer(args, dataloader, validation_loader, model, optimizer, scheduler, device)
    return test_best_model(args, model, device, dataset_test)


if __name__ == "__main__":
    main()
