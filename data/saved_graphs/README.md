# Graph datasets

Place each preprocessed dataset in this directory as `<dataset>_graph.pkl`.

## Included in the repository

- `karate_SI_graph.pkl`, `karate_SIR_graph.pkl`, `karate_IC_graph.pkl`
- `jazz_SI_graph.pkl`, `jazz_SIR_graph.pkl`, `jazz_IC_graph.pkl`

## Download the remaining datasets

The eight larger graph caches are published in
**[TGLR Datasets v1.0](https://github.com/DancinPuppet/TGLR/releases/tag/datasets-v1.0)**.

**[Download graphs.zip](https://github.com/DancinPuppet/TGLR/releases/download/datasets-v1.0/graphs.zip)** (330,415,497 bytes, approximately 315 MiB).

The archive contains these files directly at its root:

- `cora_ml_SI_graph.pkl`, `cora_ml_SIR_graph.pkl`
- `facebook_SI_graph.pkl`, `facebook_SIR_graph.pkl`
- `twitter15_graph.pkl`, `twitter16_graph.pkl`
- `twitter25_graph.pkl`, `weibo_graph.pkl`

Extract all eight `.pkl` files into `data/saved_graphs/` without changing their
names or creating an extra `graphs/` subdirectory. If `graphs.zip` is saved in the
repository root, run from that directory:

```bash
python -m zipfile -e graphs.zip data/saved_graphs/
```

For example, the resulting Twitter25 path must be
`data/saved_graphs/twitter25_graph.pkl`. The extracted files occupy approximately
2.4 GB. The loader reads the extracted `.pkl` files, not the ZIP archive.

On the release page, select **graphs.zip** under **Assets**. GitHub's automatically
generated **Source code (zip)** and **Source code (tar.gz)** downloads contain the
repository files, not these larger datasets.

SHA-256 of `graphs.zip`:

```text
2426d2aba9d022bd35cd8f08e16129b6fb9a6b7ebf70225f1f82c44ad28f3649
```

These caches and the archive are intentionally excluded from Git. Their download
is provided through the Release attachment.

## Cache format

Each pickle contains a list of NetworkX graphs. The loader assigns a graph ID
before splitting the dataset. Graphs store propagation snapshots and their node
order in graph attributes; nodes store structural features, observed states,
and binary source labels. See `create_graphs.py`, `node_feature.py`, and `data.py`
for the preprocessing and tensor conversion code. Only load pickle files from
trusted sources.

The IC caches are supplied as additional examples; the manuscript's simulated
propagation experiments use SI and SIR.
