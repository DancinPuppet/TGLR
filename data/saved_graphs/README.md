# Graph datasets

Place each preprocessed dataset in this directory as `<dataset>_graph.pkl`.

Included in this repository:

- `karate_SI_graph.pkl`, `karate_SIR_graph.pkl`, `karate_IC_graph.pkl`
- `jazz_SI_graph.pkl`, `jazz_SIR_graph.pkl`, `jazz_IC_graph.pkl`

Distributed separately; download links will be added here:

- `cora_ml_SI_graph.pkl`, `cora_ml_SIR_graph.pkl`
- `facebook_SI_graph.pkl`, `facebook_SIR_graph.pkl`
- `twitter15_graph.pkl`, `twitter16_graph.pkl`
- `twitter25_graph.pkl`, `weibo_graph.pkl`

The external download links are not available yet. These files are intentionally
excluded from Git; adding this directory to a commit will not upload them.

Each pickle contains a list of NetworkX graphs. The loader assigns a graph ID
before splitting the dataset. Graphs store propagation snapshots and their node
order in graph attributes; nodes store structural features, observed states,
and binary source labels. See `create_graphs.py`, `node_feature.py`, and `data.py`
for the preprocessing and tensor conversion code. Only load pickle files from
trusted sources.

The IC caches are supplied as additional examples; the manuscript's simulated
propagation experiments use SI and SIR.
