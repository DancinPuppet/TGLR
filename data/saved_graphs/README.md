# Graph datasets

Place each preprocessed dataset in this directory as `<dataset>_graph.pkl`.

## Included in the repository

- `karate_SI_graph.pkl`, `karate_SIR_graph.pkl`, `karate_IC_graph.pkl`
- `jazz_SI_graph.pkl`, `jazz_SIR_graph.pkl`, `jazz_IC_graph.pkl`

## Download the remaining datasets

The eight larger graph caches are published in
**[TGLR Datasets on OSF](https://osf.io/v4mp5/overview?view_only=5792705d28d54a19885b470123dd4598)**.

**[Download graphs.zip](https://osf.io/download/6ac0c1f683fe948c5b496b14/?view_only=5792705d28d54a19885b470123dd4598)** (324,599,348 bytes, approximately 310 MiB).

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

The OSF view-only link allows the archive to be downloaded without signing in
and hides project contributor names. Preserve the `view_only` parameter when
sharing the link. The same archive is also available as a GitHub Release attachment.

SHA-256 of `graphs.zip`:

```text
96d3a3c7fd73df60b2bf54028b591af5fc2eb48601875ae75445136ff69f1668
```

These caches and the archive are intentionally excluded from Git. Their download
is provided through OSF and the GitHub Release attachment.

## Twitter25 identifier update (October 1, 2026)

The public Twitter25 cache uses `n0`, `n1`, ... as node identifiers independently
within each cascade, and `cascade_000000`, `cascade_000001`, ... as cascade
identifiers. Node indices are local to a cascade and must not be interpreted as
shared user identities across cascades. Original identifier mappings are not
released.

All 981 cascades retain their original graph-list, node, neighbor, and edge
ordering; numerical attributes, source labels, and snapshot arrays are unchanged.
Snapshot column references and source-node references use the new identifiers.
Ordered data equivalence was checked before and after serialization, including
the loader's score-based node ordering. No training or model inference was rerun
for this identifier-only update. The other seven files in the archive are
byte-for-byte unchanged.

The download URL remains the same. If you downloaded the archive before this
update, download it again and verify the SHA-256 above.

## Cache format

Each pickle contains a list of NetworkX graphs. The loader assigns a graph ID
before splitting the dataset. Graphs store propagation snapshots and their node
order in graph attributes; nodes store structural features, observed states,
and binary source labels. See `create_graphs.py`, `node_feature.py`, and `data.py`
for the preprocessing and tensor conversion code. Only load pickle files from
trusted sources.

The IC caches are supplied as additional examples; the manuscript's simulated
propagation experiments use SI and SIR.
