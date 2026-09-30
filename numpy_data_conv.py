import pickle

def main():
    datasets = [
        'cora_ml_SI', 'facebook_SIR', 'cora_ml_SIR', 'facebook_SI'
    ]

    tglr_root = './data/saved_graphs/'
    sidsl_root = './model/SIDSL/datasets/'

    import os
    os.makedirs(sidsl_root, exist_ok=True)

    for name in datasets:
        src = tglr_root + name + '_graph.pkl'
        dst = sidsl_root + name + '_graph.pkl'

        if not os.path.exists(src):
            print(f"Skipping missing file: {src}")
            continue

        with open(src, 'rb') as f:
            graphs = pickle.load(f)

        with open(dst, 'wb') as f:
            pickle.dump(graphs, f, protocol=2)

        print(f"Converted: {name}, {len(graphs)} graphs")

if __name__ == "__main__":
    main()
