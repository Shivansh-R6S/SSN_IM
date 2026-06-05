import pandas as pd
import numpy as np
from collections import defaultdict, deque
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Step 5 - SAP Propagation")
print()

# Load data
print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

seeds_df  = pd.read_csv(os.path.join(SEEDS_DIR, "selected_seeds.csv"))
seed_nodes = set(seeds_df["node"].tolist())

print(f"  Edges      : {len(df):,}")
print(f"  Seed nodes : {len(seed_nodes)}")
print(f"  Seeds      : {sorted(seed_nodes)}")
print(f"  Alpha (decay parameter) : {DECAY_ALPHA}")

# Build adjacency list
print("\nBuilding adjacency list...")
adj = defaultdict(list)
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    adj[u].append((v, s))

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
print(f"  Nodes: {len(all_nodes):,}")

def run_sap(seed_nodes, adj, alpha, score_threshold):
    
    scores         = defaultdict(float)
    arrival_counts = defaultdict(int)

    # Queue entries: (node, arriving_sign, weight)
    queue = deque()

    # Initialise seeds with positive influence, weight 1.0
    for seed in seed_nodes:
        scores[seed] += 1.0
        arrival_counts[seed] += 1
        # Propagate outward from seed
        for (nbr, edge_sign) in adj[seed]:
            queue.append((nbr, edge_sign, 1.0))

    visited_edges = set()

    while queue:
        node, arriving_sign, parent_weight = queue.popleft()

        # Compute attenuated weight for this arrival
        count  = arrival_counts[node]
        weight = parent_weight * (alpha ** count)

        # Stop propagating if weight is negligibly small
        if abs(weight) < 1e-6:
            continue

        # Update score
        scores[node]         += arriving_sign * weight
        arrival_counts[node] += 1

        # Propagate further
        for (nbr, edge_sign) in adj[node]:
            edge_key = (node, nbr, arriving_sign)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            new_sign   = arriving_sign * edge_sign
            new_weight = weight * alpha
            if abs(new_weight) >= 1e-6:
                queue.append((nbr, new_sign, new_weight))

    return dict(scores)

# Run SAP
print("\nRunning SAP propagation...")
print(f"  Decay alpha : {DECAY_ALPHA}")

scores = run_sap(seed_nodes, adj, DECAY_ALPHA, SCORE_THRESH)
print(f"  Nodes reached : {len(scores):,}")

# Build results
print("\nBuilding results...")
results = []
for node in all_nodes:
    score      = scores.get(node, 0.0)
    is_seed    = node in seed_nodes
    activated  = score > SCORE_THRESH
    results.append({
        "node"      : node,
        "sap_score" : round(score, 6),
        "is_seed"   : is_seed,
        "activated" : activated
    })

results_df = pd.DataFrame(results)

# Metrics
activated_df  = results_df[results_df["activated"]]
pos_activated = results_df[results_df["sap_score"] > SCORE_THRESH]
neg_influenced = results_df[results_df["sap_score"] < -SCORE_THRESH]

print("\nSAP Results:")
print(f"  Total nodes              : {len(results_df):,}")
print(f"  Nodes reached (score!=0) : {(results_df['sap_score'] != 0).sum():,}")
print(f"  Positively activated     : {len(pos_activated):,}  ({len(pos_activated)/len(results_df)*100:.2f}%)")
print(f"  Negatively influenced    : {len(neg_influenced):,}  ({len(neg_influenced)/len(results_df)*100:.2f}%)")
print(f"  Neutral (score=0)        : {(results_df['sap_score'] == 0).sum():,}")
print(f"\n  Metric 1 - Influence Spread      : {len(pos_activated):,} nodes")
print(f"  Metric 2 - Net Influence Score   : {pos_activated['sap_score'].sum():.4f}")
print(f"\n  Score distribution:")
print(f"    Mean  : {results_df['sap_score'].mean():.6f}")
print(f"    Max   : {results_df['sap_score'].max():.6f}")
print(f"    Min   : {results_df['sap_score'].min():.6f}")
print(f"    Std   : {results_df['sap_score'].std():.6f}")

print("\nTop 15 positively activated nodes:")
print(results_df[~results_df["is_seed"]].nlargest(15, "sap_score")[
    ["node", "sap_score", "activated"]].to_string(index=False))

print("\nTop 10 negatively influenced nodes:")
print(results_df.nsmallest(10, "sap_score")[
    ["node", "sap_score", "activated"]].to_string(index=False))

# Save
print("\nSaving outputs...")
results_df.to_csv(os.path.join(EVAL_DIR, "sap_results.csv"), index=False)
print(f"  sap_results.csv: {len(results_df):,} nodes saved")
