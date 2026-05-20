import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Step 3 - Node Scoring")
print()

# Load data
print("Loading data...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

pos_df = df[df["sign"] ==  1]
neg_df = df[df["sign"] == -1]

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)
print(f"  Nodes: {n:,}  |  Edges: {len(df):,}")

# Load community assignments
print("Loading community assignments...")
assignments = pd.read_csv(os.path.join(COMMUNITY_DIR, "community_assignments.csv"))
community_map   = dict(zip(assignments["node"], assignments["final_community"]))
boundary_map    = dict(zip(assignments["node"], assignments["is_boundary_node"]))
confidence_map  = dict(zip(assignments["node"], assignments["confidence_score"]))
print(f"  Community assignments loaded for {len(assignments):,} nodes")

# Build signed graph
print("Building signed graph...")
G_signed = nx.DiGraph()
for _, row in df.iterrows():
    G_signed.add_edge(int(row["from_node"]), int(row["to_node"]),
                      sign=int(row["sign"]))

# Compute PageRank on full graph for influence weighting
print("Computing PageRank...")
pagerank = nx.pagerank(G_signed, alpha=0.85, max_iter=200)
print(f"  PageRank computed for {len(pagerank):,} nodes")
print(f"  Top 5 PageRank nodes: {sorted(pagerank, key=pagerank.get, reverse=True)[:5]}")

# Build signed adjacency for fast lookup
print("Building adjacency lookup...")
pos_neighbors = {}
neg_neighbors = {}
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    if s == 1:
        pos_neighbors.setdefault(u, set()).add(v)
    else:
        neg_neighbors.setdefault(u, set()).add(v)

# Compute scores for all nodes
print("Computing node scores...")

def compute_scores(node):
    pos_nbrs = pos_neighbors.get(node, set())
    neg_nbrs = neg_neighbors.get(node, set())

    # Opportunity: sum of PageRank of positive neighbours
    opportunity = sum(pagerank.get(v, 0) for v in pos_nbrs)

    # Danger: sum of PageRank of negative neighbours
    danger = sum(pagerank.get(v, 0) for v in neg_nbrs)

    # Enemy-of-enemy bonus: foes who are also foes with each other
    # Path: node -foe-> A -foe-> B means B is weakly aligned with node
    eoe_bonus = 0
    for foe in neg_nbrs:
        foe_foes = neg_neighbors.get(foe, set())
        # Each foe-of-foe path contributes their PageRank as a bonus
        eoe_bonus += sum(pagerank.get(ff, 0) for ff in foe_foes
                         if ff != node and ff not in neg_nbrs)

    # Seed score: risk-return ratio with eoe bonus
    # Small epsilon avoids division by zero
    epsilon    = 1e-9
    seed_score = opportunity + 0.1 * eoe_bonus - 0.5 * danger

    return {
        "node"           : node,
        "community"      : community_map.get(node, -1),
        "is_boundary"    : boundary_map.get(node, False),
        "confidence"     : round(confidence_map.get(node, 0), 4),
        "pagerank"       : round(pagerank.get(node, 0), 8),
        "pos_out_degree" : len(pos_nbrs),
        "neg_out_degree" : len(neg_nbrs),
        "opportunity"    : round(opportunity, 6),
        "danger"         : round(danger, 6),
        "eoe_bonus"      : round(eoe_bonus, 6),
        "seed_score"     : round(seed_score, 6)
    }

results = []
for i, node in enumerate(all_nodes):
    results.append(compute_scores(node))
    if (i + 1) % 10000 == 0:
        print(f"  Scored {i+1:,} / {n:,} nodes...")

scores_df = pd.DataFrame(results)
print(f"  Scoring complete for {len(scores_df):,} nodes")

# Summary stats
print("\nScore distribution summary:")
print(f"  Opportunity  mean: {scores_df['opportunity'].mean():.6f}  max: {scores_df['opportunity'].max():.6f}")
print(f"  Danger       mean: {scores_df['danger'].mean():.6f}  max: {scores_df['danger'].max():.6f}")
print(f"  EoE Bonus    mean: {scores_df['eoe_bonus'].mean():.6f}  max: {scores_df['eoe_bonus'].max():.6f}")
print(f"  Seed Score   mean: {scores_df['seed_score'].mean():.6f}  max: {scores_df['seed_score'].max():.6f}")

print("\nTop 10 nodes by seed score (excluding boundary nodes):")
top_seeds = scores_df[~scores_df["is_boundary"]].nlargest(10, "seed_score")
print(top_seeds[["node", "community", "seed_score", "opportunity",
                  "danger", "eoe_bonus", "pagerank"]].to_string(index=False))

print("\nTop 10 nodes by PageRank (most influential overall):")
top_pr = scores_df.nlargest(10, "pagerank")
print(top_pr[["node", "community", "pagerank", "seed_score",
               "is_boundary"]].to_string(index=False))

print(f"\nBoundary nodes: {scores_df['is_boundary'].sum():,}  ({scores_df['is_boundary'].mean()*100:.1f}%)")

# Save
print("\nSaving outputs...")
scores_df.to_csv(os.path.join(SCORING_DIR, "node_scores.csv"), index=False)
print(f"  node_scores.csv: {len(scores_df):,} nodes saved")
