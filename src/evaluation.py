import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Step 6 - Evaluation Against Baselines")
print()

# Load data
print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

scores_df  = pd.read_csv(os.path.join(SCORING_DIR, "node_scores.csv"))
seeds_df   = pd.read_csv(os.path.join(SEEDS_DIR, "selected_seeds.csv"))
our_seeds  = list(seeds_df["node"].astype(int))

all_nodes  = sorted(set(df["from_node"]) | set(df["to_node"]))
n          = len(all_nodes)

print(f"  Nodes: {n:,}  |  Edges: {len(df):,}")
print(f"  Our seeds ({len(our_seeds)}): {our_seeds}")

# Build adjacency
print("Building adjacency...")
adj_signed = defaultdict(list)
adj_all    = defaultdict(list)

for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    adj_signed[u].append((v, s))
    adj_all[u].append(v)

# Propagation models
def run_sap(seed_nodes, adj, alpha):
    scores         = defaultdict(float)
    arrival_counts = defaultdict(int)
    queue          = deque()

    for seed in seed_nodes:
        scores[seed] += 1.0
        arrival_counts[seed] += 1
        for (nbr, edge_sign) in adj[seed]:
            queue.append((nbr, edge_sign, 1.0))

    visited_edges = set()
    while queue:
        node, arriving_sign, parent_weight = queue.popleft()
        count  = arrival_counts[node]
        weight = parent_weight * (alpha ** count)
        if abs(weight) < 1e-6:
            continue
        scores[node]         += arriving_sign * weight
        arrival_counts[node] += 1
        for (nbr, edge_sign) in adj[node]:
            edge_key = (node, nbr, arriving_sign)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            new_weight = weight * alpha
            if abs(new_weight) >= 1e-6:
                queue.append((nbr, arriving_sign * edge_sign, new_weight))

    return dict(scores)

def run_ic(seed_nodes, adj, n_simulations=SIM_RUNS):
    """Standard Independent Cascade — unsigned, all edges positive."""
    total_activated = defaultdict(int)

    for _ in range(n_simulations):
        activated = set(seed_nodes)
        frontier  = list(seed_nodes)

        while frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in activated:
                        # Propagation probability = 1 / in_degree of v
                        in_deg = max(len([x for x in all_nodes
                                         if v in adj[x]]), 1)
                        prob = 1.0 / in_deg
                        if random.random() < prob:
                            activated.add(v)
                            next_frontier.append(v)
            frontier = next_frontier

        for node in activated:
            total_activated[node] += 1

    # Score = fraction of simulations in which node was activated
    scores = {node: total_activated[node] / n_simulations
              for node in total_activated}
    return scores

def compute_metrics(scores, threshold=SCORE_THRESH):
    spread    = sum(1 for s in scores.values() if s > threshold)
    net_score = sum(s for s in scores.values() if s > threshold)
    return spread, round(net_score, 4)

# Precompute in-degrees for IC
print("Precomputing in-degrees...")
in_degree = defaultdict(int)
for u in adj_all:
    for v in adj_all[u]:
        in_degree[v] += 1

def run_ic_fast(seed_nodes, adj, n_simulations=SIM_RUNS):
    """Faster IC using precomputed in-degrees."""
    total_activated = defaultdict(int)
    for _ in range(n_simulations):
        activated = set(seed_nodes)
        frontier  = list(seed_nodes)
        while frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in activated:
                        prob = 1.0 / max(in_degree[v], 1)
                        if random.random() < prob:
                            activated.add(v)
                            next_frontier.append(v)
            frontier = next_frontier
        for node in activated:
            total_activated[node] += 1
    return {node: cnt / n_simulations for node, cnt in total_activated.items()}

# Greedy seed selection
def greedy_seeds(adj, all_nodes, k, propagation_fn):
    selected = []
    for i in range(k):
        best_node  = None
        best_score = -1
        candidates = [n for n in all_nodes if n not in selected]
        # Sample 500 candidates for tractability
        sample = random.sample(candidates, min(500, len(candidates)))
        for node in sample:
            trial_seeds  = selected + [node]
            trial_scores = propagation_fn(trial_seeds, adj)
            spread, _    = compute_metrics(trial_scores)
            if spread > best_score:
                best_score = spread
                best_node  = node
        selected.append(best_node)
        print(f"    Greedy step {i+1}/{k}: selected node {best_node} (spread={best_score})")
    return selected

results = []

# Baseline 1: Random + IC
print("\nBaseline 1: Random Selection + Standard IC...")
random_seeds = random.sample(all_nodes, len(our_seeds))
scores_b1    = run_ic_fast(random_seeds, adj_all)
s1, n1       = compute_metrics(scores_b1)
results.append({"method": "Random + IC", "influence_spread": s1, "net_influence_score": n1})
print(f"  Seeds: {random_seeds}")
print(f"  Spread: {s1}  Net Score: {n1}")

# Baseline 2: Degree Centrality + IC
print("\nBaseline 2: Degree Centrality + Standard IC...")
degree_map    = {node: len(adj_all[node]) for node in all_nodes}
degree_seeds  = sorted(degree_map, key=degree_map.get, reverse=True)[:len(our_seeds)]
scores_b2     = run_ic_fast(degree_seeds, adj_all)
s2, n2        = compute_metrics(scores_b2)
results.append({"method": "Degree Centrality + IC", "influence_spread": s2, "net_influence_score": n2})
print(f"  Seeds: {degree_seeds}")
print(f"  Spread: {s2}  Net Score: {n2}")

# Baseline 3: Greedy + IC
print("\nBaseline 3: Greedy + Standard IC...")
greedy_ic_seeds = greedy_seeds(adj_all, all_nodes, len(our_seeds),
                                lambda s, a: run_ic_fast(s, a, n_simulations=10))
scores_b3       = run_ic_fast(greedy_ic_seeds, adj_all)
s3, n3          = compute_metrics(scores_b3)
results.append({"method": "Greedy + IC", "influence_spread": s3, "net_influence_score": n3})
print(f"  Spread: {s3}  Net Score: {n3}")

# Baseline 4: Greedy + SAP
print("\nBaseline 4: Greedy + SAP...")
greedy_sap_seeds = greedy_seeds(adj_signed, all_nodes, len(our_seeds),
                                 lambda s, a: run_sap(s, a, DECAY_ALPHA))
scores_b4        = run_sap(greedy_sap_seeds, adj_signed, DECAY_ALPHA)
s4, n4           = compute_metrics(scores_b4)
results.append({"method": "Greedy + SAP", "influence_spread": s4, "net_influence_score": n4})
print(f"  Spread: {s4}  Net Score: {n4}")

# Baseline 5: Our Selection + IC
print("\nBaseline 5: Our Selection + Standard IC...")
scores_b5 = run_ic_fast(our_seeds, adj_all)
s5, n5    = compute_metrics(scores_b5)
results.append({"method": "Our Selection + IC", "influence_spread": s5, "net_influence_score": n5})
print(f"  Spread: {s5}  Net Score: {n5}")

# Our Approach: Our Selection + SAP
print("\nOur Approach: Our Selection + SAP...")
scores_ours = run_sap(our_seeds, adj_signed, DECAY_ALPHA)
s_our, n_our = compute_metrics(scores_ours)
results.append({"method": "Our Selection + SAP", "influence_spread": s_our, "net_influence_score": n_our})
print(f"  Spread: {s_our}  Net Score: {n_our}")

# Results table
print("\nFinal Comparison:")
results_df = pd.DataFrame(results).sort_values("influence_spread", ascending=False)
results_df["spread_rank"]    = results_df["influence_spread"].rank(ascending=False).astype(int)
results_df["netscore_rank"]  = results_df["net_influence_score"].rank(ascending=False).astype(int)
print(results_df.to_string(index=False))

# Improvement over baselines
our_row = results_df[results_df["method"] == "Our Selection + SAP"].iloc[0]
print("\nImprovement of Our Approach over baselines:")
for _, row in results_df[results_df["method"] != "Our Selection + SAP"].iterrows():
    spread_imp   = (s_our - row["influence_spread"]) / max(row["influence_spread"], 1) * 100
    netscore_imp = (n_our - row["net_influence_score"]) / max(row["net_influence_score"], 1) * 100
    print(f"  vs {row['method']:<30} Spread: {spread_imp:+.1f}%  NetScore: {netscore_imp:+.1f}%")

# Save
print("\nSaving outputs...")
results_df.to_csv(os.path.join(EVAL_DIR, "metrics_comparison.csv"), index=False)
print(f"  metrics_comparison.csv saved")
