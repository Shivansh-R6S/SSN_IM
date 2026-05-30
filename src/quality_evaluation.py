"""
Quality Evaluation - Influence Quality Metrics
Algorithms evaluated:
  1. CRISP + SAP              (our full approach)
  2. CRISP + Signed IC
  3. CRISP + Signed LT
  4. Random + SAP
  5. Degree Centrality + SAP
  6. CELF Greedy + SAP        (fast version: 50 candidates, alpha=0.3)
  7. PageRank + Signed IC

Metrics:
  PIR  - Positive Influence Ratio  = positive / (positive + negative)
  WNS  - Weighted Net Score        = sum of all positive scores
  CICR - Counter-Influence Containment Rate = 1 - (negative / total reached)
  Net Spread                       = positive - negative count

Output: outputs/evaluation/quality_metrics.csv
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from scipy.stats import entropy
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Quality Evaluation - Influence Quality Metrics")
print()

# Load
print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

seeds_df  = pd.read_csv(os.path.join(SEEDS_DIR, "selected_seeds.csv"))
our_seeds = list(seeds_df["node"].astype(int))
k         = len(our_seeds)

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)

print(f"  Nodes: {n:,}  |  Edges: {len(df):,}  |  Seed budget k: {k}")

# Build adjacency
print("Building adjacency structures...")
adj_all    = defaultdict(list)
adj_signed = defaultdict(list)
adj_pos    = defaultdict(list)
adj_neg    = defaultdict(list)
in_degree  = defaultdict(int)

for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    adj_all[u].append(v)
    adj_signed[u].append((v, s))
    in_degree[v] += 1
    if s == 1:
        adj_pos[u].append(v)
    else:
        adj_neg[u].append(v)

# Quality metrics
def compute_quality_metrics(scores, method_name):
    """
    Computes four quality metrics from a score dictionary.

    PIR  = positive_inf / (positive_inf + negative_inf)
         → measures how clean the influence spread was
    WNS  = sum of all positive scores
         → measures strength of positive influence, not just count
    CICR = 1 - (negative_inf / total_reached)
         → measures how well counter-influence was contained
    Net Spread = positive_inf - negative_inf
         → width advantage of positive over negative
    """
    positive_inf  = sum(1 for s in scores.values() if s >  SCORE_THRESH)
    negative_inf  = sum(1 for s in scores.values() if s < -SCORE_THRESH)
    total_reached = sum(1 for s in scores.values() if abs(s) > SCORE_THRESH)
    net_spread    = positive_inf - negative_inf

    pir  = (positive_inf / (positive_inf + negative_inf)
            if (positive_inf + negative_inf) > 0 else 0)
    wns  = sum(s for s in scores.values() if s > SCORE_THRESH)
    cicr = (1 - negative_inf / total_reached
            if total_reached > 0 else 1)

    # Score band distribution
    strong_pos  = sum(1 for s in scores.values() if s >  1.0)
    moderate_pos= sum(1 for s in scores.values() if 0.5 < s <= 1.0)
    weak_pos    = sum(1 for s in scores.values() if 0.0 < s <= 0.5)
    weak_neg    = sum(1 for s in scores.values() if -0.5 <= s < 0.0)
    strong_neg  = sum(1 for s in scores.values() if s < -1.0)

    print(f"  PIR: {pir:.4f}  WNS: {wns:.2f}  CICR: {cicr:.4f}  Net Spread: {net_spread:,}")
    print(f"  Score bands — Strong+: {strong_pos:,}  Mod+: {moderate_pos:,}  Weak+: {weak_pos:,}  Weak-: {weak_neg:,}  Strong-: {strong_neg:,}")

    return {
        "method"        : method_name,
        "positive_inf"  : positive_inf,
        "negative_inf"  : negative_inf,
        "total_reached" : total_reached,
        "net_spread"    : net_spread,
        "PIR"           : round(pir, 4),
        "WNS"           : round(wns, 4),
        "CICR"          : round(cicr, 4),
        "strong_pos"    : strong_pos,
        "moderate_pos"  : moderate_pos,
        "weak_pos"      : weak_pos,
        "weak_neg"      : weak_neg,
        "strong_neg"    : strong_neg
    }

# Propagation models
def run_sap(seed_nodes, alpha=DECAY_ALPHA):
    scores         = defaultdict(float)
    arrival_counts = defaultdict(int)
    queue          = deque()
    for seed in seed_nodes:
        scores[seed] += 1.0
        arrival_counts[seed] += 1
        for (nbr, sign) in adj_signed[seed]:
            queue.append((nbr, sign, 1.0))
    visited = set()
    while queue:
        node, arriving_sign, parent_weight = queue.popleft()
        count  = arrival_counts[node]
        weight = parent_weight * (alpha ** count)
        if abs(weight) < 1e-6:
            continue
        scores[node]         += arriving_sign * weight
        arrival_counts[node] += 1
        for (nbr, sign) in adj_signed[node]:
            key = (node, nbr, arriving_sign)
            if key in visited:
                continue
            visited.add(key)
            new_weight = weight * alpha
            if abs(new_weight) >= 1e-6:
                queue.append((nbr, arriving_sign * sign, new_weight))
    return dict(scores)

def run_signed_ic(seed_nodes, n_sim=SIM_RUNS):
    pos_counts = defaultdict(float)
    neg_counts = defaultdict(float)
    for _ in range(n_sim):
        pos_active = set(seed_nodes)
        neg_active = set()
        pos_front  = list(seed_nodes)
        neg_front  = []
        while pos_front or neg_front:
            next_pos = []
            next_neg = []
            for u in pos_front:
                for v in adj_pos[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            pos_active.add(v); next_pos.append(v)
                for v in adj_neg[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            neg_active.add(v); next_neg.append(v)
            for u in neg_front:
                for v in adj_pos[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            neg_active.add(v); next_neg.append(v)
                for v in adj_neg[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            pos_active.add(v); next_pos.append(v)
            pos_front = next_pos
            neg_front = next_neg
        for node in pos_active: pos_counts[node] += 1
        for node in neg_active: neg_counts[node] += 1
    scores = {}
    for node in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        p = pos_counts.get(node, 0) / n_sim
        q = neg_counts.get(node, 0) / n_sim
        scores[node] = p - q
    return scores

def run_signed_lt(seed_nodes, n_sim=10):
    pos_counts = defaultdict(float)
    neg_counts = defaultdict(float)
    for _ in range(n_sim):
        thresholds    = {node: random.random() for node in all_nodes}
        pos_active    = set(seed_nodes)
        neg_active    = set()
        pos_influence = defaultdict(float)
        neg_influence = defaultdict(float)
        frontier      = list(seed_nodes)
        while frontier:
            next_frontier = []
            for u in frontier:
                for (v, s) in adj_signed[u]:
                    if v in pos_active or v in neg_active:
                        continue
                    contribution = 1.0 / max(in_degree[v], 1)
                    if s == 1:
                        pos_influence[v] += contribution
                    else:
                        neg_influence[v] += contribution
                    net = pos_influence[v] - neg_influence[v]
                    if net > thresholds[v]:
                        pos_active.add(v)
                        next_frontier.append(v)
                    elif -net > thresholds[v]:
                        neg_active.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
        for node in pos_active: pos_counts[node] += 1
        for node in neg_active: neg_counts[node] += 1
    scores = {}
    for node in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        p = pos_counts.get(node, 0) / n_sim
        q = neg_counts.get(node, 0) / n_sim
        scores[node] = p - q
    return scores

# Selection methods
def random_selection(k):
    return random.sample(all_nodes, k)

def degree_centrality_selection(k):
    degree = {node: len(adj_all[node]) for node in all_nodes}
    return sorted(degree, key=degree.get, reverse=True)[:k]

def pagerank_selection(k):
    G_temp = nx.DiGraph()
    for _, row in df.iterrows():
        G_temp.add_edge(int(row["from_node"]), int(row["to_node"]))
    pr = nx.pagerank(G_temp, alpha=0.85, max_iter=200)
    return sorted(pr, key=pr.get, reverse=True)[:k]

def celf_greedy_selection_fast(k, propagation_fn, alpha=0.3):
    """
    CELF fast version: 50 candidate sample, alpha=0.3 for faster SAP.
    Suboptimal but tractable. Noted in paper as approximate CELF.
    """
    print(f"    CELF fast: 50 candidates, alpha={alpha}...")
    sample_nodes   = random.sample(all_nodes, min(50, len(all_nodes)))
    gains          = {}
    for node in sample_nodes:
        s           = propagation_fn([node], alpha)
        m           = compute_quality_metrics(s, "temp")
        gains[node] = m["positive_inf"]

    heap           = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    selected       = []
    current_spread = 0

    while len(selected) < k and heap:
        node, gain = heap[0]
        s          = propagation_fn(selected + [node], alpha)
        m          = compute_quality_metrics(s, "temp")
        new_spread = m["positive_inf"]
        marginal   = new_spread - current_spread
        heap[0]    = (node, marginal)
        heap.sort(key=lambda x: x[1], reverse=True)
        if heap[0][0] == node:
            selected.append(node)
            current_spread = new_spread
            heap.pop(0)
            print(f"    CELF step {len(selected)}/{k}: node {node}  marginal={marginal}")

    if len(selected) < k:
        remaining  = [nd for nd in all_nodes if nd not in selected]
        selected  += random.sample(remaining, k - len(selected))

    return selected

# Run all algorithms
results = []

print("\n1. CRISP + SAP (Our Full Approach)...")
scores = run_sap(our_seeds)
results.append(compute_quality_metrics(scores, "CRISP + SAP"))

print("\n2. CRISP + Signed IC...")
scores = run_signed_ic(our_seeds)
results.append(compute_quality_metrics(scores, "CRISP + Signed IC"))

print("\n3. CRISP + Signed LT...")
scores = run_signed_lt(our_seeds)
results.append(compute_quality_metrics(scores, "CRISP + Signed LT"))

print("\n4. Random + SAP...")
rand_seeds = random_selection(k)
print(f"  Seeds: {rand_seeds}")
scores = run_sap(rand_seeds)
results.append(compute_quality_metrics(scores, "Random + SAP"))

print("\n5. Degree Centrality + SAP...")
deg_seeds = degree_centrality_selection(k)
print(f"  Seeds: {deg_seeds}")
scores = run_sap(deg_seeds)
results.append(compute_quality_metrics(scores, "Degree Centrality + SAP"))

print("\n6. CELF Greedy + SAP (fast)...")
celf_seeds = celf_greedy_selection_fast(k, run_sap, alpha=0.3)
print(f"  Seeds: {celf_seeds}")
scores = run_sap(celf_seeds)
results.append(compute_quality_metrics(scores, "CELF Greedy + SAP"))

print("\n7. PageRank + Signed IC...")
pr_seeds = pagerank_selection(k)
print(f"  Seeds: {pr_seeds}")
scores = run_signed_ic(pr_seeds)
results.append(compute_quality_metrics(scores, "PageRank + Signed IC"))

# Results table
print(f"\n{'='*100}")
print("Quality Evaluation Results")
print(f"{'='*100}")
results_df = pd.DataFrame(results)

display_cols = ["method", "positive_inf", "negative_inf",
                "net_spread", "PIR", "WNS", "CICR"]
print(results_df[display_cols].to_string(index=False))

print(f"\nScore Band Distribution:")
band_cols = ["method", "strong_pos", "moderate_pos", "weak_pos", "weak_neg", "strong_neg"]
print(results_df[band_cols].to_string(index=False))

# Improvement over each baseline
our = results_df[results_df["method"] == "CRISP + SAP"].iloc[0]
print(f"\nCRISP + SAP improvement over each baseline:")
for _, row in results_df[results_df["method"] != "CRISP + SAP"].iterrows():
    pir_imp  = our["PIR"]  - row["PIR"]
    wns_imp  = our["WNS"]  - row["WNS"]
    cicr_imp = our["CICR"] - row["CICR"]
    net_imp  = our["net_spread"] - row["net_spread"]
    print(f"  vs {row['method']:<30}  PIR: {pir_imp:+.4f}  WNS: {wns_imp:+.2f}  "
          f"CICR: {cicr_imp:+.4f}  Net: {net_imp:+,}")

# Save
print("\nSaving outputs...")
results_df.to_csv(os.path.join(EVAL_DIR, "quality_metrics.csv"), index=False)
print(f"  quality_metrics.csv saved")
print("\nQuality evaluation complete.")