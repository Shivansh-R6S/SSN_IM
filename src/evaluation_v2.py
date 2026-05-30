"""
Step 6 v2 - Evaluation Against Baselines
Two-component evaluation framework:
  - Our Selection (CRISP) vs different propagation methods (B1, B2, B3)
  - Different selection methods vs our SAP (B4, B5, B6)
  - Independent baseline (B7)
  - Our full approach: CRISP + SAP

Metrics:
  - Total nodes activated
  - Positive influence
  - Negative influence
  - Net spread

Output: outputs/evaluation/metrics_comparison.csv
"""

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

print("Step 6 v2 - Evaluation Against Baselines")
print()

# Load data
print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

pos_df    = df[df["sign"] ==  1]
neg_df    = df[df["sign"] == -1]
scores_df = pd.read_csv(os.path.join(SCORING_DIR, "node_scores.csv"))
seeds_df  = pd.read_csv(os.path.join(SEEDS_DIR, "selected_seeds.csv"))
our_seeds = list(seeds_df["node"].astype(int))

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)

print(f"  Nodes: {n:,}  |  Edges: {len(df):,}")
print(f"  CRISP seeds ({len(our_seeds)}): {our_seeds}")

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

# Metrics
def compute_metrics(scores):
    total_activated = sum(1 for s in scores.values() if abs(s) > SCORE_THRESH)
    positive_inf    = sum(1 for s in scores.values() if s >  SCORE_THRESH)
    negative_inf    = sum(1 for s in scores.values() if s < -SCORE_THRESH)
    net_spread      = positive_inf - negative_inf
    return {
        "total_activated": total_activated,
        "positive_inf"   : positive_inf,
        "negative_inf"   : negative_inf,
        "net_spread"     : net_spread
    }

# Propagation models
def run_sap(seed_nodes, alpha=DECAY_ALPHA):
    """Signed Attenuated Propagation — CRISP propagation model."""
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

def run_unsigned_ic(seed_nodes, n_sim=SIM_RUNS):
    """Standard unsigned Independent Cascade."""
    total = defaultdict(float)
    for _ in range(n_sim):
        activated = set(seed_nodes)
        frontier  = list(seed_nodes)
        while frontier:
            next_f = []
            for u in frontier:
                for v in adj_all[u]:
                    if v not in activated:
                        prob = 1.0 / max(in_degree[v], 1)
                        if random.random() < prob:
                            activated.add(v)
                            next_f.append(v)
            frontier = next_f
        for node in activated:
            total[node] += 1
    return {node: cnt / n_sim for node, cnt in total.items()}

def run_signed_ic(seed_nodes, n_sim=SIM_RUNS):
    """
    Signed Independent Cascade.
    Positive edges propagate positive activation.
    Negative edges propagate negative activation.
    Negatively activated nodes spread counter-influence through their positive edges.
    """
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
                            pos_active.add(v)
                            next_pos.append(v)
                for v in adj_neg[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            neg_active.add(v)
                            next_neg.append(v)

            for u in neg_front:
                for v in adj_pos[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            neg_active.add(v)
                            next_neg.append(v)
                for v in adj_neg[u]:
                    if v not in pos_active and v not in neg_active:
                        if random.random() < 1.0 / max(in_degree[v], 1):
                            pos_active.add(v)
                            next_pos.append(v)

            pos_front = next_pos
            neg_front = next_neg

        for node in pos_active:
            pos_counts[node] += 1
        for node in neg_active:
            neg_counts[node] += 1

    scores = {}
    for node in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        p = pos_counts.get(node, 0) / n_sim
        q = neg_counts.get(node, 0) / n_sim
        scores[node] = p - q
    return scores

def run_signed_lt(seed_nodes, n_sim=10):
    """
    Signed Linear Threshold — frontier-based for speed.
    Node activates positively when net signed influence exceeds threshold.
    """
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

        for node in pos_active:
            pos_counts[node] += 1
        for node in neg_active:
            neg_counts[node] += 1

    scores = {}
    for node in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        p = pos_counts.get(node, 0) / n_sim
        q = neg_counts.get(node, 0) / n_sim
        scores[node] = p - q
    return scores

def run_signed_voter(seed_nodes, n_steps=500, n_sim=20):
    """
    Signed Voter Model.
    Nodes iteratively adopt opinions from random signed neighbours.
    Seeds maintain fixed opinion of +1.
    """
    opinion_sum = defaultdict(float)
    node_list   = list(all_nodes)

    for _ in range(n_sim):
        opinion = {node: 0.0 for node in all_nodes}
        for seed in seed_nodes:
            opinion[seed] = 1.0

        for _ in range(n_steps):
            node = random.choice(node_list)
            if node in seed_nodes:
                continue
            neighbours = adj_signed.get(node, [])
            if not neighbours:
                continue
            nbr, sign = random.choice(neighbours)
            opinion[node] = max(-1.0, min(1.0,
                opinion[node] + 0.1 * sign * opinion[nbr]))

        for node in all_nodes:
            opinion_sum[node] += opinion[node]

    return {node: opinion_sum[node] / n_sim for node in all_nodes}

# Selection methods
def random_selection(k):
    return random.sample(all_nodes, k)

def degree_centrality_selection(k):
    degree = {node: len(adj_all[node]) for node in all_nodes}
    return sorted(degree, key=degree.get, reverse=True)[:k]

def celf_greedy_selection(k, propagation_fn):
    """
    CELF optimised greedy with 200 candidate sample and lazy evaluation.
    Suboptimal but tractable for large networks.
    """
    print(f"    Initialising CELF with 200 candidate sample...")
    sample_nodes = random.sample(all_nodes, min(50, len(all_nodes)))

    gains = {}
    for node in sample_nodes:
        s     = propagation_fn([node])
        m     = compute_metrics(s)
        gains[node] = m["positive_inf"]

    heap           = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    selected       = []
    current_spread = 0

    while len(selected) < k and heap:
        node, gain = heap[0]
        s          = propagation_fn(selected + [node])
        m          = compute_metrics(s)
        new_spread = m["positive_inf"]
        marginal   = new_spread - current_spread

        heap[0] = (node, marginal)
        heap.sort(key=lambda x: x[1], reverse=True)

        if heap[0][0] == node:
            selected.append(node)
            current_spread = new_spread
            heap.pop(0)
            print(f"    CELF step {len(selected)}/{k}: node {node}  marginal={marginal}")

    return selected

def voterank_selection(k):
    """
    VoteRank: iterative voting with ability reduction to prevent clustering.
    """
    voting_ability = {node: 1.0 for node in all_nodes}
    selected       = []
    avg_degree     = sum(len(adj_all[node]) for node in all_nodes) / max(n, 1)
    reduction      = 1.0 / avg_degree if avg_degree > 0 else 0

    for _ in range(k):
        vote_score = defaultdict(float)
        for u in all_nodes:
            for v in adj_all[u]:
                vote_score[v] += voting_ability[u]
        if not vote_score:
            break
        best = max(vote_score, key=vote_score.get)
        selected.append(best)
        for nbr in adj_all[best]:
            voting_ability[nbr] = max(0.0, voting_ability[nbr] - reduction)
        voting_ability[best] = 0.0

    print(f"    VoteRank selected: {selected}")
    return selected

def pagerank_selection(k):
    """Seeds selected by highest PageRank."""
    G_temp = nx.DiGraph()
    for _, row in df.iterrows():
        G_temp.add_edge(int(row["from_node"]), int(row["to_node"]))
    pr = nx.pagerank(G_temp, alpha=0.85)
    return sorted(pr, key=pr.get, reverse=True)[:k]

# Run all baselines
k       = len(our_seeds)
results = []

def record(method, category, scores):
    m = compute_metrics(scores)
    m["method"]   = method
    m["category"] = category
    results.append(m)
    print(f"  Total activated: {m['total_activated']:,}  Positive: {m['positive_inf']:,}  "
          f"Negative: {m['negative_inf']:,}  Net spread: {m['net_spread']:,}")

# B1: Our Selection + Unsigned IC
print("\nBaseline 1: CRISP + Unsigned IC...")
record("CRISP + Unsigned IC", "Propagation Comparison", run_unsigned_ic(our_seeds))

# B2: Our Selection + Signed IC
print("\nBaseline 2: CRISP + Signed IC...")
record("CRISP + Signed IC", "Propagation Comparison", run_signed_ic(our_seeds))

# B3: Our Selection + Signed LT
print("\nBaseline 3: CRISP + Signed LT...")
record("CRISP + Signed LT", "Propagation Comparison", run_signed_lt(our_seeds))

# B4: Random + SAP
print("\nBaseline 4: Random Selection + SAP...")
rand_seeds = random_selection(k)
print(f"  Seeds: {rand_seeds}")
record("Random + SAP", "Selection Comparison", run_sap(rand_seeds))

# B5: Degree Centrality + SAP
print("\nBaseline 5: Degree Centrality + SAP...")
deg_seeds = degree_centrality_selection(k)
print(f"  Seeds: {deg_seeds}")
record("Degree Centrality + SAP", "Selection Comparison", run_sap(deg_seeds))

# B6: CELF Greedy + SAP
print("\nBaseline 6: CELF Greedy + SAP...")
celf_seeds = celf_greedy_selection(k, lambda s: run_sap(s, alpha=0.3))
print(f"  Seeds: {celf_seeds}")
record("CELF Greedy + SAP", "Selection Comparison", run_sap(celf_seeds))



# B7: VoteRank + Signed Voter Model
print("\nBaseline 7: VoteRank + Signed Voter Model...")
vr_seeds = voterank_selection(k)
record("VoteRank + Signed Voter Model", "Independent Baseline", run_signed_voter(vr_seeds))

# B8: PageRank + Signed IC
print("\nBaseline 7b: PageRank Selection + Signed IC...")
pr_seeds = pagerank_selection(k)
print(f"  Seeds: {pr_seeds}")
record("PageRank + Signed IC", "Independent Baseline", run_signed_ic(pr_seeds))

# Our Full Approach
print("\nOur Full Approach: CRISP + SAP...")
record("CRISP + SAP", "Our Approach", run_sap(our_seeds))

# Results
print(f"\n{'='*90}")
print("Final Results")
print(f"{'='*90}")
results_df = pd.DataFrame(results)[["method", "category", "total_activated",
                                     "positive_inf", "negative_inf", "net_spread"]]
print(results_df.to_string(index=False))

# Improvement summary
our = results_df[results_df["method"] == "CRISP + SAP"].iloc[0]
print(f"\nImprovement of CRISP + SAP over each baseline:")
for _, row in results_df[results_df["method"] != "CRISP + SAP"].iterrows():
    net_imp = our["net_spread"] - row["net_spread"]
    pos_imp = ((our["positive_inf"] - row["positive_inf"]) /
               max(row["positive_inf"], 1) * 100)
    print(f"  vs {row['method']:<35}  Net spread: {net_imp:+,}  Positive inf: {pos_imp:+.1f}%")

# Save
print("\nSaving outputs...")
results_df.to_csv(os.path.join(EVAL_DIR, "metrics_comparison.csv"), index=False)
print(f"  metrics_comparison.csv saved")
print("\nStep 6 v2 complete.")