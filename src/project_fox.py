import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Evaluation v3 - Propagation vs Selection Strategy Analysis")
print()

# Load data
print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

scores_df   = pd.read_csv(os.path.join(SCORING_DIR, "node_scores.csv"))
comm_scores = pd.read_csv(os.path.join(COMMUNITY_DIR, "community_scores.csv"))
comm_assign = pd.read_csv(os.path.join(COMMUNITY_DIR, "community_assignments.csv"))

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)

print(f"  Nodes: {n:,}  |  Edges: {len(df):,}")

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

# Precompute for CRISP selection
print("Precomputing CRISP selection structures...")
pagerank_map  = dict(zip(scores_df["node"], scores_df["pagerank"]))
seed_score_map= dict(zip(scores_df["node"], scores_df["seed_score"]))
boundary_map  = dict(zip(comm_assign["node"], comm_assign["is_boundary_node"]))
community_map = dict(zip(comm_assign["node"], comm_assign["final_community"]))

pos_neighbors = {}
neg_neighbors = {}
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    if row["sign"] == 1:
        pos_neighbors.setdefault(u, set()).add(v)
    else:
        neg_neighbors.setdefault(u, set()).add(v)

# PageRank for PageRank selection
print("Computing PageRank for PageRank selection...")
G_temp = nx.DiGraph()
for _, row in df.iterrows():
    G_temp.add_edge(int(row["from_node"]), int(row["to_node"]))
full_pagerank = nx.pagerank(G_temp, alpha=0.85, max_iter=200)

# Metrics
def compute_metrics(scores):
    positive_inf   = sum(1 for s in scores.values() if s >  SCORE_THRESH)
    negative_inf   = sum(1 for s in scores.values() if s < -SCORE_THRESH)
    total_activated= positive_inf + negative_inf
    net_influence  = positive_inf - negative_inf
    strong_pos     = sum(1 for s in scores.values() if s >  1.0)
    strong_neg     = sum(1 for s in scores.values() if s < -1.0)
    wns            = sum(s for s in scores.values() if s > SCORE_THRESH)
    cicr           = (1 - negative_inf / total_activated
                      if total_activated > 0 else 1.0)
    return {
        "total_activated": total_activated,
        "positive_inf"   : positive_inf,
        "net_influence"  : net_influence,
        "strong_positive": strong_pos,
        "strong_negative": strong_neg,
        "WNS"            : round(wns, 4),
        "CICR"           : round(cicr, 4)
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
        scores[node] = pos_counts.get(node, 0)/n_sim - neg_counts.get(node, 0)/n_sim
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
                        pos_active.add(v); next_frontier.append(v)
                    elif -net > thresholds[v]:
                        neg_active.add(v); next_frontier.append(v)
            frontier = next_frontier
        for node in pos_active: pos_counts[node] += 1
        for node in neg_active: neg_counts[node] += 1
    scores = {}
    for node in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        scores[node] = pos_counts.get(node, 0)/n_sim - neg_counts.get(node, 0)/n_sim
    return scores

# Selection strategies
def crisp_selection(k):
    
    eligible = scores_df[~scores_df["is_boundary"]].copy()

    def has_sleeping_giant(node):
        return any(pagerank_map.get(f, 0) > GIANT_THRESH
                   for f in neg_neighbors.get(node, set()))

    eligible["sleeping_giant_risk"] = eligible["node"].apply(has_sleeping_giant)
    safe_candidates = eligible[~eligible["sleeping_giant_risk"]].copy()

    n_communities    = len(comm_scores)
    remaining_budget = k - n_communities

    if remaining_budget < 0:
        comm_scores["allocation"] = 1
    else:
        total_quality             = comm_scores["quality_score"].sum()
        proportional              = comm_scores["quality_score"] / total_quality * remaining_budget
        comm_scores["allocation"] = 1 + proportional.apply(np.floor).astype(int)
        leftover = k - comm_scores["allocation"].sum()
        if leftover > 0:
            top_idx = comm_scores.nlargest(int(leftover), "quality_score").index
            comm_scores.loc[top_idx, "allocation"] += 1

    OVERLAP_THRESHOLD     = 0.3
    FOE_OVERLAP_THRESHOLD = 0.3
    selected_seeds        = []

    for _, comm_row in comm_scores.iterrows():
        comm_id = int(comm_row["community_id"])
        n_seeds = int(comm_row["allocation"])

        candidates = safe_candidates[safe_candidates["community"] == comm_id].sort_values(
            "seed_score", ascending=False)
        if candidates.empty:
            candidates = eligible[eligible["community"] == comm_id].sort_values(
                "seed_score", ascending=False)
        if candidates.empty:
            candidates = scores_df[scores_df["community"] == comm_id].sort_values(
                "seed_score", ascending=False)
        if candidates.empty:
            continue

        selected_in_comm = []
        for _, candidate in candidates.iterrows():
            if len(selected_in_comm) >= n_seeds:
                break
            node     = int(candidate["node"])
            node_pos = pos_neighbors.get(node, set())
            node_neg = neg_neighbors.get(node, set())
            too_similar = False
            for prev_node in selected_in_comm:
                prev_pos = pos_neighbors.get(prev_node, set())
                prev_neg = neg_neighbors.get(prev_node, set())
                pos_union   = node_pos | prev_pos
                pos_jaccard = len(node_pos & prev_pos) / len(pos_union) if pos_union else 0
                neg_union   = node_neg | prev_neg
                neg_jaccard = len(node_neg & prev_neg) / len(neg_union) if neg_union else 0
                if pos_jaccard > OVERLAP_THRESHOLD or neg_jaccard > FOE_OVERLAP_THRESHOLD:
                    too_similar = True
                    break
            if not too_similar:
                selected_in_comm.append(node)
                selected_seeds.append(node)

    if len(selected_seeds) < k:
        already = set(selected_seeds)
        remaining_needed = k - len(selected_seeds)
        global_top = safe_candidates[~safe_candidates["node"].isin(already)].sort_values(
            "seed_score", ascending=False)
        for _, row in global_top.iterrows():
            if len(selected_seeds) >= k:
                break
            selected_seeds.append(int(row["node"]))

    return selected_seeds[:k]

def random_selection(k):
    return random.sample(all_nodes, k)

def degree_centrality_selection(k):
    degree = {node: len(adj_all[node]) for node in all_nodes}
    return sorted(degree, key=degree.get, reverse=True)[:k]

def pagerank_selection(k):
    return sorted(full_pagerank, key=full_pagerank.get, reverse=True)[:k]

def celf_greedy_selection(k, propagation_fn, alpha=0.3):
    """Fast CELF: 50 candidates, alpha=0.3 for speed."""
    sample_nodes   = random.sample(all_nodes, min(50, len(all_nodes)))
    gains          = {}
    for node in sample_nodes:
        s           = propagation_fn([node], alpha)
        m           = compute_metrics(s)
        gains[node] = m["positive_inf"]

    heap           = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    selected       = []
    current_spread = 0

    while len(selected) < k and heap:
        node, gain = heap[0]
        s          = propagation_fn(selected + [node], alpha)
        m          = compute_metrics(s)
        new_spread = m["positive_inf"]
        marginal   = new_spread - current_spread
        heap[0]    = (node, marginal)
        heap.sort(key=lambda x: x[1], reverse=True)
        if heap[0][0] == node:
            selected.append(node)
            current_spread = new_spread
            heap.pop(0)

    if len(selected) < k:
        remaining  = [nd for nd in all_nodes if nd not in selected]
        selected  += random.sample(remaining, k - len(selected))

    return selected[:k]

# Selection strategy definitions
SELECTION_STRATEGIES = {
    "CRISP"            : lambda k: crisp_selection(k),
    "Random"           : lambda k: random_selection(k),
    "Degree Centrality": lambda k: degree_centrality_selection(k),
    "CELF Greedy"      : lambda k: celf_greedy_selection(k, run_sap),
    "PageRank"         : lambda k: pagerank_selection(k),
}

# Propagation model definitions
PROPAGATION_MODELS = {
    "SAP"      : run_sap,
    "Signed IC": run_signed_ic,
    "Signed LT": run_signed_lt,
}

K_VALUES = range(20, 26)

# Run evaluation
all_results = {model: [] for model in PROPAGATION_MODELS}

for prop_name, prop_fn in PROPAGATION_MODELS.items():
    print(f"\nPropagation: {prop_name}")
    print(f"{'─'*50}")
    for sel_name, sel_fn in SELECTION_STRATEGIES.items():
        for k in K_VALUES:
            print(f"  {sel_name} | k={k} ...", end=" ", flush=True)

            seeds  = sel_fn(k)
            scores = prop_fn(seeds)
            m      = compute_metrics(scores)
            m["selection_strategy"] = sel_name
            m["k"]                  = k

            all_results[prop_name].append(m)
            print(f"pos={m['positive_inf']:,}  net={m['net_influence']:,}  "
                  f"WNS={m['WNS']:.1f}  CICR={m['CICR']:.4f}")


# Build and save CSVs
print(" Saving outputs...")

strategy_order = ["CRISP", "Random", "CELF Greedy", "Degree Centrality", "PageRank"]
k_values = list(range(20, 26))

for prop_name, results in all_results.items():
    results_df = pd.DataFrame(results)

    fname = prop_name.lower().replace(" ", "_") + "_eval.csv"
    fpath = os.path.join(EVAL_DIR, fname)

    rows = []

    for k in k_values:
        row = {"Dataset": DATASET_NAME if "DATASET_NAME" in globals() else "Slashdot",
               "Seed Size": k}

        for strategy in strategy_order:
            r = results_df[
                (results_df["selection_strategy"] == strategy) &
                (results_df["k"] == k)
            ]

            if len(r) == 0:
                continue

            r = r.iloc[0]

            if prop_name == "SAP":
                prefix = strategy
                row[f"{prefix}_Strong_Positive"] = r["strong_positive"]
                row[f"{prefix}_Strong_Negative"] = r["strong_negative"]
                row[f"{prefix}_WNS"] = r["WNS"]
                row[f"{prefix}_CICR"] = r["CICR"]

            else:  # Signed IC and Signed LT
                prefix = strategy
                row[f"{prefix}_Total_Influence"] = r["total_activated"]
                row[f"{prefix}_Positive_Influence"] = r["positive_inf"]
                row[f"{prefix}_Net_Influence"] = r["net_influence"]

        rows.append(row)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(fpath, index=False)

    print(f"  {fname} saved ({len(output_df)} rows)")

print(" Output files:")
for prop_name in PROPAGATION_MODELS:
    fname = prop_name.lower().replace(" ", "_") + "_eval.csv"
    print(f"  outputs/evaluation/{fname}")
