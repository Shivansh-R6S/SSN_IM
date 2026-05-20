"""
Step 4 - Seed Selection
Community-based proportional allocation with guaranteed minimum 1 seed per community,
portfolio diversification, sleeping giant avoidance, and enemy-of-enemy bonus.

Output: outputs/seeds/selected_seeds.csv
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Step 4 - Seed Selection")
print()

print("Loading inputs...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

scores_df   = pd.read_csv(os.path.join(SCORING_DIR, "node_scores.csv"))
comm_scores = pd.read_csv(os.path.join(COMMUNITY_DIR, "community_scores.csv"))

print(f"  Node scores    : {len(scores_df):,} nodes")
print(f"  Communities    : {len(comm_scores)}")

print("Building adjacency lookup...")
pos_neighbors = {}
neg_neighbors = {}
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    if row["sign"] == 1:
        pos_neighbors.setdefault(u, set()).add(v)
    else:
        neg_neighbors.setdefault(u, set()).add(v)

eligible = scores_df[~scores_df["is_boundary"]].copy()
print(f"\nEligible (non-boundary): {len(eligible):,}")

print(f"Applying sleeping giant filter (PageRank threshold: {GIANT_THRESH})...")
pagerank_map = dict(zip(scores_df["node"], scores_df["pagerank"]))

def has_sleeping_giant(node):
    return any(pagerank_map.get(f, 0) > GIANT_THRESH
               for f in neg_neighbors.get(node, set()))

eligible["sleeping_giant_risk"] = eligible["node"].apply(has_sleeping_giant)
safe_candidates = eligible[~eligible["sleeping_giant_risk"]].copy()
print(f"  Sleeping giant risks : {eligible['sleeping_giant_risk'].sum():,}")
print(f"  Safe candidates      : {len(safe_candidates):,}")

print(f"\nAllocating {SEED_BUDGET} seeds across {len(comm_scores)} communities...")

n_communities    = len(comm_scores)
remaining_budget = SEED_BUDGET - n_communities

if remaining_budget < 0:
    print(f"  Warning: SEED_BUDGET ({SEED_BUDGET}) < communities ({n_communities}). Setting 1 per community.")
    comm_scores["allocation"] = 1
else:
    total_quality             = comm_scores["quality_score"].sum()
    proportional              = comm_scores["quality_score"] / total_quality * remaining_budget
    comm_scores["allocation"] = 1 + proportional.apply(np.floor).astype(int)
    leftover = SEED_BUDGET - comm_scores["allocation"].sum()
    if leftover > 0:
        top_idx = comm_scores.nlargest(int(leftover), "quality_score").index
        comm_scores.loc[top_idx, "allocation"] += 1

print("\n  Community seed allocation:")
for _, row in comm_scores.sort_values("quality_score", ascending=False).iterrows():
    print(f"    Community {int(row['community_id'])}: {int(row['allocation'])} seeds  (quality: {row['quality_score']:.1f})")

print("\nSelecting seeds with portfolio diversification...")

OVERLAP_THRESHOLD     = 0.3
FOE_OVERLAP_THRESHOLD = 0.3
selected_seeds        = []

for _, comm_row in comm_scores.iterrows():
    comm_id = int(comm_row["community_id"])
    n_seeds = int(comm_row["allocation"])

    candidates = safe_candidates[safe_candidates["community"] == comm_id].sort_values("seed_score", ascending=False)
    if candidates.empty:
        candidates = eligible[eligible["community"] == comm_id].sort_values("seed_score", ascending=False)
    if candidates.empty:
        candidates = scores_df[scores_df["community"] == comm_id].sort_values("seed_score", ascending=False)
    if candidates.empty:
        print(f"  Community {comm_id}: no candidates, skipping.")
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
            selected_seeds.append({
                "node"          : node,
                "community"     : comm_id,
                "seed_score"    : round(float(candidate["seed_score"]), 6),
                "opportunity"   : round(float(candidate["opportunity"]), 6),
                "danger"        : round(float(candidate["danger"]), 6),
                "eoe_bonus"     : round(float(candidate["eoe_bonus"]), 6),
                "pagerank"      : round(float(candidate["pagerank"]), 8),
                "pos_out_degree": int(candidate["pos_out_degree"]),
                "neg_out_degree": int(candidate["neg_out_degree"])
            })

    print(f"  Community {comm_id}: {len(selected_in_comm)}/{n_seeds} seeds selected")

seeds_df = pd.DataFrame(selected_seeds)

print(f"\nFinal seed set ({len(seeds_df)} seeds):")
print(seeds_df[["node", "community", "seed_score", "opportunity",
                 "danger", "eoe_bonus", "pagerank",
                 "pos_out_degree", "neg_out_degree"]].to_string(index=False))

all_pos_reach = set()
all_foe_sets  = set()
for node in seeds_df["node"]:
    all_pos_reach |= pos_neighbors.get(node, set())
    all_foe_sets  |= neg_neighbors.get(node, set())

print("\nPortfolio diversity check:")
print(f"  Total unique positive reach : {len(all_pos_reach):,} nodes")
print(f"  Total unique foes mobilised : {len(all_foe_sets):,} nodes")
print(f"  Communities represented     : {seeds_df['community'].nunique()}")
print(f"  Seeds per community         : {seeds_df.groupby('community').size().to_dict()}")
print(f"  Seeds with danger > 0       : {(seeds_df['danger'] > 0).sum()}")
print(f"  Seeds with eoe_bonus > 0    : {(seeds_df['eoe_bonus'] > 0).sum()}")

print("\nSaving outputs...")
seeds_df.to_csv(os.path.join(SEEDS_DIR, "selected_seeds.csv"), index=False)
print(f"  selected_seeds.csv: {len(seeds_df)} seeds saved")
print("\nStep 4 complete. Ready for Step 5.")