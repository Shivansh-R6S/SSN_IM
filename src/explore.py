

import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = r"D:\Research Internship\Signed Social Network\SSN_IM\Modified Datasets\soc-sign-Slashdot081106.txt"

print("=" * 60)
print("STEP 1 — LOADING AND EXPLORING THE SLASHDOT DATASET")
print("=" * 60)

# ── LOAD ─────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH, sep="\t", header=None, names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

print(f"\n[1] RAW FILE")
print(f"    Rows: {len(df):,}  |  Columns: {list(df.columns)}")
print(df.head())

# ── FORMAT VALIDATION ────────────────────────────────────────
print(f"\n[2] FORMAT VALIDATION")
print(f"    Unique signs    : {set(df['sign'].unique())}")
print(f"    Missing values  : {df.isnull().sum().sum()}")

# ── BASIC STATS ──────────────────────────────────────────────
total_edges = len(df)
pos_edges   = (df["sign"] ==  1).sum()
neg_edges   = (df["sign"] == -1).sum()
all_nodes   = set(df["from_node"]) | set(df["to_node"])

print(f"\n[3] BASIC NETWORK STATISTICS")
print(f"    Nodes           : {len(all_nodes):,}")
print(f"    Edges           : {total_edges:,}")
print(f"    Positive (+1)   : {pos_edges:,}  ({pos_edges/total_edges*100:.1f}%)")
print(f"    Negative (-1)   : {neg_edges:,}  ({neg_edges/total_edges*100:.1f}%)")
print(f"    Imbalance ratio : {pos_edges/neg_edges:.2f}:1")

# ── DEGREE ANALYSIS ──────────────────────────────────────────
pos_df = df[df["sign"] ==  1]
neg_df = df[df["sign"] == -1]

total_out = df.groupby("from_node").size()
total_in  = df.groupby("to_node").size()
pos_in    = pos_df.groupby("to_node").size()
neg_in    = neg_df.groupby("to_node").size()

print(f"\n[4] DEGREE ANALYSIS")
print(f"    Out-degree  mean={total_out.mean():.2f}  median={total_out.median():.0f}  max={total_out.max()}")
print(f"    In-degree   mean={total_in.mean():.2f}  median={total_in.median():.0f}  max={total_in.max()}")

print(f"\n[5] TOP 10 HUB NODES (by in-degree)")
top_hubs = total_in.sort_values(ascending=False).head(10)
for node, deg in top_hubs.items():
    p = pos_in.get(node, 0)
    n = neg_in.get(node, 0)
    print(f"    Node {node:>6} | in-degree: {deg:>5} | friends: {p:>5} | foes: {n:>5}")

# ── RECIPROCITY ──────────────────────────────────────────────
sign_dict = dict(zip(zip(df["from_node"], df["to_node"]), df["sign"]))
mutual = agree = disagree = 0
for (u, v), s_uv in sign_dict.items():
    if (v, u) in sign_dict:
        mutual += 1
        agree    += 1 if s_uv == sign_dict[(v, u)] else 0
        disagree += 1 if s_uv != sign_dict[(v, u)] else 0
mutual //= 2; agree //= 2; disagree //= 2

print(f"\n[6] RECIPROCITY")
print(f"    Mutual edges    : {mutual:,}")
print(f"    Sign agreement  : {agree:,}  ({agree/max(mutual,1)*100:.1f}%)")
print(f"    Sign disagreement:{disagree:,}  ({disagree/max(mutual,1)*100:.1f}%)")

# ── TRIANGLE BALANCE ─────────────────────────────────────────
print(f"\n[7] BALANCE — TRIANGLE ANALYSIS (sample of 50,000)")
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(int(row["from_node"]), int(row["to_node"]), sign=int(row["sign"]))

adj = {}
for _, row in df.iterrows():
    u, v, s = int(row["from_node"]), int(row["to_node"]), int(row["sign"])
    adj.setdefault(u, {})[v] = s

balanced = unbalanced = sampled = 0
rng = np.random.default_rng(42)
for u in rng.choice(list(G.nodes()), size=min(5000, G.number_of_nodes()), replace=False):
    if u not in adj: continue
    for v in adj[u]:
        if v not in adj: continue
        for w in adj[v]:
            if w in adj and u in adj.get(w, {}):
                product = adj[u][v] * adj[v][w] * adj[w][u]
                if product > 0: balanced += 1
                else: unbalanced += 1
                sampled += 1
            if sampled >= 50000: break
        if sampled >= 50000: break
    if sampled >= 50000: break

total_tri = balanced + unbalanced
print(f"    Sampled triangles   : {total_tri:,}")
print(f"    Balanced            : {balanced:,}  ({balanced/max(total_tri,1)*100:.1f}%)")
print(f"    Unbalanced          : {unbalanced:,}  ({unbalanced/max(total_tri,1)*100:.1f}%)")

# ── CONNECTED COMPONENTS ─────────────────────────────────────
wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
print(f"\n[8] CONNECTED COMPONENTS")
print(f"    Total components    : {len(wcc):,}")
print(f"    Largest component   : {len(wcc[0]):,} nodes  ({len(wcc[0])/len(all_nodes)*100:.1f}%)")

print(f"\n{'='*60}")
print(f"  >> Data clean and ready for Step 2")
print(f"{'='*60}")