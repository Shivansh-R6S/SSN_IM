

import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 2 — COMMUNITY DETECTION + LOGISTIC REGRESSION")
print("=" * 60)

# ── 1. LOAD ──────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

pos_df = df[df["sign"] ==  1]
neg_df = df[df["sign"] == -1]

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)

print(f"    Nodes : {n:,}  |  Edges : {len(df):,}")

# ── 2. BUILD GRAPHS ──────────────────────────────────────────
print("\n[2] Building graphs...")

# Full signed directed graph for scoring and features
G_signed = nx.DiGraph()
for _, row in df.iterrows():
    G_signed.add_edge(int(row["from_node"]), int(row["to_node"]),
                      sign=int(row["sign"]))

print(f"    Signed graph : {G_signed.number_of_nodes():,} nodes, "
      f"{G_signed.number_of_edges():,} edges")

# ── 3. INFOMAP COMMUNITY DETECTION ───────────────────────────
print("\n[3] Running Infomap on positive-only directed subgraph...")

# Infomap works with integer node IDs starting from 1
# We create a mapping from original node IDs to 1-based IDs
node_to_infomap = {node: i+1 for i, node in enumerate(all_nodes)}
infomap_to_node = {i+1: node for i, node in enumerate(all_nodes)}

im = Infomap(
    directed=True,       # respect edge direction
    num_trials=10,       # run 10 times and take best result
    silent=True          # suppress verbose output
)

# Add only positive edges to Infomap
# Infomap finds communities based on information flow
# Positive edges = channels through which influence flows
print("    Adding positive edges to Infomap...")
for _, row in pos_df.iterrows():
    u = node_to_infomap[int(row["from_node"])]
    v = node_to_infomap[int(row["to_node"])]
    im.add_link(u, v)

# Add all nodes to ensure isolated nodes are included
for node in all_nodes:
    im.add_node(node_to_infomap[node])

print("    Running Infomap optimisation (10 trials)...")
im.run()

# Extract partition
partition = {}
for node_id, module in im.modules:
    original_node = infomap_to_node[node_id]
    partition[original_node] = module

# Handle any nodes not assigned by Infomap
for node in all_nodes:
    if node not in partition:
        partition[node] = 0

# ── 4. ANALYSE RAW INFOMAP COMMUNITIES ───────────────────────
print("\n[4] Analysing Infomap communities...")

raw_communities = {}
for node, comm_id in partition.items():
    raw_communities.setdefault(comm_id, set()).add(node)

raw_sizes = sorted([len(v) for v in raw_communities.values()], reverse=True)
print(f"    Total communities found : {len(raw_communities)}")
print(f"    Largest community       : {raw_sizes[0]:,} nodes")
print(f"    Smallest community      : {raw_sizes[-1]:,} nodes")
print(f"    Top 10 sizes            : {raw_sizes[:10]}")

# Keep top N_CLUSTERS communities by size
# Remap remaining nodes to nearest large community
print(f"\n    Keeping top {N_CLUSTERS} communities by size...")

top_comms = sorted(raw_communities.items(),
                   key=lambda x: len(x[1]), reverse=True)[:N_CLUSTERS]
top_comm_ids  = [comm_id for comm_id, _ in top_comms]
top_comm_set  = set(top_comm_ids)
id_remap      = {old: new for new, old in enumerate(top_comm_ids)}

remapped_partition = {}
for node, comm_id in partition.items():
    if comm_id in top_comm_set:
        remapped_partition[node] = id_remap[comm_id]
    else:
        # Assign to largest community
        remapped_partition[node] = 0

# Final community sizes after remapping
final_communities = {}
for node, comm_id in remapped_partition.items():
    final_communities.setdefault(comm_id, set()).add(node)

final_sizes = sorted([len(v) for v in final_communities.values()], reverse=True)
print(f"    Final community sizes   : {final_sizes}")

# ── 5. EXTRACT NODE FEATURES ─────────────────────────────────
print("\n[5] Extracting node features...")

pos_out   = pos_df.groupby("from_node").size().to_dict()
neg_out   = neg_df.groupby("from_node").size().to_dict()
pos_in    = pos_df.groupby("to_node").size().to_dict()
neg_in    = neg_df.groupby("to_node").size().to_dict()
total_out = df.groupby("from_node").size().to_dict()
total_in  = df.groupby("to_node").size().to_dict()

def get_features(node):
    po      = pos_out.get(node, 0)
    no      = neg_out.get(node, 0)
    pi      = pos_in.get(node, 0)
    ni      = neg_in.get(node, 0)
    tot_out = total_out.get(node, 0)
    tot_in  = total_in.get(node, 0)
    tot     = tot_out + tot_in

    pos_out_rate     = po / tot_out if tot_out > 0 else 0
    pos_in_rate      = pi / tot_in  if tot_in  > 0 else 0
    signed_out       = po - no
    signed_in        = pi - ni
    overall_pos_rate = (po + pi) / tot if tot > 0 else 0

    return [
        po, no, pi, ni,
        tot_out, tot_in,
        signed_out, signed_in,
        pos_out_rate, pos_in_rate,
        overall_pos_rate
    ]

feature_names = [
    "pos_out_deg", "neg_out_deg", "pos_in_deg", "neg_in_deg",
    "total_out_deg", "total_in_deg", "signed_out_deg", "signed_in_deg",
    "pos_out_rate", "pos_in_rate", "overall_pos_rate"
]

print(f"    Building feature matrix for {n:,} nodes...")
X_data = []
y_data = []
nodes_ordered = []

for node in all_nodes:
    X_data.append(get_features(node))
    y_data.append(remapped_partition.get(node, 0))
    nodes_ordered.append(node)

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data, dtype=np.int32)

print(f"    Feature matrix shape : {X.shape}")
print(f"    Label distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"      Community {u:>3} : {c:>6,} nodes  ({c/n*100:.1f}%)")

# ── 6. TRAIN LOGISTIC REGRESSION ─────────────────────────────
print("\n[6] Training Logistic Regression classifier...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"    Train : {len(X_train):,}  |  Test : {len(X_test):,}")

clf = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    solver="lbfgs",
    multi_class="multinomial"
)
clf.fit(X_train, y_train)

y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred)

print(f"\n    Test Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"\n    Classification Report:")
print(report)

# Predict community for all nodes
y_all_pred = clf.predict(X_scaled)

# ── 7. COMMUNITY QUALITY SCORING ─────────────────────────────
print("\n[7] Scoring communities...")

predicted_communities = {}
for node, label in zip(nodes_ordered, y_all_pred):
    predicted_communities.setdefault(int(label), set()).add(node)

def score_communities(communities, G_signed):
    results = []
    for comm_id, members in communities.items():
        members_set = set(members)
        int_pos = int_neg = bnd_pos = bnd_neg = 0

        for u in members_set:
            if u not in G_signed:
                continue
            for v, data in G_signed[u].items():
                s = data["sign"]
                if v in members_set:
                    if s ==  1: int_pos += 1
                    else:       int_neg += 1
                else:
                    if s ==  1: bnd_pos += 1
                    else:       bnd_neg += 1

        total_int = int_pos + int_neg
        total_bnd = bnd_pos + bnd_neg

        int_positivity = int_pos / total_int if total_int > 0 else 0
        insulation     = 1 - (bnd_neg / total_bnd) if total_bnd > 0 else 1
        quality        = len(members_set) * int_positivity * insulation

        results.append({
            "community_id"        : comm_id,
            "size"                : len(members_set),
            "internal_pos_edges"  : int_pos,
            "internal_neg_edges"  : int_neg,
            "internal_positivity" : round(int_positivity, 4),
            "insulation"          : round(insulation, 4),
            "quality_score"       : round(quality, 2)
        })

    return pd.DataFrame(results).sort_values(
        "quality_score", ascending=False
    ).reset_index(drop=True)

community_scores = score_communities(predicted_communities, G_signed)

print("\n    Top 10 Communities by Quality Score:")
print(community_scores.head(10).to_string(index=False))

# ── 8. FEATURE IMPORTANCE ────────────────────────────────────
print("\n[8] Feature importance:")
importance = np.abs(clf.coef_).mean(axis=0)
feat_imp   = sorted(zip(feature_names, importance),
                    key=lambda x: x[1], reverse=True)
for fname, imp in feat_imp:
    print(f"    {fname:<22} : {imp:.4f}")

# ── 9. SAVE OUTPUTS ──────────────────────────────────────────
print("\n[9] Saving outputs...")

assignments = pd.DataFrame({
    "node"                : nodes_ordered,
    "infomap_community"   : y_data,
    "predicted_community" : y_all_pred
})

assignments.to_csv(
    os.path.join(COMMUNITY_DIR, "community_assignments.csv"), index=False
)
community_scores.to_csv(
    os.path.join(COMMUNITY_DIR, "community_scores.csv"), index=False
)

report_path = os.path.join(COMMUNITY_DIR, "classifier_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nFeature Importance:\n")
    for fname, imp in feat_imp:
        f.write(f"  {fname:<22} : {imp:.4f}\n")

print(f"    community_assignments.csv : {len(assignments):,} nodes")
print(f"    community_scores.csv      : {len(community_scores)} communities")
print(f"    classifier_report.txt     : saved")

print(f"\n{'='*60}")
print("  >> Step 2 complete. Ready for Step 3.")
print(f"{'='*60}")