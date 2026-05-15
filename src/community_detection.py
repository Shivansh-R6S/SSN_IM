

import pandas as pd
import numpy as np
import networkx as nx
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

# Positive-only undirected for community detection
G_pos = nx.Graph()
G_pos.add_nodes_from(all_nodes)
for _, row in pos_df.iterrows():
    G_pos.add_edge(int(row["from_node"]), int(row["to_node"]))

# Full signed directed for feature extraction and scoring
G_signed = nx.DiGraph()
for _, row in df.iterrows():
    G_signed.add_edge(int(row["from_node"]), int(row["to_node"]),
                      sign=int(row["sign"]))

print(f"    Positive-only graph : {G_pos.number_of_edges():,} edges")
print(f"    Signed graph        : {G_signed.number_of_edges():,} edges")

# ── 3. LABEL PROPAGATION ─────────────────────────────────────
print("\n[3] Running Label Propagation community detection...")

communities_generator = nx.algorithms.community.label_propagation_communities(G_pos)
communities_list      = list(communities_generator)

# Build partition dict {node: community_id}
partition = {}
for comm_id, members in enumerate(communities_list):
    for node in members:
        partition[node] = comm_id

# Assign untagged nodes (nodes with no positive edges) to community -1
for node in all_nodes:
    if node not in partition:
        partition[node] = -1

n_communities = len(communities_list)
sizes         = sorted([len(c) for c in communities_list], reverse=True)

print(f"    Communities found : {n_communities}")
print(f"    Largest           : {sizes[0]:,} nodes")
print(f"    Smallest          : {sizes[-1]:,} nodes")
print(f"    Top 5 sizes       : {sizes[:5]}")

# Keep only top N_CLUSTERS communities by size for manageability
# Remap small communities to their nearest large community later
top_comm_ids = [comm_id for comm_id, _ in
                sorted(enumerate([len(c) for c in communities_list]),
                       key=lambda x: x[1], reverse=True)[:N_CLUSTERS]]

# Remap: nodes in top communities keep their id, others mapped to closest top
top_comm_set = set(top_comm_ids)
id_remap     = {old: new for new, old in enumerate(top_comm_ids)}

remapped_partition = {}
for node, comm_id in partition.items():
    if comm_id in top_comm_set:
        remapped_partition[node] = id_remap[comm_id]
    else:
        # Assign to community 0 (largest) as default for small/isolated
        remapped_partition[node] = 0

print(f"\n    Keeping top {N_CLUSTERS} communities by size.")
print(f"    Nodes remapped to top communities: {len(remapped_partition):,}")

# ── 4. EXTRACT NODE FEATURES ─────────────────────────────────
print("\n[4] Extracting node features for Logistic Regression...")

# Precompute degree stats per node
pos_out = pos_df.groupby("from_node").size().to_dict()
neg_out = neg_df.groupby("from_node").size().to_dict()
pos_in  = pos_df.groupby("to_node").size().to_dict()
neg_in  = neg_df.groupby("to_node").size().to_dict()
total_out = df.groupby("from_node").size().to_dict()
total_in  = df.groupby("to_node").size().to_dict()

def get_features(node):
    """
    Extract structural features for a node.
    These capture its position in the signed network.
    """
    po  = pos_out.get(node, 0)
    no  = neg_out.get(node, 0)
    pi  = pos_in.get(node, 0)
    ni  = neg_in.get(node, 0)
    tot_out = total_out.get(node, 0)
    tot_in  = total_in.get(node, 0)
    tot     = tot_out + tot_in

    # Positivity rates
    pos_out_rate = po / tot_out if tot_out > 0 else 0
    pos_in_rate  = pi / tot_in  if tot_in  > 0 else 0

    # Signed degree = positive degree - negative degree
    signed_out = po - no
    signed_in  = pi - ni

    # Neighbour agreement: how often do neighbours agree in sign
    # approximated as ratio of positive to total edges
    overall_pos_rate = (po + pi) / tot if tot > 0 else 0

    return [
        po,              # positive out-degree
        no,              # negative out-degree
        pi,              # positive in-degree
        ni,              # negative in-degree
        tot_out,         # total out-degree
        tot_in,          # total in-degree
        signed_out,      # signed out-degree
        signed_in,       # signed in-degree
        pos_out_rate,    # fraction of outgoing that are positive
        pos_in_rate,     # fraction of incoming that are positive
        overall_pos_rate # overall positivity rate
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
    features = get_features(node)
    label    = remapped_partition.get(node, 0)
    X_data.append(features)
    y_data.append(label)
    nodes_ordered.append(node)

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data, dtype=np.int32)

print(f"    Feature matrix shape : {X.shape}")
print(f"    Label distribution   :")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"      Community {u:>3} : {c:>6,} nodes  ({c/n*100:.1f}%)")

# ── 5. TRAIN LOGISTIC REGRESSION ─────────────────────────────
print("\n[5] Training Logistic Regression classifier...")

# Scale features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"    Train size : {len(X_train):,}")
print(f"    Test size  : {len(X_test):,}")

clf = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    class_weight="balanced",   # handles community size imbalance
    solver="lbfgs",
    multi_class="multinomial"
)
clf.fit(X_train, y_train)

y_pred    = clf.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
report    = classification_report(y_test, y_pred)

print(f"\n    Test Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"\n    Classification Report:")
print(report)

# Predict community for ALL nodes using the trained classifier
y_all_pred = clf.predict(X_scaled)

# ── 6. COMMUNITY QUALITY SCORING ─────────────────────────────
print("\n[6] Scoring communities...")

# Build community membership from classifier predictions
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

# ── 7. FEATURE IMPORTANCE ────────────────────────────────────
print("\n[7] Feature importance (mean absolute coefficient across classes):")
importance = np.abs(clf.coef_).mean(axis=0)
feat_imp   = sorted(zip(feature_names, importance),
                    key=lambda x: x[1], reverse=True)
for fname, imp in feat_imp:
    print(f"    {fname:<22} : {imp:.4f}")

# ── 8. SAVE OUTPUTS ──────────────────────────────────────────
print("\n[8] Saving outputs...")

# Community assignments — one row per node
assignments = pd.DataFrame({
    "node"                   : nodes_ordered,
    "label_prop_community"   : y_data,
    "predicted_community"    : y_all_pred
})

assignments.to_csv(
    os.path.join(COMMUNITY_DIR, "community_assignments.csv"), index=False
)
community_scores.to_csv(
    os.path.join(COMMUNITY_DIR, "community_scores.csv"), index=False
)

# Save classifier report
report_path = os.path.join(COMMUNITY_DIR, "classifier_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nFeature Importance (mean abs coefficient):\n")
    for fname, imp in feat_imp:
        f.write(f"  {fname:<22} : {imp:.4f}\n")

print(f"    community_assignments.csv : {len(assignments):,} nodes")
print(f"    community_scores.csv      : {len(community_scores)} communities")
print(f"    classifier_report.txt     : saved")

print(f"\n{'='*60}")
print("  >> Step 2 complete. Ready for Step 3.")
print(f"{'='*60}")