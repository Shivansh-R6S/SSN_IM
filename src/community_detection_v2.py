"""
Step 2 v2 - Two-Layer Stacked Logistic Regression Community Detection
Slashdot Signed Social Network

Layer 1: Logistic Regression on structural features trained on spectral core labels
Layer 2: Stacked Logistic Regression using Layer 1 probability outputs + structural
         features, trained on high-confidence Layer 1 predictions to refine
         uncertain community assignments

Output:
  outputs/community/community_assignments.csv
  outputs/community/community_scores.csv
  outputs/community/classifier_report.txt
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Step 2 v2 - Two-Layer Stacked Logistic Regression Community Detection")
print()

# Load
print("Loading data...")
df = pd.read_csv(FILE_PATH, sep="\t", header=None,
                 names=["from_node", "to_node", "sign"])
df = df.drop_duplicates(subset=["from_node", "to_node"]).reset_index(drop=True)

pos_df = df[df["sign"] ==  1]
neg_df = df[df["sign"] == -1]

all_nodes = sorted(set(df["from_node"]) | set(df["to_node"]))
n         = len(all_nodes)
print(f"  Nodes: {n:,}  |  Edges: {len(df):,}")

# Build signed graph
print("Building signed graph...")
G_signed = nx.DiGraph()
for _, row in df.iterrows():
    G_signed.add_edge(int(row["from_node"]), int(row["to_node"]),
                      sign=int(row["sign"]))

# Select core nodes by total degree
print("Selecting core nodes...")
SAMPLE_SIZE  = 5000
total_degree = {node: G_signed.in_degree(node) + G_signed.out_degree(node)
                for node in all_nodes}
core_nodes   = sorted(total_degree, key=total_degree.get, reverse=True)[:SAMPLE_SIZE]
core_nodes   = sorted(core_nodes)
core_index   = {node: i for i, node in enumerate(core_nodes)}
core_set     = set(core_nodes)
m            = len(core_nodes)
print(f"  Core nodes: {m:,}  |  Min degree: {total_degree[core_nodes[-1]]}  |  Max degree: {total_degree[core_nodes[0]]}")

# Build signed Laplacian on core nodes
print("Building signed Laplacian on core nodes...")
A_pos_core = lil_matrix((m, m), dtype=np.float32)
A_neg_core = lil_matrix((m, m), dtype=np.float32)

for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    if u in core_set and v in core_set:
        i = core_index[u]
        j = core_index[v]
        if s == 1:
            A_pos_core[i, j] = 1.0
            A_pos_core[j, i] = 1.0
        else:
            A_neg_core[i, j] = 1.0
            A_neg_core[j, i] = 1.0

A_pos_core  = A_pos_core.tocsr()
A_neg_core  = A_neg_core.tocsr()
degree_core = np.array((A_pos_core + A_neg_core).sum(axis=1)).flatten()
D_core      = diags(degree_core)
L_signed    = D_core - A_pos_core + A_neg_core

# Signed spectral clustering on core
print(f"\nRunning signed spectral clustering (k={N_CLUSTERS})...")
try:
    eigenvalues, eigenvectors = eigsh(
        L_signed, k=N_CLUSTERS, which="SM",
        sigma=0.01, tol=1e-2, maxiter=5000,
        ncv=min(m, max(3 * N_CLUSTERS + 1, 100))
    )
    print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
    X_core      = normalize(eigenvectors, norm="l2")
    kmeans      = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=20)
    core_labels = kmeans.fit_predict(X_core)
    print(f"  Core community sizes: {sorted(np.bincount(core_labels).tolist(), reverse=True)}")
except Exception as e:
    print(f"  Eigensolver failed: {e}. Falling back to degree-based KMeans.")
    core_features = np.array([[total_degree[node]] for node in core_nodes], dtype=np.float32)
    kmeans        = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=20)
    core_labels   = kmeans.fit_predict(core_features)

core_partition = {node: int(core_labels[i]) for i, node in enumerate(core_nodes)}

# Extract structural features for all nodes
print("\nExtracting structural features for all nodes...")
pos_out   = pos_df.groupby("from_node").size().to_dict()
neg_out   = neg_df.groupby("from_node").size().to_dict()
pos_in    = pos_df.groupby("to_node").size().to_dict()
neg_in    = neg_df.groupby("to_node").size().to_dict()
total_out = df.groupby("from_node").size().to_dict()
total_in  = df.groupby("to_node").size().to_dict()

def get_structural_features(node):
    po      = pos_out.get(node, 0)
    no      = neg_out.get(node, 0)
    pi      = pos_in.get(node, 0)
    ni      = neg_in.get(node, 0)
    tot_out = total_out.get(node, 0)
    tot_in  = total_in.get(node, 0)
    tot     = tot_out + tot_in
    return [
        po, no, pi, ni,
        tot_out, tot_in,
        po - no, pi - ni,
        po / tot_out if tot_out > 0 else 0,
        pi / tot_in  if tot_in  > 0 else 0,
        (po + pi) / tot if tot > 0 else 0,
        no / tot_out if tot_out > 0 else 0,
        ni / tot_in  if tot_in  > 0 else 0
    ]

structural_feature_names = [
    "pos_out_deg", "neg_out_deg", "pos_in_deg", "neg_in_deg",
    "total_out_deg", "total_in_deg", "signed_out_deg", "signed_in_deg",
    "pos_out_rate", "pos_in_rate", "overall_pos_rate",
    "foe_ratio", "danger_signal"
]

X_struct      = []
y_labels      = []
nodes_ordered = []

for node in all_nodes:
    X_struct.append(get_structural_features(node))
    y_labels.append(core_partition.get(node, -1))
    nodes_ordered.append(node)

X_struct = np.array(X_struct, dtype=np.float32)
y        = np.array(y_labels, dtype=np.int32)
print(f"  Structural feature matrix: {X_struct.shape}")

# LAYER 1 - Logistic Regression on structural features
print("\nLayer 1 - Training Logistic Regression on structural features...")

core_mask      = np.array([node in core_set for node in nodes_ordered])
X_core_struct  = X_struct[core_mask]
y_core         = y[core_mask]

scaler_l1      = StandardScaler()
X_core_scaled  = scaler_l1.fit_transform(X_core_struct)

X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(
    X_core_scaled, y_core,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_core
)

clf_l1 = LogisticRegression(
    max_iter=2000, random_state=RANDOM_SEED,
    class_weight="balanced", solver="lbfgs",
    multi_class="multinomial", C=1.0
)
clf_l1.fit(X_train_l1, y_train_l1)

y_pred_l1  = clf_l1.predict(X_test_l1)
acc_l1     = accuracy_score(y_test_l1, y_pred_l1)
report_l1  = classification_report(y_test_l1, y_pred_l1)

print(f"  Layer 1 Test Accuracy: {acc_l1:.4f}")
print(report_l1)

# Get Layer 1 probabilities for ALL nodes
X_all_scaled_l1  = scaler_l1.transform(X_struct)
y_proba_l1       = clf_l1.predict_proba(X_all_scaled_l1)
y_pred_all_l1    = clf_l1.predict(X_all_scaled_l1)

# Override core nodes with spectral labels
for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        y_pred_all_l1[i] = core_partition[node]

# Layer 1 confidence and entropy
l1_confidence = y_proba_l1.max(axis=1)
l1_entropy    = np.array([entropy(p + 1e-10) for p in y_proba_l1])

print(f"\n  Layer 1 confidence stats:")
print(f"    Mean confidence : {l1_confidence.mean():.4f}")
print(f"    High conf (>0.6): {(l1_confidence > 0.6).sum():,} nodes")
print(f"    Low conf (<0.4) : {(l1_confidence < 0.4).sum():,} nodes  (boundary candidates)")

# LAYER 2 - Stacked Logistic Regression
# Trained on high-confidence Layer 1 predictions
# Uses structural features + Layer 1 probability outputs + entropy as input
print("\nLayer 2 - Building stacked feature matrix...")

# Stacked features = structural features + L1 probability distribution + L1 entropy
# This gives Layer 2 visibility into Layer 1's uncertainty patterns
X_stacked = np.hstack([
    X_struct,           # 13 structural features
    y_proba_l1,         # k probability outputs from Layer 1
    l1_entropy.reshape(-1, 1),    # Layer 1 prediction entropy
    l1_confidence.reshape(-1, 1)  # Layer 1 max confidence
])

print(f"  Stacked feature matrix: {X_stacked.shape}")
print(f"  Features breakdown: {X_struct.shape[1]} structural + {y_proba_l1.shape[1]} L1 probs + 2 uncertainty = {X_stacked.shape[1]} total")

# Training set for Layer 2:
# High confidence Layer 1 predictions on non-core nodes
# + spectral labels on core nodes
# This gives Layer 2 a cleaner signal to learn from
HIGH_CONF_THRESH = 0.6

high_conf_mask = (l1_confidence > HIGH_CONF_THRESH) | core_mask
y_l2_labels    = y_pred_all_l1.copy()

# For core nodes use spectral labels as ground truth
for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        y_l2_labels[i] = core_partition[node]

X_l2_train_pool = X_stacked[high_conf_mask]
y_l2_train_pool = y_l2_labels[high_conf_mask]

print(f"\n  Layer 2 training pool: {len(X_l2_train_pool):,} nodes")
print(f"    Core nodes (spectral labels): {core_mask.sum():,}")
print(f"    High confidence non-core    : {(high_conf_mask & ~core_mask).sum():,}")

scaler_l2 = StandardScaler()
X_l2_scaled = scaler_l2.fit_transform(X_stacked)
X_l2_train_pool_scaled = scaler_l2.transform(X_l2_train_pool)

# Refit scaler on training pool for proper scaling
scaler_l2_fit = StandardScaler()
X_l2_train_pool_scaled = scaler_l2_fit.fit_transform(X_l2_train_pool)

X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(
    X_l2_train_pool_scaled, y_l2_train_pool,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_l2_train_pool
)

print(f"\nLayer 2 - Training stacked Logistic Regression...")
print(f"  Train: {len(X_train_l2):,}  |  Test: {len(X_test_l2):,}")

clf_l2 = LogisticRegression(
    max_iter=3000, random_state=RANDOM_SEED,
    class_weight="balanced", solver="lbfgs",
    multi_class="multinomial", C=0.5
)
clf_l2.fit(X_train_l2, y_train_l2)

y_pred_l2_test = clf_l2.predict(X_test_l2)
acc_l2         = accuracy_score(y_test_l2, y_pred_l2_test)
report_l2      = classification_report(y_test_l2, y_pred_l2_test)

print(f"  Layer 2 Test Accuracy: {acc_l2:.4f}")
print(report_l2)

# Layer 2 predictions for ALL nodes
X_all_l2_scaled  = scaler_l2_fit.transform(X_stacked)
y_proba_l2       = clf_l2.predict_proba(X_all_l2_scaled)
y_pred_all_l2    = clf_l2.predict(X_all_l2_scaled)

# Core nodes always retain spectral labels
for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        y_pred_all_l2[i] = core_partition[node]

l2_confidence = y_proba_l2.max(axis=1)
l2_entropy    = np.array([entropy(p + 1e-10) for p in y_proba_l2])

print(f"\n  Layer 2 confidence stats:")
print(f"    Mean confidence : {l2_confidence.mean():.4f}")
print(f"    High conf (>0.6): {(l2_confidence > 0.6).sum():,} nodes")
print(f"    Low conf (<0.4) : {(l2_confidence < 0.4).sum():,} nodes")

# Accuracy improvement analysis
print(f"\n  Accuracy improvement: Layer 1 = {acc_l1:.4f}  →  Layer 2 = {acc_l2:.4f}  (Δ = {acc_l2-acc_l1:+.4f})")

# Boundary node detection using Layer 2 confidence
BOUNDARY_THRESH = 0.4
is_boundary     = l2_confidence < BOUNDARY_THRESH

for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        is_boundary[i] = False

print(f"\n  Final boundary nodes: {is_boundary.sum():,}  ({is_boundary.mean()*100:.1f}%)")

# Balance theory label refinement on Layer 2 predictions
print("\nRunning balance theory label refinement on Layer 2 predictions...")

adj_sign = {}
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    adj_sign.setdefault(u, {})[v] = s

node_to_idx    = {node: i for i, node in enumerate(nodes_ordered)}
refined_labels = y_pred_all_l2.copy()
changed        = 0

for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        continue
    neighbors = adj_sign.get(node, {})
    if not neighbors:
        continue
    community_votes = np.zeros(N_CLUSTERS, dtype=np.float32)
    for nbr, sign in neighbors.items():
        if nbr in node_to_idx:
            nbr_community = refined_labels[node_to_idx[nbr]]
            community_votes[nbr_community] += sign
    best_community = int(np.argmax(community_votes))
    if best_community != refined_labels[i]:
        refined_labels[i] = best_community
        changed += 1

print(f"  Nodes reassigned during refinement: {changed:,}")

# Final community distribution
print("\nFinal community size distribution:")
final_communities = {}
for node, label in zip(nodes_ordered, refined_labels):
    final_communities.setdefault(int(label), set()).add(node)

for cid in sorted(final_communities.keys()):
    print(f"  Community {cid}: {len(final_communities[cid]):,} nodes")

# Community quality scoring
print("\nScoring communities...")

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
        total_int      = int_pos + int_neg
        total_bnd      = bnd_pos + bnd_neg
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

community_scores = score_communities(final_communities, G_signed)
print(community_scores.to_string(index=False))

# Feature importance for both layers
print("\nLayer 1 feature importance (structural features):")
all_feature_names = structural_feature_names
importance_l1 = np.abs(clf_l1.coef_).mean(axis=0)
feat_imp_l1   = sorted(zip(all_feature_names, importance_l1), key=lambda x: x[1], reverse=True)
for fname, imp in feat_imp_l1:
    print(f"  {fname:<22}: {imp:.4f}")

print("\nLayer 2 top stacked features (structural + L1 probabilities + uncertainty):")
l2_feature_names = (structural_feature_names +
                    [f"L1_prob_comm_{i}" for i in range(N_CLUSTERS)] +
                    ["L1_entropy", "L1_confidence"])
importance_l2    = np.abs(clf_l2.coef_).mean(axis=0)
feat_imp_l2      = sorted(zip(l2_feature_names, importance_l2), key=lambda x: x[1], reverse=True)
print("  Top 15 features:")
for fname, imp in feat_imp_l2[:15]:
    print(f"  {fname:<22}: {imp:.4f}")

# Save outputs
print("\nSaving outputs...")

assignments = pd.DataFrame({
    "node"              : nodes_ordered,
    "spectral_community": y_labels,
    "l1_community"      : y_pred_all_l1,
    "l2_community"      : y_pred_all_l2,
    "final_community"   : refined_labels,
    "l1_confidence"     : l1_confidence,
    "l2_confidence"     : l2_confidence,
    "l1_entropy"        : l1_entropy,
    "l2_entropy"        : l2_entropy,
    "is_boundary_node"  : is_boundary,
    "confidence_score"  : l2_confidence
})

assignments.to_csv(os.path.join(COMMUNITY_DIR, "community_assignments.csv"), index=False)
community_scores.to_csv(os.path.join(COMMUNITY_DIR, "community_scores.csv"), index=False)

report_path = os.path.join(COMMUNITY_DIR, "classifier_report.txt")
with open(report_path, "w") as f:
    f.write("TWO-LAYER STACKED LOGISTIC REGRESSION COMMUNITY DETECTION\n\n")
    f.write(f"Layer 1 Test Accuracy: {acc_l1:.4f}\n\n")
    f.write("Layer 1 Classification Report:\n")
    f.write(report_l1)
    f.write(f"\nLayer 2 Test Accuracy: {acc_l2:.4f}\n\n")
    f.write("Layer 2 Classification Report:\n")
    f.write(report_l2)
    f.write(f"\nAccuracy Improvement: {acc_l2 - acc_l1:+.4f}\n\n")
    f.write("Layer 1 Feature Importance:\n")
    for fname, imp in feat_imp_l1:
        f.write(f"  {fname:<22}: {imp:.4f}\n")
    f.write("\nLayer 2 Top 15 Features:\n")
    for fname, imp in feat_imp_l2[:15]:
        f.write(f"  {fname:<22}: {imp:.4f}\n")

print(f"  community_assignments.csv: {len(assignments):,} nodes")
print(f"  community_scores.csv     : {len(community_scores)} communities")
print(f"  classifier_report.txt    : saved")
