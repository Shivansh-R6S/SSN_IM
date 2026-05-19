

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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Step 2 - Community Detection + Confidence Scoring + Balance Refinement")
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
print(f"  Nodes: {n:,}  |  Edges: {len(df):,}  |  Pos: {len(pos_df):,}  |  Neg: {len(neg_df):,}")

# Build signed graph
print("\nBuilding signed graph...")
G_signed = nx.DiGraph()
for _, row in df.iterrows():
    G_signed.add_edge(int(row["from_node"]), int(row["to_node"]),
                      sign=int(row["sign"]))

# Select core nodes by total degree
print("\nSelecting core nodes...")
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
print("\nBuilding signed Laplacian on core nodes...")
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

# Extract features for all nodes
print("\nExtracting node features...")
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

feature_names = [
    "pos_out_deg", "neg_out_deg", "pos_in_deg", "neg_in_deg",
    "total_out_deg", "total_in_deg", "signed_out_deg", "signed_in_deg",
    "pos_out_rate", "pos_in_rate", "overall_pos_rate",
    "foe_ratio", "danger_signal"
]

X_data, y_data, nodes_ordered = [], [], []
for node in all_nodes:
    X_data.append(get_features(node))
    y_data.append(core_partition.get(node, -1))
    nodes_ordered.append(node)

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data, dtype=np.int32)
print(f"  Feature matrix: {X.shape}")

# Train logistic regression on core nodes
print("\nTraining Logistic Regression on core nodes...")
core_mask     = np.array([node in core_set for node in nodes_ordered])
X_core_all    = X[core_mask]
y_core_all    = y[core_mask]

scaler        = StandardScaler()
X_core_scaled = scaler.fit_transform(X_core_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_core_scaled, y_core_all,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_core_all
)

clf = LogisticRegression(
    max_iter=2000, random_state=RANDOM_SEED,
    class_weight="balanced", solver="lbfgs",
    multi_class="multinomial", C=1.0
)
clf.fit(X_train, y_train)

y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred)
print(f"  Test accuracy: {accuracy:.4f}")
print(report)

# Predict probabilities for ALL nodes
X_all_scaled = scaler.transform(X)
y_proba      = clf.predict_proba(X_all_scaled)
y_all_pred   = clf.predict(X_all_scaled)

# Trust spectral labels for core nodes
for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        y_all_pred[i] = core_partition[node]

# Confidence scoring - Idea 2
print("\nComputing confidence scores and identifying boundary nodes...")
CONFIDENCE_THRESHOLD = 0.4

confidence_scores = y_proba.max(axis=1)
is_boundary       = confidence_scores < CONFIDENCE_THRESHOLD

# For core nodes override boundary flag - they have confirmed labels
for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        is_boundary[i] = False

n_boundary = is_boundary.sum()
print(f"  Boundary nodes identified: {n_boundary:,}  ({n_boundary/n*100:.1f}% of all nodes)")
print(f"  High confidence nodes    : {(~is_boundary).sum():,}  ({(~is_boundary).sum()/n*100:.1f}% of all nodes)")

# Balance theory refinement - Idea 4
print("\nRunning balance theory label refinement...")

adj_sign = {}
for _, row in df.iterrows():
    u = int(row["from_node"])
    v = int(row["to_node"])
    s = int(row["sign"])
    adj_sign.setdefault(u, {})[v] = s

node_to_idx   = {node: i for i, node in enumerate(nodes_ordered)}
refined_labels = y_all_pred.copy()
changed        = 0

for i, node in enumerate(nodes_ordered):
    if node in core_partition:
        continue

    neighbors = adj_sign.get(node, {})
    if not neighbors:
        continue

    # Count votes per community weighted by sign
    # Positive neighbour in community X votes for X
    # Negative neighbour in community X votes against X
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

# Final community assignments
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

# Feature importance
print("\nFeature importance:")
importance = np.abs(clf.coef_).mean(axis=0)
feat_imp   = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
for fname, imp in feat_imp:
    print(f"  {fname:<22}: {imp:.4f}")

# Save outputs
print("\nSaving outputs...")
assignments = pd.DataFrame({
    "node"              : nodes_ordered,
    "spectral_community": y_data,
    "final_community"   : refined_labels,
    "confidence_score"  : confidence_scores,
    "is_boundary_node"  : is_boundary
})

assignments.to_csv(os.path.join(COMMUNITY_DIR, "community_assignments.csv"), index=False)
community_scores.to_csv(os.path.join(COMMUNITY_DIR, "community_scores.csv"), index=False)

report_path = os.path.join(COMMUNITY_DIR, "classifier_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nFeature Importance:\n")
    for fname, imp in feat_imp:
        f.write(f"  {fname:<22}: {imp:.4f}\n")

print(f"  community_assignments.csv: {len(assignments):,} nodes")
print(f"  community_scores.csv     : {len(community_scores)} communities")
print(f"  classifier_report.txt    : saved")
