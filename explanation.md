# Signed Attenuated Propagation — Pipeline Explanation

## Overview

This document explains every step of the influence maximisation pipeline built for the Slashdot signed social network. Each section covers what the step does, why it was designed that way, and what its output means for the next step.

---

## Dataset

**Slashdot Signed Social Network**
- 77,350 nodes (users)
- 516,575 directed edges
- Sign +1 = friend tag, Sign -1 = foe tag
- 76.7% positive edges, 23.3% negative edges

The dataset is directed — User A tagging User B as a friend does not imply the reverse. Users explicitly chose friend or foe, making the signs reliable and intentional rather than inferred.

---

## Step 1 — Exploratory Data Analysis

**What it does**

Loads the raw dataset, validates format, and computes eight structural properties of the network: basic counts, degree distributions, top hub nodes, reciprocity, triangle balance ratios, and connected components.

**Key findings**

- Zero duplicates and zero missing values — the file is clean
- 90.5% of sampled triangles are balanced under Heider's balance theory — strong community structure exists
- 96.1% of mutual edges agree on sign — the network is socially consistent
- 100% of nodes are in one weakly connected component — seeds can potentially reach the entire network
- Power law degree distribution — a small number of hub nodes are extremely well connected

**Significance**

The high triangle balance ratio (90.5%) validates our theoretical foundation. A highly balanced network means communities with positive internal edges and negative external edges naturally exist — exactly the structure our community detection and seed selection exploit. The single connected component means no nodes are isolated from influence propagation.

---

## Step 2 — Community Detection with Confidence Scoring and Balance Refinement

**What it does**

Partitions 77,350 nodes into 10 communities using three sequential mechanisms:

1. **Signed Spectral Clustering on core nodes** — Takes the 5,000 most connected nodes, builds a signed Laplacian matrix (L = D − A+ + A−), computes its 10 smallest eigenvectors, and runs KMeans. The signed Laplacian simultaneously attracts positively connected nodes and repels negatively connected nodes — directly encoding balance theory into the clustering objective.

2. **Logistic Regression for remaining nodes** — Trains on the 5,000 spectral-clustered core nodes using 13 structural features per node (positive/negative degree, signed degree, positivity rates, foe ratio, danger signal). Predicts community membership for the remaining 72,350 nodes.

3. **Confidence scoring and boundary node detection** — Uses the logistic regression probability output. Nodes where the highest community probability is below 0.4 are flagged as boundary nodes — they sit between communities and are exposed to cross-community negative edges.

4. **Balance theory label refinement** — Each non-core node examines its signed neighbourhood. Positive neighbours vote for their community, negative neighbours vote against theirs. Nodes are reassigned if the vote suggests a different community. This encodes the balance theory principle directly as a post-processing step.

**Community quality scoring**

Each community is scored on three dimensions multiplied together:

- **Internal positivity** = positive internal edges / total internal edges
- **Insulation** = 1 − (negative boundary edges / total boundary edges)  
- **Quality score** = size × internal positivity × insulation

**Final community quality scores**

| Community | Size | Internal Positivity | Insulation | Quality Score |
|---|---|---|---|---|
| 7 | 33,961 | 92.0% | 67.7% | 21,157 |
| 1 | 19,043 | 68.7% | 72.4% | 9,476 |
| 0 | 10,652 | 95.1% | 66.5% | 6,741 |
| 2 | 5,426 | 89.2% | 74.7% | 3,613 |
| 6 | 2,402 | 86.7% | 76.5% | 1,593 |
| 3 | 1,768 | 90.3% | 75.5% | 1,205 |
| 4 | 1,404 | 90.4% | 76.1% | 967 |
| 8 | 1,124 | 96.4% | 82.8% | 898 |
| 5 | 993 | 86.7% | 76.8% | 661 |
| 9 | 577 | 90.1% | 76.9% | 404 |

**Significance**

Communities are the structural foundation for seed allocation. High internal positivity means SAP influence flows cleanly within a community with minimal score subtraction. High insulation means external negative edges don't erode internally propagated influence. The boundary node flag is used in Step 4 to exclude nodes that are structurally vulnerable to cross-community counter-influence.

**Output:** `outputs/community/community_assignments.csv`, `outputs/community/community_scores.csv`

---

## Step 3 — Node Scoring

**What it does**

Computes four scores for every node using PageRank-weighted neighbourhood analysis:

**PageRank** — Computed on the full signed directed graph. Used as an influence weight for neighbours — a node connected to high-PageRank neighbours has more opportunity and more danger than a node connected to low-PageRank neighbours.

**Opportunity score** — Sum of PageRank values of all positive out-neighbours. Measures how much positive influence this node can spread weighted by how influential those friends are.

**Danger score** — Sum of PageRank values of all negative out-neighbours. Measures how much counter-influence this node could accidentally mobilise, weighted by how influential those foes are.

**Enemy-of-enemy bonus** — For each foe of the node, counts the PageRank of that foe's own foes (excluding the original node and nodes already in the foe set). This captures the divide-and-conquer principle — foes who fight each other weaken the opposition rather than unifying it against the seed.

**Seed score** — The final ranking score combining all three:

```
seed_score = opportunity + 0.1 × eoe_bonus − 0.5 × danger
```

The additive formulation prevents zero-danger nodes from dominating through division by near-zero values, while still penalising high-danger nodes and rewarding enemy-of-enemy structural advantages.

**Significance**

Node scoring is the implementation of the risk-return tradeoff from financial portfolio theory. Every node gets a single comparable score that balances its positive influence potential against its opposition mobilisation risk. Step 4 uses these scores to make seed selection decisions within each community.

**Output:** `outputs/scoring/node_scores.csv`

---

## Step 4 — Seed Selection

**What it does**

Selects the final 20 seed nodes using five sequential mechanisms:

**1. Boundary node exclusion** — Nodes flagged as boundary nodes in Step 2 are removed from the candidate pool. These nodes sit between communities and are exposed to negative edges from multiple directions — seeding them risks activating cross-community counter-influence.

**2. Sleeping giant filter** — Any candidate node whose foe set contains at least one node with PageRank above the threshold (0.005) is flagged as a sleeping giant risk and excluded. Seeding such a node would mobilise a highly influential hub against the campaign.

**3. Community-based proportional allocation** — The seed budget of 20 is allocated across communities as follows: every community gets 1 guaranteed seed, then the remaining 10 seeds are distributed proportionally to community quality scores. This ensures no community is left unseeded while still concentrating resources in the highest quality communities.

**4. Portfolio diversification within communities** — Within each community, seeds are selected greedily by seed score with two Jaccard overlap constraints:
- Positive reach overlap between any two seeds must be below 0.3 — prevents wasting seeds on nodes that reach the same people
- Foe set overlap between any two seeds must be below 0.3 — prevents double-mobilising the same opposition

**5. Fallback mechanisms** — If a community has no safe candidates after filtering, the script progressively relaxes constraints: first allows sleeping giant risk nodes, then allows boundary nodes, ensuring every community gets its allocated seeds.

**Final allocation**

| Community | Seeds Allocated | Quality Score |
|---|---|---|
| 7 | 6 | 21,157 |
| 1 | 4 | 9,476 |
| 0 | 3 | 6,741 |
| 2–9 | 1 each | varies |

**Portfolio diversity results**

- Total unique positive reach: 2,252 nodes
- Total unique foes mobilised: 1,469 nodes
- Seeds with danger > 0: 16 out of 20
- Seeds with eoe_bonus > 0: 16 out of 20
- All 10 communities represented

**Significance**

Seed selection is the core decision-making step of the entire pipeline. The combination of community-based allocation (borrowed from military resource allocation and sports team selection), portfolio diversification (borrowed from Markowitz portfolio theory), sleeping giant avoidance (borrowed from military strategy), and enemy-of-enemy bonus (borrowed from balance theory and Roman military strategy) produces a seed set that maximises positive reach while minimising coordinated opposition.

**Output:** `outputs/seeds/selected_seeds.csv`

---

## Step 5 — Signed Attenuated Propagation (SAP)

**What it does**

Propagates influence from the 20 seed nodes through the entire signed network using the SAP model — the novel propagation model developed for this research.

**SAP model rules**

- Every node starts with score 0
- Seed nodes are initialised with score +1.0
- When influence arrives at a node via a path with cumulative sign σ and parent weight w:
  - Attenuation weight = w × α^(arrival_count of this node)
  - Score update = score + σ × attenuation_weight
  - Arrival count incremented by 1
- Propagation stops when attenuation weight drops below 1e-6

**Sign multiplication along paths**

The cumulative sign of a path is the product of all edge signs along it:
- Positive × Positive = Positive (friend of friend)
- Negative × Negative = Positive (enemy of enemy)
- Positive × Negative = Negative (friend of enemy)
- Negative × Positive = Negative (enemy of friend)

**Attenuation**

With α = 0.5, each subsequent message arriving at the same node is halved. This is grounded in the psychological Law of Diminishing Marginal Utility and Shannon's Information Theory — repeated messages carry less new information and therefore less influence impact. The geometric series 1 + 0.5 + 0.25 + ... converges to 2, giving a natural theoretical ceiling on node scores.

**SAP results on Slashdot**

| Metric | Value |
|---|---|
| Nodes reached | 63,102 (81.6%) |
| Positively activated (score > 0) | 38,636 (49.95%) |
| Negatively influenced (score < 0) | 24,466 (31.63%) |
| Neutral | 14,248 (18.4%) |
| Metric 1 — Influence Spread | 38,636 nodes |
| Metric 2 — Net Influence Score | 16,686.84 |
| Max node score | 1.999 (near theoretical ceiling of 2.0) |
| Min node score | −1.935 (strong counter-influence) |

**Significance**

SAP is the novel theoretical contribution of this research. Unlike standard Independent Cascade or Linear Threshold models which treat all edges as positive conduits, SAP explicitly models the dual nature of signed relationships — positive edges amplify influence while negative edges flip and attenuate it. The geometric decay prevents runaway influence accumulation and models realistic human message fatigue. The near-ceiling max score of 1.999 confirms the mathematical model is behaving exactly as theorised.

**Output:** `outputs/evaluation/sap_results.csv`

---

## Step 6 — Evaluation Against Baselines

**What it does**

Runs five baseline methods on the same network with the same seed budget and compares all six approaches on both evaluation metrics.

**Baselines**

| Baseline | Seed Selection | Propagation Model | Purpose |
|---|---|---|---|
| Random + IC | Random | Standard IC | Placebo — better than nothing? |
| Degree Centrality + IC | Highest degree | Standard IC | Simple structural heuristic |
| Greedy + IC | Greedy marginal gain | Standard IC | Current state of the art |
| Greedy + SAP | Greedy marginal gain | SAP | Isolates seed selection contribution |
| Our Selection + IC | Our method | Standard IC | Isolates SAP contribution |
| **Our Selection + SAP** | **Our method** | **SAP** | **Our full approach** |

**Evaluation metrics**

- **Metric 1 — Influence Spread**: Number of nodes with final score above threshold. Measures width of influence.
- **Metric 2 — Net Influence Score**: Sum of all positive scores across activated nodes. Measures depth and strength of influence — a node with score 1.9 is far more genuinely influenced than a node with score 0.01.

**What each comparison proves**

- Beating Random + IC — we are better than no intelligence
- Beating Degree + IC — our method outperforms simple structural heuristics
- Beating Greedy + IC — our method outperforms the current research standard
- Beating Greedy + SAP — our seed selection adds value beyond SAP alone
- Beating Our Selection + IC — SAP adds value beyond our seed selection alone

**Output:** `outputs/evaluation/metrics_comparison.csv`

---

## Cross-Domain Inspirations Summary

| Component | Inspired By | Concept Borrowed |
|---|---|---|
| SAP sign multiplication | Balance Theory / Mathematics | Negative × Negative = Positive |
| SAP geometric decay | Psychology / Information Theory | Message fatigue, diminishing returns |
| Node opportunity score | Financial theory | Return on investment |
| Node danger score | Military strategy | Sleeping giant — local win, global cost |
| Seed score formula | Risk-return tradeoff | Maximise return relative to risk |
| Portfolio diversification | Markowitz Portfolio Theory | Diversified portfolio beats individual picks |
| Community allocation | Military / Sports | Proportional resource deployment |
| Enemy-of-enemy bonus | Roman military / Balance Theory | Divide and conquer |
| Boundary node exclusion | Civil engineering | Insulation from external interference |
| Evaluation metrics | Marketing science | Reach (width) vs engagement (depth) |

---

## Final Project Structure

```
SSN_IM/
├── data/
│   └── soc-sign-Slashdot081106.txt
├── src/
│   ├── config.py
│   ├── step1_explore.py
│   ├── step2_community.py
│   ├── step3_node_scoring.py
│   ├── step4_seed_selection.py
│   ├── step5_sap_propagation.py
│   └── step6_evaluation.py
├── outputs/
│   ├── community/
│   │   ├── community_assignments.csv
│   │   ├── community_scores.csv
│   │   └── classifier_report.txt
│   ├── scoring/
│   │   └── node_scores.csv
│   ├── seeds/
│   │   └── selected_seeds.csv
│   └── evaluation/
│       ├── sap_results.csv
│       └── metrics_comparison.csv
└── requirements.txt
```