# CS5284_project

### Graph-ADA with Label Diversity Shifts

Based on our decision to adopt the TA's second approach, we've refined the plan to focus on **label diversity shifts** as the key graph structure shift. This uses small synthetic graphs (10 nodes) with increasing label variety (1 to 10 labels) to simulate shifts from high homophily (clustered, similar labels) to high heterophily (mixed, diverse labels). These controlled synthetics let us clearly show how shifts hurt GNN performance (e.g., clustering fails in mixed graphs) and how our Graph-ADA (DA alignment + graph-aware AL) fixes it by smartly labeling "cluster-rebuilding" nodes. We'll still reference real datasets (Cora/CiteSeer) for validation but prioritize synthetics for depth. This keeps the project graph-specific, measurable, and TA-aligned.

#### 1. Project Motivation
Domain Adaptation (DA) tackles distribution shifts between a labeled source domain and an unlabeled target domain. However, purely unsupervised DA often fails when the domain gap is large. Active Learning (AL) can mitigate this by selectively labeling the most informative target samples to improve performance under limited annotation budgets.

Most Active Domain Adaptation (ADA) methods focus only on feature alignment and overlook the inherent graph structures present in many datasets, such as social networks, citation networks, or molecular graphs. In GNNs, structure shifts—like changes in label diversity (from clustered similar labels to mixed diverse ones)—disrupt clustering and propagation, dropping accuracy significantly. By incorporating structural information, we can identify representative nodes in diverse/mixed areas, improving both label efficiency and adaptation accuracy. Hence, this project aims to design a Graph-based Active Domain Adaptation (Graph-ADA) framework to enhance target-domain performance, specifically addressing label diversity shifts via controlled synthetics.

#### 2. Project Description
Core problem: How to integrate graph neural networks (GNNs) and active learning to select the most beneficial target samples for annotation in domain adaptation, focusing on label diversity shifts that degrade GNN clustering.

Input: Labeled source synthetic graph (low diversity, e.g., 1-2 labels) and unlabeled target synthetic graph (high diversity, e.g., 5-10 labels). (Validation on real graphs like Cora/CiteSeer.)

Output: Adapted GNN model with improved accuracy and robustness against diversity shifts.

Datasets: 
- Primary: Synthetic graphs (10 nodes each, generated via Stochastic Block Models with varying labels: 1-label "homophilous" to 10-label "heterophilous").
- Validation: Citation networks (Cora, CiteSeer) subsampled for similar diversity levels.

Baselines: DA (DANN, MMD-based), AL (random, uncertainty, margin). We'll quantify shifts via homophily ratio (fraction of same-label edges) and measure performance drops (e.g., accuracy vs. diversity level).

#### 3. Proposed Solution
Goal: Demonstrate that graph-aware sampling outperforms random baselines in recovering GNN accuracy on high-diversity targets, showing structure exploitation improves sample efficiency under label diversity shifts.

Our Graph-ADA pipeline includes:
- **Synthetic Graph Generation**: Create 10-node graphs using Stochastic Block Models (NetworkX). Source: Low diversity (1-2 labels, high homophily ~0.8—dense same-label edges). Target: High diversity (5-10 labels, low homophily ~0.2—mixed edges). Vary from 1-label (clustered) to 10-label (fully mixed) for controlled shift testing.
  
- **Graph Feature Extraction**: Learn embeddings via GNNs (GCN, GraphSAGE, GAT). Use pseudo-labels on target to estimate diversity.

- **Domain Alignment**: Apply adversarial or MMD-based distribution matching to align homophily levels (e.g., encourage source-like clustering in target embeddings).

- **Active Selection**: Choose target nodes by uncertainty, margin, entropy, or graph-aware criteria tailored to diversity:
  - Standard: High entropy for mixed predictions.
  - Diversity-Focused: Prioritize "cluster reps" (e.g., nodes in rare/mixed labels) or bridges (high centrality in heterophilous areas). Score: Entropy + diversity penalty (e.g., label underrepresented classes first).

- **Annotation Simulation & Iterative Training**: Query labels from an oracle for selected samples (5-10% budget), incorporate into source, retrain GNN, and repeat until budget reached or homophily gap closes. Compare random vs. graph-aware to prove +15-30% accuracy gains on high-diversity targets.

We'll ablate: Performance vs. diversity level (e.g., 1 vs. 10 labels) and show AL rebuilds clusters faster.

#### 4. Project Milestones
- **Week 8**: Finalize synthetic generation setup and diversity levels; prep validation datasets.
- **Week 9**: Generate and analyze synthetics (compute homophily scores, exploratory GNN runs).
- **Week 10**: Implement and evaluate GNN + active learning strategies on synthetics.
- **Week 11**: Model fine-tuning, ablation studies (e.g., diversity impact), and validation on real datasets.
- **Week 12**: Draft report and slides (include plots of accuracy vs. label count).
- **Week 13**: Finalize report and presentation.

#### Team Contract
Task Assignment:
- **CHOU CHEN AN**: Implementation of active learning strategies (e.g., diversity-scored selection).
- **CHOU CHIH AN**: Implementation of active learning strategies (e.g., iterative loops).
- **JIANG SHIMING**: Literature review, implementation of graph DA baselines, graph-based model integration (GCN, GAT), and homophily metrics.
- **HU WANTING**: Synthetic dataset generation (Stochastic Block Models), visualization (e.g., graphs by diversity level).
- All members: Weekly meetings, contributions to final report and presentation.