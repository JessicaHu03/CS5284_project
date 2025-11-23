"""
Data exploration and visualization utilities for the Cora dataset without DGL.

This script downloads the raw Cora citation network, parses the files directly,
computes exploratory statistics, and generates README-ready visualizations.
"""

import json
import os
import random
import shutil
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

CONTENT_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/cora/cora.content"
CITES_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/cora/cora.cites"
DEFAULT_DATA_DIR = "data"
OUTPUT_DIR = "output_cora_exploration"


def _download_with_headers(url: str, destination: str) -> None:
    """Download file with a browser-like user agent to avoid 403 errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def download_cora_raw(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Download the raw Cora files directly (no DGL dependency)."""
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, "cora")
    content_file = os.path.join(extract_path, "cora.content")
    cites_file = os.path.join(extract_path, "cora.cites")

    if os.path.exists(content_file) and os.path.exists(cites_file):
        return extract_path

    os.makedirs(extract_path, exist_ok=True)
    print("Downloading Cora dataset (cora.content & cora.cites) ...")
    try:
        _download_with_headers(CONTENT_URL, content_file)
        _download_with_headers(CITES_URL, cites_file)
    except Exception as err:
        raise RuntimeError(
            "Automatic download failed. Please download cora.content and "
            "cora.cites manually and place them under data/cora/."
        ) from err
    return extract_path


def load_cora_without_dgl(
    data_dir: str = DEFAULT_DATA_DIR,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray, Dict[str, int]]:
    """Parse the Cora dataset directly from cora.content and cora.cites."""
    cora_path = download_cora_raw(data_dir)
    content_file = os.path.join(cora_path, "cora.content")
    cites_file = os.path.join(cora_path, "cora.cites")

    node_ids: List[str] = []
    features: List[List[float]] = []
    label_strings: List[str] = []

    print("Parsing cora.content ...")
    with open(content_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            node_ids.append(parts[0])
            features.append([float(x) for x in parts[1:-1]])
            label_strings.append(parts[-1])

    unique_labels = sorted(set(label_strings))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_idx[label] for label in label_strings], dtype=np.int32)
    features = np.array(features, dtype=np.float32)

    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    print("Parsing cora.cites ...")
    edges: List[Tuple[int, int]] = []
    with open(cites_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cited, citing = parts[0], parts[1]
            if cited in node_id_to_idx and citing in node_id_to_idx:
                u = node_id_to_idx[cited]
                v = node_id_to_idx[citing]
                if u != v:
                    edges.append((u, v))

    graph = nx.Graph()
    for idx, node_id in enumerate(node_ids):
        graph.add_node(idx, paper_id=node_id, label=int(labels[idx]))
    graph.add_edges_from(edges)

    print(
        f"Loaded Cora without DGL: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {features.shape[1]}-dim features, "
        f"{len(unique_labels)} classes."
    )
    return graph, labels, features, label_to_idx


def calculate_homophily_ratio(graph: nx.Graph, labels: np.ndarray) -> float:
    """Fraction of edges connecting nodes with identical labels."""
    if graph.number_of_edges() == 0:
        return 0.0
    same_label_edges = sum(
        1 for u, v in graph.edges() if labels[u] == labels[v]
    )
    return same_label_edges / graph.number_of_edges()


def class_homophily(graph: nx.Graph, labels: np.ndarray) -> Dict[int, float]:
    """Class-wise homophily ratio."""
    same_count = Counter()
    total_count = Counter()
    for u, v in graph.edges():
        total_count[labels[u]] += 1
        total_count[labels[v]] += 1
        if labels[u] == labels[v]:
            same_count[labels[u]] += 1
            same_count[labels[v]] += 1

    ratios = {}
    for cls in total_count:
        ratios[cls] = same_count[cls] / total_count[cls] if total_count[cls] else 0.0
    return ratios


def plot_label_distribution(labels: np.ndarray, label_to_idx: Dict[str, int], save_path: str) -> None:
    """Plot the distribution of labels."""
    sns.set_theme(style="whitegrid")
    counts = Counter(labels)
    ordered_classes = sorted(counts.keys())
    heights = [counts[idx] for idx in ordered_classes]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[list(label_to_idx.keys())[cls] for cls in ordered_classes], y=heights, palette="Set3")
    plt.xlabel("Class")
    plt.ylabel("Number of Papers")
    plt.title("Cora Label Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_degree_distribution(graph: nx.Graph, save_path: str) -> None:
    """Plot the degree distribution on a log scale."""
    degrees = [deg for _, deg in graph.degree()]
    plt.figure(figsize=(8, 5))
    sns.histplot(degrees, bins=50, log_scale=(False, True), color="#4C72B0")
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency (log scale)")
    plt.title("Cora Degree Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_class_homophily(
    homophily_per_class: Dict[int, float],
    label_to_idx: Dict[str, int],
    save_path: str,
) -> None:
    """Visualize homophily per class."""
    class_names = list(label_to_idx.keys())
    ordered = sorted(homophily_per_class.items(), key=lambda kv: kv[0])
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=[class_names[idx] for idx, _ in ordered],
        y=[ratio for _, ratio in ordered],
        palette="viridis",
    )
    plt.ylim(0, 1)
    plt.ylabel("Homophily Ratio")
    plt.xlabel("Class")
    plt.title("Class-wise Homophily in Cora")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def visualize_graph_sample(
    graph: nx.Graph,
    labels: np.ndarray,
    label_to_idx: Dict[str, int],
    save_path: str,
    max_nodes: int = 400,
) -> None:
    """Visualize a sampled subgraph."""
    if graph.number_of_nodes() > max_nodes:
        sampled_nodes = set(random.sample(list(graph.nodes()), max_nodes))
        subgraph = graph.subgraph(sampled_nodes).copy()
    else:
        subgraph = graph

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.15)
    class_names = list(label_to_idx.keys())
    unique_labels = sorted(set(labels[list(subgraph.nodes())]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    color_map = {cls: colors[idx] for idx, cls in enumerate(range(len(class_names)))}
    node_colors = [color_map[labels[node]] for node in subgraph.nodes()]

    nx.draw_networkx_nodes(
        subgraph, pos, node_color=node_colors, node_size=50, alpha=0.9
    )
    nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.4)
    plt.title("Cora Citation Network (Sampled Subgraph)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def summarize_graph(graph: nx.Graph, labels: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for the graph."""
    degrees = [deg for _, deg in graph.degree()]
    avg_degree = float(np.mean(degrees))
    median_degree = float(np.median(degrees))
    homophily = calculate_homophily_ratio(graph, labels)
    clustering = float(nx.average_clustering(graph))

    result = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "average_degree": avg_degree,
        "median_degree": median_degree,
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "homophily_ratio": homophily,
        "average_clustering": clustering,
        "density": nx.density(graph),
        "is_connected": nx.is_connected(graph),
    }
    if result["is_connected"]:
        result["avg_shortest_path_length"] = float(nx.average_shortest_path_length(graph))
    else:
        largest_component = graph.subgraph(max(nx.connected_components(graph), key=len))
        result["avg_shortest_path_length_lcc"] = float(
            nx.average_shortest_path_length(largest_component)
        )
    return result


def run_exploration(data_dir: str = DEFAULT_DATA_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Main entry point to run the data exploration pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    graph, labels, features, label_to_idx = load_cora_without_dgl(data_dir)

    summary = summarize_graph(graph, labels)
    class_hom = class_homophily(graph, labels)
    summary["class_homophily"] = {
        list(label_to_idx.keys())[cls]: float(ratio) for cls, ratio in class_hom.items()
    }
    summary["num_features"] = int(features.shape[1])
    summary["num_classes"] = len(label_to_idx)

    summary_path = os.path.join(output_dir, "cora_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved summary statistics to {summary_path}")

    plot_label_distribution(
        labels,
        label_to_idx,
        os.path.join(output_dir, "label_distribution.png"),
    )
    plot_degree_distribution(
        graph,
        os.path.join(output_dir, "degree_distribution.png"),
    )
    plot_class_homophily(
        class_hom,
        label_to_idx,
        os.path.join(output_dir, "class_homophily.png"),
    )
    visualize_graph_sample(
        graph,
        labels,
        label_to_idx,
        os.path.join(output_dir, "graph_sample.png"),
    )

    print("Exploration complete. Artifacts saved to:", os.path.abspath(output_dir))


if __name__ == "__main__":
    run_exploration()

