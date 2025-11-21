"""
Cora Dataset Generator for Graph-ADA with Label Diversity Shifts
Author: HU WANTING
Purpose: Generate graphs with varying homophily from the Cora citation dataset
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import warnings
import urllib.request
import zipfile
import shutil
warnings.filterwarnings('ignore')

try:
    import dgl
    import torch
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("Warning: DGL not available. Please install with: pip install dgl torch")


class CoraDatasetGenerator:
    """
    Generate graphs with varying homophily levels from the Cora citation dataset.
    
    The generator modifies the original Cora graph structure to simulate shifts from
    high homophily (clustered, similar labels) to high heterophily (mixed, diverse labels).
    """
    
    def __init__(self, data_dir: str = "data", random_seed: int = 42):
        """
        Initialize the Cora dataset generator.
        
        Args:
            data_dir: Directory to store/download the Cora dataset
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_seed = random_seed
        np.random.seed(random_seed)
        if DGL_AVAILABLE:
            torch.manual_seed(random_seed)
        
        self.original_graph = None
        self.original_labels = None
        self.original_features = None
        
    def download_cora(self) -> None:
        """
        Download the Cora dataset if not already present.
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required. Install with: pip install dgl torch")
        
        cora_url = "https://data.dgl.ai/datasets/cora.zip"
        zip_path = os.path.join(self.data_dir, "cora.zip")
        extract_path = os.path.join(self.data_dir, "cora")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Download if not exists
        if not os.path.exists(extract_path):
            print(f"Downloading Cora dataset from {cora_url}...")
            urllib.request.urlretrieve(cora_url, zip_path)
            
            print(f"Extracting to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip file
            os.remove(zip_path)
            print("Download complete!")
        else:
            print(f"Cora dataset already exists at {extract_path}")
    
    def load_cora(self) -> Tuple[nx.Graph, np.ndarray, Optional[np.ndarray]]:
        """
        Load the Cora dataset and convert to NetworkX format.
        
        Returns:
            Tuple of (graph, node_labels, node_features)
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required. Install with: pip install dgl torch")
        
        # Download if needed
        self.download_cora()
        
        # Load using DGL
        dataset = dgl.data.CoraGraphDataset(raw_dir=self.data_dir)
        graph_dgl = dataset[0]
        
        # Extract labels and features
        labels = graph_dgl.ndata['label'].numpy()
        features = graph_dgl.ndata['feat'].numpy()
        
        # Convert to NetworkX (undirected)
        graph_nx = dgl.to_networkx(graph_dgl, edge_attrs=None, node_attrs=None)
        graph_nx = graph_nx.to_undirected()
        
        # Store original data
        self.original_graph = graph_nx.copy()
        self.original_labels = labels.copy()
        self.original_features = features.copy()
        
        # Add node attributes
        for i, node in enumerate(graph_nx.nodes()):
            graph_nx.nodes[node]['label'] = int(labels[i])
            graph_nx.nodes[node]['original_label'] = int(labels[i])
            if features is not None:
                graph_nx.nodes[node]['features'] = features[i]
        
        print(f"Loaded Cora dataset:")
        print(f"  - Nodes: {graph_nx.number_of_nodes()}")
        print(f"  - Edges: {graph_nx.number_of_edges()}")
        print(f"  - Labels: {len(np.unique(labels))} unique classes")
        print(f"  - Features: {features.shape[1] if features is not None else 0} dimensions")
        
        return graph_nx, labels, features
    
    def calculate_homophily_ratio(self, graph: nx.Graph) -> float:
        """
        Calculate the homophily ratio: fraction of edges connecting nodes with same labels.
        
        Args:
            graph: NetworkX graph with node labels
            
        Returns:
            Homophily ratio (0.0-1.0)
        """
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            return 0.0
            
        same_label_edges = 0
        for edge in graph.edges():
            node1, node2 = edge
            if graph.nodes[node1]['label'] == graph.nodes[node2]['label']:
                same_label_edges += 1
                
        return same_label_edges / total_edges
    
    def modify_homophily(self, 
                        graph: nx.Graph, 
                        target_homophily: float,
                        method: str = "rewire") -> nx.Graph:
        """
        Modify the graph structure to achieve a target homophily level.
        
        Args:
            graph: Input NetworkX graph
            target_homophily: Target homophily ratio (0.0-1.0)
            method: Method to use ('rewire' or 'add_remove')
            
        Returns:
            Modified graph with target homophily level
        """
        current_homophily = self.calculate_homophily_ratio(graph)
        
        if abs(current_homophily - target_homophily) < 0.01:
            return graph.copy()
        
        modified_graph = graph.copy()
        
        if method == "rewire":
            modified_graph = self._rewire_edges(modified_graph, target_homophily)
        elif method == "add_remove":
            modified_graph = self._add_remove_edges(modified_graph, target_homophily)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rewire' or 'add_remove'")
        
        return modified_graph
    
    def _rewire_edges(self, graph: nx.Graph, target_homophily: float) -> nx.Graph:
        """
        Rewire edges to achieve target homophily by swapping edge endpoints.
        
        Args:
            graph: Input graph
            target_homophily: Target homophily ratio
            
        Returns:
            Rewired graph
        """
        current_homophily = self.calculate_homophily_ratio(graph)
        edges = list(graph.edges())
        
        # Separate edges into same-label and different-label
        same_label_edges = []
        diff_label_edges = []
        
        for edge in edges:
            node1, node2 = edge
            if graph.nodes[node1]['label'] == graph.nodes[node2]['label']:
                same_label_edges.append(edge)
            else:
                diff_label_edges.append(edge)
        
        total_edges = len(edges)
        target_same_label = int(target_homophily * total_edges)
        current_same_label = len(same_label_edges)
        
        # Calculate how many edges to rewire
        if target_same_label > current_same_label:
            # Need more same-label edges: rewire diff-label edges
            num_to_rewire = target_same_label - current_same_label
            num_to_rewire = min(num_to_rewire, len(diff_label_edges))
            
            # Remove diff-label edges and add same-label edges
            edges_to_remove = diff_label_edges[:num_to_rewire]
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            
            # Add same-label edges by finding nodes with same labels
            nodes_by_label = {}
            for node in graph.nodes():
                label = graph.nodes[node]['label']
                if label not in nodes_by_label:
                    nodes_by_label[label] = []
                nodes_by_label[label].append(node)
            
            added = 0
            for label, nodes in nodes_by_label.items():
                if len(nodes) >= 2:
                    # Try to add edges between nodes with same label
                    for i in range(len(nodes)):
                        if added >= num_to_rewire:
                            break
                        for j in range(i + 1, len(nodes)):
                            if added >= num_to_rewire:
                                break
                            if not graph.has_edge(nodes[i], nodes[j]):
                                graph.add_edge(nodes[i], nodes[j])
                                added += 1
                        if added >= num_to_rewire:
                            break
                    if added >= num_to_rewire:
                        break
            
            # If we couldn't add enough, add random edges to maintain edge count
            while added < num_to_rewire:
                # Find a random same-label pair
                label = np.random.choice(list(nodes_by_label.keys()))
                nodes = nodes_by_label[label]
                if len(nodes) >= 2:
                    n1, n2 = np.random.choice(nodes, 2, replace=False)
                    if not graph.has_edge(n1, n2):
                        graph.add_edge(n1, n2)
                        added += 1
                else:
                    break
                    
        else:
            # Need fewer same-label edges: rewire same-label edges
            num_to_rewire = current_same_label - target_same_label
            num_to_rewire = min(num_to_rewire, len(same_label_edges))
            
            # Remove same-label edges
            edges_to_remove = same_label_edges[:num_to_rewire]
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            
            # Add diff-label edges
            nodes_by_label = {}
            for node in graph.nodes():
                label = graph.nodes[node]['label']
                if label not in nodes_by_label:
                    nodes_by_label[label] = []
                nodes_by_label[label].append(node)
            
            added = 0
            labels_list = list(nodes_by_label.keys())
            max_iterations = num_to_rewire * 10  # Prevent infinite loop
            
            while added < num_to_rewire and max_iterations > 0:
                max_iterations -= 1
                label1, label2 = np.random.choice(labels_list, 2, replace=False)
                if label1 != label2:
                    n1 = np.random.choice(nodes_by_label[label1])
                    n2 = np.random.choice(nodes_by_label[label2])
                    if not graph.has_edge(n1, n2):
                        graph.add_edge(n1, n2)
                        added += 1
        
        return graph
    
    def _add_remove_edges(self, graph: nx.Graph, target_homophily: float) -> nx.Graph:
        """
        Add/remove edges to achieve target homophily.
        
        Args:
            graph: Input graph
            target_homophily: Target homophily ratio
            
        Returns:
            Modified graph
        """
        # Similar to rewire but simpler: just add/remove edges
        return self._rewire_edges(graph, target_homophily)
    
    def calculate_graph_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Calculate various graph metrics for analysis.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        metrics['homophily_ratio'] = self.calculate_homophily_ratio(graph)
        
        # Connectivity metrics
        if nx.is_connected(graph):
            metrics['diameter'] = nx.diameter(graph)
            metrics['average_path_length'] = nx.average_shortest_path_length(graph)
        else:
            metrics['diameter'] = float('inf')
            metrics['average_path_length'] = float('inf')
        
        # Centrality metrics
        degree_centrality = nx.degree_centrality(graph)
        metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        
        # Clustering
        metrics['clustering_coefficient'] = nx.average_clustering(graph)
        
        return metrics
    
    def generate_homophily_series(self, 
                                  homophily_levels: List[float],
                                  method: str = "rewire") -> Dict:
        """
        Generate a series of graphs with varying homophily levels from Cora.
        
        Args:
            homophily_levels: List of target homophily levels (0.0-1.0)
            method: Method to modify homophily ('rewire' or 'add_remove')
            
        Returns:
            Dictionary containing graphs and metrics organized by homophily level
        """
        # Load original Cora if not already loaded
        if self.original_graph is None:
            self.load_cora()
        
        homophily_series = {}
        
        for target_homophily in homophily_levels:
            # Modify graph to achieve target homophily
            modified_graph = self.modify_homophily(
                self.original_graph.copy(),
                target_homophily,
                method=method
            )
            
            # Calculate metrics
            metrics = self.calculate_graph_metrics(modified_graph)
            actual_homophily = self.calculate_homophily_ratio(modified_graph)
            
            homophily_series[target_homophily] = {
                'graph': modified_graph,
                'metrics': metrics,
                'target_homophily': target_homophily,
                'actual_homophily': actual_homophily
            }
            
            print(f"Target homophily: {target_homophily:.2f}, "
                  f"Actual: {actual_homophily:.3f}, "
                  f"Edges: {modified_graph.number_of_edges()}")
        
        return homophily_series
    
    def generate_source_target_pairs(self, 
                                    num_pairs: int = 10,
                                    source_homophily: float = 0.8,
                                    target_homophily: float = 0.3) -> List[Tuple[nx.Graph, nx.Graph, np.ndarray, np.ndarray]]:
        """
        Generate source-target graph pairs for domain adaptation experiments.
        
        Source graphs: High homophily (clustered, similar labels)
        Target graphs: Low homophily (mixed, diverse labels)
        
        Args:
            num_pairs: Number of source-target pairs to generate
            source_homophily: Homophily level for source graphs
            target_homophily: Homophily level for target graphs
            
        Returns:
            List of tuples: (source_graph, target_graph, source_labels, target_labels)
        """
        # Load original Cora if not already loaded
        if self.original_graph is None:
            self.load_cora()
        
        pairs = []
        
        for _ in range(num_pairs):
            # Generate source graph (high homophily)
            source_graph = self.modify_homophily(
                self.original_graph.copy(),
                source_homophily,
                method="rewire"
            )
            source_labels = np.array([source_graph.nodes[node]['label'] for node in source_graph.nodes()])
            
            # Generate target graph (low homophily)
            target_graph = self.modify_homophily(
                self.original_graph.copy(),
                target_homophily,
                method="rewire"
            )
            target_labels = np.array([target_graph.nodes[node]['label'] for node in target_graph.nodes()])
            
            pairs.append((source_graph, target_graph, source_labels, target_labels))
        
        return pairs


def visualize_graph(graph: nx.Graph, 
                   title: str = "Cora Graph", 
                   save_path: Optional[str] = None,
                   max_nodes: int = 500) -> None:
    """
    Visualize a graph with node colors representing labels.
    
    Args:
        graph: NetworkX graph to visualize
        title: Title for the plot
        save_path: Optional path to save the figure
        max_nodes: Maximum number of nodes to visualize (for large graphs)
    """
    # For large graphs, sample nodes
    if graph.number_of_nodes() > max_nodes:
        nodes_to_plot = np.random.choice(list(graph.nodes()), max_nodes, replace=False)
        subgraph = graph.subgraph(nodes_to_plot)
    else:
        subgraph = graph
    
    plt.figure(figsize=(12, 10))
    
    # Get node labels and positions
    labels = [subgraph.nodes[node]['label'] for node in subgraph.nodes()]
    pos = nx.spring_layout(subgraph, seed=42, k=0.5, iterations=50)
    
    # Create color map
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    node_colors = [color_map[label] for label in labels]
    
    # Draw graph
    nx.draw(subgraph, pos, 
            node_color=node_colors,
            node_size=50,
            with_labels=False,
            edge_color='gray',
            alpha=0.6,
            width=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Legend
    legend_elements = [plt.scatter([], [], c=colors[i], s=100, label=f'Class {unique_labels[i]}') 
                      for i in range(len(unique_labels))]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_homophily_analysis(homophily_series: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot analysis of graph homophily vs. metrics.
    
    Args:
        homophily_series: Results from generate_homophily_series()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    target_homophily = sorted(homophily_series.keys())
    actual_homophily = [homophily_series[k]['actual_homophily'] for k in target_homophily]
    density = [homophily_series[k]['metrics']['density'] for k in target_homophily]
    clustering = [homophily_series[k]['metrics']['clustering_coefficient'] for k in target_homophily]
    num_edges = [homophily_series[k]['metrics']['num_edges'] for k in target_homophily]
    
    # Plot 1: Target vs Actual Homophily
    axes[0, 0].plot(target_homophily, actual_homophily, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Match')
    axes[0, 0].set_xlabel('Target Homophily', fontsize=12)
    axes[0, 0].set_ylabel('Actual Homophily', fontsize=12)
    axes[0, 0].set_title('Target vs. Actual Homophily', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Graph Density vs. Homophily
    axes[0, 1].plot(target_homophily, density, 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Target Homophily', fontsize=12)
    axes[0, 1].set_ylabel('Graph Density', fontsize=12)
    axes[0, 1].set_title('Graph Density vs. Homophily', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Clustering Coefficient vs. Homophily
    axes[1, 0].plot(target_homophily, clustering, '^-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_xlabel('Target Homophily', fontsize=12)
    axes[1, 0].set_ylabel('Clustering Coefficient', fontsize=12)
    axes[1, 0].set_title('Clustering vs. Homophily', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of Edges vs. Homophily
    axes[1, 1].plot(target_homophily, num_edges, 'd-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Target Homophily', fontsize=12)
    axes[1, 1].set_ylabel('Number of Edges', fontsize=12)
    axes[1, 1].set_title('Edge Count vs. Homophily', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """
    Demonstrate the Cora dataset generator with various examples.
    """
    if not DGL_AVAILABLE:
        print("Error: DGL is required. Please install with:")
        print("  pip install dgl torch")
        return
    
    # Create output directory
    output_dir = "output_cora"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize the generator
    generator = CoraDatasetGenerator(data_dir="data", random_seed=42)
    
    # Load Cora dataset
    print("\n=== Loading Cora Dataset ===")
    original_graph, labels, features = generator.load_cora()
    
    # Calculate original homophily
    original_homophily = generator.calculate_homophily_ratio(original_graph)
    original_metrics = generator.calculate_graph_metrics(original_graph)
    
    print(f"\nOriginal Cora Graph:")
    print(f"  - Nodes: {original_graph.number_of_nodes()}")
    print(f"  - Edges: {original_graph.number_of_edges()}")
    print(f"  - Homophily ratio: {original_homophily:.3f}")
    print(f"  - Density: {original_metrics['density']:.3f}")
    print(f"  - Clustering coefficient: {original_metrics['clustering_coefficient']:.3f}")
    
    # Generate graphs with different homophily levels
    print("\n=== Generating Graphs with Different Homophily Levels ===")
    homophily_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
    homophily_series = generator.generate_homophily_series(
        homophily_levels=homophily_levels,
        method="rewire"
    )
    
    # Visualize different homophily levels
    print("\n=== Visualizing Graphs ===")
    for target_homophily in [0.9, 0.5, 0.1]:
        if target_homophily in homophily_series:
            graph = homophily_series[target_homophily]['graph']
            actual_homophily = homophily_series[target_homophily]['actual_homophily']
            save_path = os.path.join(output_dir, f"cora_homophily_{target_homophily:.1f}.png")
            visualize_graph(
                graph,
                title=f"Cora Graph with Homophily={actual_homophily:.3f} (Target: {target_homophily:.1f})",
                save_path=save_path,
                max_nodes=500
            )
            print(f"Saved graph visualization: {save_path}")
    
    # Plot comprehensive analysis
    print("\n=== Generating Analysis Plots ===")
    analysis_path = os.path.join(output_dir, "cora_homophily_analysis.png")
    plot_homophily_analysis(homophily_series, save_path=analysis_path)
    print(f"Saved homophily analysis: {analysis_path}")
    
    # Print summary statistics
    print("\n=== Homophily Series Summary ===")
    for target_homophily in sorted(homophily_series.keys()):
        actual = homophily_series[target_homophily]['actual_homophily']
        metrics = homophily_series[target_homophily]['metrics']
        print(f"  Target: {target_homophily:.2f}, Actual: {actual:.3f}, "
              f"Edges: {metrics['num_edges']}, Density: {metrics['density']:.3f}")
    
    # Source-Target pairs for domain adaptation
    print("\n=== Generating Source-Target Pairs ===")
    pairs = generator.generate_source_target_pairs(
        num_pairs=5,
        source_homophily=0.8,
        target_homophily=0.3
    )
    
    print("Generated source-target pairs:")
    for i, (source_graph, target_graph, source_labels, target_labels) in enumerate(pairs):
        source_homophily = generator.calculate_homophily_ratio(source_graph)
        target_homophily = generator.calculate_homophily_ratio(target_graph)
        
        print(f"Pair {i+1}:")
        print(f"  Source: homophily={source_homophily:.3f}, "
              f"nodes={source_graph.number_of_nodes()}, edges={source_graph.number_of_edges()}")
        print(f"  Target: homophily={target_homophily:.3f}, "
              f"nodes={target_graph.number_of_nodes()}, edges={target_graph.number_of_edges()}")
        print(f"  Homophily shift: {source_homophily - target_homophily:.3f}")
        print()
    
    print("\n=== Demo Complete ===")
    print("All generated files saved in output_cora/ directory")


if __name__ == "__main__":
    main()

