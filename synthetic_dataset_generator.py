"""
Synthetic Dataset Generator for Graph-ADA with Label Diversity Shifts
Author: HU WANTING
Purpose: Generate synthetic graphs using Stochastic Block Models (SBM) with varying label diversity
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
# import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SBMDatasetGenerator:
    """
    Generate synthetic graphs using Stochastic Block Models (SBM) for Graph-ADA experiments.
    
    The generator creates graphs with controlled label diversity to simulate shifts from
    high homophily (clustered, similar labels) to high heterophily (mixed, diverse labels).
    """
    
    def __init__(self, num_nodes: int = 10, random_seed: int = 42):
        """
        Initialize the SBM dataset generator.
        
        Args:
            num_nodes: Number of nodes in each generated graph (default: 10)
            random_seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_sbm_graph(self, 
                          num_labels: int, 
                          homophily_level: float = 0.8,
                          edge_probability: float = 0.3,
                          use_advanced_calculation: bool = False) -> Tuple[nx.Graph, np.ndarray]:
        """
        Generate a single SBM graph with specified label diversity and homophily.
        
        Args:
            num_labels: Number of different labels/communities (1-10)
            homophily_level: Desired homophily ratio (0.0-1.0) - only used when use_advanced_calculation=True
            edge_probability: Base probability for edge generation
            use_advanced_calculation: Use direct homophily control (default: False - uses diversity-based approach)
            
        Returns:
            Tuple of (graph, node_labels)
            
        Note:
            Default behavior uses diversity-based approach where homophily is controlled by num_labels:
            - 1-2 labels: High homophily (~0.8-1.0) - clustered structure
            - 3-5 labels: Medium homophily (~0.3-0.6) - moderate mixing  
            - 6-10 labels: Low homophily (~0.0-0.3) - highly mixed structure
        """
        # Evenly distribute nodes across labels
        base_size = self.num_nodes // num_labels
        remainder = self.num_nodes % num_labels
        
        community_sizes = [base_size] * num_labels
        for i in range(remainder):
            community_sizes[i] += 1
            
        # Create node labels
        node_labels = []
        for i, size in enumerate(community_sizes):
            node_labels.extend([i] * size)
        node_labels = np.array(node_labels)
        
        # Create probability matrix for SBM
        prob_matrix = np.zeros((num_labels, num_labels))
        
        if use_advanced_calculation:
            # Advanced calculation: Estimate probabilities to achieve target homophily
            # This is based on the expected number of within vs between edges
            
            # Calculate expected edges for each community size
            total_possible_within = 0
            total_possible_between = 0
            
            for i, size in enumerate(community_sizes):
                # Within-community possible edges
                total_possible_within += size * (size - 1) // 2
                # Between-community possible edges
                for j in range(i + 1, len(community_sizes)):
                    total_possible_between += size * community_sizes[j]
            
            # Calculate target ratios
            target_within_edges = homophily_level * (total_possible_within + total_possible_between)
            target_between_edges = (1 - homophily_level) * (total_possible_within + total_possible_between)
            
            # Set probabilities (with safety bounds)
            within_prob = min(0.99, max(0.01, target_within_edges / total_possible_within)) if total_possible_within > 0 else 0.01
            between_prob = min(0.99, max(0.01, target_between_edges / total_possible_between)) if total_possible_between > 0 else 0.01
            
        else:
            # Diversity-based approach: Use multipliers based on num_labels
            if num_labels <= 2:  # High homophily (low diversity)
                within_prob = edge_probability * 3.0
                between_prob = edge_probability * 0.1
            elif num_labels <= 5:  # Medium homophily
                within_prob = edge_probability * 2.0
                between_prob = edge_probability * 0.3
            else:  # Low homophily (high diversity)
                within_prob = edge_probability * 1.2
                between_prob = edge_probability * 0.8
        
        # Ensure probabilities are within valid range [0, 1]
        within_prob = max(0.01, min(0.99, within_prob))
        between_prob = max(0.01, min(0.99, between_prob))
        
        # Set within-community probabilities (diagonal)
        np.fill_diagonal(prob_matrix, within_prob)
        
        # Set between-community probabilities (off-diagonal)
        prob_matrix[prob_matrix == 0] = between_prob
        
        # Generate the SBM graph
        graph = nx.stochastic_block_model(
            community_sizes, 
            prob_matrix, 
            seed=self.random_seed
        )
        
        # Add node attributes
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]['label'] = node_labels[i]
            graph.nodes[node]['community'] = node_labels[i]
        
        return graph, node_labels
    
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
    
    def generate_diversity_series(self, 
                                 num_graphs_per_level: int = 5,
                                 homophily_levels: Optional[List[float]] = None) -> Dict:
        """
        Generate a series of graphs with varying label diversity and homophily levels.
        
        Args:
            num_graphs_per_level: Number of graphs to generate per diversity level
            homophily_levels: List of homophily levels to test (if None, uses default range)
            
        Returns:
            Dictionary containing graphs and metrics organized by diversity level
        """
        if homophily_levels is None:
            homophily_levels = [0.8, 0.6, 0.4, 0.2]  # High to low homophily
        
        diversity_series = {}
        
        for num_labels in range(1, 11):  # 1 to 10 labels
            diversity_series[num_labels] = {
                'graphs': [],
                'metrics': [],
                'homophily_ratios': []
            }
            
            for i in range(num_graphs_per_level):
                # More labels => more mixed => lower homophily
                if num_labels <= 2:
                    homophily = 0.8  # High homophily 
                elif num_labels <= 5:
                    homophily = 0.6  # Medium homophily
                else:
                    homophily = 0.3  # Low homophily 
                
                # Generate graph
                graph, labels = self.generate_sbm_graph(
                    num_labels=num_labels,
                    homophily_level=homophily,
                    edge_probability=0.4
                )
                
                # Calculate metrics
                metrics = self.calculate_graph_metrics(graph)
                homophily_ratio = self.calculate_homophily_ratio(graph)
                
                # Store results
                diversity_series[num_labels]['graphs'].append(graph)
                diversity_series[num_labels]['metrics'].append(metrics)
                diversity_series[num_labels]['homophily_ratios'].append(homophily_ratio)
        
        return diversity_series
    
    def generate_source_target_pairs(self, 
                                   num_pairs: int = 10) -> List[Tuple[nx.Graph, nx.Graph, np.ndarray, np.ndarray]]:
        """
        Generate source-target graph pairs for domain adaptation experiments.
        
        Source graphs: Low diversity (1-2 labels, high homophily)
        Target graphs: High diversity (5-10 labels, low homophily)
        
        Args:
            num_pairs: Number of source-target pairs to generate
            
        Returns:
            List of tuples: (source_graph, target_graph, source_labels, target_labels)
        """
        pairs = []
        
        for _ in range(num_pairs):
            # Generate source graph 
            source_labels_count = np.random.choice([1, 2])
            source_graph, source_labels = self.generate_sbm_graph(
                num_labels=source_labels_count,
                homophily_level=0.8,  # High homophily
                edge_probability=0.5
            )
            
            # Generate target graph 
            target_labels_count = np.random.choice([5, 6, 7, 8, 9, 10])
            target_graph, target_labels = self.generate_sbm_graph(
                num_labels=target_labels_count,
                homophily_level=0.3,  # Low homophily
                edge_probability=0.4
            )
            
            pairs.append((source_graph, target_graph, source_labels, target_labels))
        
        return pairs

def visualize_graph(graph: nx.Graph, 
                   title: str = "SBM Graph", 
                   save_path: Optional[str] = None) -> None:
    """
    Visualize a single graph with node colors representing labels.
    
    Args:
        graph: NetworkX graph to visualize
        title: title for the plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Get node labels and positions
    labels = [graph.nodes[node]['label'] for node in graph.nodes()]
    pos = nx.spring_layout(graph, seed=42)
    
    # Create color map
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    node_colors = [color_map[label] for label in labels]
    
    # graph
    nx.draw(graph, pos, 
            node_color=node_colors,
            node_size=500,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # legend
    legend_elements = [plt.scatter([], [], c=colors[i], s=100, label=f'Label {unique_labels[i]}') 
                      for i in range(len(unique_labels))]
    plt.legend(handles=legend_elements, loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
    else:
        plt.show()

def plot_diversity_analysis(diversity_series: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot analysis of graph diversity vs. homophily ratio.
    
    Args:
        diversity_series: Results from generate_diversity_series()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    num_labels = list(diversity_series.keys())
    avg_homophily = [np.mean(diversity_series[k]['homophily_ratios']) for k in num_labels]
    std_homophily = [np.std(diversity_series[k]['homophily_ratios']) for k in num_labels]
    avg_density = [np.mean([m['density'] for m in diversity_series[k]['metrics']]) for k in num_labels]
    avg_clustering = [np.mean([m['clustering_coefficient'] for m in diversity_series[k]['metrics']]) for k in num_labels]
    
    # Plot 1: Homophily vs. Number of Labels
    axes[0, 0].errorbar(num_labels, avg_homophily, yerr=std_homophily, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
    axes[0, 0].set_xlabel('Number of Labels', fontsize=12)
    axes[0, 0].set_ylabel('Homophily Ratio', fontsize=12)
    axes[0, 0].set_title('Homophily vs. Label Diversity', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Graph Density vs. Number of Labels
    axes[0, 1].plot(num_labels, avg_density, marker='s', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Labels', fontsize=12)
    axes[0, 1].set_ylabel('Graph Density', fontsize=12)
    axes[0, 1].set_title('Graph Density vs. Label Diversity', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Clustering Coefficient vs. Number of Labels
    axes[1, 0].plot(num_labels, avg_clustering, marker='^', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Number of Labels', fontsize=12)
    axes[1, 0].set_ylabel('Clustering Coefficient', fontsize=12)
    axes[1, 0].set_title('Clustering vs. Label Diversity', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of Homophily Ratios
    all_homophily = []
    for k in num_labels:
        all_homophily.extend(diversity_series[k]['homophily_ratios'])
    
    axes[1, 1].hist(all_homophily, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Homophily Ratio', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Homophily Ratios', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
    else:
        plt.show()


def main():
    """
    Demonstrate the SBM dataset generator with various examples.
    """
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize the generator
    generator = SBMDatasetGenerator(num_nodes=10, random_seed=42)
    
    # Generate graphs with different diversity levels
    diversity_levels = [1, 3, 5, 10]
    graphs = {}
    
    for num_labels in diversity_levels:
        graph, labels = generator.generate_sbm_graph(
            num_labels=num_labels,
            homophily_level=0.8 if num_labels <= 2 else 0.3,
            edge_probability=0.4
        )
        
        homophily = generator.calculate_homophily_ratio(graph)
        metrics = generator.calculate_graph_metrics(graph)
        
        graphs[num_labels] = graph
        
        print(f"Graph with {num_labels} labels:")
        print(f"  - Nodes: {graph.number_of_nodes()}")
        print(f"  - Edges: {graph.number_of_edges()}")
        print(f"  - Homophily ratio: {homophily:.3f}")
        print(f"  - Density: {metrics['density']:.3f}")
        print(f"  - Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
        print()
    
    # Visualize different diversity levels
    for num_labels in [1, 5, 10]:
        graph = graphs[num_labels]
        save_path = os.path.join(output_dir, f"graph_{num_labels}_labels.png")
        visualize_graph(
            graph, 
            title=f"SBM Graph with {num_labels} Labels (Diversity Level: {num_labels})",
            save_path=save_path
        )
        print(f"Saved graph visualization: {save_path}")
    
    # Generate diversity series and analysis
    diversity_series = generator.generate_diversity_series(num_graphs_per_level=5)
    
    # Plot comprehensive analysis
    diversity_analysis_path = os.path.join(output_dir, "diversity_analysis.png")
    plot_diversity_analysis(diversity_series, save_path=diversity_analysis_path)
    print(f"Saved diversity analysis: {diversity_analysis_path}")
    
    # Print summary statistics
    print("Diversity Series Summary:")
    for num_labels in range(1, 11):
        avg_homophily = np.mean(diversity_series[num_labels]['homophily_ratios'])
        std_homophily = np.std(diversity_series[num_labels]['homophily_ratios'])
        print(f"  {num_labels} labels: homophily = {avg_homophily:.3f} ± {std_homophily:.3f}")
    
    # Source-Target pairs for domain adaptation
    pairs = generator.generate_source_target_pairs(num_pairs=5)
    
    print("Generated source-target pairs:")
    for i, (source_graph, target_graph, source_labels, target_labels) in enumerate(pairs):
        source_homophily = generator.calculate_homophily_ratio(source_graph)
        target_homophily = generator.calculate_homophily_ratio(target_graph)
        source_labels_count = len(np.unique(source_labels))
        target_labels_count = len(np.unique(target_labels))
        
        print(f"Pair {i+1}:")
        print(f"  Source: {source_labels_count} labels, homophily={source_homophily:.3f}")
        print(f"  Target: {target_labels_count} labels, homophily={target_homophily:.3f}")
        print(f"  Homophily shift: {source_homophily - target_homophily:.3f}")
        print()
    
    # Custom parameter exploration

    # Test different homophily levels
    homophily_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
    homophily_results = []
    
    for homophily in homophily_levels:
        graph, _ = generator.generate_sbm_graph(
            num_labels=5,
            homophily_level=homophily,
            edge_probability=0.4
        )
        actual_homophily = generator.calculate_homophily_ratio(graph)
        homophily_results.append(actual_homophily)
        
        print(f"Target homophily: {homophily:.1f}, Actual homophily: {actual_homophily:.3f}")
    
    # Plot homophily comparison
    plt.figure(figsize=(10, 6))
    plt.plot(homophily_levels, homophily_results, 'bo-', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Match')
    plt.xlabel('Target Homophily Level', fontsize=12)
    plt.ylabel('Actual Homophily Ratio', fontsize=12)
    plt.title('Target vs. Actual Homophily Levels', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    homophily_validation_path = os.path.join(output_dir, "homophily_validation.png")
    plt.savefig(homophily_validation_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    print(f"Saved homophily validation: {homophily_validation_path}")
    
    print("\n=== Demo Complete ===")
    print("All generated files saved in output/ directory:")
    print(f"- {os.path.join(output_dir, 'graph_1_labels.png')}: Low diversity graph (1 label)")
    print(f"- {os.path.join(output_dir, 'graph_5_labels.png')}: Medium diversity graph (5 labels)")
    print(f"- {os.path.join(output_dir, 'graph_10_labels.png')}: High diversity graph (10 labels)")
    print(f"- {os.path.join(output_dir, 'diversity_analysis.png')}: Comprehensive diversity analysis")
    print(f"- {os.path.join(output_dir, 'homophily_validation.png')}: Homophily parameter validation")

if __name__ == "__main__":

    main()
