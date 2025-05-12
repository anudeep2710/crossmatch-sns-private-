"""
Module for visualizing results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for visualizing results.
    """
    
    def __init__(self, use_plotly: bool = True):
        """
        Initialize the Visualizer.
        
        Args:
            use_plotly (bool): Whether to use Plotly for interactive visualizations
        """
        self.use_plotly = use_plotly
        logger.info(f"Visualizer initialized with {'Plotly' if use_plotly else 'Matplotlib'}")
    
    def plot_network(self, network: nx.Graph, title: str = "Network Visualization", 
                    save_path: Optional[str] = None) -> None:
        """
        Visualize a network.
        
        Args:
            network (nx.Graph): NetworkX graph to visualize
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        logger.info(f"Plotting network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
        
        # Check if network is empty
        if network.number_of_nodes() == 0:
            logger.warning("Empty network. Nothing to plot.")
            return
        
        # Limit the size of the network for visualization
        if network.number_of_nodes() > 100:
            logger.warning(f"Network is too large ({network.number_of_nodes()} nodes). Sampling 100 nodes.")
            nodes = list(network.nodes())
            sampled_nodes = np.random.choice(nodes, size=100, replace=False)
            network = network.subgraph(sampled_nodes)
        
        # Compute layout
        pos = nx.spring_layout(network)
        
        if self.use_plotly:
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in network.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node trace
            node_x = []
            node_y = []
            for node in network.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )
            
            # Color nodes by degree
            node_adjacencies = []
            node_text = []
            for node in network.nodes():
                node_adjacencies.append(len(list(network.neighbors(node))))
                node_text.append(f'Node: {node}<br>Connections: {len(list(network.neighbors(node)))}')
            
            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=title,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            # Save or show
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Network visualization saved to {save_path}")
            else:
                fig.show()
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 8))
            
            # Draw nodes
            node_sizes = [300 * (1 + len(list(network.neighbors(node)))) for node in network.nodes()]
            nx.draw_networkx_nodes(network, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(network, pos, width=1.0, alpha=0.5)
            
            # Draw labels
            if network.number_of_nodes() <= 50:
                nx.draw_networkx_labels(network, pos, font_size=10)
            
            plt.title(title)
            plt.axis('off')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Network visualization saved to {save_path}")
            else:
                plt.show()
    
    def plot_embeddings(self, embeddings: Dict[str, np.ndarray], labels: Optional[Dict[str, str]] = None, 
                       method: str = 'tsne', title: str = "Embedding Visualization", 
                       save_path: Optional[str] = None) -> None:
        """
        Visualize embeddings in 2D space.
        
        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary mapping user IDs to embeddings
            labels (Dict[str, str], optional): Dictionary mapping user IDs to labels
            method (str): Dimensionality reduction method ('tsne' or 'pca')
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        logger.info(f"Plotting {len(embeddings)} embeddings using {method}")
        
        # Check if embeddings dictionary is empty
        if not embeddings:
            logger.warning("Empty embeddings dictionary. Nothing to plot.")
            return
        
        # Extract user IDs and embeddings
        user_ids = list(embeddings.keys())
        emb_matrix = np.vstack([embeddings[uid] for uid in user_ids])
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        reduced_emb = reducer.fit_transform(emb_matrix)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'user_id': user_ids,
            'x': reduced_emb[:, 0],
            'y': reduced_emb[:, 1]
        })
        
        # Add labels if provided
        if labels:
            df['label'] = df['user_id'].map(labels)
        else:
            df['label'] = 'Unknown'
        
        if self.use_plotly:
            # Plotly version
            fig = px.scatter(
                df, x='x', y='y', color='label', hover_data=['user_id'],
                title=title
            )
            
            # Save or show
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Embedding visualization saved to {save_path}")
            else:
                fig.show()
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 8))
            
            if 'label' in df.columns and df['label'].nunique() > 1:
                # Color by label
                sns.scatterplot(data=df, x='x', y='y', hue='label', alpha=0.7)
                plt.legend(title='Label')
            else:
                # Single color
                sns.scatterplot(data=df, x='x', y='y', alpha=0.7)
            
            # Add annotations for a subset of points
            if len(df) <= 50:
                for i, row in df.iterrows():
                    plt.annotate(row['user_id'], (row['x'], row['y']), fontsize=8)
            
            plt.title(title)
            plt.xlabel(f"{method.upper()} Dimension 1")
            plt.ylabel(f"{method.upper()} Dimension 2")
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Embedding visualization saved to {save_path}")
            else:
                plt.show()
    
    def plot_matching_results(self, matches: pd.DataFrame, embeddings1: Dict[str, np.ndarray], 
                            embeddings2: Dict[str, np.ndarray], platform1_name: str, 
                            platform2_name: str, title: str = "Matching Results", 
                            save_path: Optional[str] = None) -> None:
        """
        Visualize matching results.
        
        Args:
            matches (pd.DataFrame): DataFrame with matches
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            platform1_name (str): Name of platform 1
            platform2_name (str): Name of platform 2
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        logger.info(f"Plotting matching results between {platform1_name} and {platform2_name}")
        
        # Check if matches DataFrame is empty
        if len(matches) == 0:
            logger.warning("Empty matches DataFrame. Nothing to plot.")
            return
        
        # Extract user IDs and embeddings
        user_ids1 = list(embeddings1.keys())
        user_ids2 = list(embeddings2.keys())
        
        emb_matrix1 = np.vstack([embeddings1[uid] for uid in user_ids1])
        emb_matrix2 = np.vstack([embeddings2[uid] for uid in user_ids2])
        
        # Apply dimensionality reduction
        reducer = TSNE(n_components=2, random_state=42)
        
        # Combine embeddings for dimensionality reduction
        combined_emb = np.vstack([emb_matrix1, emb_matrix2])
        reduced_emb = reducer.fit_transform(combined_emb)
        
        # Split back into platform-specific embeddings
        reduced_emb1 = reduced_emb[:len(user_ids1)]
        reduced_emb2 = reduced_emb[len(user_ids1):]
        
        # Create DataFrames for plotting
        df1 = pd.DataFrame({
            'user_id': user_ids1,
            'x': reduced_emb1[:, 0],
            'y': reduced_emb1[:, 1],
            'platform': platform1_name
        })
        
        df2 = pd.DataFrame({
            'user_id': user_ids2,
            'x': reduced_emb2[:, 0],
            'y': reduced_emb2[:, 1],
            'platform': platform2_name
        })
        
        # Combine DataFrames
        df = pd.concat([df1, df2])
        
        if self.use_plotly:
            # Plotly version
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Add scatter plots for each platform
            fig.add_trace(
                go.Scatter(
                    x=df1['x'], y=df1['y'],
                    mode='markers',
                    name=platform1_name,
                    marker=dict(size=10, color='blue'),
                    text=df1['user_id'],
                    hoverinfo='text'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df2['x'], y=df2['y'],
                    mode='markers',
                    name=platform2_name,
                    marker=dict(size=10, color='red'),
                    text=df2['user_id'],
                    hoverinfo='text'
                )
            )
            
            # Add lines for matches
            for _, row in matches.iterrows():
                user_id1 = row['user_id1']
                user_id2 = row['user_id2']
                confidence = row['confidence']
                
                if user_id1 in df1['user_id'].values and user_id2 in df2['user_id'].values:
                    x1 = df1[df1['user_id'] == user_id1]['x'].values[0]
                    y1 = df1[df1['user_id'] == user_id1]['y'].values[0]
                    x2 = df2[df2['user_id'] == user_id2]['x'].values[0]
                    y2 = df2[df2['user_id'] == user_id2]['y'].values[0]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x1, x2], y=[y1, y2],
                            mode='lines',
                            line=dict(width=1, color='rgba(0, 0, 0, 0.3)'),
                            hoverinfo='text',
                            text=f"Match: {user_id1} - {user_id2}<br>Confidence: {confidence:.3f}",
                            showlegend=False
                        )
                    )
            
            fig.update_layout(
                title=title,
                xaxis=dict(title='TSNE Dimension 1'),
                yaxis=dict(title='TSNE Dimension 2')
            )
            
            # Save or show
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Matching results visualization saved to {save_path}")
            else:
                fig.show()
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 8))
            
            # Plot points for each platform
            sns.scatterplot(data=df, x='x', y='y', hue='platform', alpha=0.7)
            
            # Add lines for matches
            for _, row in matches.iterrows():
                user_id1 = row['user_id1']
                user_id2 = row['user_id2']
                
                if user_id1 in df1['user_id'].values and user_id2 in df2['user_id'].values:
                    x1 = df1[df1['user_id'] == user_id1]['x'].values[0]
                    y1 = df1[df1['user_id'] == user_id1]['y'].values[0]
                    x2 = df2[df2['user_id'] == user_id2]['x'].values[0]
                    y2 = df2[df2['user_id'] == user_id2]['y'].values[0]
                    
                    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
            
            plt.title(title)
            plt.xlabel('TSNE Dimension 1')
            plt.ylabel('TSNE Dimension 2')
            plt.legend(title='Platform')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Matching results visualization saved to {save_path}")
            else:
                plt.show()
    
    def plot_evaluation_metrics(self, metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Visualize evaluation metrics.
        
        Args:
            metrics (Dict[str, Any]): Dictionary with evaluation metrics
            save_path (str, optional): Path to save the plot
        """
        logger.info("Plotting evaluation metrics")
        
        # Check if metrics dictionary is empty
        if not metrics:
            logger.warning("Empty metrics dictionary. Nothing to plot.")
            return
        
        # Check if threshold metrics are available
        if 'threshold_metrics' not in metrics:
            logger.warning("Threshold metrics not found in metrics dictionary.")
            return
        
        # Extract threshold metrics
        threshold_metrics = metrics['threshold_metrics']
        thresholds = sorted(threshold_metrics.keys())
        precision = [threshold_metrics[t]['precision'] for t in thresholds]
        recall = [threshold_metrics[t]['recall'] for t in thresholds]
        f1 = [threshold_metrics[t]['f1'] for t in thresholds]
        
        if self.use_plotly:
            # Plotly version
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=precision,
                mode='lines+markers',
                name='Precision'
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=recall,
                mode='lines+markers',
                name='Recall'
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=f1,
                mode='lines+markers',
                name='F1 Score'
            ))
            
            # Add vertical line for best threshold
            if 'best_threshold' in metrics:
                best_threshold = metrics['best_threshold']
                fig.add_vline(x=best_threshold, line_dash="dash", line_color="green",
                             annotation_text=f"Best Threshold: {best_threshold:.2f}")
            
            fig.update_layout(
                title='Evaluation Metrics vs. Threshold',
                xaxis_title='Threshold',
                yaxis_title='Score',
                legend_title='Metric'
            )
            
            # Save or show
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Evaluation metrics visualization saved to {save_path}")
            else:
                fig.show()
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 6))
            
            plt.plot(thresholds, precision, 'o-', label='Precision')
            plt.plot(thresholds, recall, 'o-', label='Recall')
            plt.plot(thresholds, f1, 'o-', label='F1 Score')
            
            # Add vertical line for best threshold
            if 'best_threshold' in metrics:
                best_threshold = metrics['best_threshold']
                plt.axvline(x=best_threshold, linestyle='--', color='green', 
                           label=f'Best Threshold: {best_threshold:.2f}')
            
            plt.title('Evaluation Metrics vs. Threshold')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Evaluation metrics visualization saved to {save_path}")
            else:
                plt.show()
