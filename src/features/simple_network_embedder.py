"""
Simple network embedder that doesn't rely on problematic dependencies.
Uses basic graph features and GCN for network embeddings.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Network Layer."""
    
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        """Forward pass for GCN layer."""
        # Simple adjacency normalization
        rowsum = adj.sum(1, keepdim=True)
        rowsum[rowsum == 0] = 1  # Avoid division by zero
        normalized_adj = adj / rowsum
        
        # GCN propagation
        support = torch.mm(normalized_adj, x)
        output = self.linear(support)
        output = self.dropout(output)
        return F.relu(output)

class SimpleGCN(nn.Module):
    """Simple Graph Convolutional Network."""
    
    def __init__(self, nfeat, nhid, nout, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.gc1 = SimpleGCNLayer(nfeat, nhid)
        self.gc2 = SimpleGCNLayer(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        """Forward pass for GCN model."""
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class SimpleNetworkEmbedder:
    """
    Simple network embedder using basic graph features and GCN.
    Fallback option when other graph embedding libraries have issues.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        """
        Initialize the SimpleNetworkEmbedder.

        Args:
            embedding_dim (int): Dimension of the embeddings
            hidden_dim (int): Hidden dimension for GCN
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.models = {}
        self.embeddings = {}
        self.scaler = StandardScaler()
        logger.info("SimpleNetworkEmbedder initialized")

    def fit_transform(self, network: nx.Graph, platform_name: str) -> np.ndarray:
        """
        Fit and transform network to embeddings.

        Args:
            network (nx.Graph): NetworkX graph
            platform_name (str): Name of the platform

        Returns:
            np.ndarray: Network embeddings
        """
        logger.info(f"Generating simple embeddings for {platform_name} network with {network.number_of_nodes()} nodes")

        # Check if network is empty
        if network.number_of_nodes() == 0:
            logger.warning(f"Empty network for {platform_name}. Returning empty embeddings.")
            return np.array([])

        # Extract basic graph features
        features = self._extract_graph_features(network)
        
        # Use GCN if we have enough nodes, otherwise use features directly
        if network.number_of_nodes() > 10:
            embeddings = self._gcn_embeddings(network, features, platform_name)
        else:
            # For small networks, just use the features directly
            embeddings = self._feature_embeddings(features)

        # Store embeddings
        self.embeddings[platform_name] = embeddings
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def _extract_graph_features(self, network: nx.Graph) -> np.ndarray:
        """Extract basic graph features for each node."""
        nodes = list(network.nodes())
        features = []

        for node in nodes:
            node_features = []
            
            # Degree centrality
            node_features.append(network.degree(node))
            
            # Clustering coefficient
            try:
                node_features.append(nx.clustering(network, node))
            except:
                node_features.append(0.0)
            
            # Betweenness centrality (approximate for large graphs)
            if network.number_of_nodes() < 1000:
                try:
                    betweenness = nx.betweenness_centrality(network)
                    node_features.append(betweenness.get(node, 0.0))
                except:
                    node_features.append(0.0)
            else:
                node_features.append(0.0)
            
            # Closeness centrality (approximate for large graphs)
            if network.number_of_nodes() < 1000:
                try:
                    closeness = nx.closeness_centrality(network)
                    node_features.append(closeness.get(node, 0.0))
                except:
                    node_features.append(0.0)
            else:
                node_features.append(0.0)
            
            # PageRank
            try:
                pagerank = nx.pagerank(network, max_iter=50)
                node_features.append(pagerank.get(node, 0.0))
            except:
                node_features.append(0.0)
            
            features.append(node_features)

        features_array = np.array(features)
        
        # Handle NaN values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize features
        if features_array.shape[0] > 1:
            features_array = self.scaler.fit_transform(features_array)
        
        return features_array

    def _feature_embeddings(self, features: np.ndarray) -> np.ndarray:
        """Create embeddings directly from features for small networks."""
        if features.shape[1] >= self.embedding_dim:
            # Use PCA-like dimensionality reduction
            U, s, Vt = np.linalg.svd(features, full_matrices=False)
            embeddings = U[:, :self.embedding_dim] * s[:self.embedding_dim]
        else:
            # Pad with zeros if we have fewer features than embedding dimension
            embeddings = np.zeros((features.shape[0], self.embedding_dim))
            embeddings[:, :features.shape[1]] = features
        
        return embeddings

    def _gcn_embeddings(self, network: nx.Graph, features: np.ndarray, platform_name: str) -> np.ndarray:
        """Generate GCN embeddings."""
        # Convert to adjacency matrix
        adj_matrix = nx.to_numpy_array(network)
        
        # Add self-loops
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
        
        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_tensor = torch.FloatTensor(adj_matrix).to(device)
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Initialize model
        model = SimpleGCN(
            nfeat=features.shape[1],
            nhid=self.hidden_dim,
            nout=self.embedding_dim
        ).to(device)
        
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        # Simple training loop
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(features_tensor, adj_tensor)
            
            # Reconstruction loss
            loss = F.mse_loss(torch.mm(output, output.t()), adj_tensor)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Generate embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(features_tensor, adj_tensor).cpu().numpy()
        
        # Store model
        self.models[platform_name] = model
        
        return embeddings

    def transform(self, network: nx.Graph, platform_name: str) -> np.ndarray:
        """Transform network using existing model."""
        return self.fit_transform(network, platform_name)

    def get_embeddings(self, platform_name: str) -> Optional[np.ndarray]:
        """Get stored embeddings for a platform."""
        return self.embeddings.get(platform_name)
