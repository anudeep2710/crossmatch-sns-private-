"""
Module for generating embeddings from social network structures.
"""

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import karateclub for alternative graph embedding methods
try:
    import karateclub
    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False
    logger.warning("karateclub package not available. Some embedding methods may not work.")

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer implementation.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """Forward pass for GCN layer."""
        # Normalize adjacency matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        # GCN propagation rule
        support = torch.mm(normalized_adj, x)
        output = self.linear(support)
        return F.relu(output)

class GCN(nn.Module):
    """
    Graph Convolutional Network model.
    """
    def __init__(self, nfeat, nhid, nout, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        """Forward pass for GCN model."""
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class NetworkEmbedder:
    """
    Class for generating embeddings from social network structures.

    Attributes:
        models (Dict): Dictionary to store trained models for each platform
        embeddings (Dict): Dictionary to store generated embeddings for each platform
        params (Dict): Dictionary to store parameters for each platform
    """

    def __init__(self, embedding_dim: int = 64, walk_length: int = 30,
                num_walks: int = 200, p: float = 1.0, q: float = 1.0,
                gcn_hidden_dim: int = 128, gcn_dropout: float = 0.5):
        """
        Initialize the NetworkEmbedder.

        Args:
            embedding_dim (int): Dimension of the embeddings
            walk_length (int): Length of each random walk for Node2Vec
            num_walks (int): Number of random walks per node for Node2Vec
            p (float): Return parameter for Node2Vec
            q (float): In-out parameter for Node2Vec
            gcn_hidden_dim (int): Hidden dimension for GCN
            gcn_dropout (float): Dropout rate for GCN
        """
        self.models = {}
        self.embeddings = {}
        self.params = {
            'embedding_dim': embedding_dim,
            'walk_length': walk_length,
            'num_walks': num_walks,
            'p': p,
            'q': q,
            'gcn_hidden_dim': gcn_hidden_dim,
            'gcn_dropout': gcn_dropout
        }
        logger.info("NetworkEmbedder initialized")

    def fit_transform(self, network: nx.Graph, platform_name: str,
                     method: str = 'node2vec', save_path: Optional[str] = None) -> np.ndarray:
        """
        Fit a model to the network and transform it to embeddings.

        Args:
            network (nx.Graph): NetworkX graph representing user connections
            platform_name (str): Name of the platform
            method (str): Embedding method ('node2vec', 'gcn', 'deepwalk', 'role2vec', or 'graph2vec')
            save_path (str, optional): Path to save the model and embeddings

        Returns:
            np.ndarray: Network embeddings
        """
        logger.info(f"Generating {method} embeddings for {platform_name} network with {network.number_of_nodes()} nodes")

        if method == 'node2vec':
            embeddings = self._node2vec_embeddings(network, platform_name)
        elif method == 'gcn':
            embeddings = self._gcn_embeddings(network, platform_name)
        elif method in ['deepwalk', 'role2vec', 'graph2vec']:
            if KARATECLUB_AVAILABLE:
                embeddings = self._karateclub_embeddings(network, platform_name, method)
            else:
                logger.warning(f"karateclub package not available. Falling back to node2vec.")
                embeddings = self._node2vec_embeddings(network, platform_name)
        else:
            raise ValueError(f"Unsupported embedding method: {method}")

        # Store embeddings
        self.embeddings[platform_name] = embeddings

        # Save model and embeddings if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Save embeddings
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.index = list(network.nodes())
            embeddings_df.to_csv(os.path.join(save_path, f"{platform_name}_{method}_embeddings.csv"))

            # Save model
            if method == 'node2vec' and platform_name in self.models:
                self.models[platform_name].save(os.path.join(save_path, f"{platform_name}_{method}_model"))

        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def transform(self, network: nx.Graph, platform_name: str,
                 method: str = 'node2vec') -> np.ndarray:
        """
        Transform a network to embeddings using a pre-trained model.

        Args:
            network (nx.Graph): NetworkX graph representing user connections
            platform_name (str): Name of the platform
            method (str): Embedding method ('node2vec', 'gcn', 'deepwalk', 'role2vec', or 'graph2vec')

        Returns:
            np.ndarray: Network embeddings
        """
        logger.info(f"Transforming {platform_name} network with {network.number_of_nodes()} nodes using {method}")

        if platform_name not in self.models:
            logger.warning(f"No pre-trained model found for {platform_name}. Training a new model.")
            return self.fit_transform(network, platform_name, method)

        if method == 'node2vec':
            # For Node2Vec, we need to retrain if the network structure changes
            return self._node2vec_embeddings(network, platform_name, retrain=True)
        elif method == 'gcn':
            # For GCN, we can reuse the model
            return self._gcn_embeddings(network, platform_name, retrain=False)
        elif method in ['deepwalk', 'role2vec', 'graph2vec']:
            if KARATECLUB_AVAILABLE:
                # For karateclub methods, we need to retrain
                return self._karateclub_embeddings(network, platform_name, method)
            else:
                logger.warning(f"karateclub package not available. Falling back to node2vec.")
                return self._node2vec_embeddings(network, platform_name, retrain=True)
        else:
            raise ValueError(f"Unsupported embedding method: {method}")

    def _node2vec_embeddings(self, network: nx.Graph, platform_name: str,
                            retrain: bool = True) -> np.ndarray:
        """
        Generate Node2Vec embeddings for a network.

        Args:
            network (nx.Graph): NetworkX graph
            platform_name (str): Name of the platform
            retrain (bool): Whether to retrain the model

        Returns:
            np.ndarray: Node2Vec embeddings
        """
        # Check if network is empty
        if network.number_of_nodes() == 0:
            logger.warning(f"Empty network for {platform_name}. Returning empty embeddings.")
            return np.array([])

        # Check if we can reuse existing model
        if not retrain and platform_name in self.models:
            logger.info(f"Reusing existing Node2Vec model for {platform_name}")
            model = self.models[platform_name]
        else:
            # Initialize Node2Vec with optimized parameters
            logger.info(f"Training new Node2Vec model for {platform_name}")

            # Use more workers for better parallelization
            num_workers = min(os.cpu_count() or 4, 8)

            node2vec = Node2Vec(
                network,
                dimensions=self.params['embedding_dim'],
                walk_length=self.params['walk_length'],
                num_walks=self.params['num_walks'],
                p=self.params['p'],
                q=self.params['q'],
                workers=num_workers,
                quiet=True  # Reduce logging noise
            )

            # Train model with optimized parameters
            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=10000,  # Larger batch size for better performance
                epochs=5,           # Fewer epochs for faster training
                compute_loss=False  # Disable loss computation for speed
            )

            # Store model
            self.models[platform_name] = model

        # Get embeddings for all nodes efficiently
        # Pre-allocate arrays for better performance
        nodes = list(network.nodes())
        embeddings_array = np.zeros((len(nodes), self.params['embedding_dim']))

        # Convert nodes to strings once
        node_strs = [str(node) for node in nodes]

        # Use vectorized operations where possible
        for i, (node, node_str) in enumerate(zip(nodes, node_strs)):
            try:
                if node_str in model.wv:
                    embeddings_array[i] = model.wv[node_str]
                else:
                    # Use zero vector for missing nodes
                    logger.debug(f"Node {node} not found in model vocabulary. Using zero vector.")
            except KeyError:
                logger.debug(f"Node {node} not found in model vocabulary. Using zero vector.")

        # Normalize embeddings using a faster method
        # Calculate mean and std in one pass
        mean = np.mean(embeddings_array, axis=0)
        std = np.std(embeddings_array, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        # Normalize
        embeddings_array = (embeddings_array - mean) / std

        return embeddings_array

    def _gcn_embeddings(self, network: nx.Graph, platform_name: str,
                       retrain: bool = True) -> np.ndarray:
        """
        Generate GCN embeddings for a network.

        Args:
            network (nx.Graph): NetworkX graph
            platform_name (str): Name of the platform
            retrain (bool): Whether to retrain the model

        Returns:
            np.ndarray: GCN embeddings
        """
        # Check if network is empty
        if network.number_of_nodes() == 0:
            logger.warning(f"Empty network for {platform_name}. Returning empty embeddings.")
            return np.array([])

        # Convert network to adjacency matrix efficiently
        adj = nx.to_scipy_sparse_array(network)  # Use sparse matrix for better memory efficiency

        # Check if we can use GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to PyTorch tensors
        adj_tensor = torch.FloatTensor(adj.todense()).to(device)

        # Add self-loops efficiently
        adj_tensor = adj_tensor + torch.eye(adj_tensor.shape[0], device=device)

        # Create feature matrix (identity matrix if no features)
        # Use sparse identity matrix for memory efficiency
        num_nodes = network.number_of_nodes()
        features_tensor = torch.eye(num_nodes, device=device)

        # Initialize or retrieve model
        if retrain or platform_name not in self.models:
            model = GCN(
                nfeat=num_nodes,
                nhid=self.params['gcn_hidden_dim'],
                nout=self.params['embedding_dim'],
                dropout=self.params['gcn_dropout']
            ).to(device)

            # Train model with early stopping
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            model.train()

            # Early stopping parameters
            best_loss = float('inf')
            patience = 20
            patience_counter = 0

            # Use fewer epochs and early stopping for faster training
            max_epochs = 100

            for epoch in range(max_epochs):
                optimizer.zero_grad()
                output = model(features_tensor, adj_tensor)

                # Use reconstruction loss with L2 regularization
                loss = F.mse_loss(torch.mm(output, output.t()), adj_tensor)

                loss.backward()
                optimizer.step()

                # Update learning rate
                scheduler.step(loss)

                # Early stopping check
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    # Save best model
                    best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        # Restore best model
                        model.load_state_dict(best_model)
                        break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f}")

            # Store model
            self.models[platform_name] = model
        else:
            model = self.models[platform_name].to(device)

        # Generate embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(features_tensor, adj_tensor).cpu().numpy()

        # Normalize embeddings using a faster method
        # Calculate mean and std in one pass
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        # Normalize
        embeddings = (embeddings - mean) / std

        return embeddings

    def _karateclub_embeddings(self, network: nx.Graph, platform_name: str,
                              method: str = 'deepwalk') -> np.ndarray:
        """
        Generate embeddings using karateclub methods.

        Args:
            network (nx.Graph): NetworkX graph
            platform_name (str): Name of the platform
            method (str): Embedding method ('deepwalk', 'role2vec', or 'graph2vec')

        Returns:
            np.ndarray: Embeddings
        """
        # Check if network is empty
        if network.number_of_nodes() == 0:
            logger.warning(f"Empty network for {platform_name}. Returning empty embeddings.")
            return np.array([])

        # Check if karateclub is available
        if not KARATECLUB_AVAILABLE:
            logger.warning("karateclub package not available. Falling back to node2vec.")
            return self._node2vec_embeddings(network, platform_name)

        # Create a copy of the network to ensure it has consecutive node indices
        # Use more efficient conversion
        original_nodes = list(network.nodes())
        G = nx.convert_node_labels_to_integers(network, label_attribute='original_label')

        # Map original labels to indices more efficiently
        node_map = {G.nodes[i]['original_label']: i for i in range(G.number_of_nodes())}

        # Check if we can reuse existing model
        if platform_name in self.models and hasattr(self.models[platform_name], 'get_embedding'):
            logger.info(f"Reusing existing {method} model for {platform_name}")
            model = self.models[platform_name]
            # Refit the model with the current graph
            model.fit(G)
        else:
            # Initialize the appropriate model with optimized parameters
            logger.info(f"Training new {method} model for {platform_name}")

            # Use more workers for better parallelization
            num_workers = min(os.cpu_count() or 4, 8)

            if method == 'deepwalk':
                model = karateclub.DeepWalk(
                    walk_number=min(self.params['num_walks'], 50),  # Reduce walks for speed
                    walk_length=self.params['walk_length'],
                    dimensions=self.params['embedding_dim'],
                    workers=num_workers,
                    window_size=5,  # Smaller window for faster training
                    epochs=5        # Fewer epochs for faster training
                )
            elif method == 'role2vec':
                model = karateclub.Role2Vec(
                    walk_number=min(self.params['num_walks'], 50),  # Reduce walks for speed
                    walk_length=self.params['walk_length'],
                    dimensions=self.params['embedding_dim'],
                    workers=num_workers,
                    window_size=5,  # Smaller window for faster training
                    epochs=5        # Fewer epochs for faster training
                )
            elif method == 'graph2vec':
                # Graph2Vec works on a list of graphs, so we need to adapt
                # For simplicity, we'll just use DeepWalk instead
                logger.warning("graph2vec requires a list of graphs. Using DeepWalk instead.")
                model = karateclub.DeepWalk(
                    walk_number=min(self.params['num_walks'], 50),  # Reduce walks for speed
                    walk_length=self.params['walk_length'],
                    dimensions=self.params['embedding_dim'],
                    workers=num_workers,
                    window_size=5,  # Smaller window for faster training
                    epochs=5        # Fewer epochs for faster training
                )
            else:
                raise ValueError(f"Unsupported karateclub method: {method}")

            # Fit the model
            model.fit(G)

            # Store model
            self.models[platform_name] = model

        # Get embeddings
        embeddings_array = model.get_embedding()

        # Create a mapping back to original node labels more efficiently
        # Pre-allocate result array
        result_embeddings = np.zeros((len(original_nodes), self.params['embedding_dim']))

        # Create a mapping from original node to index in original_nodes list
        original_node_to_idx = {node: i for i, node in enumerate(original_nodes)}

        # Map embeddings back to original node order efficiently
        for orig_node, idx in node_map.items():
            if orig_node in original_node_to_idx:
                result_embeddings[original_node_to_idx[orig_node]] = embeddings_array[idx]

        # Normalize embeddings using a faster method
        # Calculate mean and std in one pass
        mean = np.mean(result_embeddings, axis=0)
        std = np.std(result_embeddings, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        # Normalize
        result_embeddings = (result_embeddings - mean) / std

        return result_embeddings
