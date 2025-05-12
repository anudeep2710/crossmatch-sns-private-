"""
Module for matching users across platforms.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSMUAModel(nn.Module):
    """
    Gradient-based Similarity Matching Using Attention model.
    """
    def __init__(self, embedding_dim: int):
        """
        Initialize the GSMUA model.

        Args:
            embedding_dim (int): Dimension of the embeddings
        """
        super(GSMUAModel, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.similarity = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            emb1 (torch.Tensor): Embedding from platform 1
            emb2 (torch.Tensor): Embedding from platform 2

        Returns:
            torch.Tensor: Similarity score
        """
        # Compute attention weights
        att1 = self.attention(emb1)
        att2 = self.attention(emb2)

        # Apply attention
        weighted_emb1 = att1 * emb1
        weighted_emb2 = att2 * emb2

        # Concatenate embeddings
        concat = torch.cat([weighted_emb1, weighted_emb2], dim=0)

        # Compute similarity
        sim = self.similarity(concat)

        return sim

class UserMatcher:
    """
    Class for matching users across platforms.

    Attributes:
        method (str): Method for matching users
        threshold (float): Threshold for matching
        model (GSMUAModel): GSMUA model for matching
        device (torch.device): Device to use for computation
    """

    def __init__(self, method: str = 'cosine', threshold: float = 0.05, device: Optional[str] = None):
        """
        Initialize the UserMatcher.

        Args:
            method (str): Method for matching users ('cosine', 'frui-p', or 'gsmua')
            threshold (float): Threshold for matching
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.method = method
        self.threshold = threshold
        self.model = None

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"UserMatcher initialized with {method} method and threshold {threshold}")

    def fit(self, embeddings1: Dict[str, np.ndarray], embeddings2: Dict[str, np.ndarray],
           ground_truth: Optional[pd.DataFrame] = None) -> None:
        """
        Train the matcher using ground truth data.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            ground_truth (pd.DataFrame, optional): Ground truth matches
        """
        if ground_truth is None:
            logger.info("No ground truth provided. Skipping training.")
            return

        logger.info(f"Training matcher with {len(ground_truth)} ground truth matches")

        if self.method == 'gsmua':
            self._train_gsmua(embeddings1, embeddings2, ground_truth)
        elif self.method == 'frui-p':
            self._learn_frui_p_params(embeddings1, embeddings2, ground_truth)
        else:
            logger.info(f"Method {self.method} does not require training")

    def predict(self, embeddings1: Dict[str, np.ndarray], embeddings2: Dict[str, np.ndarray],
               threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Match users across platforms.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            threshold (float, optional): Threshold for matching

        Returns:
            pd.DataFrame: DataFrame with matches and confidence scores
        """
        logger.info(f"Matching users with {self.method} method")

        # Use provided threshold or default
        threshold = threshold if threshold is not None else self.threshold

        if self.method == 'cosine':
            matches = self._cosine_matching(embeddings1, embeddings2, threshold)
        elif self.method == 'frui-p':
            matches = self._frui_p_matching(embeddings1, embeddings2, threshold)
        elif self.method == 'gsmua':
            matches = self._gsmua_matching(embeddings1, embeddings2, threshold)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        logger.info(f"Found {len(matches)} matches with threshold {threshold}")

        return matches

    def _compute_similarity_matrix(self, embeddings1: Dict[str, np.ndarray],
                        embeddings2: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute similarity matrix between two sets of embeddings.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2

        Returns:
            Tuple[np.ndarray, List[str], List[str]]: Similarity matrix, user IDs from platform 1, user IDs from platform 2
        """
        # Get user IDs
        user_ids1 = list(embeddings1.keys())
        user_ids2 = list(embeddings2.keys())

        # Early check for zero embeddings
        zero_embeddings1 = {uid: np.all(emb == 0) for uid, emb in embeddings1.items()}
        zero_embeddings2 = {uid: np.all(emb == 0) for uid, emb in embeddings2.items()}

        # Initialize similarity matrix
        sim_matrix = np.zeros((len(user_ids1), len(user_ids2)))

        # Check if all embeddings have the same dimension
        dim1 = set(embeddings1[uid].shape[0] for uid in user_ids1)
        dim2 = set(embeddings2[uid].shape[0] for uid in user_ids2)

        # Check if any embeddings are 2D arrays
        has_2d = any(len(embeddings1[uid].shape) > 1 for uid in user_ids1) or \
                any(len(embeddings2[uid].shape) > 1 for uid in user_ids2)

        # Fast path: all embeddings have same dimension and are 1D
        if len(dim1) == 1 and len(dim2) == 1 and not has_2d:
            try:
                # Skip zero embeddings check for better performance
                # Create embedding matrices
                emb_matrix1 = np.vstack([embeddings1[uid] for uid in user_ids1])
                emb_matrix2 = np.vstack([embeddings2[uid] for uid in user_ids2])

                # Normalize matrices for cosine similarity
                norms1 = np.linalg.norm(emb_matrix1, axis=1, keepdims=True)
                norms2 = np.linalg.norm(emb_matrix2, axis=1, keepdims=True)

                # Replace zero norms with 1 to avoid division by zero
                norms1[norms1 == 0] = 1
                norms2[norms2 == 0] = 1

                # Normalize
                emb_matrix1_normalized = emb_matrix1 / norms1
                emb_matrix2_normalized = emb_matrix2 / norms2

                # Compute similarity matrix using matrix multiplication (much faster than cosine_similarity)
                sim_matrix = np.dot(emb_matrix1_normalized, emb_matrix2_normalized.T)

                # Apply zero embedding mask
                for i, uid1 in enumerate(user_ids1):
                    if zero_embeddings1[uid1]:
                        sim_matrix[i, :] = 0

                for j, uid2 in enumerate(user_ids2):
                    if zero_embeddings2[uid2]:
                        sim_matrix[:, j] = 0

                return sim_matrix, user_ids1, user_ids2
            except Exception as e:
                logger.error(f"Error in fast path: {e}")
                # Fall through to slow path

        # Slow path: handle inconsistent dimensions or 2D arrays
        logger.warning("Using slow path for similarity computation")

        # Use parallel processing for large matrices
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        def compute_similarity(i_uid1, embeddings1, embeddings2, user_ids2, zero_embeddings1, zero_embeddings2):
            i, uid1 = i_uid1
            row = np.zeros(len(user_ids2))

            # Skip computation if embedding1 is all zeros
            if zero_embeddings1[uid1]:
                return i, row

            emb1 = embeddings1[uid1]
            # Handle 2D arrays
            if len(emb1.shape) > 1:
                emb1 = emb1[0]

            for j, uid2 in enumerate(user_ids2):
                # Skip computation if embedding2 is all zeros
                if zero_embeddings2[uid2]:
                    row[j] = 0
                    continue

                emb2 = embeddings2[uid2]
                # Handle 2D arrays
                if len(emb2.shape) > 1:
                    emb2 = emb2[0]

                # Use the minimum common dimensions
                min_dim = min(emb1.shape[0], emb2.shape[0])
                emb1_truncated = emb1[:min_dim]
                emb2_truncated = emb2[:min_dim]

                # Compute cosine similarity on the truncated embeddings
                norm1 = np.linalg.norm(emb1_truncated)
                norm2 = np.linalg.norm(emb2_truncated)

                if norm1 == 0 or norm2 == 0:
                    row[j] = 0
                else:
                    # Compute dot product and normalize
                    dot_product = np.dot(emb1_truncated, emb2_truncated)
                    row[j] = dot_product / (norm1 * norm2)

            return i, row

        # Use ThreadPoolExecutor for parallel computation
        compute_func = partial(
            compute_similarity,
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            user_ids2=user_ids2,
            zero_embeddings1=zero_embeddings1,
            zero_embeddings2=zero_embeddings2
        )

        # Only use parallel processing for larger matrices
        if len(user_ids1) * len(user_ids2) > 1000:
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
                for i, row in executor.map(compute_func, enumerate(user_ids1)):
                    sim_matrix[i] = row
        else:
            # For small matrices, sequential is faster due to overhead
            for i_uid1 in enumerate(user_ids1):
                i, row = compute_func(i_uid1)
                sim_matrix[i] = row

        return sim_matrix, user_ids1, user_ids2

    def _cosine_matching(self, embeddings1: Dict[str, np.ndarray],
                        embeddings2: Dict[str, np.ndarray],
                        threshold: float) -> pd.DataFrame:
        """
        Match users using cosine similarity.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            threshold (float): Threshold for matching

        Returns:
            pd.DataFrame: DataFrame with matches and confidence scores
        """
        # Compute similarity matrix
        sim_matrix, user_ids1, user_ids2 = self._compute_similarity_matrix(embeddings1, embeddings2)

        # Find matches using vectorized operations
        # Get indices where similarity is above threshold
        above_threshold = sim_matrix >= threshold

        if not np.any(above_threshold):
            # No matches found
            return pd.DataFrame(columns=['user_id1', 'user_id2', 'confidence'])

        # Get row and column indices of matches
        match_indices = np.where(above_threshold)
        row_indices, col_indices = match_indices

        # Create matches dataframe directly
        matches_data = {
            'user_id1': [user_ids1[i] for i in row_indices],
            'user_id2': [user_ids2[j] for j in col_indices],
            'confidence': sim_matrix[match_indices]
        }

        # Convert to DataFrame
        matches_df = pd.DataFrame(matches_data)

        # Sort by confidence
        if not matches_df.empty:
            matches_df = matches_df.sort_values('confidence', ascending=False)

        return matches_df

    def _frui_p_matching(self, embeddings1: Dict[str, np.ndarray],
                        embeddings2: Dict[str, np.ndarray],
                        threshold: float) -> pd.DataFrame:
        """
        Match users using FRUI-P algorithm (Friend Relationship-based User Identification with Propagation).

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            threshold (float): Threshold for matching

        Returns:
            pd.DataFrame: DataFrame with matches and confidence scores
        """
        # Compute initial similarity matrix
        sim_matrix, user_ids1, user_ids2 = self._compute_similarity_matrix(embeddings1, embeddings2)

        # Create a bipartite graph
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from([f"p1_{uid}" for uid in user_ids1], bipartite=0)
        G.add_nodes_from([f"p2_{uid}" for uid in user_ids2], bipartite=1)

        # Add edges with similarity weights
        for i, user_id1 in enumerate(user_ids1):
            for j, user_id2 in enumerate(user_ids2):
                similarity = sim_matrix[i, j]
                if similarity >= threshold:
                    G.add_edge(f"p1_{user_id1}", f"p2_{user_id2}", weight=similarity)

        # Propagate similarities through the graph
        # This is a simplified version of FRUI-P
        for _ in range(3):  # Number of propagation iterations
            # For each edge, update its weight based on neighboring edges
            new_weights = {}

            for u, v, data in G.edges(data=True):
                # Get neighbors of u and v
                u_neighbors = set(G.neighbors(u))
                v_neighbors = set(G.neighbors(v))

                # Compute propagation score
                prop_score = 0
                for u_neigh in u_neighbors:
                    for v_neigh in v_neighbors:
                        if G.has_edge(u_neigh, v_neigh):
                            prop_score += G[u_neigh][v_neigh]['weight']

                # Normalize by number of possible connections
                norm_factor = max(1, len(u_neighbors) * len(v_neighbors))
                prop_score /= norm_factor

                # Update weight (combine original similarity with propagation)
                new_weight = 0.7 * data['weight'] + 0.3 * prop_score
                new_weights[(u, v)] = new_weight

            # Update edge weights
            for (u, v), weight in new_weights.items():
                G[u][v]['weight'] = weight

        # Extract matches from the graph
        matches = []

        for u, v, data in G.edges(data=True):
            if u.startswith("p1_") and v.startswith("p2_"):
                user_id1 = u[3:]  # Remove "p1_" prefix
                user_id2 = v[3:]  # Remove "p2_" prefix
                confidence = data['weight']

                matches.append({
                    'user_id1': user_id1,
                    'user_id2': user_id2,
                    'confidence': confidence
                })

        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)

        # Sort by confidence
        if not matches_df.empty:
            matches_df = matches_df.sort_values('confidence', ascending=False)

        return matches_df

    def _gsmua_matching(self, embeddings1: Dict[str, np.ndarray],
                       embeddings2: Dict[str, np.ndarray],
                       threshold: float) -> pd.DataFrame:
        """
        Match users using GSMUA (Gradient-based Similarity Matching Using Attention).

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            threshold (float): Threshold for matching

        Returns:
            pd.DataFrame: DataFrame with matches and confidence scores
        """
        # Check if model exists
        if self.model is None:
            logger.warning("GSMUA model not trained. Using cosine similarity instead.")
            return self._cosine_matching(embeddings1, embeddings2, threshold)

        # Get user IDs
        user_ids1 = list(embeddings1.keys())
        user_ids2 = list(embeddings2.keys())

        # Compute similarities using the model
        matches = []

        self.model.eval()
        with torch.no_grad():
            for user_id1 in user_ids1:
                for user_id2 in user_ids2:
                    # Convert embeddings to tensors
                    emb1 = torch.FloatTensor(embeddings1[user_id1]).to(self.device)
                    emb2 = torch.FloatTensor(embeddings2[user_id2]).to(self.device)

                    # Compute similarity
                    similarity = self.model(emb1, emb2).item()

                    if similarity >= threshold:
                        matches.append({
                            'user_id1': user_id1,
                            'user_id2': user_id2,
                            'confidence': similarity
                        })

        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)

        # Sort by confidence
        if not matches_df.empty:
            matches_df = matches_df.sort_values('confidence', ascending=False)

        return matches_df

    def _train_gsmua(self, embeddings1: Dict[str, np.ndarray],
                    embeddings2: Dict[str, np.ndarray],
                    ground_truth: pd.DataFrame) -> None:
        """
        Train GSMUA model using ground truth data.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            ground_truth (pd.DataFrame): Ground truth matches
        """
        logger.info("Training GSMUA model")

        # Check if ground truth has required columns
        required_columns = ['user_id1', 'user_id2']
        if not all(col in ground_truth.columns for col in required_columns):
            raise ValueError(f"Ground truth must have columns: {required_columns}")

        # Filter ground truth to include only users with embeddings
        valid_gt = ground_truth[
            ground_truth['user_id1'].isin(embeddings1.keys()) &
            ground_truth['user_id2'].isin(embeddings2.keys())
        ]

        if len(valid_gt) == 0:
            logger.warning("No valid ground truth matches found. Cannot train GSMUA model.")
            return

        logger.info(f"Training with {len(valid_gt)} ground truth matches")

        # Get embedding dimension
        emb_dim = next(iter(embeddings1.values())).shape[0]

        # Initialize model
        self.model = GSMUAModel(emb_dim).to(self.device)

        # Prepare training data
        positive_pairs = []
        for _, row in valid_gt.iterrows():
            user_id1 = row['user_id1']
            user_id2 = row['user_id2']
            positive_pairs.append((user_id1, user_id2))

        # Generate negative pairs
        negative_pairs = []
        user_ids1 = list(embeddings1.keys())
        user_ids2 = list(embeddings2.keys())

        # Create a set of positive pairs for quick lookup
        positive_set = set(positive_pairs)

        # Generate random negative pairs
        for _ in range(len(positive_pairs) * 3):  # 3x negative samples
            user_id1 = np.random.choice(user_ids1)
            user_id2 = np.random.choice(user_ids2)

            if (user_id1, user_id2) not in positive_set:
                negative_pairs.append((user_id1, user_id2))

        # Train the model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(100):
            total_loss = 0

            # Train on positive pairs
            for user_id1, user_id2 in positive_pairs:
                optimizer.zero_grad()

                # Convert embeddings to tensors
                emb1 = torch.FloatTensor(embeddings1[user_id1]).to(self.device)
                emb2 = torch.FloatTensor(embeddings2[user_id2]).to(self.device)

                # Forward pass
                similarity = self.model(emb1, emb2)

                # Compute loss (target = 1 for positive pairs)
                loss = criterion(similarity, torch.ones_like(similarity))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Train on negative pairs
            for user_id1, user_id2 in negative_pairs:
                optimizer.zero_grad()

                # Convert embeddings to tensors
                emb1 = torch.FloatTensor(embeddings1[user_id1]).to(self.device)
                emb2 = torch.FloatTensor(embeddings2[user_id2]).to(self.device)

                # Forward pass
                similarity = self.model(emb1, emb2)

                # Compute loss (target = 0 for negative pairs)
                loss = criterion(similarity, torch.zeros_like(similarity))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/100, Loss: {total_loss / (len(positive_pairs) + len(negative_pairs)):.4f}")

        logger.info("GSMUA model training completed")

    def _learn_frui_p_params(self, embeddings1: Dict[str, np.ndarray],
                            embeddings2: Dict[str, np.ndarray],
                            ground_truth: pd.DataFrame) -> None:
        """
        Learn parameters for FRUI-P algorithm using ground truth data.

        Args:
            embeddings1 (Dict[str, np.ndarray]): Embeddings from platform 1
            embeddings2 (Dict[str, np.ndarray]): Embeddings from platform 2
            ground_truth (pd.DataFrame): Ground truth matches
        """
        # This is a placeholder for learning FRUI-P parameters
        # In a real implementation, you would optimize the propagation parameters
        logger.info("Learning FRUI-P parameters (placeholder)")

        # For now, we'll just use default parameters
        pass

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding

        Returns:
            np.ndarray: Similarity score
        """
        # Check if embeddings are all zeros (default embeddings)
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            logger.warning("One or both embeddings are zero vectors. Returning zero similarity.")
            return np.array([[0.0]])

        if self.method == 'cosine':
            return cosine_similarity(embedding1, embedding2)
        elif self.method == 'gsmua':
            if self.model is None:
                return cosine_similarity(embedding1, embedding2)
            else:
                # Convert to torch tensors
                emb1 = torch.FloatTensor(embedding1).to(self.device)
                emb2 = torch.FloatTensor(embedding2).to(self.device)

                # Compute similarity
                with torch.no_grad():
                    similarity = self.model(emb1, emb2).item()
                return np.array([[similarity]])
        else:
            return cosine_similarity(embedding1, embedding2)
