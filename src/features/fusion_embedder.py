"""
Module for combining different types of embeddings.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GatedFusion(nn.Module):
    """
    Neural network module for gated fusion of embeddings.
    """
    def __init__(self, input_dims: List[int], output_dim: int):
        """
        Initialize the GatedFusion module.

        Args:
            input_dims (List[int]): List of input dimensions for each embedding type
            output_dim (int): Dimension of the output embedding
        """
        super(GatedFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim

        # Create projection layers for each embedding type
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Create gate networks for each embedding type
        self.gates = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for gated fusion.

        Args:
            embeddings (List[torch.Tensor]): List of embeddings to fuse

        Returns:
            torch.Tensor: Fused embedding
        """
        # Project each embedding to the output dimension
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]

        # Compute gate values for each embedding
        gates = [torch.sigmoid(gate(emb)) for gate, emb in zip(self.gates, embeddings)]

        # Apply gates to projected embeddings
        gated = [g * p for g, p in zip(gates, projected)]

        # Sum the gated embeddings
        fused = sum(gated)

        return fused

class FusionEmbedder:
    """
    Class for combining different types of embeddings.

    Attributes:
        embeddings (Dict): Dictionary to store generated embeddings for each platform
        models (Dict): Dictionary to store fusion models for each platform
        device (torch.device): Device to use for computation
    """

    def __init__(self, output_dim: int = 64, fusion_method: str = 'concat',
                weights: Optional[List[float]] = None, device: Optional[str] = None):
        """
        Initialize the FusionEmbedder.

        Args:
            output_dim (int): Dimension of the output embeddings
            fusion_method (str): Method for fusion ('concat', 'average', or 'gated')
            weights (List[float], optional): Weights for weighted average fusion
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        self.weights = weights
        self.embeddings = {}
        self.models = {}

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"FusionEmbedder initialized with {fusion_method} fusion method")

    def fit_transform(self, embeddings_dict: Dict[str, Union[Dict[str, np.ndarray], np.ndarray]],
                     platform_name: str, save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Combine different types of embeddings for a platform.

        Args:
            embeddings_dict (Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]): Dictionary mapping embedding types to user embeddings
            platform_name (str): Name of the platform
            save_path (str, optional): Path to save the embeddings

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to fused embeddings
        """
        logger.info(f"Fusing embeddings for {platform_name} using {self.fusion_method} method")

        # Check if embeddings dictionary is empty
        if not embeddings_dict:
            logger.warning(f"Empty embeddings dictionary for {platform_name}")
            return {}

        # Get all user IDs across all embedding types
        all_user_ids = set()
        for emb_type, user_embs in embeddings_dict.items():
            # Check if user_embs is a dictionary or a numpy array
            if isinstance(user_embs, dict):
                all_user_ids.update(user_embs.keys())
            elif isinstance(user_embs, np.ndarray):
                # If it's a numpy array, we assume it's a single user's embedding
                # or we don't have user IDs, so we'll use a default user ID
                all_user_ids.add('user_0')
                logger.warning(f"Embedding type {emb_type} is a numpy array, not a dictionary. Using default user ID.")

        logger.info(f"Found {len(all_user_ids)} unique users across {len(embeddings_dict)} embedding types")

        # Create fused embeddings for each user
        fused_embeddings = {}

        for user_id in all_user_ids:
            # Collect embeddings for this user
            user_embeddings = []
            embedding_types = []

            for emb_type, user_embs in embeddings_dict.items():
                if isinstance(user_embs, dict) and user_id in user_embs:
                    user_embeddings.append(user_embs[user_id])
                    embedding_types.append(emb_type)
                elif isinstance(user_embs, np.ndarray) and user_id == 'user_0':
                    user_embeddings.append(user_embs)
                    embedding_types.append(emb_type)

            # If no embeddings found, create a default embedding
            if not user_embeddings:
                logger.warning(f"No embeddings found for user {user_id}")
                # Create a default embedding
                default_embedding = np.zeros(64)  # Use a reasonable default size
                fused_embeddings[user_id] = default_embedding
                continue

            # Fuse embeddings
            if self.fusion_method == 'concat':
                fused = self._concat_fusion(user_embeddings)
            elif self.fusion_method == 'average':
                fused = self._average_fusion(user_embeddings, self.weights)
            elif self.fusion_method == 'gated':
                fused = self._gated_fusion(user_embeddings, embedding_types, platform_name)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

            fused_embeddings[user_id] = fused

        # Store embeddings
        self.embeddings[platform_name] = fused_embeddings

        # Save embeddings if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Convert to DataFrame
            embeddings_df = pd.DataFrame.from_dict(fused_embeddings, orient='index')
            embeddings_df.index.name = 'user_id'

            # Save to CSV
            embeddings_df.to_csv(os.path.join(save_path, f"{platform_name}_fused_embeddings.csv"))

            # Save model if using gated fusion
            if self.fusion_method == 'gated' and platform_name in self.models:
                torch.save(self.models[platform_name], os.path.join(save_path, f"{platform_name}_fusion_model.pt"))

        logger.info(f"Generated fused embeddings for {len(fused_embeddings)} users")
        return fused_embeddings
    def transform(self, embeddings_dict: Dict[str, Union[Dict[str, np.ndarray], np.ndarray]],
                 platform_name: str) -> Dict[str, np.ndarray]:
        """
        Combine different types of embeddings for new data.

        Args:
            embeddings_dict (Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]): Dictionary mapping embedding types to user embeddings
            platform_name (str): Name of the platform

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to fused embeddings
        """
        logger.info(f"Transforming embeddings for {platform_name} using {self.fusion_method} method")

        # Check if embeddings dictionary is empty
        if not embeddings_dict:
            logger.warning(f"Empty embeddings dictionary for {platform_name}")
            return {}

        # Get all user IDs across all embedding types
        all_user_ids = set()
        for emb_type, user_embs in embeddings_dict.items():
            # Check if user_embs is a dictionary or a numpy array
            if isinstance(user_embs, dict):
                all_user_ids.update(user_embs.keys())
            elif isinstance(user_embs, np.ndarray):
                # If it's a numpy array, we assume it's a single user's embedding
                # or we don't have user IDs, so we'll use a default user ID
                all_user_ids.add('user_0')
                logger.warning(f"Embedding type {emb_type} is a numpy array, not a dictionary. Using default user ID.")

        # Create fused embeddings for each user
        fused_embeddings = {}

        for user_id in all_user_ids:
            # Collect embeddings for this user
            user_embeddings = []
            embedding_types = []

            for emb_type, user_embs in embeddings_dict.items():
                if isinstance(user_embs, dict) and user_id in user_embs:
                    user_embeddings.append(user_embs[user_id])
                    embedding_types.append(emb_type)
                elif isinstance(user_embs, np.ndarray) and user_id == 'user_0':
                    user_embeddings.append(user_embs)
                    embedding_types.append(emb_type)

            # If no embeddings found, create a default embedding
            if not user_embeddings:
                logger.warning(f"No embeddings found for user {user_id}")
                # Create a default embedding
                default_embedding = np.zeros(64)  # Use a reasonable default size
                fused_embeddings[user_id] = default_embedding
                continue

            # Fuse embeddings
            if self.fusion_method == 'concat':
                fused = self._concat_fusion(user_embeddings)
            elif self.fusion_method == 'average':
                fused = self._average_fusion(user_embeddings, self.weights)
            elif self.fusion_method == 'gated':
                # Check if model exists
                if platform_name not in self.models:
                    logger.warning(f"No fusion model found for {platform_name}. Using average fusion instead.")
                    fused = self._average_fusion(user_embeddings, self.weights)
                else:
                    fused = self._gated_fusion(user_embeddings, embedding_types, platform_name)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

            fused_embeddings[user_id] = fused

        return fused_embeddings

    def _concat_fusion(self, embeddings_list: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate embeddings.

        Args:
            embeddings_list (List[np.ndarray]): List of embeddings to concatenate

        Returns:
            np.ndarray: Concatenated embedding
        """
        # Concatenate embeddings
        concatenated = np.concatenate(embeddings_list)

        # Normalize to unit length (L2 normalization)
        norm = np.linalg.norm(concatenated)
        if norm > 0:
            normalized = concatenated / norm
        else:
            normalized = concatenated

        return normalized

    def _average_fusion(self, embeddings_list: List[np.ndarray],
                       weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Average embeddings with optional weights.

        Args:
            embeddings_list (List[np.ndarray]): List of embeddings to average
            weights (List[float], optional): Weights for each embedding

        Returns:
            np.ndarray: Averaged embedding
        """
        # Normalize each embedding to unit length (L2 normalization)
        normalized_embeddings = []
        for emb in embeddings_list:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized = emb / norm
            else:
                normalized = emb
            normalized_embeddings.append(normalized)

        # Apply weights if provided
        if weights and len(weights) == len(embeddings_list):
            weighted_sum = np.zeros_like(normalized_embeddings[0])
            for i, emb in enumerate(normalized_embeddings):
                weighted_sum += weights[i] * emb
            averaged = weighted_sum / sum(weights)
        else:
            # Simple average
            averaged = np.mean(normalized_embeddings, axis=0)

        # Normalize the result to unit length
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged
    def _gated_fusion(self, embeddings_list: List[np.ndarray],
                     embedding_types: List[str], platform_name: str) -> np.ndarray:
        """
        Fuse embeddings using a gated mechanism.

        Args:
            embeddings_list (List[np.ndarray]): List of embeddings to fuse
            embedding_types (List[str]): Types of embeddings
            platform_name (str): Name of the platform

        Returns:
            np.ndarray: Fused embedding
        """
        # Convert numpy arrays to torch tensors
        embeddings_tensors = [torch.FloatTensor(emb) for emb in embeddings_list]

        # Check if model exists for this platform
        if platform_name not in self.models:
            # Create a new model
            input_dims = [emb.shape[0] for emb in embeddings_list]
            model = GatedFusion(input_dims, self.output_dim)
            model.to(self.device)

            # Train the model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()

            # Move tensors to device
            embeddings_tensors = [emb.to(self.device) for emb in embeddings_tensors]

            # Simple training loop (in a real scenario, you would use a proper loss function)
            for epoch in range(100):
                optimizer.zero_grad()
                output = model(embeddings_tensors)

                # Use a simple reconstruction loss
                loss = 0
                for i, emb in enumerate(embeddings_tensors):
                    proj = nn.Linear(emb.shape[0], output.shape[0]).to(self.device)
                    loss += F.mse_loss(proj(emb), output)

                loss.backward()
                optimizer.step()

            # Store the model
            model.eval()
            self.models[platform_name] = model
        else:
            # Use existing model
            model = self.models[platform_name]
            model.eval()

            # Move tensors to device
            embeddings_tensors = [emb.to(self.device) for emb in embeddings_tensors]

        # Generate fused embedding
        with torch.no_grad():
            fused = model(embeddings_tensors).cpu().numpy()

        return fused

    def get_user_embeddings(self, platform_name: str) -> Dict[str, np.ndarray]:
        """
        Get the stored user embeddings for a platform.

        Args:
            platform_name (str): Name of the platform

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        if platform_name not in self.embeddings:
            logger.warning(f"No embeddings found for {platform_name}")
            return {}

        return self.embeddings[platform_name]
