"""
Ensemble matcher with multiple algorithms for cross-platform user identification.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class GSMUAMatcher(nn.Module):
    """
    Enhanced Graph-based Social Media User Alignment (GSMUA) matcher
    with multi-head attention and improved architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GSMUA matcher.
        
        Args:
            config: Configuration dictionary
        """
        super(GSMUAMatcher, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Network parameters
        self.embedding_dim = config.get('fusion_output_dim', 512)
        self.hidden_dim = config.get('gsmua_hidden_dim', 256)
        self.num_heads = config.get('gsmua_attention_heads', 8)
        self.dropout_prob = config.get('gsmua_dropout', 0.1)
        
        # Multi-head attention for cross-platform comparison
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        
        # Similarity computation
        self.similarity_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between embeddings.
        
        Args:
            emb1: Embeddings from platform 1 [batch_size, embedding_dim]
            emb2: Embeddings from platform 2 [batch_size, embedding_dim]
            
        Returns:
            Similarity scores [batch_size]
        """
        batch_size = emb1.size(0)
        
        # Cross-attention between platforms
        emb1_attended, _ = self.cross_attention(
            emb1.unsqueeze(1), emb2.unsqueeze(1), emb2.unsqueeze(1)
        )
        emb2_attended, _ = self.cross_attention(
            emb2.unsqueeze(1), emb1.unsqueeze(1), emb1.unsqueeze(1)
        )
        
        emb1_attended = emb1_attended.squeeze(1)
        emb2_attended = emb2_attended.squeeze(1)
        
        # Concatenate original and attended embeddings
        combined_features = torch.cat([
            emb1_attended, emb2_attended
        ], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined_features)
        
        # Compute similarity
        similarity_scores = self.similarity_head(features).squeeze(1)
        
        return similarity_scores
    
    def predict(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Predict matches using learned threshold."""
        with torch.no_grad():
            scores = self.forward(emb1, emb2)
            predictions = (scores > self.threshold).float()
        return predictions
    
    def get_threshold(self) -> float:
        """Get the current threshold value."""
        return self.threshold.item()


class FRUIPMatcher:
    """
    Advanced Feature-Rich User Identification across Platforms (FRUI-P) matcher
    with enhanced propagation and weighted features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FRUI-P matcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Propagation parameters
        self.num_iterations = config.get('fruip_iterations', 5)
        self.damping_factor = config.get('fruip_damping', 0.85)
        self.convergence_threshold = config.get('fruip_convergence', 1e-6)
        
        # Feature weights
        self.feature_weights = {
            'semantic': config.get('fruip_semantic_weight', 0.4),
            'network': config.get('fruip_network_weight', 0.3),
            'temporal': config.get('fruip_temporal_weight', 0.2),
            'profile': config.get('fruip_profile_weight', 0.1)
        }
        
        self.similarity_matrix = None
        self.is_fitted = False
        
    def fit(self, embeddings1: Dict[str, np.ndarray], embeddings2: Dict[str, np.ndarray],
            ground_truth: Optional[np.ndarray] = None) -> 'FRUIPMatcher':
        """
        Fit the FRUI-P matcher.
        
        Args:
            embeddings1: Embeddings for platform 1
            embeddings2: Embeddings for platform 2
            ground_truth: Ground truth matching matrix (optional)
            
        Returns:
            self
        """
        self.logger.info("Fitting FRUI-P matcher...")
        
        # Compute weighted similarity matrix
        self.similarity_matrix = self._compute_weighted_similarity(embeddings1, embeddings2)
        
        # Apply iterative propagation
        self.similarity_matrix = self._propagate_similarities(self.similarity_matrix)
        
        self.is_fitted = True
        self.logger.info("FRUI-P matcher fitted successfully")
        
        return self
    
    def _compute_weighted_similarity(self, embeddings1: Dict[str, np.ndarray], 
                                   embeddings2: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted similarity matrix across all features."""
        
        similarity_matrices = {}
        
        for modality in embeddings1.keys():
            if modality in embeddings2 and modality in self.feature_weights:
                # Compute cosine similarity
                emb1 = embeddings1[modality]
                emb2 = embeddings2[modality]
                
                # Normalize embeddings
                emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
                emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
                
                # Compute similarity matrix
                similarity = np.dot(emb1_norm, emb2_norm.T)
                similarity_matrices[modality] = similarity
        
        # Weighted combination
        weighted_similarity = np.zeros((len(embeddings1[list(embeddings1.keys())[0]]), 
                                      len(embeddings2[list(embeddings2.keys())[0]])))
        
        total_weight = 0
        for modality, similarity in similarity_matrices.items():
            weight = self.feature_weights.get(modality, 0)
            weighted_similarity += weight * similarity
            total_weight += weight
        
        if total_weight > 0:
            weighted_similarity /= total_weight
        
        return weighted_similarity
    
    def _propagate_similarities(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply iterative similarity propagation."""
        
        current_matrix = similarity_matrix.copy()
        
        for iteration in range(self.num_iterations):
            # Row-wise normalization (platform 1 perspective)
            row_sums = current_matrix.sum(axis=1, keepdims=True)
            row_normalized = current_matrix / (row_sums + 1e-8)
            
            # Column-wise normalization (platform 2 perspective)
            col_sums = current_matrix.sum(axis=0, keepdims=True)
            col_normalized = current_matrix / (col_sums + 1e-8)
            
            # Update with damping
            new_matrix = (self.damping_factor * (row_normalized + col_normalized) / 2 + 
                         (1 - self.damping_factor) * similarity_matrix)
            
            # Check convergence
            change = np.abs(new_matrix - current_matrix).max()
            if change < self.convergence_threshold:
                self.logger.info(f"FRUI-P converged after {iteration + 1} iterations")
                break
            
            current_matrix = new_matrix
        
        return current_matrix
    
    def predict(self, threshold: float = 0.5) -> np.ndarray:
        """
        Predict matches using the fitted model.
        
        Args:
            threshold: Similarity threshold for matching
            
        Returns:
            Binary matching matrix
        """
        if not self.is_fitted:
            raise ValueError("FRUIPMatcher must be fitted before prediction")
        
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix is None")
        return (self.similarity_matrix > threshold).astype(int)
    
    def get_similarity_scores(self) -> np.ndarray:
        """Get the computed similarity matrix."""
        if not self.is_fitted:
            raise ValueError("FRUIPMatcher must be fitted before getting scores")
        
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix is None")
        return self.similarity_matrix


class GBMatcher:
    """
    Gradient Boosting Matcher using LightGBM for user matching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GB matcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Try to use LightGBM, fall back to sklearn GradientBoosting
        try:
            import lightgbm as lgb
            self.use_lightgbm = True
        except ImportError:
            lgb = None
            self.use_lightgbm = False
            
        if self.use_lightgbm and lgb is not None:
            self.model = lgb.LGBMClassifier(
                n_estimators=config.get('gb_n_estimators', 500),
                learning_rate=config.get('gb_learning_rate', 0.1),
                max_depth=config.get('gb_max_depth', 6),
                num_leaves=config.get('gb_num_leaves', 31),
                subsample=config.get('gb_subsample', 0.8),
                colsample_bytree=config.get('gb_colsample_bytree', 0.8),
                random_state=config.get('random_state', 42),
                verbosity=-1
            )
        else:
            self.use_lightgbm = False
            self.model = GradientBoostingClassifier(
                n_estimators=config.get('gb_n_estimators', 500),
                learning_rate=config.get('gb_learning_rate', 0.1),
                max_depth=config.get('gb_max_depth', 6),
                subsample=config.get('gb_subsample', 0.8),
                random_state=42
            )
        
        self.feature_names = []
        self.is_fitted = False
        
    def _create_features(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Create feature vectors for pairs of embeddings."""
        
        features = []
        
        # Element-wise operations
        features.append(emb1 * emb2)  # Element-wise product
        features.append(np.abs(emb1 - emb2))  # Element-wise absolute difference
        features.append((emb1 + emb2) / 2)  # Element-wise average
        
        # Statistical features
        features.append(np.maximum(emb1, emb2))  # Element-wise maximum
        features.append(np.minimum(emb1, emb2))  # Element-wise minimum
        
        # Similarity measures
        cosine_sim = np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1) + 1e-8
        )
        euclidean_dist = np.linalg.norm(emb1 - emb2, axis=1)
        manhattan_dist = np.sum(np.abs(emb1 - emb2), axis=1)
        
        # Add scalar features
        scalar_features = np.column_stack([
            cosine_sim,
            euclidean_dist,
            manhattan_dist,
            np.linalg.norm(emb1, axis=1),  # L2 norm of emb1
            np.linalg.norm(emb2, axis=1),  # L2 norm of emb2
        ])
        
        # Concatenate all features
        combined_features = np.concatenate(features + [scalar_features], axis=1)
        
        return combined_features
    
    def fit(self, emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray) -> 'GBMatcher':
        """
        Fit the gradient boosting matcher.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            labels: Binary labels (1 for match, 0 for non-match)
            
        Returns:
            self
        """
        self.logger.info("Fitting GB matcher...")
        
        # Create feature vectors
        features = self._create_features(emb1, emb2)
        
        # Fit the model
        self.model.fit(features, labels)
        
        # Store feature names for interpretability
        self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        self.is_fitted = True
        self.logger.info("GB matcher fitted successfully")
        
        return self
    
    def predict(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Predict matches.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("GBMatcher must be fitted before prediction")
        
        features = self._create_features(emb1, emb2)
        predictions = self.model.predict(features)
        return np.array(predictions) if predictions is not None else np.array([])
    
    def predict_proba(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Predict match probabilities.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            
        Returns:
            Match probabilities
        """
        if not self.is_fitted:
            raise ValueError("GBMatcher must be fitted before prediction")
        
        features = self._create_features(emb1, emb2)
        probabilities = self.model.predict_proba(features)
        return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.ravel()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        if self.use_lightgbm:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        return dict(zip(self.feature_names, importance))


class EnsembleCombiner:
    """
    Ensemble combiner using stacking meta-learner and dynamic confidence weighting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble combiner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Base matcher weights (learned dynamically)
        self.matcher_weights = None
        self.use_dynamic_weighting = config.get('use_dynamic_weighting', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        self.is_fitted = False
        
    def fit(self, base_predictions: Dict[str, np.ndarray], 
            base_confidences: Dict[str, np.ndarray],
            labels: np.ndarray) -> 'EnsembleCombiner':
        """
        Fit the ensemble combiner.
        
        Args:
            base_predictions: Dictionary of base matcher predictions
            base_confidences: Dictionary of base matcher confidence scores
            labels: Ground truth labels
            
        Returns:
            self
        """
        self.logger.info("Fitting ensemble combiner...")
        
        # Prepare meta-features
        meta_features = []
        matcher_names = []
        
        for matcher_name in base_predictions.keys():
            if matcher_name in base_confidences:
                meta_features.append(base_predictions[matcher_name])
                meta_features.append(base_confidences[matcher_name])
                matcher_names.extend([f"{matcher_name}_pred", f"{matcher_name}_conf"])
        
        meta_features = np.column_stack(meta_features)
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, labels)
        
        # Compute individual matcher performance for weighting
        self.matcher_weights = {}
        for matcher_name, predictions in base_predictions.items():
            f1 = f1_score(labels, predictions > 0.5)
            self.matcher_weights[matcher_name] = f1
        
        # Normalize weights
        total_weight = sum(self.matcher_weights.values())
        if total_weight > 0:
            self.matcher_weights = {
                name: weight / total_weight 
                for name, weight in self.matcher_weights.items()
            }
        
        self.is_fitted = True
        self.logger.info("Ensemble combiner fitted successfully")
        
        return self
    
    def predict(self, base_predictions: Dict[str, np.ndarray],
                base_confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict using ensemble combination.
        
        Args:
            base_predictions: Dictionary of base matcher predictions
            base_confidences: Dictionary of base matcher confidence scores
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("EnsembleCombiner must be fitted before prediction")
        
        # Meta-learner prediction
        meta_features = []
        for matcher_name in base_predictions.keys():
            if matcher_name in base_confidences:
                meta_features.append(base_predictions[matcher_name])
                meta_features.append(base_confidences[matcher_name])
        
        meta_features = np.column_stack(meta_features)
        meta_predictions = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        # Weighted voting
        if self.use_dynamic_weighting:
            weighted_predictions = self._dynamic_weighted_voting(
                base_predictions, base_confidences
            )
        else:
            weighted_predictions = self._static_weighted_voting(base_predictions)
        
        # Combine meta-learner and weighted voting
        ensemble_predictions = (meta_predictions + weighted_predictions) / 2
        
        return ensemble_predictions
    
    def _static_weighted_voting(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Static weighted voting based on training performance."""
        
        weighted_sum = np.zeros(len(list(base_predictions.values())[0]))
        total_weight = 0
        
        for matcher_name, predictions in base_predictions.items():
            weight = self.matcher_weights.get(matcher_name, 0) if self.matcher_weights else 1.0
            weighted_sum += weight * predictions
            total_weight += weight
        
        return weighted_sum / (total_weight + 1e-8)
    
    def _dynamic_weighted_voting(self, base_predictions: Dict[str, np.ndarray],
                                base_confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """Dynamic weighted voting based on prediction confidence."""
        
        num_samples = len(list(base_predictions.values())[0])
        ensemble_predictions = np.zeros(num_samples)
        
        for i in range(num_samples):
            sample_weights = []
            sample_predictions = []
            
            for matcher_name in base_predictions.keys():
                if matcher_name in base_confidences:
                    confidence = base_confidences[matcher_name][i]
                    base_weight = self.matcher_weights.get(matcher_name, 0) if self.matcher_weights else 1.0
                    
                    # Boost weight for high-confidence predictions
                    if confidence > self.confidence_threshold:
                        dynamic_weight = base_weight * (1 + confidence)
                    else:
                        dynamic_weight = base_weight * confidence
                    
                    sample_weights.append(dynamic_weight)
                    sample_predictions.append(base_predictions[matcher_name][i])
            
            # Weighted average for this sample
            if sum(sample_weights) > 0:
                ensemble_predictions[i] = np.average(sample_predictions, weights=sample_weights)
            else:
                ensemble_predictions[i] = np.mean(sample_predictions)
        
        return ensemble_predictions
    
    def get_matcher_weights(self) -> Dict[str, float]:
        """Get the learned matcher weights."""
        return self.matcher_weights.copy() if self.matcher_weights else {}
    
    def get_meta_learner_coef(self) -> np.ndarray:
        """Get meta-learner coefficients."""
        if self.is_fitted:
            return self.meta_learner.coef_[0]
        return np.array([])


class OptimizedCosineMatcher:
    """
    Optimized cosine similarity matcher with learned threshold and score normalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized cosine matcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.threshold = config.get('cosine_threshold', 0.7)
        self.normalize_scores = config.get('normalize_cosine_scores', True)
        
        # Score normalization parameters
        self.score_mean = 0.0
        self.score_std = 1.0
        
        self.is_fitted = False
        
    def fit(self, emb1: np.ndarray, emb2: np.ndarray, 
            labels: np.ndarray) -> 'OptimizedCosineMatcher':
        """
        Fit the cosine matcher by optimizing threshold.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            labels: Ground truth labels
            
        Returns:
            self
        """
        self.logger.info("Fitting optimized cosine matcher...")
        
        # Compute cosine similarities
        similarities = self._compute_similarities(emb1, emb2)
        
        # Optimize threshold
        self.threshold = self._optimize_threshold(similarities, labels)
        
        # Fit score normalization
        if self.normalize_scores:
            self.score_mean = np.mean(similarities)
            self.score_std = np.std(similarities)
        
        self.is_fitted = True
        self.logger.info(f"Optimized cosine matcher fitted with threshold: {self.threshold:.3f}")
        
        return self
    
    def _compute_similarities(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between embeddings."""
        
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.sum(emb1_norm * emb2_norm, axis=1)
        
        return similarities
    
    def _optimize_threshold(self, similarities: np.ndarray, labels: np.ndarray) -> float:
        """Optimize threshold for best F1 score."""
        
        def objective(threshold):
            predictions = (similarities > threshold[0]).astype(int)
            return -f1_score(labels, predictions)
        
        result = minimize(
            objective, 
            x0=[self.threshold], 
            bounds=[(0.0, 1.0)],
            method='L-BFGS-B'
        )
        
        return result.x[0]
    
    def predict(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Predict matches using optimized cosine similarity.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            
        Returns:
            Binary predictions
        """
        similarities = self._compute_similarities(emb1, emb2)
        
        if self.normalize_scores and self.is_fitted:
            similarities = (similarities - self.score_mean) / (self.score_std + 1e-8)
        
        return (similarities > self.threshold).astype(int)
    
    def predict_proba(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Get similarity scores as probabilities.
        
        Args:
            emb1: Embeddings from platform 1
            emb2: Embeddings from platform 2
            
        Returns:
            Similarity scores
        """
        similarities = self._compute_similarities(emb1, emb2)
        
        if self.normalize_scores and self.is_fitted:
            similarities = (similarities - self.score_mean) / (self.score_std + 1e-8)
            # Apply sigmoid to convert to probabilities
            similarities = 1 / (1 + np.exp(-similarities))
        
        return similarities
    
    def get_threshold(self) -> float:
        """Get the optimized threshold."""
        return self.threshold