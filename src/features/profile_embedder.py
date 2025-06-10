"""
Profile embedder for demographics and metadata features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class ProfileEmbedder(nn.Module):
    """
    Neural network for embedding user profile features including demographics,
    account metadata, and activity patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize profile embedder.
        
        Args:
            config: Configuration dictionary
        """
        super(ProfileEmbedder, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Embedding dimensions
        self.embedding_dim = config.get('profile_embedding_dim', 128)
        self.categorical_embedding_dim = config.get('categorical_embedding_dim', 32)
        
        # Scalers and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        
        # Feature categories
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
        
        # Network layers (will be initialized after seeing data)
        self.categorical_embeddings = nn.ModuleDict()
        self.numerical_projection = None
        self.text_projection = None
        self.fusion_layer = None
        self.output_projection = None
        
        self.is_fitted = False
        
    def fit(self, data: Dict[str, pd.DataFrame]) -> 'ProfileEmbedder':
        """
        Fit the embedder on training data.
        
        Args:
            data: Dictionary of DataFrames for each platform
            
        Returns:
            self
        """
        self.logger.info("Fitting profile embedder...")
        
        # Combine all data for feature analysis
        all_data = []
        for platform, df in data.items():
            platform_df = df.copy()
            platform_df['platform'] = platform
            all_data.append(platform_df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Identify feature types
        self._identify_feature_types(combined_df)
        
        # Fit preprocessing components
        self._fit_preprocessing(combined_df)
        
        # Initialize neural network layers
        self._initialize_networks()
        
        self.is_fitted = True
        self.logger.info("Profile embedder fitted successfully")
        
        return self
    
    def _identify_feature_types(self, df: pd.DataFrame):
        """Identify numerical, categorical, and text features."""
        
        # Predefined feature categories based on common social media profile fields
        predefined_numerical = [
            'followers_count', 'following_count', 'posts_count', 'likes_count',
            'comments_count', 'shares_count', 'account_age_days', 'avg_post_length',
            'avg_engagement_rate', 'posting_frequency', 'activity_score'
        ]
        
        predefined_categorical = [
            'platform', 'account_type', 'verification_status', 'gender',
            'age_group', 'country', 'language', 'timezone'
        ]
        
        predefined_text = [
            'bio', 'description', 'interests', 'occupation', 'education'
        ]
        
        # Find existing features in data
        available_columns = set(df.columns)
        
        self.numerical_features = [f for f in predefined_numerical if f in available_columns]
        self.categorical_features = [f for f in predefined_categorical if f in available_columns]
        self.text_features = [f for f in predefined_text if f in available_columns]
        
        # Auto-detect additional numerical features
        for col in df.columns:
            if col not in self.numerical_features + self.categorical_features + self.text_features:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique() > 10:  # Likely continuous
                        self.numerical_features.append(col)
                    else:  # Likely categorical
                        self.categorical_features.append(col)
                elif pd.api.types.is_string_dtype(df[col]):
                    if df[col].nunique() < 50:  # Likely categorical
                        self.categorical_features.append(col)
                    else:  # Likely text
                        self.text_features.append(col)
        
        self.logger.info(f"Identified features - Numerical: {len(self.numerical_features)}, "
                        f"Categorical: {len(self.categorical_features)}, "
                        f"Text: {len(self.text_features)}")
    
    def _fit_preprocessing(self, df: pd.DataFrame):
        """Fit preprocessing components."""
        
        # Fit numerical scaler
        if self.numerical_features:
            numerical_data = df[self.numerical_features].fillna(0)
            self.scaler.fit(numerical_data)
        
        # Fit categorical encoders
        for feature in self.categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                # Handle missing values
                feature_data = df[feature].fillna('unknown').astype(str)
                le.fit(feature_data)
                self.label_encoders[feature] = le
        
        # Fit text vectorizers
        for feature in self.text_features:
            if feature in df.columns:
                tfidf = TfidfVectorizer(
                    max_features=100,  # Limit features for embedding
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2
                )
                # Handle missing values
                text_data = df[feature].fillna('').astype(str)
                tfidf.fit(text_data)
                self.tfidf_vectorizers[feature] = tfidf
    
    def _initialize_networks(self):
        """Initialize neural network layers based on fitted data."""
        
        total_dim = 0
        
        # Categorical embeddings
        for feature in self.categorical_features:
            if feature in self.label_encoders:
                vocab_size = len(self.label_encoders[feature].classes_)
                embedding_dim = min(self.categorical_embedding_dim, vocab_size // 2 + 1)
                self.categorical_embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
                total_dim += embedding_dim
        
        # Numerical features projection
        if self.numerical_features:
            numerical_dim = len(self.numerical_features)
            self.numerical_projection = nn.Sequential(
                nn.Linear(numerical_dim, numerical_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(numerical_dim * 2, self.categorical_embedding_dim)
            )
            total_dim += self.categorical_embedding_dim
        
        # Text features projection
        if self.text_features:
            text_input_dim = sum(
                self.tfidf_vectorizers[f].max_features 
                for f in self.text_features 
                if f in self.tfidf_vectorizers
            )
            if text_input_dim > 0:
                self.text_projection = nn.Sequential(
                    nn.Linear(text_input_dim, text_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(text_input_dim // 2, self.categorical_embedding_dim)
                )
                total_dim += self.categorical_embedding_dim
        
        # Fusion layer
        if total_dim > 0:
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(total_dim // 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
        
        self.logger.info(f"Initialized network with total input dim: {total_dim}, "
                        f"output dim: {self.embedding_dim}")
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Transform profile data to embeddings.
        
        Args:
            data: Dictionary of DataFrames for each platform
            
        Returns:
            Dictionary of embeddings for each platform
        """
        if not self.is_fitted:
            raise ValueError("ProfileEmbedder must be fitted before transform")
        
        embeddings = {}
        
        for platform, df in data.items():
            if len(df) == 0:
                embeddings[platform] = np.array([]).reshape(0, self.embedding_dim)
                continue
            
            platform_embeddings = self._transform_single_platform(df)
            embeddings[platform] = platform_embeddings
        
        return embeddings
    
    def _transform_single_platform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data for a single platform."""
        
        feature_tensors = []
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature in df.columns and feature in self.label_encoders:
                # Handle missing values and unknown categories
                feature_data = df[feature].fillna('unknown').astype(str)
                
                # Transform with label encoder, handling unknown values
                encoded_values = []
                for value in feature_data:
                    try:
                        encoded_values.append(self.label_encoders[feature].transform([value])[0])
                    except ValueError:
                        # Unknown category, use 0 (first class)
                        encoded_values.append(0)
                
                encoded_tensor = torch.LongTensor(encoded_values)
                embedded = self.categorical_embeddings[feature](encoded_tensor)
                feature_tensors.append(embedded)
        
        # Process numerical features
        if self.numerical_features:
            numerical_data = df[self.numerical_features].fillna(0).values
            scaled_data = self.scaler.transform(numerical_data)
            numerical_tensor = torch.FloatTensor(scaled_data)
            
            if self.numerical_projection is not None:
                projected = self.numerical_projection(numerical_tensor)
                feature_tensors.append(projected)
        
        # Process text features
        text_features_list = []
        for feature in self.text_features:
            if feature in df.columns and feature in self.tfidf_vectorizers:
                text_data = df[feature].fillna('').astype(str)
                tfidf_features = self.tfidf_vectorizers[feature].transform(text_data)
                text_features_list.append(tfidf_features.toarray())
        
        if text_features_list:
            combined_text_features = np.hstack(text_features_list)
            text_tensor = torch.FloatTensor(combined_text_features)
            
            if self.text_projection is not None:
                projected = self.text_projection(text_tensor)
                feature_tensors.append(projected)
        
        # Combine all features
        if feature_tensors:
            combined_features = torch.cat(feature_tensors, dim=1)
            
            # Apply fusion layer
            if self.fusion_layer is not None:
                final_embeddings = self.fusion_layer(combined_features)
            else:
                final_embeddings = combined_features
            
            return final_embeddings.detach().numpy()
        else:
            # No features available, return zero embeddings
            return np.zeros((len(df), self.embedding_dim))
    
    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Fit the embedder and transform data in one step.
        
        Args:
            data: Dictionary of DataFrames for each platform
            
        Returns:
            Dictionary of embeddings for each platform
        """
        return self.fit(data).transform(data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on embedding layer weights.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            return {}
        
        importance_scores = {}
        
        # Categorical feature importance (based on embedding variance)
        for feature, embedding_layer in self.categorical_embeddings.items():
            if hasattr(embedding_layer, 'weight'):
                weights = embedding_layer.weight.data
                variance = torch.var(weights, dim=0).mean().item()
                importance_scores[f"categorical_{feature}"] = variance
        
        # Numerical feature importance (based on projection layer weights)
        if self.numerical_projection is not None and hasattr(self.numerical_projection, '__getitem__'):
            try:
                first_layer = self.numerical_projection[0]
                if hasattr(first_layer, 'weight'):
                    for i, feature in enumerate(self.numerical_features):
                        if i < first_layer.weight.data.shape[1]:
                            # Get first layer weights for this feature
                            weight = first_layer.weight.data[:, i]
                            importance = torch.abs(weight).mean().item()
                            importance_scores[f"numerical_{feature}"] = importance
            except (IndexError, AttributeError):
                pass
        
        return importance_scores
    
    def save_state(self, filepath: str):
        """Save the embedder state."""
        state = {
            'config': self.config,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'text_features': self.text_features,
            'is_fitted': self.is_fitted,
            'model_state_dict': self.state_dict(),
            'scaler_state': {
                'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') and hasattr(self.scaler.mean_, 'tolist') else None,
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') and hasattr(self.scaler.scale_, 'tolist') else None
            },
            'label_encoders': {
                name: {
                    'classes_': encoder.classes_.tolist() if hasattr(encoder.classes_, 'tolist') else list(encoder.classes_)
                }
                for name, encoder in self.label_encoders.items()
            }
        }
        
        torch.save(state, filepath)
        self.logger.info(f"Profile embedder state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the embedder state."""
        state = torch.load(filepath)
        
        self.config = state['config']
        self.numerical_features = state['numerical_features']
        self.categorical_features = state['categorical_features']
        self.text_features = state['text_features']
        self.is_fitted = state['is_fitted']
        
        # Restore scaler
        if state['scaler_state']['mean_'] is not None:
            self.scaler.mean_ = np.array(state['scaler_state']['mean_'])
            self.scaler.scale_ = np.array(state['scaler_state']['scale_'])
        
        # Restore label encoders
        for name, encoder_state in state['label_encoders'].items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_state['classes_'])
            self.label_encoders[name] = encoder
        
        # Initialize networks and load weights
        self._initialize_networks()
        self.load_state_dict(state['model_state_dict'])
        
        self.logger.info(f"Profile embedder state loaded from {filepath}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        return {
            'embedding_dim': self.embedding_dim,
            'num_numerical_features': len(self.numerical_features),
            'num_categorical_features': len(self.categorical_features),
            'num_text_features': len(self.text_features),
            'is_fitted': self.is_fitted,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }