"""
Simple semantic embedder using basic TF-IDF and word embeddings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SimpleSemanticEmbedder:
    """
    Simple semantic embedder using TF-IDF and SVD for dimensionality reduction.
    Falls back option when sentence-transformers is not available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simple semantic embedder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.embedding_dim = config.get('semantic_embedding_dim', 384)
        self.max_features = config.get('tfidf_max_features', 10000)
        self.min_df = config.get('tfidf_min_df', 1)  # More flexible for small datasets
        self.max_df = config.get('tfidf_max_df', 0.95)
        self.ngram_range = config.get('tfidf_ngram_range', (1, 2))
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.svd = TruncatedSVD(
            n_components=min(self.embedding_dim, self.max_features),
            random_state=42
        )
        
        # Download NLTK data if needed
        self._download_nltk_data()
        
        # Get stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        self.is_fitted = False
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        nltk_downloads = ['punkt', 'stopwords']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                try:
                    nltk.download(item, quiet=True)
                except:
                    self.logger.warning(f"Failed to download NLTK data: {item}")
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove hashtags and mentions but keep the text
        text = re.sub(r'[#@](\w+)', r'\1', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def fit(self, texts: List[str]) -> 'SimpleSemanticEmbedder':
        """
        Fit the semantic embedder on texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        self.logger.info("Fitting simple semantic embedder...")
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]  # Remove empty texts
        
        if not processed_texts:
            self.logger.warning("No valid texts to fit on")
            return self
        
        # Fit TF-IDF vectorizer with error handling
        try:
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        except ValueError as e:
            if "After pruning" in str(e):
                # Adjust parameters for small datasets
                self.logger.warning(f"TF-IDF pruning error: {e}. Adjusting parameters...")
                self.vectorizer = TfidfVectorizer(
                    max_features=min(self.max_features, len(processed_texts) * 10),
                    min_df=1,  # Most permissive
                    max_df=1.0,  # Most permissive
                    ngram_range=(1, 1),  # Simpler n-grams
                    stop_words=None,  # No stop words for small datasets
                    lowercase=True,
                    strip_accents='unicode'
                )
                tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            else:
                raise e
        
        # Fit SVD for dimensionality reduction
        # Adjust n_components if necessary
        n_features = tfidf_matrix.shape[1]
        if self.svd.n_components > n_features:
            self.logger.warning(f"Reducing SVD components from {self.svd.n_components} to {n_features}")
            self.svd.n_components = min(n_features, self.embedding_dim)

        # Ensure minimum components for SVD
        if self.svd.n_components < 2 and n_features >= 2:
            self.svd.n_components = min(2, n_features)
        elif n_features < 2:
            # For very small feature sets, skip SVD and use TF-IDF directly
            self.logger.warning("Feature set too small for SVD, using TF-IDF directly")
            self.svd = None

        if self.svd is not None:
            self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
        self.logger.info(f"Simple semantic embedder fitted on {len(processed_texts)} texts")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to embeddings.
        
        Args:
            texts: List of text documents
            
        Returns:
            Embedding matrix
        """
        if not self.is_fitted:
            raise ValueError("SimpleSemanticEmbedder must be fitted before transform")
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Handle empty texts
        valid_indices = [i for i, text in enumerate(processed_texts) if text]
        valid_texts = [processed_texts[i] for i in valid_indices]
        
        if not valid_texts:
            # Return zero embeddings for all empty texts
            embedding_dim = self.svd.n_components if self.svd else self.embedding_dim
            return np.zeros((len(texts), embedding_dim))

        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(valid_texts)

        # Apply SVD if available, otherwise use TF-IDF directly
        if self.svd is not None:
            embeddings = self.svd.transform(tfidf_matrix)
        else:
            # Use TF-IDF directly for very small feature sets
            embeddings = tfidf_matrix.toarray()
        
        # Ensure embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2')
        
        # Create full embedding matrix with zeros for empty texts
        full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
        for i, valid_idx in enumerate(valid_indices):
            full_embeddings[valid_idx] = embeddings[i]
        
        return full_embeddings
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the embedder and transform texts in one step.
        
        Args:
            texts: List of text documents
            
        Returns:
            Embedding matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_embeddings_for_dataframe(self, df: pd.DataFrame, 
                                   text_column: str = 'content') -> np.ndarray:
        """
        Get embeddings for a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            Embedding matrix
        """
        if text_column not in df.columns:
            self.logger.warning(f"Column '{text_column}' not found in DataFrame")
            return np.zeros((len(df), self.embedding_dim))
        
        texts = df[text_column].fillna('').astype(str).tolist()
        
        if not self.is_fitted:
            return self.fit_transform(texts)
        else:
            return self.transform(texts)
    
    def get_similarity_matrix(self, embeddings1: np.ndarray, 
                            embeddings2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (optional)
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # Normalize embeddings
        norm1 = normalize(embeddings1, norm='l2')
        norm2 = normalize(embeddings2, norm='l2')
        
        # Ensure arrays are proper numpy arrays
        if not isinstance(norm1, np.ndarray):
            norm1 = np.array(norm1)
        if not isinstance(norm2, np.ndarray):
            norm2 = np.array(norm2)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(norm1, norm2.T)
        
        return similarity_matrix
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the TF-IDF vectorizer."""
        if not self.is_fitted:
            return []
        
        try:
            return self.vectorizer.get_feature_names_out().tolist()
        except AttributeError:
            # Older sklearn versions
            try:
                return self.vectorizer.get_feature_names().tolist()
            except AttributeError:
                return []
    
    def get_top_features(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a given text.
        
        Args:
            text: Input text
            top_k: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if not self.is_fitted:
            return []
        
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return []
        
        # Get TF-IDF scores
        tfidf_scores = self.vectorizer.transform([processed_text])
        feature_names = self.get_feature_names()
        
        if len(feature_names) == 0:
            return []
        
        # Handle sparse matrix
        if hasattr(tfidf_scores, 'nonzero'):
            # Get non-zero features
            non_zero_indices = tfidf_scores.nonzero()[1]
            scores = tfidf_scores.data
            
            # Create feature-score pairs
            feature_scores = []
            for idx, score in zip(non_zero_indices, scores):
                if idx < len(feature_names):
                    feature_scores.append((str(feature_names[idx]), float(score)))
        else:
            # Dense matrix fallback
            scores_array = np.array(tfidf_scores).flatten()
            feature_scores = [(str(feature_names[i]), float(score)) 
                             for i, score in enumerate(scores_array) if score > 0]
        
        # Sort by score and return top k
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_k]
    
    def save_model(self, filepath: str):
        """Save the fitted model."""
        import joblib
        
        model_data = {
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.svd = model_data['svd']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        if not self.is_fitted:
            return {'is_fitted': False}
        
        stats = {
            'is_fitted': True,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'max_features': self.max_features,
        }

        if self.svd is not None:
            stats.update({
                'embedding_dim': self.svd.n_components,
                'explained_variance_ratio': self.svd.explained_variance_ratio_.sum(),
                'n_components': self.svd.n_components
            })
        else:
            stats.update({
                'embedding_dim': len(self.vectorizer.vocabulary_),
                'explained_variance_ratio': 1.0,
                'n_components': len(self.vectorizer.vocabulary_)
            })

        return stats