"""
Module for generating embeddings from textual content using BERT.
"""

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticEmbedder:
    """
    Class for generating embeddings from textual content using BERT.

    Attributes:
        models (Dict): Dictionary to store models for each platform
        tokenizers (Dict): Dictionary to store tokenizers for each platform
        embeddings (Dict): Dictionary to store generated embeddings for each platform
        device (torch.device): Device to use for computation
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                use_sentence_transformer: bool = True, device: Optional[str] = None):
        """
        Initialize the SemanticEmbedder.

        Args:
            model_name (str): Name of the pre-trained model to use
            use_sentence_transformer (bool): Whether to use SentenceTransformer
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.use_sentence_transformer = use_sentence_transformer
        self.models = {}
        self.tokenizers = {}
        self.embeddings = {}

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"SemanticEmbedder initialized with model {model_name} on {self.device}")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        try:
            if self.use_sentence_transformer:
                # Fix for PyTorch meta tensor issue
                try:
                    # First try with regular loading
                    self.sentence_transformer = SentenceTransformer(self.model_name)
                    self.sentence_transformer.to(self.device)
                except RuntimeError as e:
                    if "meta tensor" in str(e).lower():
                        # If meta tensor error occurs, use CPU first then move to device
                        logger.info("Using CPU loading workaround for meta tensor issue")
                        self.sentence_transformer = SentenceTransformer(self.model_name, device="cpu")
                        self.sentence_transformer.to(self.device)
                    else:
                        raise
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Fix for PyTorch meta tensor issue
                try:
                    # First try with regular loading
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.to(self.device)
                except RuntimeError as e:
                    if "meta tensor" in str(e).lower():
                        # If meta tensor error occurs, use CPU first then move to device
                        logger.info("Using CPU loading workaround for meta tensor issue")
                        self.model = AutoModel.from_pretrained(self.model_name, device_map="cpu")
                        self.model.to(self.device)
                    else:
                        raise
                logger.info(f"Loaded BERT model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, platform_name: str, text_col: str,
                     user_id_col: str, batch_size: int = 32,
                     save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for text data and aggregate by user.

        Args:
            data (pd.DataFrame): DataFrame containing text data
            platform_name (str): Name of the platform
            text_col (str): Name of the column containing text
            user_id_col (str): Name of the column containing user IDs
            batch_size (int): Batch size for processing
            save_path (str, optional): Path to save the embeddings

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        logger.info(f"Generating embeddings for {platform_name} with {len(data)} texts")

        # Check if text column exists
        if text_col not in data.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")

        # Check if user ID column exists
        if user_id_col not in data.columns:
            raise ValueError(f"User ID column '{user_id_col}' not found in data")

        # Filter out rows with empty text
        data = data[data[text_col].notna() & (data[text_col] != '')]

        if len(data) == 0:
            logger.warning(f"No valid text data found for {platform_name}")
            return {}

        # Generate embeddings
        if self.use_sentence_transformer:
            all_embeddings = self._generate_sentence_transformer_embeddings(data[text_col].tolist(), batch_size)
        else:
            all_embeddings = self._generate_bert_embeddings(data[text_col].tolist(), batch_size)

        # Aggregate embeddings by user
        user_embeddings = {}
        for i, user_id in enumerate(data[user_id_col]):
            if user_id not in user_embeddings:
                user_embeddings[user_id] = []
            user_embeddings[user_id].append(all_embeddings[i])

        # Average embeddings for each user
        for user_id in user_embeddings:
            user_embeddings[user_id] = np.mean(user_embeddings[user_id], axis=0)

        # Store embeddings
        self.embeddings[platform_name] = user_embeddings

        # Save embeddings if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Convert to DataFrame
            embeddings_df = pd.DataFrame.from_dict(user_embeddings, orient='index')
            embeddings_df.index.name = user_id_col

            # Save to CSV
            embeddings_df.to_csv(os.path.join(save_path, f"{platform_name}_semantic_embeddings.csv"))

        logger.info(f"Generated embeddings for {len(user_embeddings)} users")
        return user_embeddings

    def transform(self, data: pd.DataFrame, platform_name: str, text_col: str,
                 user_id_col: str, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for new text data using the same model.

        Args:
            data (pd.DataFrame): DataFrame containing text data
            platform_name (str): Name of the platform
            text_col (str): Name of the column containing text
            user_id_col (str): Name of the column containing user IDs
            batch_size (int): Batch size for processing

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        logger.info(f"Transforming {len(data)} texts for {platform_name}")

        # Check if text column exists
        if text_col not in data.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")

        # Check if user ID column exists
        if user_id_col not in data.columns:
            raise ValueError(f"User ID column '{user_id_col}' not found in data")

        # Filter out rows with empty text
        data = data[data[text_col].notna() & (data[text_col] != '')]

        if len(data) == 0:
            logger.warning(f"No valid text data found for {platform_name}")
            return {}

        # Generate embeddings
        if self.use_sentence_transformer:
            all_embeddings = self._generate_sentence_transformer_embeddings(data[text_col].tolist(), batch_size)
        else:
            all_embeddings = self._generate_bert_embeddings(data[text_col].tolist(), batch_size)

        # Aggregate embeddings by user
        user_embeddings = {}
        for i, user_id in enumerate(data[user_id_col]):
            if user_id not in user_embeddings:
                user_embeddings[user_id] = []
            user_embeddings[user_id].append(all_embeddings[i])

        # Average embeddings for each user
        for user_id in user_embeddings:
            user_embeddings[user_id] = np.mean(user_embeddings[user_id], axis=0)

        return user_embeddings

    def _generate_sentence_transformer_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformer.

        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing

        Returns:
            np.ndarray: Array of embeddings
        """
        logger.info(f"Generating SentenceTransformer embeddings for {len(texts)} texts")

        # Generate embeddings
        embeddings = self.sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        return embeddings

    def _generate_bert_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Generate embeddings using BERT.

        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing

        Returns:
            np.ndarray: Array of embeddings
        """
        logger.info(f"Generating BERT embeddings for {len(texts)} texts")

        embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

                # Use CLS token embedding as sentence embedding
                batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)

        # Normalize embeddings
        scaler = StandardScaler()
        all_embeddings = scaler.fit_transform(all_embeddings)

        return all_embeddings

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

    def save_model(self, save_path: str):
        """
        Save the model and tokenizer.

        Args:
            save_path (str): Path to save the model
        """
        os.makedirs(save_path, exist_ok=True)

        if self.use_sentence_transformer:
            self.sentence_transformer.save(os.path.join(save_path, "sentence_transformer"))
        else:
            self.model.save_pretrained(os.path.join(save_path, "bert_model"))
            self.tokenizer.save_pretrained(os.path.join(save_path, "bert_tokenizer"))

        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load the model and tokenizer.

        Args:
            load_path (str): Path to load the model from
        """
        if self.use_sentence_transformer:
            self.sentence_transformer = SentenceTransformer(os.path.join(load_path, "sentence_transformer"))
            self.sentence_transformer.to(self.device)
        else:
            self.model = AutoModel.from_pretrained(os.path.join(load_path, "bert_model"))
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_path, "bert_tokenizer"))
            self.model.to(self.device)

        logger.info(f"Model loaded from {load_path}")
