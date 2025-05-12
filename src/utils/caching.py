"""
Module for caching embeddings and intermediate results.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import hashlib
from tqdm import tqdm
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Class for caching embeddings.

    Attributes:
        cache_dir (str): Directory to store cache files
        use_compression (bool): Whether to compress cache files
    """

    def __init__(self, cache_dir: str = 'cache', use_compression: bool = True):
        """
        Initialize the EmbeddingCache.

        Args:
            cache_dir (str): Directory to store cache files
            use_compression (bool): Whether to compress cache files
        """
        self.cache_dir = cache_dir
        self.use_compression = use_compression

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"EmbeddingCache initialized with cache directory: {cache_dir}")

    def save(self, key: str, embeddings: Union[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]) -> None:
        """
        Save embeddings to cache.

        Args:
            key (str): Cache key
            embeddings (Union[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]): Embeddings to cache
        """
        # Generate cache file path
        cache_file = self._get_cache_file_path(key)

        # Save embeddings
        try:
            if isinstance(embeddings, pd.DataFrame):
                # Save DataFrame as parquet (faster and smaller than CSV)
                embeddings.to_parquet(cache_file + '.parquet', compression='snappy')
                logger.info(f"Saved DataFrame to {cache_file}.parquet")
            else:
                # Save numpy array or dictionary as pickle
                if self.use_compression:
                    # Use higher compression level (5) for better compression ratio
                    joblib.dump(embeddings, cache_file + '.joblib', compress=5)
                    logger.info(f"Saved compressed embeddings to {cache_file}.joblib")
                else:
                    with open(cache_file + '.pkl', 'wb') as f:
                        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Saved embeddings to {cache_file}.pkl")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def load(self, key: str) -> Optional[Union[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]]:
        """
        Load embeddings from cache.

        Args:
            key (str): Cache key

        Returns:
            Optional[Union[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]]: Cached embeddings or None if not found
        """
        # Generate cache file path
        cache_file = self._get_cache_file_path(key)

        # Check if cache file exists
        if os.path.exists(cache_file + '.parquet'):
            # Load DataFrame from parquet
            try:
                embeddings = pd.read_parquet(cache_file + '.parquet')
                logger.info(f"Loaded DataFrame from {cache_file}.parquet")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading DataFrame: {e}")
                return None
        elif os.path.exists(cache_file + '.csv'):
            # Legacy support for CSV
            try:
                embeddings = pd.read_csv(cache_file + '.csv', index_col=0)
                logger.info(f"Loaded DataFrame from {cache_file}.csv")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading DataFrame: {e}")
                return None
        elif os.path.exists(cache_file + '.joblib'):
            # Load compressed pickle
            try:
                embeddings = joblib.load(cache_file + '.joblib')
                logger.info(f"Loaded compressed embeddings from {cache_file}.joblib")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading compressed embeddings: {e}")
                return None
        elif os.path.exists(cache_file + '.pkl'):
            # Load pickle
            try:
                with open(cache_file + '.pkl', 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from {cache_file}.pkl")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                return None
        else:
            logger.info(f"No cache found for key: {key}")
            return None

    def exists(self, key: str) -> bool:
        """
        Check if cache exists for a key.

        Args:
            key (str): Cache key

        Returns:
            bool: True if cache exists, False otherwise
        """
        # Generate cache file path
        cache_file = self._get_cache_file_path(key)

        # Check if any cache file exists
        return (os.path.exists(cache_file + '.parquet') or
                os.path.exists(cache_file + '.csv') or
                os.path.exists(cache_file + '.joblib') or
                os.path.exists(cache_file + '.pkl'))

    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache for a key or all cache.

        Args:
            key (str, optional): Cache key to clear. If None, clear all cache.
        """
        if key:
            # Clear cache for specific key
            cache_file = self._get_cache_file_path(key)

            # Remove all possible cache files
            for ext in ['.parquet', '.csv', '.joblib', '.pkl']:
                if os.path.exists(cache_file + ext):
                    os.remove(cache_file + ext)
                    logger.info(f"Removed cache file: {cache_file}{ext}")
        else:
            # Clear all cache
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed cache file: {file_path}")

            logger.info("Cleared all cache")

    def _get_cache_file_path(self, key: str) -> str:
        """
        Generate cache file path for a key.

        Args:
            key (str): Cache key

        Returns:
            str: Cache file path (without extension)
        """
        # Hash the key to create a filename
        key_hash = hashlib.md5(key.encode()).hexdigest()

        # Create cache file path
        cache_file = os.path.join(self.cache_dir, key_hash)

        return cache_file

class BatchProcessor:
    """
    Class for batch processing of large datasets.

    Attributes:
        cache (EmbeddingCache): Cache for storing intermediate results
    """

    def __init__(self, cache: Optional[EmbeddingCache] = None):
        """
        Initialize the BatchProcessor.

        Args:
            cache (EmbeddingCache, optional): Cache for storing intermediate results
        """
        self.cache = cache
        logger.info("BatchProcessor initialized")

    def process_in_batches(self, data: Union[pd.DataFrame, List, np.ndarray],
                          batch_size: int, process_fn: Callable,
                          cache_key: Optional[str] = None,
                          **kwargs) -> Any:
        """
        Process data in batches.

        Args:
            data (Union[pd.DataFrame, List, np.ndarray]): Data to process
            batch_size (int): Batch size
            process_fn (Callable): Function to process each batch
            cache_key (str, optional): Cache key for storing results
            **kwargs: Additional arguments to pass to process_fn

        Returns:
            Any: Combined results from all batches
        """
        # Check if results are already cached
        if self.cache and cache_key and self.cache.exists(cache_key):
            logger.info(f"Loading cached results for key: {cache_key}")
            return self.cache.load(cache_key)

        logger.info(f"Processing data in batches of size {batch_size}")

        # Determine data length
        if isinstance(data, pd.DataFrame):
            data_len = len(data)
        elif isinstance(data, (list, np.ndarray)):
            data_len = len(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Process in batches
        results = []

        for i in tqdm(range(0, data_len, batch_size), desc="Processing batches"):
            # Get batch
            if isinstance(data, pd.DataFrame):
                batch = data.iloc[i:i+batch_size]
            else:
                batch = data[i:i+batch_size]

            # Process batch
            batch_result = process_fn(batch, **kwargs)

            # Store result
            results.append(batch_result)

        # Combine results
        combined_results = self._combine_results(results)

        # Cache results if requested
        if self.cache and cache_key:
            logger.info(f"Caching results with key: {cache_key}")
            self.cache.save(cache_key, combined_results)

        return combined_results

    def _combine_results(self, results: List[Any]) -> Any:
        """
        Combine results from multiple batches.

        Args:
            results (List[Any]): List of batch results

        Returns:
            Any: Combined results
        """
        if not results:
            return None

        # Check type of first result
        first_result = results[0]

        if isinstance(first_result, pd.DataFrame):
            # Combine DataFrames
            return pd.concat(results, ignore_index=True)
        elif isinstance(first_result, np.ndarray):
            # Combine numpy arrays
            return np.vstack(results)
        elif isinstance(first_result, dict):
            # Combine dictionaries
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        elif isinstance(first_result, list):
            # Combine lists
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        else:
            # Return list of results
            return results
