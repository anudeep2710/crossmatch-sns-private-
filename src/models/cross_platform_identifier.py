"""
Main module for cross-platform user identification.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime

# Wrap PyTorch imports in try-except to avoid Streamlit file watcher errors
try:
    # Import project modules
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import Preprocessor
    from src.features.network_embedder import NetworkEmbedder
    from src.features.semantic_embedder import SemanticEmbedder
    from src.features.temporal_embedder import TemporalEmbedder
    from src.features.fusion_embedder import FusionEmbedder
    from src.models.user_matcher import UserMatcher
    from src.models.evaluator import Evaluator
    from src.utils.visualizer import Visualizer
    from src.utils.caching import EmbeddingCache, BatchProcessor
except RuntimeError as e:
    if "__path__._path" in str(e):
        # This is the PyTorch/Streamlit file watcher error, we can ignore it
        # and try importing again
        from src.data.data_loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.features.network_embedder import NetworkEmbedder
        from src.features.semantic_embedder import SemanticEmbedder
        from src.features.temporal_embedder import TemporalEmbedder
        from src.features.fusion_embedder import FusionEmbedder
        from src.models.user_matcher import UserMatcher
        from src.models.evaluator import Evaluator
        from src.utils.visualizer import Visualizer
        from src.utils.caching import EmbeddingCache, BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossPlatformUserIdentifier:
    """
    Main class for cross-platform user identification.

    Attributes:
        config (Dict): Configuration dictionary
        data_loader (DataLoader): Data loader instance
        preprocessor (Preprocessor): Preprocessor instance
        network_embedder (NetworkEmbedder): Network embedder instance
        semantic_embedder (SemanticEmbedder): Semantic embedder instance
        temporal_embedder (TemporalEmbedder): Temporal embedder instance
        fusion_embedder (FusionEmbedder): Fusion embedder instance
        user_matcher (UserMatcher): User matcher instance
        evaluator (Evaluator): Evaluator instance
        visualizer (Visualizer): Visualizer instance
        cache (EmbeddingCache): Cache instance
        batch_processor (BatchProcessor): Batch processor instance
        data (Dict): Dictionary to store loaded data
        embeddings (Dict): Dictionary to store generated embeddings
        matches (Dict): Dictionary to store matching results
        metrics (Dict): Dictionary to store evaluation metrics
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CrossPlatformUserIdentifier.

        Args:
            config_path (str, optional): Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor(download_nltk=self.config.get('download_nltk', True))

        # Initialize embedders
        self.network_embedder = NetworkEmbedder(
            embedding_dim=self.config.get('network_embedding_dim', 64),
            walk_length=self.config.get('walk_length', 30),
            num_walks=self.config.get('num_walks', 200),
            p=self.config.get('p', 1.0),
            q=self.config.get('q', 1.0)
        )

        self.semantic_embedder = SemanticEmbedder(
            model_name=self.config.get('semantic_model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            use_sentence_transformer=self.config.get('use_sentence_transformer', True),
            device=self.config.get('device', None)
        )

        self.temporal_embedder = TemporalEmbedder(
            num_time_bins=self.config.get('num_time_bins', 24),
            num_day_bins=self.config.get('num_day_bins', 7),
            normalize=self.config.get('normalize_temporal', True),
            timezone=self.config.get('timezone', 'UTC')
        )

        self.fusion_embedder = FusionEmbedder(
            output_dim=self.config.get('fusion_output_dim', 64),
            fusion_method=self.config.get('fusion_method', 'concat'),
            weights=self.config.get('fusion_weights', None),
            device=self.config.get('device', None)
        )

        # Initialize matcher and evaluator
        self.user_matcher = UserMatcher(
            method=self.config.get('matching_method', 'cosine'),
            threshold=self.config.get('matching_threshold', 0.7),
            device=self.config.get('device', None)
        )

        self.evaluator = Evaluator()

        # Initialize visualizer
        self.visualizer = Visualizer(
            use_plotly=self.config.get('use_plotly', True)
        )

        # Initialize caching
        cache_dir = self.config.get('cache_dir', 'cache')
        self.cache = EmbeddingCache(
            cache_dir=cache_dir,
            use_compression=self.config.get('use_compression', True)
        )

        self.batch_processor = BatchProcessor(cache=self.cache)

        # Initialize data storage
        self.data = {}
        self.embeddings = {}
        self.matches = {}
        self.metrics = {}

        logger.info("CrossPlatformUserIdentifier initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use default.

        Args:
            config_path (str, optional): Path to configuration file

        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            'download_nltk': True,
            'network_embedding_dim': 64,
            'walk_length': 30,
            'num_walks': 200,
            'p': 1.0,
            'q': 1.0,
            'semantic_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_sentence_transformer': True,
            'num_time_bins': 24,
            'num_day_bins': 7,
            'normalize_temporal': True,
            'timezone': 'UTC',
            'fusion_output_dim': 64,
            'fusion_method': 'concat',
            'matching_method': 'cosine',
            'matching_threshold': 0.7,
            'use_plotly': True,
            'cache_dir': 'cache',
            'use_compression': True,
            'batch_size': 32
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Update default config with loaded config
                default_config.update(config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            logger.info("Using default configuration")

        return default_config

    def load_data(self, platform1_path: str, platform2_path: str,
                 ground_truth_path: Optional[str] = None) -> None:
        """
        Load data from different platforms.

        Args:
            platform1_path (str): Path to platform 1 data
            platform2_path (str): Path to platform 2 data
            ground_truth_path (str, optional): Path to ground truth data
        """
        logger.info(f"Loading data from {platform1_path} and {platform2_path}")

        # Extract platform names from paths
        platform1_name = os.path.basename(platform1_path.rstrip('/')) if platform1_path else "platform1"
        platform2_name = os.path.basename(platform2_path.rstrip('/')) if platform2_path else "platform2"

        # Load platform 1 data
        profiles1_path = os.path.join(platform1_path, 'profiles.csv')
        posts1_path = os.path.join(platform1_path, 'posts.csv')
        network1_path = os.path.join(platform1_path, 'network.edgelist')

        if not os.path.exists(profiles1_path):
            raise FileNotFoundError(f"Profiles file not found: {profiles1_path}")

        platform1_data = self.data_loader.load_platform_data(
            platform_name=platform1_name,
            profiles_path=profiles1_path,
            posts_path=posts1_path if os.path.exists(posts1_path) else None,
            network_path=network1_path if os.path.exists(network1_path) else None
        )

        # Load platform 2 data if path is provided
        if platform2_path:
            profiles2_path = os.path.join(platform2_path, 'profiles.csv')
            posts2_path = os.path.join(platform2_path, 'posts.csv')
            network2_path = os.path.join(platform2_path, 'network.edgelist')

            if not os.path.exists(profiles2_path):
                logger.warning(f"Profiles file not found for platform 2: {profiles2_path}")
                platform2_data = {}
            else:
                platform2_data = self.data_loader.load_platform_data(
                    platform_name=platform2_name,
                    profiles_path=profiles2_path,
                    posts_path=posts2_path if os.path.exists(posts2_path) else None,
                    network_path=network2_path if os.path.exists(network2_path) else None
                )
        else:
            logger.warning("No path provided for platform 2. Skipping.")
            platform2_data = {}

        # Load ground truth if provided
        if ground_truth_path and os.path.exists(ground_truth_path):
            self.data_loader.load_ground_truth(ground_truth_path)

        # Store data
        self.data = {platform1_name: platform1_data}

        # Add platform2 data if it exists
        if platform2_data:
            self.data[platform2_name] = platform2_data
            logger.info(f"Loaded data for platforms: {platform1_name}, {platform2_name}")
        else:
            logger.info(f"Loaded data for platform: {platform1_name}")

    def generate_synthetic_data(self, num_users: int = 1000, overlap_ratio: float = 0.7) -> None:
        """
        Generate synthetic data for testing.

        Args:
            num_users (int): Number of users per platform
            overlap_ratio (float): Ratio of users that exist on multiple platforms
        """
        logger.info(f"Generating synthetic data with {num_users} users and {overlap_ratio} overlap ratio")

        # Generate synthetic data
        synthetic_data = self.data_loader.generate_synthetic_data(
            num_users=num_users,
            num_platforms=2,
            overlap_ratio=overlap_ratio,
            network_density=0.05,
            save_dir='data/synthetic'
        )

        # Store data
        self.data = synthetic_data

        logger.info(f"Generated synthetic data for platforms: {list(synthetic_data.keys())}")

    def preprocess(self) -> None:
        """Preprocess loaded data."""
        logger.info("Preprocessing data")

        # Preprocess each platform's data
        for platform_name, platform_data in self.data.items():
            logger.info(f"Preprocessing data for {platform_name}")

            # Preprocess profiles
            if 'profiles' in platform_data:
                platform_data['profiles'] = self.preprocessor.preprocess_profiles(platform_data['profiles'])

            # Preprocess posts
            if 'posts' in platform_data:
                platform_data['posts'] = self.preprocessor.preprocess_posts(platform_data['posts'])

            # Preprocess network
            if 'network' in platform_data:
                platform_data['network'] = self.preprocessor.preprocess_network(platform_data['network'])

        logger.info("Preprocessing completed")

    def extract_features(self) -> None:
        """Extract features and generate embeddings."""
        logger.info("Extracting features and generating embeddings")

        # Initialize embeddings dictionary
        self.embeddings = {}

        # Process each platform
        for platform_name, platform_data in self.data.items():
            logger.info(f"Generating embeddings for {platform_name}")

            platform_embeddings = {}

            # Generate network embeddings if network data is available
            if 'network' in platform_data:
                network_cache_key = f"network_embeddings_{platform_name}"

                if self.cache.exists(network_cache_key):
                    network_embeddings = self.cache.load(network_cache_key)
                else:
                    network_embeddings = self.network_embedder.fit_transform(
                        network=platform_data['network'],
                        platform_name=platform_name,
                        method=self.config.get('network_method', 'node2vec')
                    )
                    self.cache.save(network_cache_key, network_embeddings)

                platform_embeddings['network'] = network_embeddings

            # Generate semantic embeddings if posts data is available
            if 'posts' in platform_data:
                semantic_cache_key = f"semantic_embeddings_{platform_name}"

                if self.cache.exists(semantic_cache_key):
                    semantic_embeddings = self.cache.load(semantic_cache_key)
                else:
                    semantic_embeddings = self.semantic_embedder.fit_transform(
                        data=platform_data['posts'],
                        platform_name=platform_name,
                        text_col='content',
                        user_id_col='user_id',
                        batch_size=self.config.get('batch_size', 32)
                    )
                    self.cache.save(semantic_cache_key, semantic_embeddings)

                platform_embeddings['semantic'] = semantic_embeddings

            # Generate temporal embeddings if posts data is available
            if 'posts' in platform_data:
                temporal_cache_key = f"temporal_embeddings_{platform_name}"

                if self.cache.exists(temporal_cache_key):
                    temporal_embeddings = self.cache.load(temporal_cache_key)
                else:
                    temporal_embeddings = self.temporal_embedder.fit_transform(
                        activity_data=platform_data['posts'],
                        platform_name=platform_name,
                        timestamp_col='timestamp',
                        user_id_col='user_id'
                    )
                    self.cache.save(temporal_cache_key, temporal_embeddings)

                platform_embeddings['temporal'] = temporal_embeddings

            # Fuse embeddings
            fusion_cache_key = f"fusion_embeddings_{platform_name}"

            if self.cache.exists(fusion_cache_key):
                fused_embeddings = self.cache.load(fusion_cache_key)
            else:
                fused_embeddings = self.fusion_embedder.fit_transform(
                    embeddings_dict=platform_embeddings,
                    platform_name=platform_name
                )
                self.cache.save(fusion_cache_key, fused_embeddings)

            platform_embeddings['fusion'] = fused_embeddings

            # Store embeddings
            self.embeddings[platform_name] = platform_embeddings

        logger.info("Feature extraction completed")

    def match_users(self, platform1_name: str, platform2_name: str,
                   embedding_type: str = 'fusion') -> pd.DataFrame:
        """
        Match users across platforms.

        Args:
            platform1_name (str): Name of platform 1
            platform2_name (str): Name of platform 2
            embedding_type (str): Type of embeddings to use for matching

        Returns:
            pd.DataFrame: DataFrame with matches
        """
        logger.info(f"Matching users between {platform1_name} and {platform2_name} using {embedding_type} embeddings")

        # Check if embeddings are available
        if platform1_name not in self.embeddings or platform2_name not in self.embeddings:
            raise ValueError(f"Embeddings not found for platforms: {platform1_name}, {platform2_name}")

        if embedding_type not in self.embeddings[platform1_name] or embedding_type not in self.embeddings[platform2_name]:
            raise ValueError(f"Embedding type {embedding_type} not found for platforms: {platform1_name}, {platform2_name}")

        # Get embeddings
        embeddings1 = self.embeddings[platform1_name][embedding_type]
        embeddings2 = self.embeddings[platform2_name][embedding_type]

        # Train matcher if ground truth is available
        if hasattr(self.data_loader, 'ground_truth'):
            self.user_matcher.fit(embeddings1, embeddings2, self.data_loader.ground_truth)

        # Match users
        matches = self.user_matcher.predict(
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            threshold=self.config.get('matching_threshold', 0.7)
        )

        # Store matches
        match_key = f"{platform1_name}_{platform2_name}_{embedding_type}"
        self.matches[match_key] = matches

        logger.info(f"Found {len(matches)} matches between {platform1_name} and {platform2_name}")

        return matches

    def evaluate(self, ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate matching results.

        Args:
            ground_truth_path (str, optional): Path to ground truth data

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Evaluating matching results")

        # Load ground truth if provided
        if ground_truth_path:
            ground_truth = self.data_loader.load_ground_truth(ground_truth_path)
        elif hasattr(self.data_loader, 'ground_truth'):
            ground_truth = self.data_loader.ground_truth
        else:
            logger.warning("Ground truth not found. Returning default metrics.")
            # Return default metrics
            self.metrics = {
                'default': {
                    'best_threshold': 0.7,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'threshold_metrics': {}
                }
            }
            return self.metrics

        # Check if there are any matches
        if not self.matches:
            logger.warning("No matches found. Returning default metrics.")
            # Return default metrics
            self.metrics = {
                'default': {
                    'best_threshold': 0.7,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': len(ground_truth),
                    'threshold_metrics': {}
                }
            }
            return self.metrics

        # Evaluate each match
        for match_key, matches in self.matches.items():
            logger.info(f"Evaluating matches for {match_key}")

            # Check if matches is a DataFrame
            if not isinstance(matches, pd.DataFrame):
                logger.warning(f"Matches for {match_key} is not a DataFrame. Skipping evaluation.")
                continue

            # Evaluate matches
            metrics = self.evaluator.evaluate(matches, ground_truth)

            # Compute precision-recall curve
            self.evaluator.compute_precision_recall_curve(matches, ground_truth)

            # Compute ROC curve
            self.evaluator.compute_roc_curve(matches, ground_truth)

            # Compute confusion matrix
            self.evaluator.compute_confusion_matrix(
                matches, ground_truth,
                threshold=metrics['best_threshold']
            )

            # Store metrics
            self.metrics[match_key] = metrics

        # If no metrics were computed, return default metrics
        if not self.metrics:
            logger.warning("No metrics computed. Returning default metrics.")
            self.metrics = {
                'default': {
                    'best_threshold': 0.7,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': len(ground_truth),
                    'threshold_metrics': {}
                }
            }

        logger.info("Evaluation completed")

        return self.metrics

    def visualize(self, output_dir: str) -> None:
        """
        Visualize results.

        Args:
            output_dir (str): Directory to save visualizations
        """
        logger.info(f"Visualizing results to {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Visualize networks
        for platform_name, platform_data in self.data.items():
            if 'network' in platform_data:
                network_path = os.path.join(output_dir, f"{platform_name}_network.html")
                self.visualizer.plot_network(
                    network=platform_data['network'],
                    title=f"{platform_name} Network",
                    save_path=network_path
                )

        # Visualize embeddings
        for platform_name, platform_embeddings in self.embeddings.items():
            for emb_type, embeddings in platform_embeddings.items():
                emb_path = os.path.join(output_dir, f"{platform_name}_{emb_type}_embeddings.html")
                self.visualizer.plot_embeddings(
                    embeddings=embeddings,
                    title=f"{platform_name} {emb_type.capitalize()} Embeddings",
                    save_path=emb_path
                )

        # Visualize matches
        for match_key, matches in self.matches.items():
            # Extract platform names from match key
            platform1_name, platform2_name, emb_type = match_key.split('_')

            # Get embeddings
            embeddings1 = self.embeddings[platform1_name][emb_type]
            embeddings2 = self.embeddings[platform2_name][emb_type]

            # Visualize matches
            match_path = os.path.join(output_dir, f"{match_key}_matches.html")
            self.visualizer.plot_matching_results(
                matches=matches,
                embeddings1=embeddings1,
                embeddings2=embeddings2,
                platform1_name=platform1_name,
                platform2_name=platform2_name,
                title=f"Matches between {platform1_name} and {platform2_name} ({emb_type})",
                save_path=match_path
            )

        # Visualize evaluation metrics
        for match_key, metrics in self.metrics.items():
            # Visualize precision-recall curve
            pr_path = os.path.join(output_dir, f"{match_key}_precision_recall.html")
            self.evaluator.plot_precision_recall_curve(save_path=pr_path)

            # Visualize ROC curve
            roc_path = os.path.join(output_dir, f"{match_key}_roc.html")
            self.evaluator.plot_roc_curve(save_path=roc_path)

            # Visualize confusion matrix
            cm_path = os.path.join(output_dir, f"{match_key}_confusion_matrix.html")
            self.evaluator.plot_confusion_matrix(save_path=cm_path)

            # Visualize evaluation metrics
            metrics_path = os.path.join(output_dir, f"{match_key}_metrics.html")
            self.visualizer.plot_evaluation_metrics(metrics, save_path=metrics_path)

            # Save metrics to JSON
            json_path = os.path.join(output_dir, f"{match_key}_metrics.json")
            self.evaluator.save_metrics(json_path)

        logger.info("Visualization completed")

    def create_user_embedding(self, user_data: Dict[str, Any], platform_name: str) -> np.ndarray:
        """
        Create embedding for a single user.

        Args:
            user_data (Dict[str, Any]): User data
            platform_name (str): Name of the platform

        Returns:
            np.ndarray: User embedding
        """
        logger.info(f"Creating embedding for user on {platform_name}")

        # Convert user data to DataFrame
        if 'profile' in user_data:
            profile_df = pd.DataFrame([user_data['profile']])
        else:
            profile_df = pd.DataFrame([user_data])

        if 'posts' in user_data:
            posts_df = pd.DataFrame(user_data['posts'])
        else:
            posts_df = pd.DataFrame()

        # Preprocess data
        profile_df = self.preprocessor.preprocess_profiles(profile_df)
        if not posts_df.empty:
            posts_df = self.preprocessor.preprocess_posts(posts_df)

        # Generate embeddings
        embeddings = {}

        # Generate semantic embeddings if posts are available
        if not posts_df.empty:
            semantic_embeddings = self.semantic_embedder.transform(
                data=posts_df,
                platform_name=platform_name,
                text_col='content',
                user_id_col='user_id',
                batch_size=1
            )
            embeddings['semantic'] = semantic_embeddings

        # Generate temporal embeddings if posts are available
        if not posts_df.empty:
            temporal_embeddings = self.temporal_embedder.transform(
                activity_data=posts_df,
                platform_name=platform_name,
                timestamp_col='timestamp',
                user_id_col='user_id'
            )
            embeddings['temporal'] = temporal_embeddings

        # Fuse embeddings
        if embeddings:
            fused_embeddings = self.fusion_embedder.transform(
                embeddings_dict=embeddings,
                platform_name=platform_name
            )

            # Get the embedding for the user
            user_id = profile_df['user_id'].iloc[0]
            if user_id in fused_embeddings:
                return fused_embeddings[user_id]

        # If no embeddings could be generated, return None
        logger.warning(f"Could not generate embedding for user on {platform_name}")
        return None
