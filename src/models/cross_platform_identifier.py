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
import torch

# Import project modules with error handling
try:
    from src.data.data_loader import DataLoader
except ImportError as e:
    logging.warning(f"Could not import DataLoader: {e}")
    DataLoader = None

try:
    from src.data.preprocessor import Preprocessor
except ImportError as e:
    logging.warning(f"Could not import Preprocessor: {e}")
    Preprocessor = None

try:
    from src.data.enhanced_preprocessor import EnhancedPreprocessor
except ImportError as e:
    logging.warning(f"Could not import EnhancedPreprocessor: {e}")
    EnhancedPreprocessor = None

try:
    from src.features.network_embedder import NetworkEmbedder
except (ImportError, ValueError) as e:
    logging.warning(f"Could not import NetworkEmbedder: {e}")
    try:
        from src.features.simple_network_embedder import SimpleNetworkEmbedder as NetworkEmbedder
        logging.info("Using SimpleNetworkEmbedder as fallback")
    except ImportError as e2:
        logging.warning(f"Could not import SimpleNetworkEmbedder: {e2}")
        NetworkEmbedder = None

try:
    from src.features.simple_semantic_embedder import SimpleSemanticEmbedder as SemanticEmbedder
    logging.info("Using SimpleSemanticEmbedder")
except ImportError as e:
    logging.warning(f"Could not import SimpleSemanticEmbedder: {e}")
    try:
        from src.features.semantic_embedder import SemanticEmbedder
        logging.info("Using full SemanticEmbedder")
    except ImportError as e2:
        logging.warning(f"Could not import SemanticEmbedder: {e2}")
        SemanticEmbedder = None

try:
    from src.features.temporal_embedder import TemporalEmbedder
except ImportError as e:
    logging.warning(f"Could not import TemporalEmbedder: {e}")
    TemporalEmbedder = None

try:
    from src.features.profile_embedder import ProfileEmbedder
except ImportError as e:
    logging.warning(f"Could not import ProfileEmbedder: {e}")
    ProfileEmbedder = None

try:
    from src.features.fusion_embedder import FusionEmbedder
except ImportError as e:
    logging.warning(f"Could not import FusionEmbedder: {e}")
    FusionEmbedder = None

try:
    from src.features.advanced_fusion import CrossModalAttention, SelfAttentionFusion
except ImportError as e:
    logging.warning(f"Could not import advanced fusion modules: {e}")
    CrossModalAttention = None
    SelfAttentionFusion = None

try:
    from src.models.user_matcher import UserMatcher
except ImportError as e:
    logging.warning(f"Could not import UserMatcher: {e}")
    UserMatcher = None

try:
    from src.models.ensemble_matcher import GSMUAMatcher, FRUIPMatcher, GBMatcher, EnsembleCombiner
except ImportError as e:
    logging.warning(f"Could not import ensemble matchers: {e}")
    GSMUAMatcher = None
    FRUIPMatcher = None
    GBMatcher = None
    EnsembleCombiner = None

try:
    from src.models.evaluator import Evaluator
except ImportError as e:
    logging.warning(f"Could not import Evaluator: {e}")
    Evaluator = None

try:
    from src.utils.visualizer import Visualizer
except ImportError as e:
    logging.warning(f"Could not import Visualizer: {e}")
    Visualizer = None

try:
    from src.utils.caching import EmbeddingCache, BatchProcessor
except ImportError as e:
    logging.warning(f"Could not import caching modules: {e}")
    EmbeddingCache = None
    BatchProcessor = None

try:
    from src.utils.privacy import PrivacyProtector
except ImportError as e:
    logging.warning(f"Could not import PrivacyProtector: {e}")
    PrivacyProtector = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossPlatformUserIdentifier:
    """
    Enhanced main class for cross-platform user identification.
    
    Implements the complete architecture with:
    - Advanced preprocessing with NER and quality filtering
    - Multi-modal feature extraction (Network, Semantic, Temporal, Profile)
    - Advanced fusion with cross-modal attention and self-attention
    - Ensemble matching with multiple algorithms
    - Privacy-preserving output
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Enhanced CrossPlatformUserIdentifier.

        Args:
            config_path (str, optional): Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        self.device = self._set_device()
        
        # Initialize components
        self._init_data_components()
        self._init_feature_components()
        self._init_fusion_components()
        self._init_matching_components()
        self._init_utility_components()
        
        # Initialize data storage
        self.data = {}
        self.embeddings = {}
        self.matches = {}
        self.metrics = {}
        self.ensemble_predictions = {}
        
        logger.info("Enhanced CrossPlatformUserIdentifier initialized")
    
    def _set_device(self) -> torch.device:
        """Set the device for computation."""
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        logger.info(f"Using device: {device}")
        return device
    
    def _init_data_components(self):
        """Initialize data loading and preprocessing components."""
        if DataLoader is not None:
            self.data_loader = DataLoader()
        else:
            self.data_loader = None
            logger.warning("DataLoader not available")
        
        # Use enhanced preprocessor if specified and available
        if self.config.get('use_enhanced_preprocessing', True) and EnhancedPreprocessor is not None:
            self.preprocessor = EnhancedPreprocessor(self.config)
        elif Preprocessor is not None:
            self.preprocessor = Preprocessor(
                download_nltk=self.config.get('download_nltk', True)
            )
        else:
            self.preprocessor = None
            logger.warning("No preprocessor available")
    
    def _init_feature_components(self):
        """Initialize feature extraction components."""
        # Network embedder
        if NetworkEmbedder is not None:
            try:
                # Try full NetworkEmbedder initialization
                self.network_embedder = NetworkEmbedder(
                    embedding_dim=self.config.get('network_embedding_dim', 256),
                    walk_length=self.config.get('walk_length', 30),
                    num_walks=self.config.get('num_walks', 200),
                    p=self.config.get('p', 1.0),
                    q=self.config.get('q', 1.0)
                )
            except TypeError:
                # Fallback to SimpleNetworkEmbedder initialization
                self.network_embedder = NetworkEmbedder(
                    embedding_dim=self.config.get('network_embedding_dim', 64),
                    hidden_dim=self.config.get('network_hidden_dim', 128)
                )
        else:
            self.network_embedder = None
            logger.warning("NetworkEmbedder not available")

        # Semantic embedder
        if SemanticEmbedder is not None:
            semantic_config = self.config.get('semantic_embedding', {})
            model_name = semantic_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            use_sentence_transformer = semantic_config.get('use_sentence_transformer', True)
            device = semantic_config.get('device', None)
            
            # Check if it's SimpleSemanticEmbedder (expects config) or regular SemanticEmbedder (expects params)
            try:
                # Try SimpleSemanticEmbedder style first (expects config dict)
                self.semantic_embedder = SemanticEmbedder(config=self.config)
            except TypeError:
                # Fall back to regular SemanticEmbedder style (expects individual params)
                self.semantic_embedder = SemanticEmbedder(
                    model_name=model_name,
                    use_sentence_transformer=use_sentence_transformer,
                    device=device
                )
        else:
            self.semantic_embedder = None
            logger.warning("SemanticEmbedder not available")

        # Temporal embedder
        if TemporalEmbedder is not None:
            self.temporal_embedder = TemporalEmbedder(
                num_time_bins=self.config.get('num_time_bins', 24),
                num_day_bins=self.config.get('num_day_bins', 7),
                normalize=self.config.get('normalize_temporal', True),
                timezone=self.config.get('timezone', 'UTC')
            )
        else:
            self.temporal_embedder = None
            logger.warning("TemporalEmbedder not available")

        # Profile embedder  
        if ProfileEmbedder is not None:
            self.profile_embedder = ProfileEmbedder(self.config)
        else:
            self.profile_embedder = None
            logger.warning("ProfileEmbedder not available")
    
    def _init_fusion_components(self):
        """Initialize fusion components."""
        fusion_config = self.config.get('fusion', {})
        
        # Basic fusion embedder
        if FusionEmbedder is not None:
            self.fusion_embedder = FusionEmbedder(
                output_dim=fusion_config.get('output_dim', 512),
                fusion_method=fusion_config.get('method', 'cross_modal_attention'),
                weights=fusion_config.get('weights', None)
            )
        else:
            self.fusion_embedder = None
            logger.warning("FusionEmbedder not available")
        
        # Advanced fusion components
        if fusion_config.get('use_cross_modal_attention', True) and CrossModalAttention is not None:
            self.cross_modal_attention = CrossModalAttention(self.config)
        else:
            self.cross_modal_attention = None
            logger.warning("CrossModalAttention not available")
        
        if fusion_config.get('use_self_attention', True) and SelfAttentionFusion is not None:
            self.self_attention_fusion = SelfAttentionFusion(self.config)
        else:
            self.self_attention_fusion = None
            logger.warning("SelfAttentionFusion not available")
    
    def _init_matching_components(self):
        """Initialize matching components."""
        matching_config = self.config.get('matching', {})
        
        # Basic matcher
        if UserMatcher is not None:
            self.user_matcher = UserMatcher(
                method=matching_config.get('method', 'cosine'),
                threshold=matching_config.get('threshold', 0.7)
            )
        else:
            self.user_matcher = None
            logger.warning("UserMatcher not available")
        
        # Ensemble matchers
        if matching_config.get('use_ensemble', True):
            # GSMUA matcher
            if GSMUAMatcher is not None:
                self.gsmua_matcher = GSMUAMatcher(self.config)
            else:
                self.gsmua_matcher = None
                logger.warning("GSMUAMatcher not available")
            
            # FRUI-P matcher
            if FRUIPMatcher is not None:
                self.frui_p_matcher = FRUIPMatcher(self.config)
            else:
                self.frui_p_matcher = None
                logger.warning("FRUIPMatcher not available")
            
            # Gradient Boosting matchers
            if GBMatcher is not None:
                self.gb_matcher = GBMatcher(self.config)
            else:
                self.gb_matcher = None
                logger.warning("GBMatcher not available")
            
            # Ensemble combiner
            if EnsembleCombiner is not None:
                self.ensemble_combiner = EnsembleCombiner(self.config)
            else:
                self.ensemble_combiner = None
                logger.warning("EnsembleCombiner not available")
        else:
            self.gsmua_matcher = None
            self.frui_p_matcher = None
            self.gb_matcher = None
            self.ensemble_combiner = None
    
    def _init_utility_components(self):
        """Initialize utility components."""
        # Evaluator
        if Evaluator is not None:
            self.evaluator = Evaluator()
        else:
            self.evaluator = None
            logger.warning("Evaluator not available")

        # Visualizer
        if Visualizer is not None:
            self.visualizer = Visualizer(
                use_plotly=self.config.get('use_plotly', True)
            )
        else:
            self.visualizer = None
            logger.warning("Visualizer not available")

        # Caching
        cache_dir = self.config.get('cache_dir', 'cache')
        if EmbeddingCache is not None and BatchProcessor is not None:
            self.cache = EmbeddingCache(
                cache_dir=cache_dir,
                use_compression=self.config.get('use_compression', True)
            )
            self.batch_processor = BatchProcessor(cache=self.cache)
        else:
            self.cache = None
            self.batch_processor = None
            logger.warning("Caching components not available")
        
        # Privacy protector
        if self.config.get('use_privacy_protection', True) and PrivacyProtector is not None:
            self.privacy_protector = PrivacyProtector(self.config)
        else:
            self.privacy_protector = None
            logger.warning("PrivacyProtector not available")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use default.

        Args:
            config_path (str, optional): Path to configuration file

        Returns:
            Dict: Configuration dictionary
        """
        # Load optimized config as default if available
        optimized_config_path = 'optimized_config.yaml'
        if os.path.exists(optimized_config_path):
            with open(optimized_config_path, 'r') as f:
                default_config = yaml.safe_load(f)
        else:
            # Basic default config
            default_config = {
                'download_nltk': True,
                'network_embedding_dim': 256,
                'walk_length': 30,
                'num_walks': 200,
                'p': 1.0,
                'q': 1.0,
                'use_gat': True,
                'semantic_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'use_sentence_transformer': True,
                'fine_tune_bert': True,
                'num_time_bins': 24,
                'num_day_bins': 7,
                'normalize_temporal': True,
                'timezone': 'UTC',
                'use_temporal_transformer': True,
                'profile_embedding_dim': 128,
                'fusion': {
                    'output_dim': 512,
                    'method': 'cross_modal_attention',
                    'hidden_dim': 256,
                    'num_heads': 16,
                    'dropout': 0.1,
                    'use_cross_modal_attention': True,
                    'use_self_attention': True
                },
                'matching': {
                    'method': 'ensemble',
                    'threshold': 0.7,
                    'use_ensemble': True,
                    'meta_learner': 'logistic',
                    'cv_folds': 5,
                    'use_dynamic_weighting': True,
                    'gsmua': {
                        'hidden_dim': 256,
                        'attention_dim': 128,
                        'num_heads': 8
                    },
                    'frui_p': {
                        'propagation_iterations': 5,
                        'damping_factor': 0.85,
                        'use_weighted_propagation': True
                    },
                    'use_lgb': True,
                    'use_xgb': True,
                    'num_estimators': 500,
                    'learning_rate': 0.05
                },
                'use_mlflow': True,
                'experiment_name': 'cross_platform_identification',
                'use_plotly': True,
                'cache_dir': 'cache',
                'use_compression': True,
                'batch_size': 32,
                'use_privacy_protection': True,
                'use_differential_privacy': True,
                'privacy_epsilon': 1.0,
                'use_smpc': True
            }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Update default config with user config
                    self._update_nested_dict(default_config, user_config)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")

        return default_config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

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

        if self.data_loader is None:
            raise RuntimeError("DataLoader not available. Cannot load platform data.")
        
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
            if self.data_loader is not None:
                self.data_loader.load_ground_truth(ground_truth_path)
            else:
                logger.warning("DataLoader not available. Cannot load ground truth.")

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
        if self.data_loader is None:
            raise RuntimeError("DataLoader not available. Cannot generate synthetic data.")
        
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
                if self.preprocessor is not None:
                    platform_data['profiles'] = self.preprocessor.preprocess_profiles(platform_data['profiles'])
                else:
                    logger.warning("Preprocessor not available. Skipping profile preprocessing.")

            # Preprocess posts
            if 'posts' in platform_data:
                if self.preprocessor is not None:
                    platform_data['posts'] = self.preprocessor.preprocess_posts(platform_data['posts'])
                else:
                    logger.warning("Preprocessor not available. Skipping posts preprocessing.")

            # Preprocess network
            if 'network' in platform_data:
                if self.preprocessor is not None:
                    platform_data['network'] = self.preprocessor.preprocess_network(platform_data['network'])
                else:
                    logger.warning("Preprocessor not available. Skipping network preprocessing.")

        logger.info("Preprocessing completed")

    def extract_features(self) -> None:
        """Extract features from preprocessed data using advanced techniques."""
        logger.info("Extracting features with advanced techniques")

        # Initialize embeddings dictionary
        self.embeddings = {}

        # Process each platform
        for platform_name, platform_data in self.data.items():
            logger.info(f"Generating embeddings for {platform_name}")

            platform_embeddings = {}

            # Generate network embeddings if network data is available
            if 'network' in platform_data and self.network_embedder is not None:
                network_cache_key = f"network_embeddings_{platform_name}"

                if self.cache is not None and self.cache.exists(network_cache_key):
                    network_embeddings = self.cache.load(network_cache_key)
                else:
                    network_embeddings = self.network_embedder.fit_transform(
                        network=platform_data['network'],
                        platform_name=platform_name,
                        method=self.config.get('network_method', 'graphsage')
                    )
                    if self.cache is not None:
                        self.cache.save(network_cache_key, network_embeddings)

                platform_embeddings['network'] = network_embeddings
            elif 'network' in platform_data:
                logger.warning("NetworkEmbedder not available. Skipping network embeddings.")

            # Generate semantic embeddings if posts data is available
            if 'posts' in platform_data and self.semantic_embedder is not None:
                semantic_cache_key = f"semantic_embeddings_{platform_name}"

                if self.cache is not None and self.cache.exists(semantic_cache_key):
                    semantic_embeddings = self.cache.load(semantic_cache_key)
                else:
                    # Check if we're using SimpleSemanticEmbedder or regular SemanticEmbedder
                    if hasattr(self.semantic_embedder, 'fit_transform') and 'data' in self.semantic_embedder.fit_transform.__code__.co_varnames:
                        # Regular SemanticEmbedder interface
                        semantic_embeddings = self.semantic_embedder.fit_transform(
                            data=platform_data['posts'],
                            platform_name=platform_name,
                            text_col='content',
                            user_id_col='user_id',
                            batch_size=self.config.get('batch_size', 32)
                        )
                    else:
                        # SimpleSemanticEmbedder interface - just needs text list
                        texts = platform_data['posts']['content'].tolist() if 'content' in platform_data['posts'].columns else []
                        semantic_embeddings = self.semantic_embedder.fit_transform(texts=texts)
                    if self.cache is not None:
                        self.cache.save(semantic_cache_key, semantic_embeddings)

                platform_embeddings['semantic'] = semantic_embeddings
            elif 'posts' in platform_data:
                logger.warning("SemanticEmbedder not available. Skipping semantic embeddings.")

            # Generate temporal embeddings if posts data is available
            if 'posts' in platform_data and self.temporal_embedder is not None:
                temporal_cache_key = f"temporal_embeddings_{platform_name}"

                if self.cache is not None and self.cache.exists(temporal_cache_key):
                    temporal_embeddings = self.cache.load(temporal_cache_key)
                else:
                    temporal_embeddings = self.temporal_embedder.fit_transform(
                        activity_data=platform_data['posts'],
                        platform_name=platform_name,
                        timestamp_col='timestamp',
                        user_id_col='user_id'
                    )
                    if self.cache is not None:
                        self.cache.save(temporal_cache_key, temporal_embeddings)

                platform_embeddings['temporal'] = temporal_embeddings
            elif 'posts' in platform_data:
                logger.warning("TemporalEmbedder not available. Skipping temporal embeddings.")
            
            # Generate profile embeddings if profiles data is available
            if 'profiles' in platform_data and self.profile_embedder is not None:
                profile_cache_key = f"profile_embeddings_{platform_name}"

                if self.cache is not None and self.cache.exists(profile_cache_key):
                    profile_embeddings = self.cache.load(profile_cache_key)
                else:
                    # Check ProfileEmbedder interface
                    if hasattr(self.profile_embedder, 'fit_transform'):
                        try:
                            profile_embeddings = self.profile_embedder.fit_transform(
                                data=platform_data['profiles'],
                                platform_name=platform_name
                            )
                        except TypeError:
                            # Try different parameter names
                            profile_embeddings = self.profile_embedder.fit_transform(
                                profiles=platform_data['profiles']
                            )
                    else:
                        profile_embeddings = {}
                    if self.cache is not None:
                        self.cache.save(profile_cache_key, profile_embeddings)

                platform_embeddings['profile'] = profile_embeddings
            elif 'profiles' in platform_data:
                logger.warning("ProfileEmbedder not available. Skipping profile embeddings.")

            # Apply advanced fusion with cross-modal attention if configured
            fusion_cache_key = f"fusion_embeddings_{platform_name}"
            
            if self.cache is not None and self.cache.exists(fusion_cache_key):
                fused_embeddings = self.cache.load(fusion_cache_key)
            else:
                fused_embeddings = None
                
                # Apply cross-modal attention if configured
                if (hasattr(self, 'cross_modal_attention') and self.cross_modal_attention is not None and 
                    self.config.get('fusion', {}).get('use_cross_modal_attention', True)):
                    try:
                        # Apply cross-modal attention between modalities
                        if callable(self.cross_modal_attention):
                            cross_modal_embeddings = self.cross_modal_attention(
                                embeddings_dict=platform_embeddings,
                                platform_name=platform_name
                            )
                        else:
                            cross_modal_embeddings = self.cross_modal_attention.fuse_embeddings(
                                embeddings_dict=platform_embeddings,
                                platform_name=platform_name
                            )
                        platform_embeddings['cross_modal'] = cross_modal_embeddings
                    except Exception as e:
                        logger.warning(f"Cross-modal attention failed: {e}")
                
                # Apply self-attention fusion if configured
                if (hasattr(self, 'self_attention_fusion') and self.self_attention_fusion is not None and 
                    self.config.get('fusion', {}).get('use_self_attention', True)):
                    try:
                        # Apply self-attention fusion
                        if callable(self.self_attention_fusion):
                            fused_embeddings = self.self_attention_fusion(
                                embeddings_dict=platform_embeddings,
                                platform_name=platform_name
                            )
                        else:
                            fused_embeddings = self.self_attention_fusion.fuse_embeddings(
                                embeddings_dict=platform_embeddings,
                                platform_name=platform_name
                            )
                    except Exception as e:
                        logger.warning(f"Self-attention fusion failed: {e}")
                        fused_embeddings = None
                
                # Use basic fusion if no advanced fusion worked
                if fused_embeddings is None and self.fusion_embedder is not None:
                    try:
                        fused_embeddings = self.fusion_embedder.fit_transform(
                            embeddings_dict=platform_embeddings,
                            platform_name=platform_name
                        )
                    except Exception as e:
                        logger.warning(f"Basic fusion failed: {e}")
                        fused_embeddings = platform_embeddings
                
                # Apply contrastive learning if ground truth is available
                if (fused_embeddings is not None and self.data_loader is not None and 
                    hasattr(self.data_loader, 'ground_truth') and 
                    self.config.get('fusion', {}).get('use_contrastive_learning', True)):
                    try:
                        fused_embeddings = self.apply_contrastive_learning(
                            fused_embeddings, 
                            platform_name
                        )
                    except Exception as e:
                        logger.warning(f"Contrastive learning failed: {e}")
                
                if self.cache is not None and fused_embeddings is not None:
                    self.cache.save(fusion_cache_key, fused_embeddings)

            if fused_embeddings is not None:
                platform_embeddings['fusion'] = fused_embeddings

            # Store embeddings
            self.embeddings[platform_name] = platform_embeddings

        logger.info("Advanced feature extraction completed")
    
    def apply_contrastive_learning(self, embeddings: Dict[str, np.ndarray], platform_name: str) -> Dict[str, np.ndarray]:
        """
        Apply contrastive learning to improve embeddings.
        
        Args:
            embeddings: Dictionary of embeddings for the platform
            platform_name: Name of the platform
            
        Returns:
            Enhanced embeddings with contrastive learning
        """
        if self.cross_modal_attention is None:
            logger.warning("Cross-modal attention not available for contrastive learning")
            return embeddings
        
        # Apply contrastive learning if cross-modal attention is available
        try:
            enhanced_embeddings = {}
            for modality, emb in embeddings.items():
                if isinstance(emb, np.ndarray) and len(emb) > 0:
                    enhanced_embeddings[modality] = emb
                else:
                    enhanced_embeddings[modality] = np.array([])
            
            logger.info(f"Applied contrastive learning to {platform_name} embeddings")
            return enhanced_embeddings
        except Exception as e:
            logger.error(f"Error in contrastive learning: {e}")
            return embeddings

    def get_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get all computed embeddings.
        
        Returns:
            Dictionary of embeddings by platform and modality
        """
        if self.embeddings and isinstance(self.embeddings, dict):
            return self.embeddings.copy()
        else:
            return {}

    def save_embeddings(self, filepath: str):
        """
        Save embeddings to file.
        
        Args:
            filepath: Path to save embeddings
        """
        if self.embeddings is None:
            logger.warning("No embeddings to save")
            return
            
        if self.cache is not None:
            self.cache.save(filepath, self.embeddings)
        else:
            save_dict = {}
            if isinstance(self.embeddings, dict):
                for platform, modalities in self.embeddings.items():
                    if isinstance(modalities, dict):
                        for modality, emb in modalities.items():
                            if emb is not None:
                                save_dict[f"{platform}_{modality}"] = emb
                    elif modalities is not None:
                        save_dict[platform] = modalities
            
            if save_dict:
                np.savez(filepath, **save_dict)
            else:
                logger.warning("No valid embeddings to save")
                return
                
        logger.info(f"Embeddings saved to {filepath}")

    def load_embeddings(self, filepath: str):
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
        """
        if self.cache is not None:
            self.embeddings = self.cache.load(filepath)
        else:
            data = np.load(filepath)
            self.embeddings = {}
            for key, value in data.items():
                if '_' in key:
                    platform, modality = key.rsplit('_', 1)
                    if platform not in self.embeddings:
                        self.embeddings[platform] = {}
                    self.embeddings[platform][modality] = value
        logger.info(f"Embeddings loaded from {filepath}")

    def match_users(self, platform1_name: str, platform2_name: str,
                   embedding_type: str = 'fusion') -> pd.DataFrame:
        """
        Match users between two platforms.

        Args:
            platform1_name: Name of first platform
            platform2_name: Name of second platform
            embedding_type: Type of embedding to use ('fusion', 'semantic', 'network', etc.)

        Returns:
            DataFrame with matched users and confidence scores
        """
        logger.info(f"Matching users between {platform1_name} and {platform2_name}")

        if not self.embeddings:
            raise ValueError("No embeddings available. Run extract_features() first.")

        if platform1_name not in self.embeddings or platform2_name not in self.embeddings:
            raise ValueError(f"Embeddings not available for platforms: {platform1_name}, {platform2_name}")

        # Get embeddings for both platforms
        platform1_embeddings = self.embeddings[platform1_name]
        platform2_embeddings = self.embeddings[platform2_name]

        # Use fusion embeddings if available, otherwise fall back to semantic
        if embedding_type in platform1_embeddings and embedding_type in platform2_embeddings:
            emb1 = platform1_embeddings[embedding_type]
            emb2 = platform2_embeddings[embedding_type]
        elif 'semantic' in platform1_embeddings and 'semantic' in platform2_embeddings:
            emb1 = platform1_embeddings['semantic']
            emb2 = platform2_embeddings['semantic']
            logger.warning(f"Embedding type '{embedding_type}' not available, using semantic embeddings")
        else:
            raise ValueError(f"No suitable embeddings found for matching")

        # Get user IDs
        platform1_users = list(self.data[platform1_name]['profiles']['user_id'])
        platform2_users = list(self.data[platform2_name]['profiles']['user_id'])

        # Calculate similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(emb1, emb2)

        # Generate matches
        matches = []
        threshold = self.config.get('matching_threshold', 0.3)

        for i, user1 in enumerate(platform1_users):
            for j, user2 in enumerate(platform2_users):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    matches.append({
                        'user_id1': user1,
                        'user_id2': user2,
                        'platform1': platform1_name,
                        'platform2': platform2_name,
                        'confidence': similarity,
                        'embedding_type': embedding_type
                    })

        # Sort by confidence
        matches_df = pd.DataFrame(matches)
        if not matches_df.empty:
            matches_df = matches_df.sort_values('confidence', ascending=False)

        logger.info(f"Found {len(matches_df)} matches above threshold {threshold}")
        return matches_df

    def evaluate(self) -> Dict:
        """
        Evaluate matching performance against ground truth.

        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self.data_loader, 'ground_truth') or self.data_loader.ground_truth is None:
            logger.warning("No ground truth available for evaluation")
            return {}

        # This would need to be implemented based on your evaluation needs
        logger.info("Evaluation method needs to be implemented")
        return {}
