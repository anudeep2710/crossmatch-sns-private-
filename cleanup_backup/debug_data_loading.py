"""
Debug script to test data loading in the cross-platform user identification project.
"""

import os
import sys
import logging
import pandas as pd
import networkx as nx
from src.models.cross_platform_identifier import CrossPlatformUserIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test data loading functionality."""
    try:
        # Initialize the identifier
        identifier = CrossPlatformUserIdentifier()
        
        # Define paths
        linkedin_path = "data/linkedin"
        instagram_path = "data/instagram"
        ground_truth_path = "data/ground_truth.csv"
        
        # Check if files exist
        logger.info("Checking if data files exist...")
        
        linkedin_profiles = os.path.join(linkedin_path, "profiles.csv")
        linkedin_posts = os.path.join(linkedin_path, "posts.csv")
        linkedin_network = os.path.join(linkedin_path, "network.edgelist")
        
        instagram_profiles = os.path.join(instagram_path, "profiles.csv")
        instagram_posts = os.path.join(instagram_path, "posts.csv")
        instagram_network = os.path.join(instagram_path, "network.edgelist")
        
        # Check LinkedIn files
        if not os.path.exists(linkedin_profiles):
            logger.error(f"LinkedIn profiles file not found: {linkedin_profiles}")
            return
        
        if not os.path.exists(linkedin_posts):
            logger.warning(f"LinkedIn posts file not found: {linkedin_posts}")
        
        if not os.path.exists(linkedin_network):
            logger.warning(f"LinkedIn network file not found: {linkedin_network}")
        
        # Check Instagram files
        if not os.path.exists(instagram_profiles):
            logger.error(f"Instagram profiles file not found: {instagram_profiles}")
            return
        
        if not os.path.exists(instagram_posts):
            logger.warning(f"Instagram posts file not found: {instagram_posts}")
        
        if not os.path.exists(instagram_network):
            logger.warning(f"Instagram network file not found: {instagram_network}")
        
        # Check ground truth file
        if not os.path.exists(ground_truth_path):
            logger.warning(f"Ground truth file not found: {ground_truth_path}")
        
        # Load data
        logger.info("Loading data...")
        identifier.load_data(
            platform1_path=linkedin_path,
            platform2_path=instagram_path,
            ground_truth_path=ground_truth_path if os.path.exists(ground_truth_path) else None
        )
        
        # Check loaded data
        logger.info(f"Loaded platforms: {list(identifier.data.keys())}")
        
        # Check if both platforms were loaded
        if 'linkedin' in identifier.data and 'instagram' in identifier.data:
            logger.info("Both LinkedIn and Instagram data were loaded successfully.")
            
            # Print some stats
            logger.info(f"LinkedIn profiles: {len(identifier.data['linkedin']['profiles'])}")
            logger.info(f"Instagram profiles: {len(identifier.data['instagram']['profiles'])}")
            
            if 'posts' in identifier.data['linkedin']:
                logger.info(f"LinkedIn posts: {len(identifier.data['linkedin']['posts'])}")
            
            if 'posts' in identifier.data['instagram']:
                logger.info(f"Instagram posts: {len(identifier.data['instagram']['posts'])}")
            
            if 'network' in identifier.data['linkedin']:
                logger.info(f"LinkedIn network nodes: {identifier.data['linkedin']['network'].number_of_nodes()}")
                logger.info(f"LinkedIn network edges: {identifier.data['linkedin']['network'].number_of_edges()}")
            
            if 'network' in identifier.data['instagram']:
                logger.info(f"Instagram network nodes: {identifier.data['instagram']['network'].number_of_nodes()}")
                logger.info(f"Instagram network edges: {identifier.data['instagram']['network'].number_of_edges()}")
            
            # Check ground truth
            if hasattr(identifier.data_loader, 'ground_truth'):
                logger.info(f"Ground truth matches: {len(identifier.data_loader.ground_truth)}")
            
            # Try to preprocess data
            logger.info("Preprocessing data...")
            identifier.preprocess()
            logger.info("Preprocessing completed.")
            
            # Try to extract features
            logger.info("Extracting features...")
            identifier.extract_features()
            logger.info("Feature extraction completed.")
            
            # Try to match users
            logger.info("Matching users...")
            matches = identifier.match_users(
                platform1_name='linkedin',
                platform2_name='instagram',
                embedding_type='fusion'
            )
            logger.info(f"Found {len(matches)} potential matches.")
            
            # Print top matches
            logger.info("Top matches:")
            print(matches.head())
            
            # Try to evaluate
            if hasattr(identifier.data_loader, 'ground_truth'):
                logger.info("Evaluating matches...")
                metrics = identifier.evaluate()
                
                # Print metrics
                for key, value in metrics.items():
                    logger.info(f"Metrics for {key}:")
                    for metric_name, metric_value in value.items():
                        logger.info(f"  {metric_name}: {metric_value}")
        else:
            logger.error("Failed to load both LinkedIn and Instagram data.")
            if 'linkedin' in identifier.data:
                logger.info("LinkedIn data was loaded.")
            if 'instagram' in identifier.data:
                logger.info("Instagram data was loaded.")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
