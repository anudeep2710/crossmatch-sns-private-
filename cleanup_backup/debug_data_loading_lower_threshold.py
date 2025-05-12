"""
Debug script to test data loading in the cross-platform user identification project with a lower threshold.
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
    """Test data loading functionality with a lower threshold."""
    try:
        # Initialize the identifier with a lower threshold
        identifier = CrossPlatformUserIdentifier()
        identifier.config['matching_threshold'] = 0.3  # Lower threshold for matching
        
        # Define paths
        linkedin_path = "data/linkedin"
        instagram_path = "data/instagram"
        ground_truth_path = "data/ground_truth.csv"
        
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
            
            # Preprocess data
            logger.info("Preprocessing data...")
            identifier.preprocess()
            logger.info("Preprocessing completed.")
            
            # Extract features
            logger.info("Extracting features...")
            identifier.extract_features()
            logger.info("Feature extraction completed.")
            
            # Try to match users with lower threshold
            logger.info(f"Matching users with threshold {identifier.config['matching_threshold']}...")
            matches = identifier.match_users(
                platform1_name='linkedin',
                platform2_name='instagram',
                embedding_type='fusion'
            )
            logger.info(f"Found {len(matches)} potential matches.")
            
            # Print top matches
            logger.info("Top matches:")
            if not matches.empty:
                print(matches.head(10))
            else:
                print("No matches found.")
            
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
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
