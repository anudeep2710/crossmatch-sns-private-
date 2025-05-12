"""
Example script for cross-platform user identification.
"""

import os
import logging
from src.models.cross_platform_identifier import CrossPlatformUserIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    # Initialize the identifier
    identifier = CrossPlatformUserIdentifier(config_path="config.yaml")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    identifier.generate_synthetic_data(num_users=500, overlap_ratio=0.7)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    identifier.preprocess()
    
    # Extract features
    logger.info("Extracting features...")
    identifier.extract_features()
    
    # Match users
    logger.info("Matching users...")
    platform_names = list(identifier.data.keys())
    matches = identifier.match_users(platform_names[0], platform_names[1])
    
    # Print matches
    logger.info(f"Found {len(matches)} matches")
    print(matches.head(10))
    
    # Evaluate results
    logger.info("Evaluating results...")
    metrics = identifier.evaluate()
    
    # Print metrics
    match_key = f"{platform_names[0]}_{platform_names[1]}_fusion"
    if match_key in metrics:
        print(f"Precision: {metrics[match_key]['precision']:.4f}")
        print(f"Recall: {metrics[match_key]['recall']:.4f}")
        print(f"F1 Score: {metrics[match_key]['f1']:.4f}")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Visualize results
    logger.info("Visualizing results...")
    identifier.visualize("output")
    
    logger.info("Example completed. Results saved to 'output' directory.")

if __name__ == "__main__":
    main()
