"""
Run analysis on LinkedIn and Instagram data and display the results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.cross_platform_identifier import CrossPlatformUserIdentifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run analysis and display results."""
    # Initialize identifier
    identifier = CrossPlatformUserIdentifier()
    
    # Update configuration
    config = {
        'network_method': 'node2vec',
        'semantic_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'matching_method': 'cosine',
        'matching_threshold': 0.3  # Lower threshold for better matching
    }
    identifier.config.update(config)
    
    # Define paths
    platform1_path = "data/linkedin"
    platform2_path = "data/instagram"
    ground_truth_path = "data/ground_truth.csv"
    
    # Check if paths exist
    if not os.path.exists(platform1_path):
        logger.error(f"Platform 1 directory not found: {platform1_path}")
        return
    
    if not os.path.exists(platform2_path):
        logger.error(f"Platform 2 directory not found: {platform2_path}")
        return
    
    # Load data
    logger.info("Loading data...")
    try:
        identifier.load_data(
            platform1_path=platform1_path,
            platform2_path=platform2_path,
            ground_truth_path=ground_truth_path if os.path.exists(ground_truth_path) else None
        )
        
        # Get platform names
        platform_names = list(identifier.data.keys())
        if len(platform_names) >= 2:
            platform1_name = platform_names[0]
            platform2_name = platform_names[1]
            
            logger.info(f"Loaded data for {platform1_name} and {platform2_name}")
            
            # Show sample data
            logger.info(f"Sample profiles from {platform1_name}:")
            print(identifier.data[platform1_name]['profiles'].head())
            
            logger.info(f"Sample profiles from {platform2_name}:")
            print(identifier.data[platform2_name]['profiles'].head())
            
            if hasattr(identifier.data_loader, 'ground_truth'):
                logger.info("Sample ground truth:")
                print(identifier.data_loader.ground_truth.head())
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Preprocess data
    logger.info("Preprocessing data...")
    identifier.preprocess()
    
    # Extract features
    logger.info("Extracting features...")
    identifier.extract_features()
    
    # Match users
    logger.info("Matching users...")
    matches = identifier.match_users(
        platform1_name=platform1_name,
        platform2_name=platform2_name,
        embedding_type='fusion'
    )
    
    # Display matches
    logger.info("User matches:")
    print(matches)
    
    # Save matches to CSV
    matches.to_csv("results/matches.csv", index=False)
    logger.info("Matches saved to results/matches.csv")
    
    # Evaluate if ground truth is available
    if hasattr(identifier.data_loader, 'ground_truth'):
        logger.info("Evaluating matches...")
        metrics = identifier.evaluate()
        
        # Display metrics
        match_key = f"{platform1_name}_{platform2_name}_fusion"
        if match_key in metrics:
            logger.info(f"Metrics for {match_key}:")
            for metric_name, metric_value in metrics[match_key].items():
                logger.info(f"  {metric_name}: {metric_value}")
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run analysis
    main()
