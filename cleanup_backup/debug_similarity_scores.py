"""
Debug script to print similarity scores between users.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
from src.models.cross_platform_identifier import CrossPlatformUserIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Print similarity scores between users."""
    try:
        # Initialize the identifier
        identifier = CrossPlatformUserIdentifier()

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

            # Preprocess data
            logger.info("Preprocessing data...")
            identifier.preprocess()
            logger.info("Preprocessing completed.")

            # Extract features
            logger.info("Extracting features...")
            identifier.extract_features()
            logger.info("Feature extraction completed.")

            # Get embeddings
            linkedin_embeddings = identifier.embeddings['linkedin']['fusion']
            instagram_embeddings = identifier.embeddings['instagram']['fusion']

            # Print embedding dimensions
            logger.info("Embedding dimensions:")
            for user_id, embedding in linkedin_embeddings.items():
                logger.info(f"LinkedIn user {user_id}: {embedding.shape}")

            for user_id, embedding in instagram_embeddings.items():
                logger.info(f"Instagram user {user_id}: {embedding.shape}")

            # Compute similarity matrix
            logger.info("Computing similarity matrix...")

            # Get user IDs
            linkedin_user_ids = list(linkedin_embeddings.keys())
            instagram_user_ids = list(instagram_embeddings.keys())

            # Create a table of similarity scores
            similarity_table = []

            for linkedin_user_id in linkedin_user_ids:
                for instagram_user_id in instagram_user_ids:
                    # Get embeddings
                    linkedin_embedding = linkedin_embeddings[linkedin_user_id]
                    instagram_embedding = instagram_embeddings[instagram_user_id]

                    # Check if embeddings are 2D arrays (network embeddings)
                    if len(linkedin_embedding.shape) > 1 or len(instagram_embedding.shape) > 1:
                        # For 2D arrays, we'll use the first row as the embedding
                        if len(linkedin_embedding.shape) > 1:
                            linkedin_embedding = linkedin_embedding[0]
                        if len(instagram_embedding.shape) > 1:
                            instagram_embedding = instagram_embedding[0]

                    # Compute similarity using cosine similarity
                    min_dim = min(linkedin_embedding.shape[0], instagram_embedding.shape[0])
                    linkedin_embedding_truncated = linkedin_embedding[:min_dim]
                    instagram_embedding_truncated = instagram_embedding[:min_dim]

                    # Compute cosine similarity
                    norm1 = np.linalg.norm(linkedin_embedding_truncated)
                    norm2 = np.linalg.norm(instagram_embedding_truncated)

                    if norm1 == 0 or norm2 == 0:
                        similarity = 0
                    else:
                        # Compute dot product and normalize
                        dot_product = np.dot(linkedin_embedding_truncated, instagram_embedding_truncated)
                        similarity = dot_product / (norm1 * norm2)

                    # Add to table
                    similarity_table.append({
                        'linkedin_user_id': linkedin_user_id,
                        'instagram_user_id': instagram_user_id,
                        'similarity': similarity
                    })

            # Convert to DataFrame
            similarity_df = pd.DataFrame(similarity_table)

            # Sort by similarity
            similarity_df = similarity_df.sort_values('similarity', ascending=False)

            # Print top 20 similarity scores
            logger.info("Top 20 similarity scores:")
            print(similarity_df.head(20))

            # Check if there are any matches in the ground truth
            if hasattr(identifier.data_loader, 'ground_truth'):
                logger.info("Checking ground truth matches...")
                ground_truth = identifier.data_loader.ground_truth

                # Print ground truth
                logger.info("Ground truth:")
                print(ground_truth)

                # Check if any of the top similarity scores match the ground truth
                for _, row in similarity_df.head(20).iterrows():
                    linkedin_user_id = row['linkedin_user_id']
                    instagram_user_id = row['instagram_user_id']

                    # Check if this pair is in the ground truth
                    match = ground_truth[
                        ((ground_truth['user_id_1'] == linkedin_user_id) & (ground_truth['user_id_2'] == instagram_user_id)) |
                        ((ground_truth['user_id_1'] == instagram_user_id) & (ground_truth['user_id_2'] == linkedin_user_id))
                    ]

                    if not match.empty:
                        is_same_user = match.iloc[0]['is_same_user']
                        logger.info(f"Match found in ground truth: {linkedin_user_id} - {instagram_user_id}, is_same_user: {is_same_user}")
        else:
            logger.error("Failed to load both LinkedIn and Instagram data.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
