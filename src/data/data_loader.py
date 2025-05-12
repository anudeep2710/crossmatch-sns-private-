"""
Module for loading data from different social media platforms.
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from faker import Faker
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and handling data from different social media platforms.

    Attributes:
        data (Dict): Dictionary to store loaded data
        platforms (List[str]): List of loaded platform names
    """

    def __init__(self):
        """Initialize the DataLoader."""
        self.data = {}
        self.platforms = []
        self.faker = Faker()
        logger.info("DataLoader initialized")

    def load_platform_data(self, platform_name: str, profiles_path: str,
                          posts_path: Optional[str] = None,
                          network_path: Optional[str] = None) -> Dict:
        """
        Load data for a specific platform.

        Args:
            platform_name (str): Name of the platform (e.g., 'linkedin', 'instagram')
            profiles_path (str): Path to profiles data file (CSV or JSON)
            posts_path (str, optional): Path to posts data file (CSV or JSON)
            network_path (str, optional): Path to network data file (CSV or JSON)

        Returns:
            Dict: Dictionary containing loaded data
        """
        logger.info(f"Loading data for platform: {platform_name}")

        platform_data = {}

        # Load profiles
        if profiles_path:
            if profiles_path.endswith('.csv'):
                profiles = pd.read_csv(profiles_path)
            elif profiles_path.endswith('.json'):
                profiles = self._load_json_as_dataframe(profiles_path)
            else:
                raise ValueError(f"Unsupported file format for profiles: {profiles_path}")

            platform_data['profiles'] = profiles
            logger.info(f"Loaded {len(profiles)} profiles for {platform_name}")

        # Load posts if provided
        if posts_path:
            if posts_path.endswith('.csv'):
                posts = pd.read_csv(posts_path)
            elif posts_path.endswith('.json'):
                posts = self._load_json_as_dataframe(posts_path)
            else:
                raise ValueError(f"Unsupported file format for posts: {posts_path}")

            platform_data['posts'] = posts
            logger.info(f"Loaded {len(posts)} posts for {platform_name}")

        # Load network if provided
        if network_path:
            if network_path.endswith('.csv'):
                network_df = pd.read_csv(network_path)
                network = self._create_network_from_dataframe(network_df)
            elif network_path.endswith('.json'):
                network = self._load_network_from_json(network_path)
            elif network_path.endswith('.edgelist'):
                try:
                    network = nx.read_edgelist(network_path)
                    logger.info(f"Successfully loaded network from edgelist: {network_path}")
                except Exception as e:
                    logger.error(f"Error loading edgelist: {str(e)}")
                    network = nx.Graph()  # Create empty graph as fallback
            else:
                raise ValueError(f"Unsupported file format for network: {network_path}")

            platform_data['network'] = network
            logger.info(f"Loaded network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges for {platform_name}")

        # Store the loaded data
        self.data[platform_name] = platform_data
        if platform_name not in self.platforms:
            self.platforms.append(platform_name)

        return platform_data

    def load_ground_truth(self, ground_truth_path: str) -> pd.DataFrame:
        """
        Load ground truth data for user matching.

        Args:
            ground_truth_path (str): Path to ground truth file (CSV)

        Returns:
            pd.DataFrame: DataFrame containing ground truth matches
        """
        logger.info(f"Loading ground truth from: {ground_truth_path}")

        if ground_truth_path.endswith('.csv'):
            ground_truth = pd.read_csv(ground_truth_path)
        elif ground_truth_path.endswith('.json'):
            ground_truth = self._load_json_as_dataframe(ground_truth_path)
        else:
            raise ValueError(f"Unsupported file format for ground truth: {ground_truth_path}")

        # Standardize column names if needed
        if 'linkedin_id' in ground_truth.columns and 'instagram_id' in ground_truth.columns:
            # Keep the original columns but add standardized ones
            ground_truth['user_id1'] = ground_truth['linkedin_id']
            ground_truth['user_id2'] = ground_truth['instagram_id']
            logger.info("Standardized ground truth column names from linkedin_id/instagram_id to user_id1/user_id2")

        self.ground_truth = ground_truth
        logger.info(f"Loaded {len(ground_truth)} ground truth matches")

        return ground_truth

    def generate_synthetic_data(self, num_users: int, num_platforms: int = 2,
                               overlap_ratio: float = 0.7, network_density: float = 0.05,
                               save_dir: Optional[str] = None) -> Dict:
        """
        Generate synthetic data for testing.

        Args:
            num_users (int): Number of users per platform
            num_platforms (int): Number of platforms to generate
            overlap_ratio (float): Ratio of users that exist on multiple platforms
            network_density (float): Density of the network connections
            save_dir (str, optional): Directory to save generated data

        Returns:
            Dict: Dictionary containing generated data
        """
        logger.info(f"Generating synthetic data for {num_platforms} platforms with {num_users} users each")

        platform_names = [f"platform_{i}" for i in range(1, num_platforms + 1)]
        synthetic_data = {}
        ground_truth = []

        # Calculate number of overlapping users
        num_overlap = int(num_users * overlap_ratio)

        # Generate common pool of users that will appear on multiple platforms
        common_users = []
        for i in range(num_overlap):
            user_id = f"user_{i}"
            name = self.faker.name()
            email = self.faker.email()
            bio = self.faker.text(max_nb_chars=200)

            common_users.append({
                "user_id": user_id,
                "name": name,
                "email": email,
                "bio": bio
            })

        # Generate data for each platform
        for platform_idx, platform_name in enumerate(platform_names):
            # Create profiles
            profiles = []

            # Add overlapping users with some variations
            for user in common_users:
                platform_user_id = f"{platform_name}_{user['user_id']}"

                # Add some noise/variations to simulate real-world differences
                if platform_idx > 0:
                    # Add to ground truth for platforms after the first one
                    ground_truth.append({
                        "platform1": platform_names[0],
                        "user_id1": f"{platform_names[0]}_{user['user_id']}",
                        "platform2": platform_name,
                        "user_id2": platform_user_id
                    })

                # Create profile with variations
                profile = {
                    "user_id": platform_user_id,
                    "name": user['name'] if np.random.random() > 0.2 else self._add_name_variation(user['name']),
                    "email": user['email'] if np.random.random() > 0.3 else self.faker.email(),
                    "bio": user['bio'] if np.random.random() > 0.4 else self.faker.text(max_nb_chars=200),
                    "followers_count": np.random.randint(10, 1000),
                    "following_count": np.random.randint(10, 500),
                    "location": self.faker.city(),
                    "join_date": self.faker.date_time_between(start_date="-5y", end_date="now").strftime("%Y-%m-%d")
                }
                profiles.append(profile)

            # Add platform-specific users
            for i in range(num_users - num_overlap):
                platform_user_id = f"{platform_name}_unique_{i}"
                profile = {
                    "user_id": platform_user_id,
                    "name": self.faker.name(),
                    "email": self.faker.email(),
                    "bio": self.faker.text(max_nb_chars=200),
                    "followers_count": np.random.randint(10, 1000),
                    "following_count": np.random.randint(10, 500),
                    "location": self.faker.city(),
                    "join_date": self.faker.date_time_between(start_date="-5y", end_date="now").strftime("%Y-%m-%d")
                }
                profiles.append(profile)

            # Create posts
            posts = []
            for profile in profiles:
                num_posts = np.random.randint(1, 10)
                for j in range(num_posts):
                    post = {
                        "post_id": f"{profile['user_id']}_post_{j}",
                        "user_id": profile['user_id'],
                        "content": self.faker.text(max_nb_chars=280),
                        "timestamp": self.faker.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
                        "likes": np.random.randint(0, 100),
                        "comments": np.random.randint(0, 20)
                    }
                    posts.append(post)

            # Create network
            network = nx.Graph()
            user_ids = [profile["user_id"] for profile in profiles]
            network.add_nodes_from(user_ids)

            # Add edges based on network density
            num_possible_edges = (len(user_ids) * (len(user_ids) - 1)) // 2
            num_edges = int(num_possible_edges * network_density)

            edges_added = 0
            while edges_added < num_edges:
                user1 = np.random.choice(user_ids)
                user2 = np.random.choice(user_ids)
                if user1 != user2 and not network.has_edge(user1, user2):
                    network.add_edge(user1, user2)
                    edges_added += 1

            # Convert to DataFrames
            profiles_df = pd.DataFrame(profiles)
            posts_df = pd.DataFrame(posts)

            # Store data
            platform_data = {
                "profiles": profiles_df,
                "posts": posts_df,
                "network": network
            }
            synthetic_data[platform_name] = platform_data

            # Save data if directory is provided
            if save_dir:
                os.makedirs(os.path.join(save_dir, platform_name), exist_ok=True)
                profiles_df.to_csv(os.path.join(save_dir, platform_name, "profiles.csv"), index=False)
                posts_df.to_csv(os.path.join(save_dir, platform_name, "posts.csv"), index=False)

                # Save network as edge list
                nx.write_edgelist(network, os.path.join(save_dir, platform_name, "network.edgelist"))

        # Save ground truth if directory is provided
        if save_dir:
            ground_truth_df = pd.DataFrame(ground_truth)
            ground_truth_df.to_csv(os.path.join(save_dir, "ground_truth.csv"), index=False)

        # Store in instance
        self.data.update(synthetic_data)
        self.platforms.extend([p for p in platform_names if p not in self.platforms])
        self.ground_truth = pd.DataFrame(ground_truth)

        logger.info(f"Generated synthetic data with {len(ground_truth)} ground truth matches")

        return synthetic_data

    def _load_json_as_dataframe(self, json_path: str) -> pd.DataFrame:
        """
        Load JSON file as pandas DataFrame.

        Args:
            json_path (str): Path to JSON file

        Returns:
            pd.DataFrame: DataFrame containing JSON data
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to convert dict of dicts to DataFrame
            if all(isinstance(v, dict) for v in data.values()):
                return pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'id'})
            else:
                # Single record, convert to DataFrame with one row
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported JSON structure in: {json_path}")

    def _create_network_from_dataframe(self, df: pd.DataFrame) -> nx.Graph:
        """
        Create a NetworkX graph from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with network data

        Returns:
            nx.Graph: NetworkX graph
        """
        G = nx.Graph()

        # Check if DataFrame has source and target columns
        if 'source' in df.columns and 'target' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['source'], row['target'])
        # Check if DataFrame has from and to columns
        elif 'from' in df.columns and 'to' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['from'], row['to'])
        # Check if DataFrame has user1 and user2 columns
        elif 'user1' in df.columns and 'user2' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['user1'], row['user2'])
        else:
            raise ValueError("DataFrame must have source/target, from/to, or user1/user2 columns")

        return G

    def _load_network_from_json(self, json_path: str) -> nx.Graph:
        """
        Load network data from JSON file.

        Args:
            json_path (str): Path to JSON file

        Returns:
            nx.Graph: NetworkX graph
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        G = nx.Graph()

        # Handle different JSON structures for networks
        if 'nodes' in data and 'links' in data:
            # D3.js format
            for node in data['nodes']:
                if isinstance(node, dict) and 'id' in node:
                    G.add_node(node['id'])
                else:
                    G.add_node(node)

            for link in data['links']:
                if isinstance(link, dict) and 'source' in link and 'target' in link:
                    G.add_edge(link['source'], link['target'])
        elif 'edges' in data:
            # Simple edge list format
            for edge in data['edges']:
                if isinstance(edge, list) and len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
                elif isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                    G.add_edge(edge['source'], edge['target'])
        else:
            # Try to interpret as adjacency list
            for node, neighbors in data.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)

        return G

    def _add_name_variation(self, name: str) -> str:
        """
        Add variation to a name to simulate differences across platforms.

        Args:
            name (str): Original name

        Returns:
            str: Name with variation
        """
        variations = [
            lambda n: n.lower(),
            lambda n: n.upper(),
            lambda n: '.'.join(n.lower().split()),
            lambda n: '_'.join(n.lower().split()),
            lambda n: n.split()[0] if ' ' in n else n,
            lambda n: n.split()[-1] if ' ' in n else n,
            lambda n: n + str(np.random.randint(1, 100)),
            lambda n: n.replace(' ', '')
        ]

        variation_func = np.random.choice(variations)
        return variation_func(name)
