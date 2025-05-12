"""
Tests for the DataLoader class.
"""

import unittest
import os
import pandas as pd
import networkx as nx
import shutil
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_generate_synthetic_data(self):
        """Test generating synthetic data."""
        # Generate synthetic data
        synthetic_data = self.data_loader.generate_synthetic_data(
            num_users=100,
            num_platforms=2,
            overlap_ratio=0.7,
            network_density=0.05,
            save_dir=self.test_dir
        )
        
        # Check if data was generated for two platforms
        self.assertEqual(len(synthetic_data), 2)
        
        # Check if platform names are correct
        platform_names = list(synthetic_data.keys())
        self.assertEqual(platform_names[0], 'platform_1')
        self.assertEqual(platform_names[1], 'platform_2')
        
        # Check if profiles were generated
        self.assertIsInstance(synthetic_data['platform_1']['profiles'], pd.DataFrame)
        self.assertEqual(len(synthetic_data['platform_1']['profiles']), 100)
        
        # Check if posts were generated
        self.assertIsInstance(synthetic_data['platform_1']['posts'], pd.DataFrame)
        self.assertGreater(len(synthetic_data['platform_1']['posts']), 0)
        
        # Check if network was generated
        self.assertIsInstance(synthetic_data['platform_1']['network'], nx.Graph)
        self.assertEqual(synthetic_data['platform_1']['network'].number_of_nodes(), 100)
        
        # Check if ground truth was generated
        self.assertIsInstance(self.data_loader.ground_truth, pd.DataFrame)
        self.assertGreater(len(self.data_loader.ground_truth), 0)
        
        # Check if files were saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'platform_1', 'profiles.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'platform_1', 'posts.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'platform_1', 'network.edgelist')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'ground_truth.csv')))

if __name__ == '__main__':
    unittest.main()
