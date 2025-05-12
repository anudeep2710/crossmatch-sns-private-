"""
Module for evaluating cross-platform user identification results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    """
    Class for evaluating cross-platform user identification results.
    
    Attributes:
        metrics (Dict): Dictionary to store evaluation metrics
    """
    
    def __init__(self):
        """Initialize the Evaluator."""
        self.metrics = {}
        logger.info("Evaluator initialized")
        
    def _get_id_columns(self, ground_truth):
        """
        Get the column names for user IDs in the ground truth DataFrame.
        
        Args:
            ground_truth (pd.DataFrame): Ground truth DataFrame
            
        Returns:
            tuple: Column names for user ID 1 and user ID 2
        """
        if 'user_id1' in ground_truth.columns and 'user_id2' in ground_truth.columns:
            return 'user_id1', 'user_id2'
        elif 'linkedin_id' in ground_truth.columns and 'instagram_id' in ground_truth.columns:
            return 'linkedin_id', 'instagram_id'
        else:
            raise ValueError("Ground truth DataFrame must have columns: ['user_id1', 'user_id2'] or ['linkedin_id', 'instagram_id']")
    
    def evaluate(self, matches: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate matching results against ground truth.
        
        Args:
            matches (pd.DataFrame): DataFrame with predicted matches
            ground_truth (pd.DataFrame): DataFrame with ground truth matches
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {len(matches)} matches against {len(ground_truth)} ground truth matches")
        
        # Check if DataFrames are empty
        if matches.empty or ground_truth.empty:
            logger.warning("Matches or ground truth DataFrame is empty. Returning default metrics.")
            # Return default metrics
            self.metrics = {
                'best_threshold': 0.7,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(ground_truth),
                'threshold_metrics': {}
            }
            return self.metrics
        
        # Check if DataFrames have required columns
        required_columns = ['user_id1', 'user_id2', 'confidence']
        if not all(col in matches.columns for col in required_columns):
            logger.warning(f"Matches DataFrame missing required columns: {required_columns}. Returning default metrics.")
            # Return default metrics
            self.metrics = {
                'best_threshold': 0.7,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(ground_truth),
                'threshold_metrics': {}
            }
            return self.metrics
        
        # Get the column names for user IDs
        id1_col, id2_col = self._get_id_columns(ground_truth)
        
        # Create a set of ground truth pairs for quick lookup
        gt_pairs = set()
        for _, row in ground_truth.iterrows():
            gt_pairs.add((row[id1_col], row[id2_col]))
            # Also add the reverse pair to handle different orderings
            gt_pairs.add((row[id2_col], row[id1_col]))
        
        # Evaluate at different thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = {}
        
        for threshold in thresholds:
            # Filter matches by threshold
            filtered_matches = matches[matches['confidence'] >= threshold]
            
            # Compute metrics
            tp = 0  # True positives
            fp = 0  # False positives
            
            for _, row in filtered_matches.iterrows():
                if (row['user_id1'], row['user_id2']) in gt_pairs or (row['user_id2'], row['user_id1']) in gt_pairs:
                    tp += 1
                else:
                    fp += 1
            
            # Compute precision and recall
            fn = len(gt_pairs) // 2 - tp  # Divide by 2 because we added both directions
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-10, precision + recall)
            
            threshold_metrics[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Find best threshold based on F1 score
        best_threshold = max(threshold_metrics.keys(), key=lambda t: threshold_metrics[t]['f1'])
        best_metrics = threshold_metrics[best_threshold]
        
        # Store metrics
        self.metrics = {
            'best_threshold': best_threshold,
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1': best_metrics['f1'],
            'tp': best_metrics['tp'],
            'fp': best_metrics['fp'],
            'fn': best_metrics['fn'],
            'threshold_metrics': threshold_metrics
        }
        
        logger.info(f"Best threshold: {best_threshold}, F1: {best_metrics['f1']:.4f}, "
                   f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
        
        return self.metrics
        
    def compute_precision_recall_curve(self, matches: pd.DataFrame, ground_truth: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.
        
        Args:
            matches (pd.DataFrame): DataFrame with predicted matches
            ground_truth (pd.DataFrame): DataFrame with ground truth matches
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Precision, recall, thresholds
        """
        logger.info("Computing precision-recall curve")
        
        # Check if matches DataFrame is empty or missing required columns
        if matches.empty or not all(col in matches.columns for col in ['user_id1', 'user_id2', 'confidence']):
            logger.warning("Matches DataFrame is empty or missing required columns. Returning default precision-recall curve.")
            # Return default values
            precision = np.array([0.0, 0.0])
            recall = np.array([0.0, 1.0])
            thresholds = np.array([])
            
            # Store in metrics
            self.metrics['precision_recall_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
            }
            
            return precision, recall, thresholds
        
        # Get the column names for user IDs
        id1_col, id2_col = self._get_id_columns(ground_truth)
        
        # Create a set of ground truth pairs for quick lookup
        gt_pairs = set()
        for _, row in ground_truth.iterrows():
            gt_pairs.add((row[id1_col], row[id2_col]))
            # Also add the reverse pair to handle different orderings
            gt_pairs.add((row[id2_col], row[id1_col]))
        
        # Create binary labels for matches
        y_true = []
        y_scores = []
        
        for _, row in matches.iterrows():
            is_match = (row['user_id1'], row['user_id2']) in gt_pairs or (row['user_id2'], row['user_id1']) in gt_pairs
            y_true.append(1 if is_match else 0)
            y_scores.append(row['confidence'])
        
        # Compute precision-recall curve
        if len(y_true) > 0 and len(y_scores) > 0:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        else:
            # Return default values if no data
            precision = np.array([0.0, 0.0])
            recall = np.array([0.0, 1.0])
            thresholds = np.array([])
        
        # Store in metrics
        self.metrics['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
        }
        
        return precision, recall, thresholds
