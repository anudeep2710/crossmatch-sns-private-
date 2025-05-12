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
    def compute_roc_curve(self, matches: pd.DataFrame, ground_truth: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve and AUC.
        
        Args:
            matches (pd.DataFrame): DataFrame with predicted matches
            ground_truth (pd.DataFrame): DataFrame with ground truth matches
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: FPR, TPR, thresholds, AUC
        """
        logger.info("Computing ROC curve")
        
        # Check if matches DataFrame is empty or missing required columns
        if matches.empty or not all(col in matches.columns for col in ['user_id1', 'user_id2', 'confidence']):
            logger.warning("Matches DataFrame is empty or missing required columns. Returning default ROC curve.")
            # Return default values
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([1.0, 0.0])
            roc_auc = 0.5
            
            # Store in metrics
            self.metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc
            }
            
            return fpr, tpr, thresholds, roc_auc
        
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
        
        # Compute ROC curve
        if len(y_true) > 0 and len(y_scores) > 0 and len(set(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            # Return default values if no data or all labels are the same
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([1.0, 0.0])
            roc_auc = 0.5
        
        # Store in metrics
        self.metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc
        }
        
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        return fpr, tpr, thresholds, roc_auc
        
    def compute_confusion_matrix(self, matches: pd.DataFrame, ground_truth: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        Compute confusion matrix at a specific threshold.
        
        Args:
            matches (pd.DataFrame): DataFrame with predicted matches
            ground_truth (pd.DataFrame): DataFrame with ground truth matches
            threshold (float): Confidence threshold
            
        Returns:
            np.ndarray: Confusion matrix
        """
        logger.info(f"Computing confusion matrix at threshold {threshold}")
        
        # Check if matches DataFrame is empty or missing required columns
        if matches.empty or not all(col in matches.columns for col in ['user_id1', 'user_id2', 'confidence']):
            logger.warning("Matches DataFrame is empty or missing required columns. Returning default confusion matrix.")
            # Return default values
            # [TN, FP]
            # [FN, TP]
            gt_count = len(ground_truth) if not ground_truth.empty else 0
            cm = np.array([[0, 0], [gt_count, 0]])
            
            # Store in metrics
            self.metrics['confusion_matrix'] = {
                'matrix': cm.tolist(),
                'threshold': threshold
            }
            
            return cm
        
        # Get the column names for user IDs
        id1_col, id2_col = self._get_id_columns(ground_truth)
        
        # Create a set of ground truth pairs for quick lookup
        gt_pairs = set()
        for _, row in ground_truth.iterrows():
            gt_pairs.add((row[id1_col], row[id2_col]))
            # Also add the reverse pair to handle different orderings
            gt_pairs.add((row[id2_col], row[id1_col]))
        
        # Filter matches by threshold
        filtered_matches = matches[matches['confidence'] >= threshold]
        
        # Create a set of predicted pairs
        pred_pairs = set()
        for _, row in filtered_matches.iterrows():
            pred_pairs.add((row['user_id1'], row['user_id2']))
        
        # Compute true positives, false positives, and false negatives
        tp = 0
        for pair in pred_pairs:
            if pair in gt_pairs or (pair[1], pair[0]) in gt_pairs:
                tp += 1
        
        fp = len(pred_pairs) - tp
        fn = len(gt_pairs) // 2 - tp  # Divide by 2 because we added both directions
        tn = 0  # True negatives are not well-defined in this context
        
        # Create confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Store in metrics
        self.metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'threshold': threshold
        }
        
        return cm
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if 'precision_recall_curve' not in self.metrics:
            logger.warning("Precision-recall curve not computed. Call compute_precision_recall_curve first.")
            return
        
        precision = self.metrics['precision_recall_curve']['precision']
        recall = self.metrics['precision_recall_curve']['recall']
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.', label=f"AUC={auc(recall, precision):.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve saved to {save_path}")
        else:
            plt.show()
            
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if 'roc_curve' not in self.metrics:
            logger.warning("ROC curve not computed. Call compute_roc_curve first.")
            return
        
        fpr = self.metrics['roc_curve']['fpr']
        tpr = self.metrics['roc_curve']['tpr']
        roc_auc = self.metrics['roc_curve']['auc']
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, marker='.', label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
            
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if 'confusion_matrix' not in self.metrics:
            logger.warning("Confusion matrix not computed. Call compute_confusion_matrix first.")
            return
        
        cm = np.array(self.metrics['confusion_matrix']['matrix'])
        threshold = self.metrics['confusion_matrix']['threshold']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
            
    def save_metrics(self, save_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            save_path (str): Path to save the metrics
        """
        import json
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")
