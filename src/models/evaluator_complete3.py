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
