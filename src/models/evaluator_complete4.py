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
