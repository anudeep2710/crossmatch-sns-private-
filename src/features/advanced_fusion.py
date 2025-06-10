"""
Advanced fusion module with cross-modal attention and self-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing different modalities.
    Implements Text ↔ Graph attention with dynamic importance weighting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cross-modal attention.
        
        Args:
            config: Configuration dictionary
        """
        super(CrossModalAttention, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Attention parameters
        self.num_heads = config.get('attention_heads', 16)
        self.hidden_dim = config.get('fusion_hidden_dim', 512)
        self.dropout_prob = config.get('attention_dropout', 0.1)
        self.temperature = config.get('attention_temperature', 1.0)
        
        # Input dimensions for different modalities
        self.text_dim = config.get('semantic_embedding_dim', 768)
        self.graph_dim = config.get('network_embedding_dim', 256)
        self.temporal_dim = config.get('temporal_embedding_dim', 256)
        self.profile_dim = config.get('profile_embedding_dim', 128)
        
        # Projection layers to common dimension
        self.text_projection = nn.Linear(self.text_dim, self.hidden_dim)
        self.graph_projection = nn.Linear(self.graph_dim, self.hidden_dim)
        self.temporal_projection = nn.Linear(self.temporal_dim, self.hidden_dim)
        self.profile_projection = nn.Linear(self.profile_dim, self.hidden_dim)
        
        # Multi-head attention layers
        self.text_graph_attention = MultiHeadCrossAttention(
            self.hidden_dim, self.num_heads, self.dropout_prob
        )
        self.graph_text_attention = MultiHeadCrossAttention(
            self.hidden_dim, self.num_heads, self.dropout_prob
        )
        self.temporal_text_attention = MultiHeadCrossAttention(
            self.hidden_dim, self.num_heads, self.dropout_prob
        )
        self.temporal_graph_attention = MultiHeadCrossAttention(
            self.hidden_dim, self.num_heads, self.dropout_prob
        )
        
        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(4) / 4)  # text, graph, temporal, profile
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Final projection
        self.output_projection = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through cross-modal attention.
        
        Args:
            embeddings: Dictionary containing embeddings for different modalities
                       Expected keys: 'text', 'graph', 'temporal', 'profile'
                       
        Returns:
            Fused embeddings tensor
        """
        # Project all modalities to common dimension
        text_emb = self.text_projection(embeddings['text']) if 'text' in embeddings else None
        graph_emb = self.graph_projection(embeddings['graph']) if 'graph' in embeddings else None
        temporal_emb = self.temporal_projection(embeddings['temporal']) if 'temporal' in embeddings else None
        profile_emb = self.profile_projection(embeddings['profile']) if 'profile' in embeddings else None
        
        # Handle missing modalities
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device
        
        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
        if graph_emb is None:
            graph_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
        if temporal_emb is None:
            temporal_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
        if profile_emb is None:
            profile_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Cross-modal attention
        # Text ↔ Graph
        text_attended = self.text_graph_attention(text_emb, graph_emb, graph_emb)
        graph_attended = self.graph_text_attention(graph_emb, text_emb, text_emb)
        
        # Temporal ↔ Text
        temporal_text_attended = self.temporal_text_attention(temporal_emb, text_emb, text_emb)
        
        # Temporal ↔ Graph
        temporal_graph_attended = self.temporal_graph_attention(temporal_emb, graph_emb, graph_emb)
        
        # Combine attended representations
        text_final = text_emb + text_attended
        graph_final = graph_emb + graph_attended
        temporal_final = temporal_emb + temporal_text_attended + temporal_graph_attended
        profile_final = profile_emb  # Profile doesn't participate in cross-attention
        
        # Apply layer normalization
        text_final = self.layer_norm(text_final)
        graph_final = self.layer_norm(graph_final)
        temporal_final = self.layer_norm(temporal_final)
        profile_final = self.layer_norm(profile_final)
        
        # Weight modalities by learned importance
        weights = F.softmax(self.modality_weights, dim=0)
        
        weighted_text = text_final * weights[0]
        weighted_graph = graph_final * weights[1]
        weighted_temporal = temporal_final * weights[2]
        weighted_profile = profile_final * weights[3]
        
        # Concatenate and project
        concatenated = torch.cat([
            weighted_text, weighted_graph, weighted_temporal, weighted_profile
        ], dim=1)
        
        fused_output = self.output_projection(concatenated)
        
        return fused_output
    
    def get_attention_weights(self) -> Dict[str, float]:
        """Get the learned modality weights."""
        weights = F.softmax(self.modality_weights, dim=0)
        return {
            'text': weights[0].item(),
            'graph': weights[1].item(),
            'temporal': weights[2].item(),
            'profile': weights[3].item()
        }


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query_projection(query)
        K = self.key_projection(key)
        V = self.value_projection(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        # Final projection
        output = self.output_projection(attended_values)
        
        return output.squeeze(1) if output.size(1) == 1 else output


class SelfAttentionFusion(nn.Module):
    """
    Self-attention fusion with modality weighting and residual connections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize self-attention fusion.
        
        Args:
            config: Configuration dictionary
        """
        super(SelfAttentionFusion, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Network parameters
        self.hidden_dim = config.get('fusion_hidden_dim', 512)
        self.num_heads = config.get('self_attention_heads', 8)
        self.num_layers = config.get('self_attention_layers', 3)
        self.dropout_prob = config.get('self_attention_dropout', 0.1)
        
        # Input dimensions
        self.input_dims = {
            'text': config.get('semantic_embedding_dim', 768),
            'graph': config.get('network_embedding_dim', 256),
            'temporal': config.get('temporal_embedding_dim', 256),
            'profile': config.get('profile_embedding_dim', 128)
        }
        
        # Input projections
        self.input_projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.hidden_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(self.hidden_dim, self.num_heads, self.dropout_prob)
            for _ in range(self.num_layers)
        ])
        
        # Modality-specific gating
        self.modality_gates = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            for modality in self.input_dims.keys()
        })
        
        # Final layers
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through self-attention fusion.
        
        Args:
            embeddings: Dictionary containing embeddings for different modalities
                       
        Returns:
            Fused embeddings tensor
        """
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device
        
        # Project inputs and create sequence
        modality_embeddings = []
        modality_masks = []
        
        for modality in ['text', 'graph', 'temporal', 'profile']:
            if modality in embeddings:
                projected = self.input_projections[modality](embeddings[modality])
                gate_score = self.modality_gates[modality](projected)
                gated_embedding = projected * gate_score
                modality_embeddings.append(gated_embedding.unsqueeze(1))
                modality_masks.append(torch.ones(batch_size, 1, device=device))
            else:
                # Create zero embedding for missing modality
                zero_embedding = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
                modality_embeddings.append(zero_embedding)
                modality_masks.append(torch.zeros(batch_size, 1, device=device))
        
        # Concatenate modalities to form sequence
        sequence = torch.cat(modality_embeddings, dim=1)  # [batch_size, num_modalities, hidden_dim]
        mask = torch.cat(modality_masks, dim=1)  # [batch_size, num_modalities]
        
        # Apply self-attention layers
        attended_sequence = sequence
        for attention_layer in self.attention_layers:
            attended_sequence = attention_layer(attended_sequence, mask)
        
        # Global pooling with masking
        masked_sequence = attended_sequence * mask.unsqueeze(-1)
        pooled = masked_sequence.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Final processing
        normalized = self.layer_norm(pooled)
        output = self.output_projection(normalized)
        
        # Residual connection
        if pooled.size(-1) == output.size(-1):
            output = output + pooled
        
        return output
    
    def get_modality_importance(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get importance scores for each modality."""
        importance_scores = {}
        
        for modality in ['text', 'graph', 'temporal', 'profile']:
            if modality in embeddings:
                projected = self.input_projections[modality](embeddings[modality])
                gate_score = self.modality_gates[modality](projected)
                importance_scores[modality] = gate_score.mean().item()
            else:
                importance_scores[modality] = 0.0
        
        return importance_scores


class SelfAttentionLayer(nn.Module):
    """Single self-attention layer with layer normalization and residual connections."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.1):
        super(SelfAttentionLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attention_output, _ = self.attention(x, x, x, key_padding_mask=mask == 0 if mask is not None else None)
        x = self.layer_norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning module with InfoNCE loss and hard negative mining.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize contrastive learning module.
        
        Args:
            config: Configuration dictionary
        """
        super(ContrastiveLearning, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Contrastive learning parameters
        self.temperature = config.get('contrastive_temperature', 0.07)
        self.hard_negative_ratio = config.get('hard_negative_ratio', 0.3)
        self.embedding_dim = config.get('fusion_output_dim', 512)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2)
        )
        
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                positive_pairs: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: Embeddings from platform 1
            embeddings2: Embeddings from platform 2
            positive_pairs: Boolean tensor indicating positive pairs
            
        Returns:
            Contrastive loss
        """
        # Project embeddings
        proj1 = self.projection_head(embeddings1)
        proj2 = self.projection_head(embeddings2)
        
        # Normalize embeddings
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(proj1, proj2.T) / self.temperature
        
        # InfoNCE loss
        batch_size = proj1.size(0)
        device = proj1.device
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=device)
        
        # Mask for positive pairs
        positive_mask = positive_pairs.float()
        
        # Compute loss for both directions
        loss_12 = F.cross_entropy(similarity_matrix, labels, reduction='none')
        loss_21 = F.cross_entropy(similarity_matrix.T, labels, reduction='none')
        
        # Weight by positive pairs
        loss_12 = (loss_12 * positive_mask.diagonal()).sum() / positive_mask.diagonal().sum().clamp(min=1)
        loss_21 = (loss_21 * positive_mask.diagonal()).sum() / positive_mask.diagonal().sum().clamp(min=1)
        
        total_loss = (loss_12 + loss_21) / 2
        
        return total_loss
    
    def mine_hard_negatives(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                           positive_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine hard negative pairs for improved contrastive learning.
        
        Args:
            embeddings1: Embeddings from platform 1
            embeddings2: Embeddings from platform 2
            positive_pairs: Boolean tensor indicating positive pairs
            
        Returns:
            Hard negative embeddings for both platforms
        """
        # Compute similarity matrix
        similarity_matrix = torch.matmul(
            F.normalize(embeddings1, dim=1),
            F.normalize(embeddings2, dim=1).T
        )
        
        # Mask positive pairs
        negative_mask = ~positive_pairs
        masked_similarity = similarity_matrix.clone()
        masked_similarity[positive_pairs] = -float('inf')
        
        # Find hard negatives (high similarity but negative pairs)
        num_hard_negatives = int(self.hard_negative_ratio * embeddings1.size(0))
        
        # Get top negative similarities
        hard_negative_scores, hard_negative_indices = torch.topk(
            masked_similarity.flatten(), num_hard_negatives
        )
        
        # Convert flat indices to 2D indices
        row_indices = hard_negative_indices // embeddings2.size(0)
        col_indices = hard_negative_indices % embeddings2.size(0)
        
        # Extract hard negative embeddings
        hard_neg_emb1 = embeddings1[row_indices]
        hard_neg_emb2 = embeddings2[col_indices]
        
        return hard_neg_emb1, hard_neg_emb2