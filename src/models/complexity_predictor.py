"""
Complexity Predictor for VLM Routing

Predicts whether a query is "simple" (use CLIP) or "complex" (use LLaVA)
based on instruction text and image features.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import timm

class ComplexityPredictor(nn.Module):
    """
    Lightweight model that predicts instruction complexity.
    
    Input: Image + Text instruction
    Output: Complexity score ∈ [0, 1]
      - Score < 0.5 → "Simple" (route to CLIP)
      - Score >= 0.5 → "Complex" (route to LLaVA)
    """
    
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Text encoder: DistilBERT (lightweight, fast)
        self.text_encoder = AutoModel.from_pretrained(
            "distilbert-base-uncased",
            load_in_8bit=False  # Use normal precision for training
        )
        
        # Vision encoder: MobileViT-XS (lightweight mobile arch)
        self.vision_encoder = timm.create_model(
            'mobilevit_xs',
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Fusion layer: Simple concatenation + MLP
        text_dim = 768  # DistilBERT hidden size
        vision_dim = 256  # MobileViT-XS output dim
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output: [0, 1]
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Text tokens [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            pixel_values: Images [batch, 3, 224, 224]
        
        Returns:
            complexity_scores: [batch, 1] (values in [0, 1])
        """
        
        # Encode text (use [CLS] token)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_repr = text_outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Encode vision
        vision_repr = self.vision_encoder(pixel_values)  # [batch, 256, 14, 14]
        vision_repr = vision_repr.mean(dim=[2, 3])  # Global average pool → [batch, 256]
        
        # Fuse and predict
        fused = torch.cat([text_repr, vision_repr], dim=1)  # [batch, 1024]
        complexity_score = self.fusion(fused)  # [batch, 1]
        
        return complexity_score
