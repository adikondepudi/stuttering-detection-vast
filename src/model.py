import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StutterDetectionModel(nn.Module):
    """BiLSTM model for stuttering detection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input dimension (Whisper + MFCC features)
        self.input_dim = (config['features']['whisper_dim'] + 
                         3 * config['features']['mfcc']['n_mfcc'])  # 768 + 39 = 807
        
        # Temporal modeling with BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config['model']['lstm_units_1'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0  # We'll use explicit dropout layers
        )
        
        self.dropout1 = nn.Dropout(config['model']['dropout_rate'])
        
        self.lstm2 = nn.LSTM(
            input_size=config['model']['lstm_units_1'] * 2,  # *2 for bidirectional
            hidden_size=config['model']['lstm_units_2'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.dropout2 = nn.Dropout(config['model']['dropout_rate'])
        
        # Classification head
        self.classifier = nn.Linear(
            config['model']['lstm_units_2'] * 2,  # *2 for bidirectional
            config['labels']['num_classes']
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'classifier' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        Returns:
            logits: [batch_size, seq_len, num_classes]
        """
        # First BiLSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # Second BiLSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Time-distributed classification
        # Reshape for linear layer
        batch_size, seq_len, hidden_dim = lstm_out2.shape
        lstm_out2_reshaped = lstm_out2.reshape(-1, hidden_dim)
        
        # Apply classifier
        logits = self.classifier(lstm_out2_reshaped)
        
        # Reshape back
        logits = logits.reshape(batch_size, seq_len, -1)
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        Args:
            logits: Model outputs [batch_size, seq_len, num_classes]
            targets: Ground truth labels [batch_size, seq_len, num_classes]
        Returns:
            loss: Scalar loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def build_model(config):
    """Build model and loss function"""
    model = StutterDetectionModel(config)
    criterion = FocalLoss(
        alpha=config['training']['focal_loss_alpha'],
        gamma=config['training']['focal_loss_gamma']
    )
    
    return model, criterion