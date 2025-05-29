import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class StutterDetectionModel(nn.Module):
    """BiLSTM model for stuttering detection with robust input handling"""
    
    def __init__(self, config, verbose=False):
        super().__init__()
        self.config = config
        self.verbose = verbose
        
        # Extract dimensions from config
        self.whisper_dim = config['features'].get('whisper_dim', 768)
        self.n_mfcc = config['features']['mfcc'].get('n_mfcc', 13)
        self.mfcc_delta_count = 2  # Always use base + delta + delta2
        
        # Calculate input dimension
        self.input_dim = self.whisper_dim + (self.mfcc_delta_count + 1) * self.n_mfcc
        
        # Expected sequence length
        self.expected_seq_len = config['labels']['pooled_frames']
        self.num_classes = config['labels']['num_classes']
        
        if self.verbose:
            print(f"Model initialized:")
            print(f"  Input dimension: {self.input_dim} (Whisper: {self.whisper_dim}, "
                  f"MFCC: {(self.mfcc_delta_count + 1) * self.n_mfcc})")
            print(f"  Expected sequence length: {self.expected_seq_len}")
            print(f"  Number of classes: {self.num_classes}")
        
        # Model hyperparameters
        self.lstm_units_1 = config['model'].get('lstm_units_1', 256)
        self.lstm_units_2 = config['model'].get('lstm_units_2', 128)
        self.dropout_rate = config['model'].get('dropout_rate', 0.3)
        
        # Input projection layer (handles dimension mismatches)
        self.input_projection = nn.Linear(self.input_dim, self.input_dim)
        
        # Temporal modeling with BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_units_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0  # We'll use explicit dropout layers
        )
        
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(self.lstm_units_1 * 2)
        
        self.lstm2 = nn.LSTM(
            input_size=self.lstm_units_1 * 2,  # *2 for bidirectional
            hidden_size=self.lstm_units_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.layer_norm2 = nn.LayerNorm(self.lstm_units_2 * 2)
        
        # Classification head
        self.classifier = nn.Linear(
            self.lstm_units_2 * 2,  # *2 for bidirectional
            self.num_classes
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
            elif 'classifier' in name or 'input_projection' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def _validate_and_fix_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and fix input dimensions"""
        batch_size = x.size(0)
        
        # Check sequence length
        if x.size(1) != self.expected_seq_len:
            if self.verbose:
                print(f"Warning: Sequence length mismatch. Got {x.size(1)}, expected {self.expected_seq_len}")
            
            if x.size(1) < self.expected_seq_len:
                # Pad sequence
                pad_len = self.expected_seq_len - x.size(1)
                padding = torch.zeros(batch_size, pad_len, x.size(2), 
                                    dtype=x.dtype, device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate sequence
                x = x[:, :self.expected_seq_len, :]
        
        # Check feature dimension
        if x.size(2) != self.input_dim:
            if self.verbose:
                print(f"Warning: Feature dimension mismatch. Got {x.size(2)}, expected {self.input_dim}")
            
            if x.size(2) < self.input_dim:
                # Pad features
                pad_dim = self.input_dim - x.size(2)
                padding = torch.zeros(batch_size, x.size(1), pad_dim, 
                                    dtype=x.dtype, device=x.device)
                x = torch.cat([x, padding], dim=2)
            else:
                # Truncate features
                x = x[:, :, :self.input_dim]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input validation
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        Returns:
            logits: [batch_size, seq_len, num_classes]
        """
        # Validate and fix input if needed
        x = self._validate_and_fix_input(x)
        
        # Input projection (helps with any remaining dimension issues)
        x = self.input_projection(x)
        
        # First BiLSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out1 = self.layer_norm1(lstm_out1)
        
        # Second BiLSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        lstm_out2 = self.layer_norm2(lstm_out2)
        
        # Time-distributed classification
        # Reshape for linear layer
        batch_size, seq_len, hidden_dim = lstm_out2.shape
        lstm_out2_reshaped = lstm_out2.reshape(-1, hidden_dim)
        
        # Apply classifier
        logits = self.classifier(lstm_out2_reshaped)
        
        # Reshape back
        logits = logits.reshape(batch_size, seq_len, self.num_classes)
        
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model
        Args:
            x: Input features
            threshold: Classification threshold
        Returns:
            Dictionary with logits, probabilities, and predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).float()
        
        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': predictions
        }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance with robust handling"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', epsilon=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon  # For numerical stability
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss with shape validation
        Args:
            logits: Model outputs [batch_size, seq_len, num_classes]
            targets: Ground truth labels [batch_size, seq_len, num_classes]
        Returns:
            loss: Scalar loss value
        """
        # Validate shapes
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")
        
        # Ensure targets are in [0, 1]
        targets = torch.clamp(targets, 0, 1)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Add epsilon for numerical stability
        probs = torch.clamp(probs, self.epsilon, 1 - self.epsilon)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, self.epsilon, 1 - self.epsilon)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Handle any NaN or Inf values
        focal_loss = torch.where(
            torch.isfinite(focal_loss),
            focal_loss,
            torch.zeros_like(focal_loss)
        )
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss as alternative to Focal Loss"""
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted BCE loss"""
        # Validate shapes
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")
        
        # Use built-in BCE with logits for stability
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        
        return loss


def build_model(config, verbose=False):
    """Build model and loss function with config validation"""
    # Validate config
    required_keys = [
        ('features.whisper_dim', 768),
        ('features.mfcc.n_mfcc', 13),
        ('labels.pooled_frames', 25),
        ('labels.num_classes', 5),
        ('model.lstm_units_1', 256),
        ('model.lstm_units_2', 128),
        ('model.dropout_rate', 0.3),
        ('training.focal_loss_alpha', 0.25),
        ('training.focal_loss_gamma', 2.0)
    ]
    
    for key_path, default in required_keys:
        keys = key_path.split('.')
        current = config
        try:
            for k in keys[:-1]:
                current = current[k]
            if keys[-1] not in current:
                current[keys[-1]] = default
                if verbose:
                    print(f"Using default value for {key_path}: {default}")
        except KeyError:
            print(f"Warning: Missing config section for {key_path}, using default: {default}")
    
    # Build model
    model = StutterDetectionModel(config, verbose=verbose)
    
    # Build loss function
    criterion = FocalLoss(
        alpha=config['training']['focal_loss_alpha'],
        gamma=config['training']['focal_loss_gamma']
    )
    
    return model, criterion


# Utility functions for model analysis
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, int, int]) -> str:
    """Get model summary"""
    total_params = count_parameters(model)
    
    summary = f"Model Summary:\n"
    summary += f"Total Parameters: {total_params:,}\n"
    summary += f"Expected Input Shape: {input_shape}\n\n"
    
    summary += "Layer Details:\n"
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                summary += f"  {name}: {module.__class__.__name__} - {params:,} params\n"
    
    return summary


# Testing function
def test_model(config_path='config/config.yaml'):
    """Test model with dummy data"""
    import yaml
    
    print("Testing model...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    model, criterion = build_model(config, verbose=True)
    
    # Print model summary
    batch_size = 4
    seq_len = config['labels']['pooled_frames']
    input_dim = 768 + 3 * 13  # Whisper + MFCC features
    
    print(get_model_summary(model, (batch_size, seq_len, input_dim)))
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_labels = torch.randint(0, 2, (batch_size, seq_len, config['labels']['num_classes'])).float()
    
    # Test model forward
    try:
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {config['labels']['num_classes']})")
        
        # Test loss calculation
        loss = criterion(output, dummy_labels)
        print(f"Loss value: {loss.item():.4f}")
        
        # Test predictions
        predictions = model.predict(dummy_input)
        print(f"Prediction shapes:")
        print(f"  Logits: {predictions['logits'].shape}")
        print(f"  Probabilities: {predictions['probabilities'].shape}")
        print(f"  Predictions: {predictions['predictions'].shape}")
        
    except Exception as e:
        print(f"Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with wrong dimensions
    print("\nTesting input validation...")
    
    # Wrong sequence length
    wrong_seq = torch.randn(batch_size, seq_len + 10, input_dim)
    try:
        output = model(wrong_seq)
        print(f"Handled wrong sequence length: {wrong_seq.shape} -> {output.shape}")
    except Exception as e:
        print(f"Error with wrong sequence length: {e}")
    
    # Wrong feature dimension
    wrong_feat = torch.randn(batch_size, seq_len, input_dim + 20)
    try:
        output = model(wrong_feat)
        print(f"Handled wrong feature dimension: {wrong_feat.shape} -> {output.shape}")
    except Exception as e:
        print(f"Error with wrong feature dimension: {e}")
    
    print("\nModel test complete!")


if __name__ == "__main__":
    test_model()