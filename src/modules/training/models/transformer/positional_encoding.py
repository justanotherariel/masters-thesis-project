import torch
import torch.nn as nn
import math
from typing import Literal


class PositionalEncoding(nn.Module):
    """
    Flexible positional encoding supporting learned, 1D sinusoidal, and 2D sinusoidal encodings.
    
    Args:
        encoding_type: Type of positional encoding ('learned', 'sinusoidal_1d', 'sinusoidal_2d')
        d_model: Dimension of the model
        max_len: Maximum sequence length (for 1D) or total number of positions (for learned)
        height: Height of the 2D grid (only for sinusoidal_2d)
        width: Width of the 2D grid (only for sinusoidal_2d)
        learnable_scale: Whether to add a learnable scaling factor for sinusoidal encodings
    """
    
    def __init__(
        self,
        encoding_type: Literal['learned', 'sinusoidal_1d', 'sinusoidal_2d'],
        d_model: int,
        max_len: int = None,
        height: int = None,
        width: int = None,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.encoding_type = encoding_type
        self.d_model = d_model
        
        if encoding_type == 'learned':
            if max_len is None:
                raise ValueError("max_len must be specified for learned positional encoding")
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model))
            
        elif encoding_type == 'sinusoidal_1d':
            if max_len is None:
                raise ValueError("max_len must be specified for 1D sinusoidal positional encoding")
            self.register_buffer('pos_embedding', self._create_sinusoidal_1d(max_len, d_model))
            
        elif encoding_type == 'sinusoidal_2d':
            if height is None or width is None:
                raise ValueError("height and width must be specified for 2D sinusoidal positional encoding")
            self.height = height
            self.width = width
            self.register_buffer('pos_embedding', self._create_sinusoidal_2d(height, width, d_model))
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
            
        # Optional learnable scaling for sinusoidal encodings
        if learnable_scale and encoding_type != 'learned':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = 1.0
    
    def _create_sinusoidal_1d(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create 1D sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _create_sinusoidal_2d(self, height: int, width: int, d_model: int) -> torch.Tensor:
        """Create 2D sinusoidal positional encoding for grid positions."""
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sinusoidal encoding")
        
        pe = torch.zeros(height * width, d_model)
        
        # Create 2D position grid
        y_pos = torch.arange(height).unsqueeze(1).repeat(1, width).flatten()
        x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1).flatten()
        
        # Split dimensions equally between x and y coordinates
        d_x = d_model // 2
        d_y = d_model // 2
        
        # Sinusoidal encoding for x coordinates
        div_term_x = torch.exp(torch.arange(0, d_x, 2).float() * 
                               (-math.log(10000.0) / d_x))
        pe[:, 0:d_x:2] = torch.sin(x_pos.unsqueeze(1) * div_term_x)
        pe[:, 1:d_x:2] = torch.cos(x_pos.unsqueeze(1) * div_term_x)
        
        # Sinusoidal encoding for y coordinates
        div_term_y = torch.exp(torch.arange(0, d_y, 2).float() * 
                               (-math.log(10000.0) / d_y))
        pe[:, d_x::2] = torch.sin(y_pos.unsqueeze(1) * div_term_y)
        pe[:, d_x+1::2] = torch.cos(y_pos.unsqueeze(1) * div_term_y)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, seq_len: int = None, include_cls_token: bool = False) -> torch.Tensor:
        """
        Get positional encoding.
        
        Args:
            seq_len: Length of the sequence (only used for learned/1D to slice the encoding)
            include_cls_token: Whether to include an additional position for a CLS token
                              (appended at the end)
        
        Returns:
            Positional encoding tensor of shape (1, seq_len, d_model)
        """
        if self.encoding_type == 'learned':
            pe = self.pos_embedding
            if seq_len is not None:
                pe = pe[:, :seq_len]
            elif not include_cls_token:
                pe = pe[:, :-1]
                
        elif self.encoding_type == 'sinusoidal_1d':
            pe = self.pos_embedding
            if seq_len is not None:
                pe = pe[:, :seq_len]
            pe = pe * self.scale
            
        else:  # sinusoidal_2d
            pe = self.pos_embedding * self.scale
        
        # Add a zero vector for CLS token position (or learned if using learned encoding)
        if self.encoding_type != 'learned' and include_cls_token:
                # For sinusoidal, append a zero vector
                cls_pe = torch.zeros(1, 1, self.d_model, device=pe.device)
                pe = torch.cat([pe, cls_pe], dim=1)
                
        return pe
