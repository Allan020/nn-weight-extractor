"""Base parser class for model weight extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class LayerInfo:
    """Information about a network layer."""
    name: str
    type: str  # 'convolutional', 'maxpool', 'route', 'shortcut', 'yolo', etc.
    index: int
    
    # Convolutional layer parameters
    filters: Optional[int] = None
    channels: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    groups: Optional[int] = None
    activation: Optional[str] = None
    batch_normalize: bool = False
    
    # Weights and biases
    weights: Optional[np.ndarray] = None
    biases: Optional[np.ndarray] = None
    
    # Batch normalization parameters
    bn_gamma: Optional[np.ndarray] = None  # scales
    bn_beta: Optional[np.ndarray] = None   # biases (in BN layer)
    bn_mean: Optional[np.ndarray] = None   # moving_mean
    bn_variance: Optional[np.ndarray] = None  # moving_variance
    
    # Maxpool parameters
    pool_size: Optional[int] = None
    pool_stride: Optional[int] = None
    
    # Route/Shortcut parameters
    layers: Optional[List[int]] = None
    
    # Additional metadata
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    
    def __repr__(self):
        if self.type == 'convolutional':
            bn_str = " +BN" if self.batch_normalize else ""
            return (f"LayerInfo(name='{self.name}', type='{self.type}', "
                   f"filters={self.filters}, size={self.kernel_size}x{self.kernel_size}, "
                   f"stride={self.stride}{bn_str}, activation='{self.activation}')")
        else:
            return f"LayerInfo(name='{self.name}', type='{self.type}')"


class BaseParser(ABC):
    """Abstract base class for model parsers."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.model = None
        self.layers: List[LayerInfo] = []
        self.input_shape: Optional[Tuple] = None
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def parse_layers(self) -> List[LayerInfo]:
        """
        Parse all layers from the model.
        
        Returns:
            List of LayerInfo objects
        """
        pass
    
    @abstractmethod
    def extract_layer_weights(self, layer_name: str) -> Optional[np.ndarray]:
        """
        Extract weights from a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Weights as numpy array or None
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Get model input shape (height, width, channels).
        
        Returns:
            Tuple of (height, width, channels)
        """
        pass
    
    def get_conv_layers(self) -> List[LayerInfo]:
        """Get all convolutional layers."""
        return [layer for layer in self.layers if layer.type == 'convolutional']
    
    def get_layer_by_name(self, name: str) -> Optional[LayerInfo]:
        """Get layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None
    
    def get_layer_by_index(self, index: int) -> Optional[LayerInfo]:
        """Get layer by index."""
        for layer in self.layers:
            if layer.index == index:
                return layer
        return None
    
    def print_summary(self):
        """Print model summary."""
        print("\nModel Summary:")
        print(f"Total layers: {len(self.layers)}")
        print(f"Input shape: {self.input_shape}")
        
        conv_layers = self.get_conv_layers()
        print(f"Convolutional layers: {len(conv_layers)}")
        
        total_params = 0
        for layer in conv_layers:
            if layer.weights is not None:
                total_params += layer.weights.size
            if layer.biases is not None:
                total_params += layer.biases.size
        
        print(f"Total parameters: {total_params:,}")
        print()
        
        if self.verbose:
            print("Layer Details:")
            for layer in self.layers:
                print(f"  {layer}")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate parsed model.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.layers:
            errors.append("No layers found in model")
        
        if not self.input_shape:
            errors.append("Input shape not determined")
        
        for layer in self.get_conv_layers():
            if layer.weights is None:
                errors.append(f"Layer {layer.name}: weights not extracted")
            # Note: Biases can be None if use_bias=False - we'll create zeros during writing
            if layer.batch_normalize:
                if layer.bn_gamma is None:
                    errors.append(f"Layer {layer.name}: BN gamma not found")
                if layer.bn_mean is None:
                    errors.append(f"Layer {layer.name}: BN mean not found")
                if layer.bn_variance is None:
                    errors.append(f"Layer {layer.name}: BN variance not found")
        
        return len(errors) == 0, errors

