"""Keras model parser for H5 format."""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any
import warnings

try:
    from tensorflow import keras
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, MaxPooling2D, 
        Add, Concatenate, UpSampling2D, ZeroPadding2D,
        LeakyReLU, ReLU, Activation
    )
except ImportError:
    try:
        import keras
        from keras.models import load_model, Model
        from keras.layers import (
            Conv2D, BatchNormalization, MaxPooling2D,
            Add, Concatenate, UpSampling2D, ZeroPadding2D,
            LeakyReLU, ReLU, Activation
        )
    except ImportError:
        raise ImportError("Neither TensorFlow nor standalone Keras found. Please install TensorFlow.")

from .base_parser import BaseParser, LayerInfo


class KerasParser(BaseParser):
    """Parser for Keras H5 models."""
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.layer_name_to_info: Dict[str, LayerInfo] = {}
        self.layer_connections: Dict[str, List[str]] = {}
        
    def load_model(self, model_path: str) -> bool:
        """Load Keras model from H5 file."""
        try:
            # Try to load as a full model first
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = load_model(model_path, compile=False)
            
            if self.verbose:
                print(f"Loaded Keras model from {model_path}")
                print(f"Model has {len(self.model.layers)} layers")
            
            return True
        except Exception as e:
            # If loading fails, it might be a weights-only file
            print(f"Note: Could not load as complete model: {e}")
            print()
            print("This H5 file appears to contain only weights (not the full model).")
            print("For YOLOv3 models, the model architecture needs to be built first.")
            print()
            print("Please use one of these approaches:")
            print("1. If you have the original training script, re-save the model with:")
            print("   model.save('model.h5')  # Saves full model")
            print()
            print("2. For YOLOv3, use the provided yolo.py to build the model first:")
            print("   - The old h5_to_weight_yolo3/check_weight.py script handles this")
            print("   - Or build the model structure, then load_weights()")
            print()
            print("3. If you already have .weights and .cfg files, skip Python step:")
            print("   cd ../cpp")
            print("   ./weights_extractor --cfg model.cfg --weights model.weights")
            return False
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get model input shape (height, width, channels)."""
        if self.model is None:
            return (0, 0, 0)
        
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Shape is typically (None, height, width, channels) or (batch, h, w, c)
        if len(input_shape) == 4:
            _, h, w, c = input_shape
            return (h if h else 416, w if w else 416, c if c else 3)
        
        return (416, 416, 3)  # Default
    
    def parse_layers(self) -> List[LayerInfo]:
        """Parse all layers from Keras model."""
        if self.model is None:
            return []
        
        self.input_shape = self.get_input_shape()
        self.layers = []
        conv_idx = 0
        
        for idx, layer in enumerate(self.model.layers):
            layer_info = self._parse_single_layer(layer, idx, conv_idx)
            
            if layer_info is not None:
                self.layers.append(layer_info)
                self.layer_name_to_info[layer.name] = layer_info
                
                if layer_info.type == 'convolutional':
                    conv_idx += 1
        
        # Second pass: identify batch normalization relationships
        self._match_conv_bn_pairs()
        
        if self.verbose:
            print(f"Parsed {len(self.layers)} layers")
            print(f"Found {len(self.get_conv_layers())} convolutional layers")
        
        return self.layers
    
    def _parse_single_layer(self, layer, index: int, conv_idx: int) -> Optional[LayerInfo]:
        """Parse a single Keras layer."""
        layer_type = type(layer).__name__
        
        # Convolutional layer
        if isinstance(layer, Conv2D):
            return self._parse_conv2d(layer, index, conv_idx)
        
        # Batch normalization
        elif isinstance(layer, BatchNormalization):
            return self._parse_batch_norm(layer, index)
        
        # Max pooling
        elif isinstance(layer, MaxPooling2D):
            return self._parse_maxpool(layer, index)
        
        # Upsampling
        elif isinstance(layer, UpSampling2D):
            return self._parse_upsample(layer, index)
        
        # Route layers (Concatenate)
        elif isinstance(layer, Concatenate):
            return self._parse_concatenate(layer, index)
        
        # Shortcut layers (Add)
        elif isinstance(layer, Add):
            return self._parse_add(layer, index)
        
        # Skip other layers (activation, zero padding, etc.)
        # These are typically fused with conv layers
        return None
    
    def _parse_conv2d(self, layer: Conv2D, index: int, conv_idx: int) -> LayerInfo:
        """Parse Conv2D layer."""
        config = layer.get_config()
        weights = layer.get_weights()
        
        # Get kernel size
        kernel_size = config['kernel_size']
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]  # Assume square kernels
        
        # Get stride
        strides = config['strides']
        if isinstance(strides, (list, tuple)):
            stride = strides[0]  # Assume same stride in both dimensions
        else:
            stride = strides
        
        # Get padding
        padding_mode = config['padding']
        if padding_mode == 'same':
            padding = kernel_size // 2
        elif padding_mode == 'valid':
            padding = 0
        else:
            padding = 0
        
        # Extract weights and biases
        conv_weights = None
        conv_biases = None
        
        if len(weights) >= 1:
            # Keras format: (height, width, in_channels, out_channels)
            # Darknet format: (out_channels, in_channels, height, width)
            conv_weights = weights[0]
            # Transpose to Darknet format
            conv_weights = np.transpose(conv_weights, [3, 2, 0, 1])
        
        if len(weights) >= 2:
            conv_biases = weights[1]
        else:
            # Layer has no biases (use_bias=False) - create zeros
            if conv_weights is not None:
                filters = conv_weights.shape[0]
                conv_biases = np.zeros(filters, dtype=np.float32)
        
        # Get activation
        activation = config.get('activation', 'linear')
        
        # Determine number of filters and channels from weights shape
        if conv_weights is not None:
            filters = conv_weights.shape[0]
            channels = conv_weights.shape[1] * config.get('groups', 1)
        else:
            filters = config['filters']
            channels = layer.input_shape[-1] if hasattr(layer, 'input_shape') else 3
        
        info = LayerInfo(
            name=layer.name,
            type='convolutional',
            index=index,
            filters=filters,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=config.get('groups', 1),
            activation=activation,
            batch_normalize=False,  # Will be updated if BN follows
            weights=conv_weights,
            biases=conv_biases,
            input_shape=layer.input_shape if hasattr(layer, 'input_shape') else None,
            output_shape=layer.output_shape if hasattr(layer, 'output_shape') else None
        )
        
        return info
    
    def _parse_batch_norm(self, layer: BatchNormalization, index: int) -> Optional[LayerInfo]:
        """Parse BatchNormalization layer."""
        weights = layer.get_weights()
        
        if len(weights) < 4:
            return None
        
        # Keras BN weights: [gamma, beta, moving_mean, moving_variance]
        gamma = weights[0]
        beta = weights[1]
        moving_mean = weights[2]
        moving_variance = weights[3]
        
        info = LayerInfo(
            name=layer.name,
            type='batch_normalization',
            index=index,
            bn_gamma=gamma,
            bn_beta=beta,
            bn_mean=moving_mean,
            bn_variance=moving_variance
        )
        
        return info
    
    def _parse_maxpool(self, layer: MaxPooling2D, index: int) -> LayerInfo:
        """Parse MaxPooling2D layer."""
        config = layer.get_config()
        
        pool_size = config['pool_size']
        if isinstance(pool_size, (list, tuple)):
            pool_size = pool_size[0]
        
        strides = config['strides']
        if isinstance(strides, (list, tuple)):
            stride = strides[0]
        else:
            stride = strides
        
        info = LayerInfo(
            name=layer.name,
            type='maxpool',
            index=index,
            pool_size=pool_size,
            pool_stride=stride
        )
        
        return info
    
    def _parse_upsample(self, layer: UpSampling2D, index: int) -> LayerInfo:
        """Parse UpSampling2D layer."""
        config = layer.get_config()
        size = config.get('size', (2, 2))
        if isinstance(size, (list, tuple)):
            size = size[0]
        
        info = LayerInfo(
            name=layer.name,
            type='upsample',
            index=index,
            stride=size  # Reuse stride field for upsample factor
        )
        
        return info
    
    def _parse_concatenate(self, layer: Concatenate, index: int) -> LayerInfo:
        """Parse Concatenate layer (route in Darknet)."""
        info = LayerInfo(
            name=layer.name,
            type='route',
            index=index
        )
        
        return info
    
    def _parse_add(self, layer: Add, index: int) -> LayerInfo:
        """Parse Add layer (shortcut in Darknet)."""
        info = LayerInfo(
            name=layer.name,
            type='shortcut',
            index=index
        )
        
        return info
    
    def _match_conv_bn_pairs(self):
        """Match convolutional layers with batch normalization layers by name."""
        # Create a dictionary of BN layers by their corresponding conv layer name
        bn_layers = {}
        for layer in self.layers:
            if layer.type == 'batch_normalization':
                # Extract number from BN name (e.g., 'batch_normalization_55' -> '55')
                # Match with corresponding conv layer (e.g., 'conv2d_55')
                import re
                match = re.search(r'_(\d+)$', layer.name)
                if match:
                    num = match.group(1)
                    conv_name_candidates = [f'conv2d_{num}', f'convolutional_{num}']
                    for conv_name in conv_name_candidates:
                        bn_layers[conv_name] = layer
                elif layer.name == 'batch_normalization':
                    # First BN layer (no number)
                    bn_layers['conv2d'] = layer
        
        # Match conv layers with their BN layers
        for i in range(len(self.layers)):
            current = self.layers[i]
            
            if current.type == 'convolutional':
                # Try to find matching BN by name
                bn_layer = bn_layers.get(current.name)
                
                # If not found by name, check if next layer is BN (fallback to sequential)
                if bn_layer is None and i + 1 < len(self.layers):
                    next_layer = self.layers[i + 1]
                    if next_layer.type == 'batch_normalization':
                        bn_layer = next_layer
                
                if bn_layer:
                    current.batch_normalize = True
                    current.bn_gamma = bn_layer.bn_gamma
                    current.bn_beta = bn_layer.bn_beta
                    current.bn_mean = bn_layer.bn_mean
                    current.bn_variance = bn_layer.bn_variance
                    
                    # In Darknet, when BN is used, the bias comes from BN beta
                    # The conv bias is typically not used (or is zero)
                    if current.biases is None or not current.biases.any():
                        current.biases = bn_layer.bn_beta
                    
                    if self.verbose:
                        print(f"Matched BN layer '{bn_layer.name}' with conv '{current.name}'")
    
    def extract_layer_weights(self, layer_name: str) -> Optional[np.ndarray]:
        """Extract weights from a specific layer."""
        if layer_name in self.layer_name_to_info:
            return self.layer_name_to_info[layer_name].weights
        return None
    
    def get_activation_after_layer(self, layer_name: str) -> str:
        """Try to determine activation function after a layer."""
        if self.model is None:
            return 'linear'
        
        # Find the layer in the model
        for i, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                # Check if next layer is an activation
                if i + 1 < len(self.model.layers):
                    next_layer = self.model.layers[i + 1]
                    if isinstance(next_layer, (LeakyReLU, ReLU, Activation)):
                        if isinstance(next_layer, LeakyReLU):
                            return 'leaky'
                        elif isinstance(next_layer, ReLU):
                            return 'relu'
                        elif isinstance(next_layer, Activation):
                            return next_layer.get_config().get('activation', 'linear')
                break
        
        return 'linear'

