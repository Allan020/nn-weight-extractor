"""Darknet weights file writer."""

import numpy as np
import struct
from typing import List, BinaryIO
from model_parsers.base_parser import LayerInfo


class DarknetWeightsWriter:
    """Write weights in Darknet format."""
    
    def __init__(self, output_path: str, verbose: bool = False):
        self.output_path = output_path
        self.verbose = verbose
        self.file_handle: BinaryIO = None
        self.total_bytes_written = 0
    
    def open(self):
        """Open output file for writing."""
        self.file_handle = open(self.output_path, 'wb')
        self.total_bytes_written = 0
    
    def close(self):
        """Close output file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def write_header(self, major: int = 0, minor: int = 2, revision: int = 0, 
                    seen: int = 0):
        """
        Write Darknet weights file header.
        
        Args:
            major: Major version number
            minor: Minor version number
            revision: Revision number
            seen: Number of images seen during training
        """
        if not self.file_handle:
            raise RuntimeError("File not opened. Call open() first.")
        
        # Write version numbers as int32
        header_data = struct.pack('iii', major, minor, revision)
        self.file_handle.write(header_data)
        self.total_bytes_written += 12
        
        # Write seen as int64 (for version >= 0.2)
        if (major * 10 + minor) >= 2:
            seen_data = struct.pack('q', seen)  # 'q' is int64
            self.file_handle.write(seen_data)
            self.total_bytes_written += 8
        else:
            # Old format uses int32 for seen
            seen_data = struct.pack('i', seen)
            self.file_handle.write(seen_data)
            self.total_bytes_written += 4
        
        if self.verbose:
            print(f"Written header: major={major}, minor={minor}, "
                  f"revision={revision}, seen={seen}")
    
    def write_convolutional_layer(self, layer: LayerInfo):
        """
        Write convolutional layer weights in Darknet format.
        
        Format:
        - biases (n floats)
        - if batch_normalize: scales, mean, variance (n floats each)
        - weights (num floats)
        
        Args:
            layer: LayerInfo object with weights
        """
        if not self.file_handle:
            raise RuntimeError("File not opened. Call open() first.")
        
        if layer.type != 'convolutional':
            raise ValueError(f"Layer {layer.name} is not convolutional")
        
        if layer.weights is None:
            raise ValueError(f"Layer {layer.name} missing weights")
        
        n = layer.filters
        num_weights = layer.weights.size
        
        # Get biases or create zeros if missing (use_bias=False)
        if layer.biases is None:
            biases = np.zeros(n, dtype=np.float32)
            if self.verbose:
                print(f"  Note: Layer {layer.name} has no biases, using zeros")
        else:
            biases = np.array(layer.biases, dtype=np.float32)
        if biases.shape[0] != n:
            raise ValueError(f"Layer {layer.name}: biases size mismatch "
                           f"(expected {n}, got {biases.shape[0]})")
        
        biases.tofile(self.file_handle)
        self.total_bytes_written += biases.nbytes
        
        if self.verbose:
            print(f"  Written {n} biases for layer {layer.name}")
        
        # Write batch normalization parameters if present
        if layer.batch_normalize:
            if (layer.bn_gamma is None or layer.bn_mean is None or 
                layer.bn_variance is None):
                raise ValueError(f"Layer {layer.name}: batch_normalize=True "
                               "but BN parameters missing")
            
            # Scales (gamma)
            scales = np.array(layer.bn_gamma, dtype=np.float32)
            if scales.shape[0] != n:
                raise ValueError(f"Layer {layer.name}: scales size mismatch")
            scales.tofile(self.file_handle)
            self.total_bytes_written += scales.nbytes
            
            # Rolling mean
            mean = np.array(layer.bn_mean, dtype=np.float32)
            if mean.shape[0] != n:
                raise ValueError(f"Layer {layer.name}: mean size mismatch")
            mean.tofile(self.file_handle)
            self.total_bytes_written += mean.nbytes
            
            # Rolling variance
            variance = np.array(layer.bn_variance, dtype=np.float32)
            if variance.shape[0] != n:
                raise ValueError(f"Layer {layer.name}: variance size mismatch")
            variance.tofile(self.file_handle)
            self.total_bytes_written += variance.nbytes
            
            if self.verbose:
                print(f"  Written BN parameters for layer {layer.name}")
        
        # Write weights
        weights = np.array(layer.weights, dtype=np.float32)
        if weights.size != num_weights:
            raise ValueError(f"Layer {layer.name}: weights size mismatch "
                           f"(expected {num_weights}, got {weights.size})")
        
        weights.tofile(self.file_handle)
        self.total_bytes_written += weights.nbytes
        
        if self.verbose:
            print(f"  Written {num_weights} weights for layer {layer.name}")
            print(f"  Weight shape: {weights.shape}")
    
    def write_layers(self, layers: List[LayerInfo]):
        """
        Write all convolutional layers to file.
        
        Args:
            layers: List of LayerInfo objects
        """
        conv_layers = [l for l in layers if l.type == 'convolutional']
        
        if self.verbose:
            print(f"\nWriting {len(conv_layers)} convolutional layers...")
        
        for i, layer in enumerate(conv_layers):
            if self.verbose:
                print(f"\nLayer {i+1}/{len(conv_layers)}: {layer.name}")
            
            self.write_convolutional_layer(layer)
        
        if self.verbose:
            print(f"\nTotal bytes written: {self.total_bytes_written:,}")
    
    def write_model(self, layers: List[LayerInfo], major: int = 0, 
                   minor: int = 2, revision: int = 0, seen: int = 0):
        """
        Write complete model to Darknet format.
        
        Args:
            layers: List of LayerInfo objects
            major: Major version
            minor: Minor version
            revision: Revision number
            seen: Images seen during training
        """
        self.open()
        try:
            self.write_header(major, minor, revision, seen)
            self.write_layers(layers)
        finally:
            self.close()
        
        if self.verbose:
            print(f"\nSuccessfully wrote weights to {self.output_path}")


def write_darknet_weights(output_path: str, layers: List[LayerInfo], 
                         verbose: bool = False):
    """
    Convenience function to write Darknet weights file.
    
    Args:
        output_path: Output file path
        layers: List of LayerInfo objects
        verbose: Enable verbose output
    """
    writer = DarknetWeightsWriter(output_path, verbose=verbose)
    writer.write_model(layers)

