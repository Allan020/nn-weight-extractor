"""Darknet configuration file (.cfg) generator."""

from typing import List, TextIO, Tuple
from model_parsers.base_parser import LayerInfo


class DarknetCfgGenerator:
    """Generate Darknet .cfg files from parsed models."""
    
    def __init__(self, output_path: str, verbose: bool = False):
        self.output_path = output_path
        self.verbose = verbose
    
    def write_net_section(self, file: TextIO, input_shape: Tuple[int, int, int],
                         batch: int = 1, subdivisions: int = 1,
                         momentum: float = 0.9, decay: float = 0.0005):
        """
        Write [net] section.
        
        Args:
            file: Output file handle
            input_shape: (height, width, channels)
            batch: Batch size
            subdivisions: Subdivisions
            momentum: Momentum
            decay: Weight decay
        """
        height, width, channels = input_shape
        
        file.write("[net]\n")
        file.write(f"batch={batch}\n")
        file.write(f"subdivisions={subdivisions}\n")
        file.write(f"width={width}\n")
        file.write(f"height={height}\n")
        file.write(f"channels={channels}\n")
        file.write(f"momentum={momentum}\n")
        file.write(f"decay={decay}\n")
        file.write("\n")
    
    def write_convolutional_layer(self, file: TextIO, layer: LayerInfo):
        """Write [convolutional] layer section."""
        file.write("[convolutional]\n")
        
        if layer.batch_normalize:
            file.write("batch_normalize=1\n")
        
        file.write(f"filters={layer.filters}\n")
        file.write(f"size={layer.kernel_size}\n")
        file.write(f"stride={layer.stride}\n")
        file.write(f"pad={1 if layer.padding else 0}\n")
        
        if layer.groups and layer.groups > 1:
            file.write(f"groups={layer.groups}\n")
        
        # Determine activation
        activation = layer.activation
        if activation == 'linear':
            activation = 'linear'
        elif activation in ['relu', 'leaky', 'leaky_relu']:
            activation = 'leaky'
        else:
            activation = 'linear'
        
        file.write(f"activation={activation}\n")
        file.write("\n")
    
    def write_maxpool_layer(self, file: TextIO, layer: LayerInfo):
        """Write [maxpool] layer section."""
        file.write("[maxpool]\n")
        file.write(f"size={layer.pool_size}\n")
        file.write(f"stride={layer.pool_stride}\n")
        file.write("\n")
    
    def write_upsample_layer(self, file: TextIO, layer: LayerInfo):
        """Write [upsample] layer section."""
        file.write("[upsample]\n")
        file.write(f"stride={layer.stride}\n")
        file.write("\n")
    
    def write_route_layer(self, file: TextIO, layer: LayerInfo):
        """Write [route] layer section."""
        file.write("[route]\n")
        if layer.layers:
            layers_str = ", ".join(str(l) for l in layer.layers)
            file.write(f"layers={layers_str}\n")
        else:
            # Will need to be filled in manually or determined from model graph
            file.write("layers=-1\n")
        file.write("\n")
    
    def write_shortcut_layer(self, file: TextIO, layer: LayerInfo):
        """Write [shortcut] layer section."""
        file.write("[shortcut]\n")
        if layer.layers:
            # Shortcut typically references one layer
            file.write(f"from={layer.layers[0]}\n")
        else:
            file.write("from=-3\n")  # Common default
        file.write("activation=linear\n")
        file.write("\n")
    
    def write_yolo_layer(self, file: TextIO, num_classes: int = 80,
                        anchors: str = None, mask: str = None):
        """Write [yolo] detection layer section."""
        file.write("[yolo]\n")
        
        if mask:
            file.write(f"mask={mask}\n")
        else:
            file.write("mask=6,7,8\n")
        
        if anchors:
            file.write(f"anchors={anchors}\n")
        else:
            # Default YOLO anchors
            file.write("anchors=10,13, 16,30, 33,23, 30,61, 62,45, "
                      "59,119, 116,90, 156,198, 373,326\n")
        
        file.write(f"classes={num_classes}\n")
        file.write("num=9\n")
        file.write("jitter=.3\n")
        file.write("ignore_thresh=.5\n")
        file.write("truth_thresh=1\n")
        file.write("random=1\n")
        file.write("\n")
    
    def write_layer(self, file: TextIO, layer: LayerInfo):
        """Write a single layer section."""
        if layer.type == 'convolutional':
            self.write_convolutional_layer(file, layer)
        elif layer.type == 'maxpool':
            self.write_maxpool_layer(file, layer)
        elif layer.type == 'upsample':
            self.write_upsample_layer(file, layer)
        elif layer.type == 'route':
            self.write_route_layer(file, layer)
        elif layer.type == 'shortcut':
            self.write_shortcut_layer(file, layer)
        # Skip batch_normalization as it's part of convolutional
    
    def generate(self, layers: List[LayerInfo], input_shape: Tuple[int, int, int],
                batch: int = 1, subdivisions: int = 1):
        """
        Generate complete .cfg file.
        
        Args:
            layers: List of LayerInfo objects
            input_shape: (height, width, channels)
            batch: Batch size
            subdivisions: Subdivisions
        """
        with open(self.output_path, 'w') as f:
            # Write network section
            self.write_net_section(f, input_shape, batch, subdivisions)
            
            # Write all layers (skip batch_normalization as they're merged with conv)
            layer_count = 0
            for layer in layers:
                if layer.type != 'batch_normalization':
                    if self.verbose:
                        print(f"Writing layer {layer_count}: {layer.name} ({layer.type})")
                    self.write_layer(f, layer)
                    layer_count += 1
        
        if self.verbose:
            print(f"\nGenerated config file: {self.output_path}")
            print(f"Total layers written: {layer_count}")


def generate_darknet_cfg(output_path: str, layers: List[LayerInfo],
                        input_shape: Tuple[int, int, int],
                        batch: int = 1, subdivisions: int = 1,
                        verbose: bool = False):
    """
    Convenience function to generate Darknet .cfg file.
    
    Args:
        output_path: Output file path
        layers: List of LayerInfo objects
        input_shape: (height, width, channels)
        batch: Batch size
        subdivisions: Subdivisions
        verbose: Enable verbose output
    """
    generator = DarknetCfgGenerator(output_path, verbose=verbose)
    generator.generate(layers, input_shape, batch, subdivisions)

