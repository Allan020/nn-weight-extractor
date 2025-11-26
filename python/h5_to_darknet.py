#!/usr/bin/env python3
"""
Keras/TensorFlow model to Darknet format converter.

Converts Keras .h5 models to Darknet .weights and .cfg format.
"""

import argparse
import os
import sys
from typing import Optional

from utils import Logger, validate_file_exists, validate_output_path, format_size, get_file_size
from model_parsers import (
    detect_model_format, ModelFormat, get_parser_for_format
)
from darknet_writer import write_darknet_weights
from cfg_generator import generate_darknet_cfg


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Keras/TensorFlow models to Darknet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert Keras H5 model to Darknet format
  python h5_to_darknet.py --input model.h5 --output-weights model.weights --output-cfg model.cfg
  
  # With custom input size
  python h5_to_darknet.py --input model.h5 --output-weights model.weights --output-cfg model.cfg --img-size 608
  
  # Verbose output
  python h5_to_darknet.py --input model.h5 --output-weights model.weights --output-cfg model.cfg --verbose
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input model file (.h5, .hdf5)'
    )
    
    parser.add_argument(
        '--output-weights', '-ow',
        required=True,
        help='Output Darknet weights file (.weights)'
    )
    
    parser.add_argument(
        '--output-cfg', '-oc',
        required=True,
        help='Output Darknet config file (.cfg)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='Input image size (width and height). If not specified, uses model input size'
    )
    
    parser.add_argument(
        '--img-width',
        type=int,
        default=None,
        help='Input image width (overrides --img-size)'
    )
    
    parser.add_argument(
        '--img-height',
        type=int,
        default=None,
        help='Input image height (overrides --img-size)'
    )
    
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='Number of input channels (default: 3)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size for .cfg file (default: 1)'
    )
    
    parser.add_argument(
        '--subdivisions',
        type=int,
        default=1,
        help='Subdivisions for .cfg file (default: 1)'
    )
    
    parser.add_argument(
        '--seen',
        type=int,
        default=0,
        help='Number of images seen during training (for .weights header)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def determine_input_shape(args, parser_input_shape):
    """Determine final input shape from arguments and parser."""
    h, w, c = parser_input_shape
    
    # Override from command-line arguments
    if args.img_width:
        w = args.img_width
    elif args.img_size:
        w = args.img_size
    
    if args.img_height:
        h = args.img_height
    elif args.img_size:
        h = args.img_size
    
    c = args.channels if args.channels else c
    
    return (h, w, c)


def main():
    """Main conversion function."""
    args = parse_arguments()
    logger = Logger(verbose=args.verbose)
    
    # Validate input file
    logger.info("Starting Keras to Darknet conversion")
    logger.info(f"Input model: {args.input}")
    
    if not validate_file_exists(args.input):
        logger.error(f"Input file does not exist or is not readable: {args.input}")
        return 1
    
    file_size = get_file_size(args.input)
    logger.info(f"Input file size: {format_size(file_size)}")
    
    # Validate output paths
    if not validate_output_path(args.output_weights):
        logger.error(f"Cannot write to output weights file: {args.output_weights}")
        return 1
    
    if not validate_output_path(args.output_cfg):
        logger.error(f"Cannot write to output cfg file: {args.output_cfg}")
        return 1
    
    # Detect model format
    logger.info("Detecting model format...")
    model_format = detect_model_format(args.input)
    logger.info(f"Detected format: {model_format.value}")
    
    if model_format == ModelFormat.UNKNOWN:
        logger.error("Unknown or unsupported model format")
        return 1
    
    if model_format == ModelFormat.DARKNET_WEIGHTS:
        logger.warning("Input is already in Darknet format, no conversion needed")
        return 1
    
    # Get appropriate parser
    logger.info("Loading model...")
    parser = get_parser_for_format(model_format, verbose=args.verbose)
    
    if parser is None:
        logger.error(f"No parser available for format: {model_format.value}")
        return 1
    
    # Load and parse model
    if not parser.load_model(args.input):
        logger.error("Failed to load model")
        return 1
    
    logger.info("Parsing model layers...")
    layers = parser.parse_layers()
    
    if not layers:
        logger.error("No layers found in model")
        return 1
    
    logger.info(f"Found {len(layers)} layers")
    conv_layers = parser.get_conv_layers()
    logger.info(f"Convolutional layers: {len(conv_layers)}")
    
    # Determine input shape
    parser_input_shape = parser.get_input_shape()
    input_shape = determine_input_shape(args, parser_input_shape)
    logger.info(f"Input shape: {input_shape[0]}x{input_shape[1]}x{input_shape[2]}")
    
    # Validate model
    logger.info("Validating model...")
    is_valid, errors = parser.validate()
    
    if not is_valid:
        logger.error("Model validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return 1
    
    logger.success("Model validation passed")
    
    # Print summary if verbose
    if args.verbose:
        parser.print_summary()
    
    # Write Darknet weights file
    logger.info(f"Writing weights to {args.output_weights}...")
    try:
        write_darknet_weights(
            args.output_weights,
            layers,
            verbose=args.verbose
        )
        weights_size = get_file_size(args.output_weights)
        logger.success(f"Weights file written ({format_size(weights_size)})")
    except Exception as e:
        logger.error(f"Failed to write weights file: {e}")
        return 1
    
    # Generate Darknet config file
    logger.info(f"Generating config file {args.output_cfg}...")
    try:
        generate_darknet_cfg(
            args.output_cfg,
            layers,
            input_shape,
            batch=args.batch,
            subdivisions=args.subdivisions,
            verbose=args.verbose
        )
        logger.success("Config file generated")
    except Exception as e:
        logger.error(f"Failed to generate config file: {e}")
        return 1
    
    # Final summary
    logger.success("\nConversion completed successfully!")
    logger.info("Output files:")
    logger.info(f"  Weights: {args.output_weights}")
    logger.info(f"  Config:  {args.output_cfg}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

