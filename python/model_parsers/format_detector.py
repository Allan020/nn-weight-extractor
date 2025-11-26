"""Model format detection utilities."""

import os
from enum import Enum
from typing import Optional
import h5py


class ModelFormat(Enum):
    """Supported model formats."""
    KERAS_H5 = "keras_h5"
    TF_SAVED_MODEL = "tf_saved_model"
    DARKNET_WEIGHTS = "darknet_weights"
    UNKNOWN = "unknown"


def detect_model_format(model_path: str) -> ModelFormat:
    """
    Detect the format of the model file.
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        ModelFormat enum value
    """
    if not os.path.exists(model_path):
        return ModelFormat.UNKNOWN
    
    # Check if it's a directory (TensorFlow SavedModel)
    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
            return ModelFormat.TF_SAVED_MODEL
        return ModelFormat.UNKNOWN
    
    # Check file extension
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    
    if ext == '.h5' or ext == '.hdf5':
        # Verify it's actually a Keras model
        if _is_keras_h5(model_path):
            return ModelFormat.KERAS_H5
        return ModelFormat.UNKNOWN
    
    elif ext == '.weights':
        # Check for Darknet weights file
        if _is_darknet_weights(model_path):
            return ModelFormat.DARKNET_WEIGHTS
        return ModelFormat.UNKNOWN
    
    return ModelFormat.UNKNOWN


def _is_keras_h5(filepath: str) -> bool:
    """Check if file is a valid Keras H5 model."""
    try:
        with h5py.File(filepath, 'r') as f:
            # Keras models have specific structure
            if 'model_weights' in f.keys() or 'layer_names' in f.attrs:
                return True
            # Check for model config
            if 'model_config' in f.attrs:
                return True
    except (OSError, IOError):
        return False
    return False


def _is_darknet_weights(filepath: str) -> bool:
    """Check if file is a valid Darknet weights file."""
    try:
        with open(filepath, 'rb') as f:
            # Darknet weights start with header: major, minor, revision (int32)
            header = f.read(12)
            if len(header) < 12:
                return False
            # Check if values look reasonable (version numbers should be small)
            import struct
            major, minor, revision = struct.unpack('iii', header)
            if 0 <= major <= 10 and 0 <= minor <= 10 and 0 <= revision <= 10:
                return True
    except (OSError, IOError):
        return False
    return False


def get_parser_for_format(format_type: ModelFormat, verbose: bool = False):
    """
    Get appropriate parser instance for the format.
    
    Args:
        format_type: ModelFormat enum value
        verbose: Enable verbose output
        
    Returns:
        Parser instance or None
    """
    if format_type == ModelFormat.KERAS_H5:
        from .keras_parser import KerasParser
        return KerasParser(verbose=verbose)
    
    elif format_type == ModelFormat.TF_SAVED_MODEL:
        # Not implemented yet
        return None
    
    elif format_type == ModelFormat.DARKNET_WEIGHTS:
        # Darknet weights don't need parsing, they're already in the right format
        return None
    
    return None

