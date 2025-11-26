"""Utility functions for the weight converter tool."""

import os
import sys
import struct
import numpy as np
from typing import Optional, Tuple, List


class Logger:
    """Simple logger with verbosity control."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def info(self, message: str):
        """Print info message."""
        print(f"[INFO] {message}")
    
    def debug(self, message: str):
        """Print debug message if verbose."""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def warning(self, message: str):
        """Print warning message."""
        print(f"[WARNING] {message}", file=sys.stderr)
    
    def error(self, message: str):
        """Print error message."""
        print(f"[ERROR] {message}", file=sys.stderr)
    
    def success(self, message: str):
        """Print success message."""
        print(f"[SUCCESS] {message}")


def validate_file_exists(filepath: str) -> bool:
    """Check if file exists and is readable."""
    if not os.path.exists(filepath):
        return False
    if not os.path.isfile(filepath):
        return False
    if not os.access(filepath, os.R_OK):
        return False
    return True


def validate_output_path(filepath: str) -> bool:
    """Check if output path is writable."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            return False
    return True


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def read_binary_floats(file_handle, count: int) -> np.ndarray:
    """Read count float32 values from binary file."""
    data = np.fromfile(file_handle, dtype=np.float32, count=count)
    if len(data) != count:
        raise ValueError(f"Expected {count} floats, got {len(data)}")
    return data


def write_binary_floats(file_handle, data: np.ndarray):
    """Write float32 array to binary file."""
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    data.tofile(file_handle)


def calculate_conv_output_size(input_size: int, kernel_size: int, 
                               stride: int, padding: int) -> int:
    """Calculate convolution output size."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def count_parameters(weights_shape: Tuple) -> int:
    """Count total parameters in weight tensor."""
    count = 1
    for dim in weights_shape:
        count *= dim
    return count


def verify_darknet_weights_header(filepath: str) -> Tuple[int, int, int, int]:
    """
    Read and verify Darknet weights file header.
    Returns: (major, minor, revision, seen)
    """
    with open(filepath, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        if len(header) < 5:
            raise ValueError("Invalid weights file: header too short")
        
        major = int(header[0])
        minor = int(header[1])
        revision = int(header[2])
        
        # Seen could be int32 or int64 depending on version
        if (major * 10 + minor) >= 2:
            # Reread with proper size
            f.seek(0)
            header = np.fromfile(f, dtype=np.int32, count=3)
            seen = np.fromfile(f, dtype=np.int64, count=1)[0]
        else:
            seen = int(header[3])
        
        return major, minor, revision, int(seen)


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                       suffix: str = '', length: int = 50):
    """Print a progress bar to terminal."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                  tolerance: float = 1e-5) -> Tuple[bool, float]:
    """
    Compare two arrays within tolerance.
    Returns: (are_close, max_difference)
    """
    if arr1.shape != arr2.shape:
        return False, float('inf')
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    are_close = np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance)
    
    return are_close, float(max_diff)

