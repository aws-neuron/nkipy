#!/usr/bin/env python3
"""
Script to convert specific layers in safetensor files to float8_e5m2 format.

This script:
1. Opens a safetensor file
2. Finds layers matching patterns like "layers.*.gate_up_weight" and "layers.*.down_weight"
3. Converts those tensors to float8_e5m2 format
4. Saves all tensors back to a new directory with the same structure
"""

import os
import re
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from typing import Dict, Any


def matches_conversion_pattern(tensor_name: str) -> bool:
    """
    Check if a tensor name matches the patterns that should be converted to float8_e5m2.
    
    Patterns to match:
    - layers.*.gate_up_weight (e.g., layers.0.gate_up_weight, layers.32.gate_up_weight)
    - layers.*.down_weight (e.g., layers.0.down_weight, layers.32.down_weight)
    
    Args:
        tensor_name: Name of the tensor to check
        
    Returns:
        True if the tensor should be converted, False otherwise
    """
    patterns = [
        r'^layers\.\d+\.gate_up_weight$',
        r'^layers\.\d+\.down_weight$'
    ]
    
    for pattern in patterns:
        if re.match(pattern, tensor_name):
            return True
    return False


def convert_safetensor_file(input_path: str, output_dir: str) -> None:
    """
    Convert specific layers in a safetensor file to float8_e5m2 format.
    
    Args:
        input_path: Path to the input safetensor file
        output_dir: Directory where the converted file will be saved
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file will have the same name as input file
    output_path = output_dir / input_path.name
    
    print(f"Converting {input_path} -> {output_path}")
    
    # Dictionary to store all tensors (converted and original)
    tensors_dict: Dict[str, torch.Tensor] = {}
    metadata_dict: Dict[str, str] = {}
    
    # Open and process the safetensor file
    with safe_open(input_path, framework="pt", device="cpu") as f:
        # Get metadata if available
        metadata = f.metadata()
        if metadata:
            metadata_dict.update(metadata)
        
        # Process each tensor
        for tensor_name in f.keys():
            tensor = f.get_tensor(tensor_name)
            
            if matches_conversion_pattern(tensor_name):
                print(f"Converting {tensor_name}: {tensor.dtype} -> float8_e5m2")
                # Convert to float8_e5m2
                converted_tensor = tensor.to(torch.float8_e5m2)
                tensors_dict[tensor_name] = converted_tensor
            else:
                # Keep original tensor unchanged
                tensors_dict[tensor_name] = tensor
    
    # Save all tensors to the new file
    print(f"Saving {len(tensors_dict)} tensors to {output_path}")
    save_file(tensors_dict, output_path, metadata=metadata_dict)
    print("Conversion completed successfully!")


def find_safetensor_files(directory: str) -> list:
    """
    Find all .safetensors files in a directory.
    
    Args:
        directory: Directory to search for safetensor files
        
    Returns:
        List of paths to safetensor files
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    safetensor_files = []
    for file_path in directory.rglob("*.safetensors"):
        if file_path.is_file():
            safetensor_files.append(file_path)
    
    return sorted(safetensor_files)


def process_directory_or_file(input_path: str, output_dir: str, dry_run: bool = False) -> int:
    """
    Process either a single file or all safetensor files in a directory.
    
    Args:
        input_path: Path to file or directory
        output_dir: Output directory
        dry_run: Whether to run in dry-run mode
        
    Returns:
        0 on success, 1 on error
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file processing
        if not input_path.suffix == '.safetensors':
            print(f"Error: {input_path} is not a safetensor file")
            return 1
        files_to_process = [input_path]
    elif input_path.is_dir():
        # Directory processing
        files_to_process = find_safetensor_files(input_path)
        if not files_to_process:
            print(f"No safetensor files found in directory: {input_path}")
            return 1
        print(f"Found {len(files_to_process)} safetensor files in {input_path}")
    else:
        print(f"Error: {input_path} does not exist")
        return 1
    
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("="*60)
    
    total_files_processed = 0
    total_conversions = 0
    
    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}")
        
        if dry_run:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    conversion_count = 0
                    total_count = 0
                    
                    for tensor_name in f.keys():
                        total_count += 1
                        if matches_conversion_pattern(tensor_name):
                            tensor = f.get_tensor(tensor_name)
                            print(f"  Would convert: {tensor_name} ({tensor.dtype} -> float8_e5m2)")
                            conversion_count += 1
                    
                    print(f"  Summary: {conversion_count}/{total_count} tensors would be converted")
                    total_conversions += conversion_count
            except Exception as e:
                print(f"  Error analyzing {file_path}: {e}")
                continue
        else:
            try:
                convert_safetensor_file(str(file_path), output_dir)
                total_files_processed += 1
            except Exception as e:
                print(f"  Error converting {file_path}: {e}")
                continue
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN SUMMARY:")
        print(f"  Files analyzed: {len(files_to_process)}")
        print(f"  Total tensors that would be converted: {total_conversions}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"CONVERSION SUMMARY:")
        print(f"  Files processed successfully: {total_files_processed}/{len(files_to_process)}")
        print(f"{'='*60}")
    
    return 0


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert specific layers in safetensor files to float8_e5m2 format"
    )
    parser.add_argument(
        "input_path",
        help="Path to the input safetensor file or directory containing safetensor files"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the converted files will be saved"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which tensors would be converted without actually converting"
    )
    
    args = parser.parse_args()
    
    return process_directory_or_file(args.input_path, args.output_dir, args.dry_run)


if __name__ == "__main__":
    exit(main())
