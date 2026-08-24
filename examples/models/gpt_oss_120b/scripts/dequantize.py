#!/usr/bin/env python3
"""
Script to dequantize model weights from MXFP4 to BF16 format with parallel processing.
Supports processing all layers or a specified number of layers, with output split into
multiple safetensors files for optimal performance on FSx filesystems.
"""

import os
import re
import time
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from safetensors.torch import save_file
from weights import Checkpoint


def extract_layer_number(tensor_name: str) -> int:
    """Extract layer number from tensor name. Returns -1 if not a layer tensor."""
    match = re.match(r'block\.(\d+)\.', tensor_name)
    return int(match.group(1)) if match else -1


def filter_tensors_by_layers(all_tensor_names: List[str], max_layers: int) -> List[str]:
    """Filter tensor names based on max_layers parameter."""
    if max_layers == -1:
        # Process all tensors
        return all_tensor_names
    
    filtered_tensors = []
    for tensor_name in all_tensor_names:
        layer_num = extract_layer_number(tensor_name)
        if layer_num == -1:
            # Non-layer tensor (e.g., embedding.weight), always include
            filtered_tensors.append(tensor_name)
        elif layer_num < max_layers:
            # Layer tensor within the limit
            filtered_tensors.append(tensor_name)
    
    return filtered_tensors


def calculate_tensor_size_bytes(tensor_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    """Calculate tensor size in bytes."""
    numel = 1
    for dim in tensor_shape:
        numel *= dim
    return numel * torch.tensor([], dtype=dtype).element_size()


def process_tensor_batch(args: Tuple[str, List[str], torch.device]) -> Dict[str, torch.Tensor]:
    """Process a batch of tensors in parallel. Returns dict of tensor_name -> tensor."""
    source_path, tensor_names, device = args
    checkpoint = Checkpoint(source_path, device)
    batch_results = {}
    
    for tensor_name in tensor_names:
        try:
            tensor = checkpoint.get(tensor_name)
            
            # Ensure tensor is in BF16
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            
            batch_results[tensor_name] = tensor
            
        except Exception as e:
            print(f"  Error processing {tensor_name}: {e}")
            continue
    
    return batch_results


def save_tensor_group(args: Tuple[Dict[str, torch.Tensor], str, int, int]) -> Tuple[str, float, int]:
    """Save a group of tensors to a safetensors file. Returns (filename, size_gb, tensor_count)."""
    group, output_file, part_num, total_parts = args
    
    try:
        group_size_gb = sum(t.numel() * t.element_size() for t in group.values()) / (1024**3)
        print(f"  Saving part {part_num}/{total_parts}: {len(group)} tensors, {group_size_gb:.2f} GiB -> {os.path.basename(output_file)}")
        
        save_file(group, output_file)
        return output_file, group_size_gb, len(group)
        
    except Exception as e:
        print(f"  Error saving {output_file}: {e}")
        return output_file, 0.0, 0


def dequantize_tensors(source_path: str, dest_dir: str, max_layers: int = -1, 
                      num_workers: int = 4, max_file_size_gb: float = 5.0):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Initialize checkpoint loader
    device = torch.device("cpu")  # Use CPU to avoid memory issues
    checkpoint = Checkpoint(source_path, device)

    print("Available tensors in checkpoint:")
    for name in sorted(checkpoint.tensor_name_to_file.keys()):
        print(f"  {name}")

    # Filter tensors based on max_layers parameter
    all_tensor_names = list(checkpoint.tensor_name_to_file.keys())
    filtered_tensor_names = filter_tensors_by_layers(all_tensor_names, max_layers)

    print(f"\nFiltered to {len(filtered_tensor_names)} tensors")
    if max_layers != -1:
        print(f"(processing layers 0-{max_layers-1} + per-model tensors)")
    else:
        print("(processing all layers)")
    
    for name in sorted(filtered_tensor_names)[:10]:  # Show first 10
        print(f"  {name}")
    if len(filtered_tensor_names) > 10:
        print(f"  ... and {len(filtered_tensor_names) - 10} more")

    # Identify which tensors are quantized weights vs regular tensors
    print("\nAnalyzing tensor types...")

    quantized_weights = set()
    regular_tensors = set()

    for tensor_name in filtered_tensor_names:
        if tensor_name.endswith(".blocks"):
            # This is a quantized weight, add the base name
            base_name = tensor_name[:-7]  # Remove '.blocks'
            quantized_weights.add(base_name)
        elif tensor_name.endswith(".scales"):
            # Skip, already handled by .blocks case
            continue
        else:
            # Check if this has corresponding .blocks and .scales
            blocks_name = tensor_name + ".blocks"
            scales_name = tensor_name + ".scales"
            if (
                blocks_name in filtered_tensor_names
                and scales_name in filtered_tensor_names
            ):
                quantized_weights.add(tensor_name)
            else:
                regular_tensors.add(tensor_name)

    print(f"Found {len(quantized_weights)} quantized weights and {len(regular_tensors)} regular tensors")

    # Combine all tensors to process
    all_tensors_to_process = list(regular_tensors) + list(quantized_weights)
    total_tensors = len(all_tensors_to_process)

    print(f"\nProcessing {total_tensors} tensors using {num_workers} workers...")
    
    # Split tensors into batches for parallel processing
    batch_size = max(1, total_tensors // num_workers)
    tensor_batches = []
    for i in range(0, total_tensors, batch_size):
        batch = all_tensors_to_process[i:i + batch_size]
        tensor_batches.append((source_path, batch, device))

    print(f"Created {len(tensor_batches)} batches (avg {len(all_tensors_to_process)//len(tensor_batches)} tensors per batch)")

    # Process batches in parallel
    start_time = time.time()
    all_params = {}

    if num_workers > 1 and len(tensor_batches) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {executor.submit(process_tensor_batch, batch): i 
                             for i, batch in enumerate(tensor_batches)}
            
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_params.update(batch_results)
                    completed_batches += 1
                    print(f"  Batch {completed_batches}/{len(tensor_batches)} completed ({len(batch_results)} tensors)")
                except Exception as e:
                    print(f"  Batch {batch_idx + 1} failed: {e}")
    else:
        # Fallback to sequential processing for small jobs
        print("Using sequential processing (single worker or small batch count)")
        for i, batch_args in enumerate(tensor_batches):
            batch_results = process_tensor_batch(batch_args)
            all_params.update(batch_results)
            print(f"  Batch {i + 1}/{len(tensor_batches)} completed ({len(batch_results)} tensors)")

    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds ({len(all_params)} tensors processed)")

    # Group tensors into files based on size limit
    print(f"\nGrouping tensors into files (max {max_file_size_gb:.1f} GiB per file)...")
    max_size_bytes = int(max_file_size_gb * (1024**3))
    tensor_groups = []
    current_group = {}
    current_size = 0

    for tensor_name, tensor in all_params.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # If adding this tensor would exceed the limit, start a new group
        if current_size + tensor_size > max_size_bytes and current_group:
            tensor_groups.append(current_group)
            current_group = {}
            current_size = 0
        
        current_group[tensor_name] = tensor
        current_size += tensor_size
    
    # Add the last group if it has any tensors
    if current_group:
        tensor_groups.append(current_group)

    print(f"Created {len(tensor_groups)} file groups")

    # Prepare file saving tasks
    save_tasks = []
    output_files = []
    
    for i, group in enumerate(tensor_groups):
        part_num = i + 1
        output_file = os.path.join(dest_dir, f"model_part_{part_num:03d}.safetensors")
        output_files.append(output_file)
        save_tasks.append((group, output_file, part_num, len(tensor_groups)))

    # Save files in parallel using ThreadPoolExecutor
    save_start_time = time.time()
    print(f"Saving {len(tensor_groups)} files in parallel...")
    
    file_results = []
    max_io_workers = min(len(tensor_groups), 8)  # Limit concurrent file writes for FSx
    
    with ThreadPoolExecutor(max_workers=max_io_workers) as executor:
        future_to_task = {executor.submit(save_tensor_group, task): i 
                         for i, task in enumerate(save_tasks)}
        
        for future in as_completed(future_to_task):
            task_idx = future_to_task[future]
            try:
                filename, size_gb, tensor_count = future.result()
                file_results.append((filename, size_gb, tensor_count))
            except Exception as e:
                print(f"  Save task {task_idx + 1} failed: {e}")

    save_time = time.time() - save_start_time
    print(f"Parallel file saving completed in {save_time:.2f} seconds")

    # Calculate performance metrics
    total_params = sum(tensor.numel() for tensor in all_params.values())
    total_size_gb = sum(
        tensor.numel() * tensor.element_size() for tensor in all_params.values()
    ) / (1024**3)
    total_time = processing_time + save_time
    
    # Throughput calculations
    processing_throughput_gb_s = total_size_gb / processing_time if processing_time > 0 else 0
    save_throughput_gb_s = total_size_gb / save_time if save_time > 0 else 0
    overall_throughput_gb_s = total_size_gb / total_time if total_time > 0 else 0
    tensors_per_second = len(all_params) / processing_time if processing_time > 0 else 0

    # Print detailed summary
    print(f"\n{'='*60}")
    print("DEQUANTIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Source:           {source_path}")
    print(f"Destination:      {dest_dir}")
    print(f"Max layers:       {'All' if max_layers == -1 else max_layers}")
    print(f"Workers used:     {num_workers}")
    
    print("\nDATA PROCESSED:")
    print(f"  Tensors processed:    {len(all_params):,}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Total data size:      {total_size_gb:.2f} GiB")
    print(f"  Files created:        {len(output_files)}")
    print(f"  Avg file size:        {total_size_gb/len(output_files):.2f} GiB")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Processing time:      {processing_time:.2f}s")
    print(f"  File save time:       {save_time:.2f}s")
    print(f"  Total time:           {total_time:.2f}s")
    print(f"  Processing throughput: {processing_throughput_gb_s:.2f} GiB/s")
    print(f"  Save throughput:      {save_throughput_gb_s:.2f} GiB/s")
    print(f"  Overall throughput:   {overall_throughput_gb_s:.2f} GiB/s")
    print(f"  Tensors per second:   {tensors_per_second:.1f} tensors/s")
    
    print("\nOUTPUT FILES:")
    for i, output_file in enumerate(output_files, 1):
        if i <= len(file_results):
            _, size_gb, tensor_count = file_results[i-1]
            print(f"  {i:2d}. {os.path.basename(output_file)} ({tensor_count} tensors, {size_gb:.2f} GiB)")
        else:
            print(f"  {i:2d}. {os.path.basename(output_file)}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    # Source and destination paths
    import argparse
    parser = argparse.ArgumentParser(description="Dequantize model weights from MXFP4 to BF16 format with parallel processing")
    parser.add_argument("--source-path", type=str, required=True, help="Path to the source checkpoint")
    parser.add_argument("--dest-dir", type=str, required=True, help="Path to the destination directory")
    parser.add_argument("--max-layers", type=int, default=-1, 
                       help="Maximum number of layers to process (-1 for all layers)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of parallel workers for processing")
    parser.add_argument("--max-file-size-gb", type=float, default=5.0,
                       help="Maximum size per output file in GiB")
    args = parser.parse_args()
    dequantize_tensors(args.source_path, args.dest_dir, args.max_layers, args.num_workers, args.max_file_size_gb)
