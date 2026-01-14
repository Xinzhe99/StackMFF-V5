# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# Script to find the best epoch based on PSNR and SSIM metrics

import os
import subprocess
import argparse
import re
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures
import threading
from typing import Dict, Tuple, Any
import numpy as np
import cv2
from PIL import Image
import glob
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Find best epoch based on PSNR and SSIM metrics')
    parser.add_argument('--checkpoints_dir', type=str, 
                        default=r'/home/ot/Students/xxz/projects_mff/StackMFFV5/train_runs/train_runs45/model_save',
                        help='Directory containing checkpoint files')
    parser.add_argument('--test_root', type=str, 
                        default=r'/home/ot/Students/xxz/datasets/test_datasets',
                        help='Root directory containing test datasets')
    parser.add_argument('--ground_truth_path', type=str, 
                        default=r'/home/ot/Students/xxz/datasets/results_of_different_methods/Ground Truth',
                        help='Path to ground truth images')
    parser.add_argument('--datasets', nargs='+', 
                        default=['Mobile Depth','Middlebury'],
                        help='List of datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='./best_epoch_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='Start epoch number')
    parser.add_argument('--end_epoch', type=int, default=21, 
                        help='End epoch number')
    return parser.parse_args()

def get_checkpoint_files(checkpoints_dir, start_epoch=1, end_epoch=25):
    checkpoint_files = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith('.pth') and file.startswith('epoch_'):
            # Extract epoch number from filename
            match = re.search(r'epoch_(\d+)\.pth', file)
            if match:
                epoch_num = int(match.group(1))
                if start_epoch <= epoch_num <= end_epoch:
                    checkpoint_files.append((epoch_num, os.path.join(checkpoints_dir, file)))
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])
    return checkpoint_files

def run_inference(model_path, test_root, datasets, output_dir, batch_size, num_workers):
    """Run inference using predict_datasets.py"""
    cmd = [
        'python', 'predict_datasets.py',
        '--model_path', model_path,
        '--data_root', test_root,  # Changed from --test_root to --data_root
        '--datasets'] + datasets + [  # Changed from --test_datasets to --datasets
        '--output_dir', output_dir,
        '--device', 'cuda:0'  # Added device parameter
    ]
    
    print(f"Running inference with command: {' '.join(cmd)}")
    
    # Run the command and stream output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    # Print output in real-time
    if process.stdout is not None:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[INF] {output.strip()}")
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Error running inference, return code: {process.returncode}")
        return None
    
    # Extract timestamped output directory from stdout
    # The predict_datasets.py script creates a directory with timestamp
    # We need to find the latest created directory in output_dir
    try:
        # Look for directories that start with 'batch_results_' (updated naming convention)
        output_dirs = [d for d in os.listdir(output_dir) if d.startswith('batch_results_')]
        if not output_dirs:
            print("No results directory found")
            return None
        
        # Get the most recent results directory
        latest_dir = max(output_dirs, key=lambda d: os.path.getctime(os.path.join(output_dir, d)))
        return os.path.join(output_dir, latest_dir)
    except Exception as e:
        print(f"Error finding results directory: {str(e)}")
        # If we can't find the specific directory pattern, try listing all directories
        try:
            all_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
            if all_dirs:
                # Return the most recently created directory
                latest_dir = max(all_dirs, key=lambda d: os.path.getctime(os.path.join(output_dir, d)))
                return os.path.join(output_dir, latest_dir)
        except Exception as fallback_e:
            print(f"Fallback also failed: {str(fallback_e)}")
        return None

def get_image_formats(folder):
    """Get all image file formats in the folder"""
    formats = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            ext = ext[1:].lower() # remove .
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                if ext not in formats:
                    formats.append(ext)
    return formats

def read_image_flexible(base_path, supported_formats=['jpg', 'jpeg', 'png']):
    """
    Read images more flexibly, trying multiple possible file extensions
    
    Args:
        base_path: File path without extension
        supported_formats: List of supported formats
    
    Returns:
        Image data or None
    """
    # First try the original path
    if os.path.exists(base_path):
        img = cv2.imread(base_path, cv2.IMREAD_COLOR)  # Read as color image
        if img is not None:
            return img
    
    # Try various supported extensions
    for fmt in supported_formats:
        path_with_ext = f"{base_path}.{fmt}"
        if os.path.exists(path_with_ext):
            img = cv2.imread(path_with_ext, cv2.IMREAD_COLOR)  # Read as color image
            if img is not None:
                return img
    
    # Try uppercase extensions
    for fmt in supported_formats:
        path_with_ext = f"{base_path}.{fmt.upper()}"
        if os.path.exists(path_with_ext):
            img = cv2.imread(path_with_ext, cv2.IMREAD_COLOR)  # Read as color image
            if img is not None:
                return img
                
    return None

def get_images_dict(folder, supported_formats=['jpg', 'jpeg', 'png']):
    """
    Get all image files in supported formats from a folder, returning a dictionary with filename (without extension) as key
    
    Args:
        folder: Folder path
        supported_formats: List of supported formats
    
    Returns:
        dict: {filename_without_ext: full_path}
    """
    images_dict = {}
    
    # Iterate through all supported formats
    for fmt in supported_formats:
        # Find files with lowercase extensions
        pattern = os.path.join(folder, f"*.{fmt}")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            images_dict[name_without_ext] = file_path
            
        # Find files with uppercase extensions
        pattern = os.path.join(folder, f"*.{fmt.upper()}")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            images_dict[name_without_ext] = file_path
    
    return images_dict

def calculate_metrics_for_dataset(result_path, gt_path, supported_formats=['jpg', 'jpeg', 'png']):
    """
    Calculate PSNR and SSIM metrics for a single dataset by comparing result images with ground truth images
    
    Args:
        result_path: Path to the result images folder
        gt_path: Path to the ground truth images folder
        supported_formats: List of supported image formats
    
    Returns:
        dict: {"PSNR": average_psnr, "SSIM": average_ssim}
    """
    # Get ground truth image dictionary {filename: full_path}
    gt_images_dict = get_images_dict(gt_path, supported_formats)
    ground_truth_names = list(gt_images_dict.keys())
    
    if not ground_truth_names:
        print(f"Warning: No ground truth images found in {gt_path}")
        return {"PSNR": 0, "SSIM": 0}
        
    print(f"Found {len(ground_truth_names)} ground truth images in {gt_path}")
    
    # Get result image dictionary
    result_images_dict = get_images_dict(result_path, supported_formats)
    
    if not result_images_dict:
        print(f"Warning: No result images found in {result_path}")
        return {"PSNR": 0, "SSIM": 0}
        
    print(f"Found {len(result_images_dict)} result images in {result_path}")
    
    # Initialize metrics
    total_psnr = 0
    total_ssim = 0
    valid_image_count = 0
    
    # Iterate through ground truth images
    for img_name in tqdm(ground_truth_names, desc=f"Processing {os.path.basename(gt_path)}"):
        # Get ground truth image path
        gt_img_path = gt_images_dict[img_name]
        img_truth = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        
        if img_truth is None:
            print(f"Warning: Could not read ground truth image {gt_img_path}")
            continue
        
        # Construct base path for result image (without extension)
        result_base_path = os.path.join(result_path, img_name)
        
        # Try to read result image
        img_result = read_image_flexible(result_base_path)
        
        # Skip if result image cannot be read
        if img_result is None:
            print(f"Warning: Could not read result image for {img_name}")
            continue
        
        # Skip if image sizes don't match
        if len(img_result.shape) != len(img_truth.shape) or img_result.shape[0:2] != img_truth.shape[0:2]:
            print(f"Warning: Image sizes don't match for {img_name}: {img_result.shape} vs {img_truth.shape}")
            continue
        
        # Calculate metrics
        try:
            # For color images, we need to handle them properly
            if len(img_truth.shape) == 3 and len(img_result.shape) == 3:  # Color images
                # Convert to grayscale for PSNR/SSIM calculation
                img_truth_gray = cv2.cvtColor(img_truth, cv2.COLOR_BGR2GRAY)
                img_result_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
                psnr_value = compare_psnr(img_truth_gray, img_result_gray)
                ssim_value = compare_ssim(img_truth_gray, img_result_gray)
            else:  # Grayscale images
                psnr_value = compare_psnr(img_truth, img_result)
                ssim_value = compare_ssim(img_truth, img_result)
            
            total_psnr += psnr_value
            total_ssim += ssim_value
            valid_image_count += 1
            
        except Exception as e:
            print(f"Error calculating metrics for {img_name}: {str(e)}")
            continue
    
    # Calculate averages
    if valid_image_count > 0:
        avg_psnr = total_psnr / valid_image_count
        avg_ssim = total_ssim / valid_image_count
        
        print(f"Dataset {os.path.basename(gt_path)}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f} (from {valid_image_count} valid images)")
        
        return {"PSNR": avg_psnr, "SSIM": avg_ssim}
    else:
        print(f"Warning: No valid images processed for dataset {os.path.basename(gt_path)}")
        return {"PSNR": 0, "SSIM": 0}

def run_evaluation(results_dir, ground_truth_path, datasets, output_dir):
    """
    Run evaluation by calculating PSNR and SSIM metrics for each dataset
    
    Args:
        results_dir: Directory containing result images (organized by dataset folders)
        ground_truth_path: Directory containing ground truth images (organized by dataset folders)
        datasets: List of dataset names to evaluate
        output_dir: Directory to save results
    
    Returns:
        dict: Metrics for all datasets {dataset: {"PSNR": value, "SSIM": value}}
    """
    print(f"Running evaluation: comparing {results_dir} with {ground_truth_path} for datasets {datasets}")
    
    all_metrics = {}
    
    for dataset in datasets:
        # Define paths for this dataset
        result_dataset_path = os.path.join(results_dir, dataset)
        gt_dataset_path = os.path.join(ground_truth_path, dataset)
        
        # Verify that both directories exist
        if not os.path.exists(result_dataset_path):
            print(f"Warning: Result directory does not exist: {result_dataset_path}")
            continue
        
        if not os.path.exists(gt_dataset_path):
            print(f"Warning: Ground truth directory does not exist: {gt_dataset_path}")
            continue
        
        # Calculate metrics for this dataset
        dataset_metrics = calculate_metrics_for_dataset(result_dataset_path, gt_dataset_path)
        all_metrics[dataset] = dataset_metrics
    
    return all_metrics

def process_single_epoch(epoch_num: int, checkpoint_path: str, args: argparse.Namespace) -> Tuple[int, Dict[str, Dict[str, float]]]:
    """Process a single epoch and return its metrics"""
    print(f"\n[{threading.current_thread().name}] Processing epoch {epoch_num}...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Create temporary directory for this epoch's inference results
    epoch_output_dir = os.path.join(args.output_dir, f'epoch_{epoch_num}_results')
    
    # Run inference
    inference_results_dir = run_inference(
        checkpoint_path, 
        args.test_root, 
        args.datasets, 
        epoch_output_dir,
        args.batch_size,
        args.num_workers
    )
    
    if inference_results_dir is None:
        print(f"Skipping epoch {epoch_num} due to inference error")
        return epoch_num, {}
        
    print(f"Inference results directory: {inference_results_dir}")
        
    # Run evaluation - this now returns the metrics directly
    metrics = run_evaluation(
        inference_results_dir,
        args.ground_truth_path,
        args.datasets,
        args.output_dir  # This parameter is now ignored in the new function
    )
    
    if metrics is None:
        print(f"Skipping epoch {epoch_num} due to evaluation error")
        return epoch_num, {}
        
    print(f"Epoch {epoch_num} results:")
    for dataset, values in metrics.items():
        print(f"  {dataset} - PSNR: {values['PSNR']:.4f}, SSIM: {values['SSIM']:.4f}")
        
    return epoch_num, metrics

def find_best_epochs(all_results):
    """Find the epochs with highest PSNR and SSIM for each dataset"""
    best_epochs = {}
    
    for dataset in all_results[list(all_results.keys())[0]].keys():
        best_epochs[dataset] = {
            'PSNR': {'epoch': -1, 'value': -1},
            'SSIM': {'epoch': -1, 'value': -1}
        }
    
    # Iterate through all epochs
    for epoch, metrics in all_results.items():
        for dataset, values in metrics.items():
            # Update best PSNR
            if values['PSNR'] > best_epochs[dataset]['PSNR']['value']:
                best_epochs[dataset]['PSNR'] = {'epoch': epoch, 'value': values['PSNR']}
                
            # Update best SSIM
            if values['SSIM'] > best_epochs[dataset]['SSIM']['value']:
                best_epochs[dataset]['SSIM'] = {'epoch': epoch, 'value': values['SSIM']}
    
    # Calculate average PSNR and SSIM for each epoch
    epoch_averages = {}
    for epoch, metrics in all_results.items():
        total_psnr = 0
        total_ssim = 0
        count = len(metrics)
        
        for dataset, values in metrics.items():
            total_psnr += values['PSNR']
            total_ssim += values['SSIM']
            
        epoch_averages[epoch] = {
            'avg_PSNR': total_psnr / count,
            'avg_SSIM': total_ssim / count
        }
    
    # Find epochs with highest average PSNR and SSIM
    best_avg_psnr_epoch = max(epoch_averages.keys(), key=lambda x: epoch_averages[x]['avg_PSNR'])
    best_avg_ssim_epoch = max(epoch_averages.keys(), key=lambda x: epoch_averages[x]['avg_SSIM'])
    
    return best_epochs, epoch_averages, best_avg_psnr_epoch, best_avg_ssim_epoch

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get checkpoint files
    checkpoint_files = get_checkpoint_files(args.checkpoints_dir, args.start_epoch, args.end_epoch)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Dictionary to store results for all epochs
    all_results = {}
    
    # Process each checkpoint in parallel
    print(f"Processing {len(checkpoint_files)} checkpoint files: {[epoch for epoch, _ in checkpoint_files]}")
    
    # Use ThreadPoolExecutor for parallel processing
    # We limit the number of concurrent threads to avoid overwhelming the system
    max_workers = min(len(checkpoint_files), 1)  # Limit to 2 concurrent processes to reduce memory usage

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_epoch = {
            executor.submit(process_single_epoch, epoch_num, checkpoint_path, args): (epoch_num, checkpoint_path)
            for epoch_num, checkpoint_path in checkpoint_files
        }
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_epoch), total=len(checkpoint_files), desc="Processing epochs"):
            epoch_num, checkpoint_path = future_to_epoch[future]
            try:
                epoch_num, metrics = future.result()
                if metrics:  # Only store results if metrics were successfully extracted
                    all_results[epoch_num] = metrics
            except Exception as exc:
                print(f'Epoch {epoch_num} generated an exception: {exc}')
    
    # Find best epochs
    if all_results:
        best_epochs, epoch_averages, best_avg_psnr_epoch, best_avg_ssim_epoch = find_best_epochs(all_results)
        
        # Print results
        print("\n" + "="*60)
        print("BEST EPOCHS RESULTS")
        print("="*60)
        
        for dataset in best_epochs.keys():
            print(f"\nDataset: {dataset}")
            print(f"  Best PSNR: Epoch {best_epochs[dataset]['PSNR']['epoch']} "
                  f"(PSNR = {best_epochs[dataset]['PSNR']['value']:.4f})")
            print(f"  Best SSIM: Epoch {best_epochs[dataset]['SSIM']['epoch']} "
                  f"(SSIM = {best_epochs[dataset]['SSIM']['value']:.4f})")
        
        # Print average results
        print("\n" + "="*60)
        print("AVERAGE METRICS RESULTS")
        print("="*60)
        print(f"\nEpoch with highest average PSNR: {best_avg_psnr_epoch} (avg_PSNR = {epoch_averages[best_avg_psnr_epoch]['avg_PSNR']:.4f})")
        print(f"Epoch with highest average SSIM: {best_avg_ssim_epoch} (avg_SSIM = {epoch_averages[best_avg_ssim_epoch]['avg_SSIM']:.4f})")
        
        # Print detailed metrics for best average PSNR epoch
        print("\n" + "-"*40)
        print(f"Detailed metrics for epoch with highest average PSNR ({best_avg_psnr_epoch}):")
        for dataset, values in all_results[best_avg_psnr_epoch].items():
            print(f"  {dataset} - PSNR: {values['PSNR']:.4f}, SSIM: {values['SSIM']:.4f}")
            
        # Print detailed metrics for best average SSIM epoch
        print("\n" + "-"*40)
        print(f"Detailed metrics for epoch with highest average SSIM ({best_avg_ssim_epoch}):")
        for dataset, values in all_results[best_avg_ssim_epoch].items():
            print(f"  {dataset} - PSNR: {values['PSNR']:.4f}, SSIM: {values['SSIM']:.4f}")
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(args.output_dir, f'best_epochs_summary_{timestamp}.txt')
        with open(results_file, 'w') as f:
            f.write("BEST EPOCHS RESULTS\n")
            f.write("="*60 + "\n")
            for dataset in best_epochs.keys():
                f.write(f"\nDataset: {dataset}\n")
                f.write(f"  Best PSNR: Epoch {best_epochs[dataset]['PSNR']['epoch']} "
                        f"(PSNR = {best_epochs[dataset]['PSNR']['value']:.4f})\n")
                f.write(f"  Best SSIM: Epoch {best_epochs[dataset]['SSIM']['epoch']} "
                        f"(SSIM = {best_epochs[dataset]['SSIM']['value']:.4f})\n")
            
            # Write average results
            f.write("\nAVERAGE METRICS RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"\nEpoch with highest average PSNR: {best_avg_psnr_epoch} (avg_PSNR = {epoch_averages[best_avg_psnr_epoch]['avg_PSNR']:.4f})\n")
            f.write(f"Epoch with highest average SSIM: {best_avg_ssim_epoch} (avg_SSIM = {epoch_averages[best_avg_ssim_epoch]['avg_SSIM']:.4f})\n")
            
            # Write detailed metrics for best average PSNR epoch
            f.write("\n" + "-"*40 + "\n")
            f.write(f"Detailed metrics for epoch with highest average PSNR ({best_avg_psnr_epoch}):\n")
            for dataset, values in all_results[best_avg_psnr_epoch].items():
                f.write(f"  {dataset} - PSNR: {values['PSNR']:.4f}, SSIM: {values['SSIM']:.4f}\n")
                
            # Write detailed metrics for best average SSIM epoch
            f.write("\n" + "-"*40 + "\n")
            f.write(f"Detailed metrics for epoch with highest average SSIM ({best_avg_ssim_epoch}):\n")
            for dataset, values in all_results[best_avg_ssim_epoch].items():
                f.write(f"  {dataset} - PSNR: {values['PSNR']:.4f}, SSIM: {values['SSIM']:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
    else:
        print("No results to process")

if __name__ == '__main__':
    main()