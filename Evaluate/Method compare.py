# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# This script compares different image fusion methods by calculating various evaluation metrics.

import numpy as np
import math
import os
import skimage
from PIL import Image
import pandas as pd
import cv2
import glob
import re
from natsort import natsorted
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from VIF import vifp_mscale
from MI import MI_function
from niqe import niqe
from sklearn.metrics.pairwise import cosine_similarity
from simple_metric import *  # Import all evaluation metric functions
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare different image fusion methods by calculating evaluation metrics')
    parser.add_argument('--base_path', type=str, default=r'/home/ot/Students/xxz/datasets/results_of_different_methods',
                        help='Base path containing the result images for different methods')
    parser.add_argument('--ground_truth_path', type=str, default=r'/home/ot/Students/xxz/datasets/results_of_different_methods/Ground Truth',
                        help='Path to the ground truth images')
    parser.add_argument('--methods', type=str, nargs='+', default=['StackMFF V5'],
                        help='List of methods to compare')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['FlyingThings3D','Middlebury','Mobile Depth','Road-MF'],
                        help='List of datasets to evaluate')
    parser.add_argument('--metrics', type=str, nargs='+', default=['SSIM','PSNR'],
                        help='List of metrics to calculate')
    parser.add_argument('--enable_registration', action='store_true',
                        help='Enable image registration')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save output results')
    
    return parser.parse_args()

# Create output directory
def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def align_images(img_result, img_truth):
    """
    Align images so that img_result aligns with img_truth
    Using ORB feature detection and matching, then affine transformation
    
    Args:
        img_result: Image to be registered
        img_truth: Reference image (Ground Truth)
        
    Returns:
        aligned_img: Registered image
    """
    # Convert to color image (if needed)
    if len(img_result.shape) == 2:
        img_result_color = cv2.cvtColor(img_result, cv2.COLOR_GRAY2BGR)
    else:
        img_result_color = img_result
        
    if len(img_truth.shape) == 2:
        img_truth_color = cv2.cvtColor(img_truth, cv2.COLOR_GRAY2BGR)
    else:
        img_truth_color = img_truth
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img_result_color, None)
    kp2, des2 = orb.detectAndCompute(img_truth_color, None)
    
    # If not enough keypoints are found, return the original image
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return img_result
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # If too few matching points, return the original image
    if len(matches) < 4:
        return img_result
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract coordinates of matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calculate homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # If transformation matrix is found, perform perspective transformation
    if M is not None:
        aligned_img = cv2.warpPerspective(img_result, M, (img_truth.shape[1], img_truth.shape[0]))
        return aligned_img
    else:
        return img_result

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
        img = cv2.imread(base_path, 0)
        if img is not None:
            return img
    
    # Try various supported extensions
    for fmt in supported_formats:
        path_with_ext = f"{base_path}.{fmt}"
        if os.path.exists(path_with_ext):
            img = cv2.imread(path_with_ext, 0)
            if img is not None:
                return img
    
    # Try uppercase extensions
    for fmt in supported_formats:
        path_with_ext = f"{base_path}.{fmt.upper()}"
        if os.path.exists(path_with_ext):
            img = cv2.imread(path_with_ext, 0)
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

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Create a dictionary to store results for all datasets and methods
    # Structure: {method: {dataset: {metric: value}}}
    all_results = {}
    
    # Get ground truth image paths
    for dataset in args.datasets:
        print('Testing:', dataset)
        
        # Get ground truth image dictionary {filename: full_path}
        gt_images_dict = get_images_dict(os.path.join(args.ground_truth_path, dataset))
        ground_truth_names = list(gt_images_dict.keys())
        
        if not ground_truth_names:
            print(f"Warning: No ground truth images found in {os.path.join(args.ground_truth_path, dataset)}")
            continue
            
        print(f"Found {len(ground_truth_names)} ground truth images")
        
        method_results = {}
        df = pd.DataFrame(columns=(['Method'] + args.metrics))
        
        # Iterate through methods
        for method_index, method in enumerate(args.methods):
            # Get image dictionary for this method's results
            method_images_dict = get_images_dict(os.path.join(args.base_path, method, dataset))
            
            if not method_images_dict:
                print(f"Warning: No result images found for method {method} in dataset {dataset}")
                continue
                
            print(f"Found {len(method_images_dict)} result images for method {method}")
            
            metric_results = {}
            # Initialize metric dictionary
            metric_dict = {}
            for metric in args.metrics:
                metric_dict[metric] = 0  # Initialize to 0
            
            # Record number of successfully processed images
            valid_image_count = 0
            
            # Iterate through ground truth images
            for img_name in tqdm(ground_truth_names):
                # Get ground truth image path
                gt_img_path = gt_images_dict[img_name]
                img_truth = cv2.imread(gt_img_path, 0)
                
                if img_truth is None:
                    print(f"Warning: Could not read ground truth image {gt_img_path}")
                    continue
                
                # Construct base path for result image (without extension)
                result_base_path = os.path.join(args.base_path, method, dataset, img_name)
                
                # Try to read result image
                img_result = read_image_flexible(result_base_path)
                
                # Skip if result image cannot be read
                if img_result is None:
                    print(f"Warning: Could not read result image for {img_name} in method {method}")
                    continue
                
                # If registration is enabled and image sizes don't match, perform registration
                if args.enable_registration and img_result.shape != img_truth.shape:
                    img_result = align_images(img_result, img_truth)
                
                # Skip if image sizes still don't match
                if img_result.shape != img_truth.shape:
                    print(f"Warning: Image sizes still don't match after alignment for {img_name} in method {method}")
                    continue
                
                # Increase valid image count
                valid_image_count += 1
                
                # Calculate all metrics
                for metric_index, metric in enumerate(args.metrics):
                    try:
                        if metric == 'SF':
                            value = SF_function(img_result)
                        elif metric == 'AVG':
                            value = AG_function(img_result)
                        elif metric == 'EN':
                            value = EN_function(img_result)
                        elif metric == 'STD':
                            value = SD_function(img_result)
                        elif metric == 'MSE':
                            value = MSE_function(img_result, img_truth)
                        elif metric == 'MAE':
                            value = MAE_function(img_result, img_truth)
                        elif metric == 'RMSE':
                            value = RMSE_function(img_result, img_truth)
                        elif metric == 'logRMS':
                            value = logRMS_function(img_result, img_truth)
                        elif metric == 'abs_rel_error':
                            value = abs_rel_error_function(img_result, img_truth)
                        elif metric == 'sqr_rel_error':
                            value = sqr_rel_error_function(img_result, img_truth)
                        elif metric == 'SSIM':
                            value = compare_ssim(img_truth, img_result, multichannel=False)
                        elif metric == 'PSNR':
                            value = compare_psnr(img_truth, img_result)
                        elif metric == "VIF":
                            value = vifp_mscale(img_truth, img_result)
                        elif metric == "MI":
                            value = MI_function(img_truth, img_result)
                        elif metric == 'mean_diff':
                            value = mean_diff(img_truth, img_result)
                        elif metric == "NIQE":
                            value = niqe(img_result)
                        elif metric == "CSG":
                            def CSG_function(img1, img2):
                                gradient_x_1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
                                gradient_y_1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
                                gradient_direction_1 = np.arctan2(gradient_y_1, gradient_x_1)
                                gradient_direction_normalized_1 = cv2.normalize(gradient_direction_1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                                gradient_x_2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
                                gradient_y_2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
                                gradient_direction_2 = np.arctan2(gradient_y_2, gradient_x_2)
                                gradient_direction_normalized_2 = cv2.normalize(gradient_direction_2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                                result = cosine_similarity(gradient_direction_normalized_1.reshape(1, -1), gradient_direction_normalized_2.reshape(1, -1))[0, 0]
                                return result
                            value = CSG_function(img_result, img_truth)
                        
                        metric_dict[metric] += value
                    except Exception as e:
                        print(f"Error calculating {metric} for {img_name}: {str(e)}")
                        # Set value to 0 on error
                        metric_dict[metric] += 0

            # Calculate averages using the actual number of successfully processed images
            if valid_image_count > 0:
                for metric in metric_dict:
                    metric_dict[metric] /= valid_image_count
                    metric_dict[metric] = round(metric_dict[metric], 4)
            else:
                print(f"Warning: No valid images processed for method {method}")

            # Store results in the all_results dictionary
            if method not in all_results:
                all_results[method] = {}
            all_results[method][dataset] = metric_dict
            
            print(f"{method} on {dataset}: {metric_dict}")
    
    # Create multi-level column DataFrame
    # Format: Method | Dataset1-Metric1 | Dataset1-Metric2 | Dataset2-Metric1 | ...
    
    if not all_results:
        print("No results to save.")
        return
    
    # Create multi-level columns
    column_tuples = [('', 'Method')]
    for dataset in args.datasets:
        for metric in args.metrics:
            column_tuples.append((dataset, metric))
    
    multi_columns = pd.MultiIndex.from_tuples(column_tuples)
    
    # Create DataFrame with multi-level columns
    rows = []
    for method in args.methods:
        row = [method]
        for dataset in args.datasets:
            for metric in args.metrics:
                if method in all_results and dataset in all_results[method]:
                    value = all_results[method][dataset].get(metric, '')
                    row.append(value)
                else:
                    row.append('')
        rows.append(row)
    
    df_final = pd.DataFrame(rows, columns=multi_columns)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'compare_result_{timestamp}.xlsx'
    output_filepath = os.path.join(args.output_dir, output_filename)
    
    # Save to Excel
    df_final.to_excel(output_filepath, index=True)
    print(f"\nResults saved to {output_filepath}")
    
    # Print summary results
    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF ALL DATASETS")
    print("=" * 60)
    print(df_final.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("END OF EVALUATION")
    print("=" * 60)

if __name__ == '__main__':
    main()