#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:39:05 2025

@author: sugyanikrishnadarsinee
"""
import os
import shutil
import random

# Paths to dataset2 and output folders
dataset2_path = "/Users/sugyanikrishnadarsinee/Downloads/dataset-2/test"  # Replace with actual dataset2 path
val_dataset_path = "/Users/sugyanikrishnadarsinee/Desktop/fruitdetection/finaldataset2/val"  # Replace with actual validation dataset path
test_dataset_path = "/Users/sugyanikrishnadarsinee/Desktop/fruitdetection/finaldataset2/test"  # Replace with actual test dataset path

# Ensure output directories exist
os.makedirs(val_dataset_path, exist_ok=True)
os.makedirs(test_dataset_path, exist_ok=True)

# Function to split and copy only images starting with "Screen Shot"
def split_filtered_dataset(source_folder, val_dest_folder, test_dest_folder):
    if not os.path.isdir(source_folder):
        print(f"Skipping {source_folder}, not a directory.")
        return
    
    # Get all images that start with "Screen Shot"
    images = [img for img in os.listdir(source_folder) if img.startswith("Screen Shot") and img.lower().endswith(('jpg', 'jpeg', 'png'))]

    if not images:
        print(f"Skipping {source_folder}, no images starting with 'Screen Shot' found.")
        return

    # Shuffle images to randomize selection
    random.shuffle(images)

    # Calculate 40% for validation and 60% for testing
    val_count = int(0.4 * len(images))
    
    val_images = images[:val_count]  # First 40%
    test_images = images[val_count:]  # Remaining 60%

    # Get category name and create corresponding folders
    category_name = os.path.basename(source_folder)
    val_category_folder = os.path.join(val_dest_folder, category_name)
    test_category_folder = os.path.join(test_dest_folder, category_name)
    os.makedirs(val_category_folder, exist_ok=True)
    os.makedirs(test_category_folder, exist_ok=True)

    # Copy images to validation dataset
    for img in val_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_category_folder, img))

    # Copy remaining images to test dataset
    for img in test_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(test_category_folder, img))

# Process each category in dataset2
for category in os.listdir(dataset2_path):
    category_path = os.path.join(dataset2_path, category)

    if os.path.exists(category_path):
        split_filtered_dataset(category_path, val_dataset_path, test_dataset_path)

print("Filtered Validation (40%) and Test (60%) datasets successfully created from dataset2!")
