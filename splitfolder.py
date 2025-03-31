#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:30:57 2025

@author: sugyanikrishnadarsinee
"""

import os
import shutil
import random

# Define dataset paths
dataset_path = "/Users/sugyanikrishnadarsinee/Desktop/fruitdetection/sppog"  # Change this to your actual dataset path
output_dir = "/Users/sugyanikrishnadarsinee/Desktop/fruitdetection/dataset20"  # Change this to where you want train/val/test folders

# Define split sizes
train_size = 1600
val_size = 150
test_size = 250

# Ensure output directories exist
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Process each category
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    if not os.path.isdir(category_path):
        continue  # Skip non-folder files

    # Create corresponding category folders in train, val, test
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, category), exist_ok=True)

    # List all images in category
    images = [img for img in os.listdir(category_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if len(images) < train_size + val_size + test_size:
        print(f"Warning: Not enough images in {category} to meet required splits.")

    # Shuffle images
    random.shuffle(images)

    # Split dataset
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:train_size + val_size + test_size]

    # Move images to respective directories
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    
    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Dataset successfully split into train, validation, and test folders!")
