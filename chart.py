#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 13:38:34 2025

@author: sugyanikrishnadarsinee
"""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Select random images from the validation set
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for ax in axes.flat:
    # Randomly select an index
    idx = random.randint(0, len(val_generator.filepaths) - 1)
    img_path = val_generator.filepaths[idx]
    label = val_generator.classes[idx]

    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    # Model prediction
    pred = model.predict(np.expand_dims(img_resized / 255.0, axis=0))
    predicted_label = class_labels[np.argmax(pred)]

    # Display the image
    ax.imshow(img)
    ax.set_title(f"True: {class_labels[label]}\nPred: {predicted_label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
