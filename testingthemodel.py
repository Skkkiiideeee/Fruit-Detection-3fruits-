#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:12:04 2025

@author: sugyanikrishnadarsinee
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("fruit_classification_model.h5")

# Extract class labels and clean them (remove unwanted suffixes)
original_labels = list(train_generator.class_indices.keys())
cleaned_labels = [label.replace("_done", "").replace("_fresh", "") for label in original_labels]

# Path to the test image (ensure path is correct)
image_path = "/Users/sugyanikrishnadarsinee/Desktop/pomegranatefrominternet.png"

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))  # Resize image
img_array = image.img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Normalize

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)  # Get the index of the highest probability
confidence = max(prediction[0]) * 100  # Convert to percentage

# Check if confidence is below 70%
if confidence < 70:
    predicted_label = "Unknown Fruit" + " closest to: " + cleaned_labels[predicted_class]
else:
    predicted_label = cleaned_labels[predicted_class]  # Use cleaned labels

# Display the image with prediction
plt.imshow(img)
plt.axis("off")
plt.title(f"Predicted: {predicted_label}")
plt.show()

# Print probability distribution
print("\nPrediction Probabilities:")
for label, prob in zip(cleaned_labels, prediction[0]):
    print(f"{label}: {prob:.4f}")

# Print final predicted class
print(f"\nFinal Prediction: {predicted_label} (Confidence: {confidence:.2f}%)")
