import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import skimage
from skimage import exposure

# Set style for visualizations
sns.set_style('darkgrid')

# Function to seed for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

seed_everything()

# Function to load image paths and labels using glob
def load_image_paths(data_dir):
    image_paths = list(Path(data_dir).rglob('*.jpg'))  # Modify extensions if needed
    labels = [path.parent.name for path in image_paths]
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return df

# Function to load and preprocess dataset
def load_data(data_dir, img_size=(224, 224), batch_size=32, shuffle=True):
    data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = data_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=shuffle
    )

    valid_generator = data_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=shuffle
    )

    return train_generator, valid_generator

# Function to visualize sample images
def plot_sample_images(data_generator, class_names, n_rows=3, n_cols=3):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()
    
    # Fetch a batch of images and labels
    img_batch, label_batch = next(data_generator)
    
    for i in range(n_rows * n_cols):
        img = img_batch[i]
        label = np.argmax(label_batch[i])  # Convert one-hot encoding to class label
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to display dataset statistics
def dataset_statistics(df):
    print(f"Total Images: {len(df)}")
    print(f"Number of Classes: {df['label'].nunique()}")
    print(f"Classes: {df['label'].unique()}")

# Function to plot class distribution
def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Class Distribution')
    plt.show()

# Function to show image sizes
def analyze_image_sizes(df):
    image_sizes = [cv2.imread(str(path)).shape[:2] for path in df['image_path']]
    df['height'] = [size[0] for size in image_sizes]
    df['width'] = [size[1] for size in image_sizes]
    print(df[['height', 'width']].describe())
    
    # Plot histogram of image dimensions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['height'], kde=True, color='blue', label='Height')
    sns.histplot(df['width'], kde=True, color='red', label='Width')
    plt.legend()
    plt.title('Image Dimension Distribution')
    plt.show()

# Function to analyze color channel distributions
def analyze_color_channels(df):
    means = []
    stds = []
    for path in df['image_path']:
        img = cv2.imread(str(path))
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    
    means = np.array(means)
    stds = np.array(stds)
    
    plt.figure(figsize=(10, 6))
    plt.plot(['B', 'G', 'R'], np.mean(means, axis=0), label='Mean Intensity')
    plt.fill_between(['B', 'G', 'R'], np.mean(means, axis=0) - np.mean(stds, axis=0),
                     np.mean(means, axis=0) + np.mean(stds, axis=0), alpha=0.2, color='gray')
    plt.title('Color Channel Mean and Standard Deviation')
    plt.legend()
    plt.show()

# Function to perform PCA for feature correlation analysis
def pca_analysis(train_generator, n_components=2):
    batch = next(train_generator)
    images = batch[0]
    images_flat = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images_flat)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.argmax(batch[1], axis=1), cmap='viridis', alpha=0.6)
    plt.title('PCA Visualization of Image Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Function to visualize augmented images
def plot_augmented_images(data_generator, class_names, n_rows=3, n_cols=3):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()
    
    img_batch, label_batch = next(data_generator)
    for i in range(n_rows * n_cols):
        img = img_batch[i]
        label = np.argmax(label_batch[i])
        axes[i].imshow(img)
        axes[i].set_title(f'Augmented Label: {class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# EDA function that wraps all the steps
def perform_eda(data_dir):
    print(f"Performing EDA on dataset: {data_dir}")
    
    # Load image paths and labels into DataFrame
    df = load_image_paths(data_dir)
    
    # Display dataset statistics
    dataset_statistics(df)
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Analyze and display image sizes
    analyze_image_sizes(df)
    
    # Analyze color channels
    analyze_color_channels(df)
    
    # Load data generators for sample visualization
    train_gen, valid_gen = load_data(data_dir)
    class_names = list(train_gen.class_indices.keys())

    # Display random sample images from the training data
    plot_sample_images(train_gen, class_names)
    
    # PCA analysis for feature correlation
    pca_analysis(train_gen)
    
    # Visualize augmented images
    augmented_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ).flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    plot_augmented_images(augmented_data_gen, class_names)

# Path to your dataset directory
data_dir = '/kaggle/working/Birds/train/'

# Perform EDA
perform_eda(data_dir)