#!/usr/bin/env python3
"""
Face Recognition Feature Comparison: Eigenfaces vs Fisherfaces

This script creates visualizations comparing eigenfaces (PCA-based) and 
fisherfaces (LDA-based) feature extraction methods for facial recognition.
It uses the Olivetti faces dataset from scikit-learn, which is a subset 
of the AT&T Database of Faces.

The script generates three images:
1. A side-by-side comparison of top eigenfaces and fisherfaces
2. Sample original faces from the dataset for reference
3. Face reconstruction using eigenfaces

All images are saved as high-resolution PNG files suitable for presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
import os

def load_face_data():
    """Load the Olivetti faces dataset and return data and metadata."""
    print("Loading Olivetti faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    
    # Get dataset information
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    h, w = faces.images[0].shape  # Image dimensions
    
    print(f"Dataset loaded: {n_samples} samples, {n_features} features, {n_classes} individuals")
    print(f"Each face image is {h}x{w} pixels")
    
    return faces, X, y, h, w

def extract_eigenfaces(X, n_components=25):
    """Extract eigenfaces using PCA."""
    print(f"Extracting {n_components} eigenfaces...")
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to get eigenfaces
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get variance explained
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Total variance explained by {n_components} components: {cumulative_variance[-1]:.2%}")
    
    return pca, X_pca, cumulative_variance

def extract_fisherfaces(X, y, n_components_pca=None):
    """Extract fisherfaces using PCA followed by LDA."""
    print("Extracting fisherfaces...")
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # First use PCA to reduce dimensions (to avoid singularity issues)
    if n_components_pca is None:
        # By default, keep enough components to avoid singularity in LDA
        n_components_pca = min(n_samples, n_features, n_classes - 1)
    
    pca_lda = PCA(n_components=n_components_pca, whiten=True)
    X_pca_lda = pca_lda.fit_transform(X_scaled)
    
    # Then apply LDA
    n_components_lda = min(n_components_pca, n_classes - 1)  # Maximum possible LDA components
    lda = LDA(n_components=n_components_lda)
    X_lda = lda.fit_transform(X_pca_lda, y)
    
    # Get fisherfaces (transform LDA components back to image space)
    fisherfaces = np.dot(lda.scalings_.T, pca_lda.components_)
    
    print(f"Extracted {n_components_lda} fisherfaces")
    
    return lda, pca_lda, X_lda, fisherfaces

def create_comparison_image(eigenfaces, fisherfaces, h, w, n_display=5, cmap='plasma'):
    """Create a side-by-side comparison of eigenfaces and fisherfaces."""
    print("Creating comparison visualization...")
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, n_display + 1)
    
    # Plot titles
    ax_title = plt.subplot(gs[0, 0])
    ax_title.text(0.5, 0.5, "Eigenfaces\n(PCA)", fontsize=16, ha='center')
    ax_title.axis('off')
    
    ax_title2 = plt.subplot(gs[1, 0])
    ax_title2.text(0.5, 0.5, "Fisherfaces\n(LDA)", fontsize=16, ha='center')
    ax_title2.axis('off')
    
    # Plot eigenfaces
    for i in range(n_display):
        ax = plt.subplot(gs[0, i + 1])
        eigenface_img = eigenfaces[i].reshape(h, w)
        
        # Normalize for better visualization
        eigenface_img = (eigenface_img - eigenface_img.min()) / (eigenface_img.max() - eigenface_img.min())
        
        ax.imshow(eigenface_img, cmap=cmap)
        ax.set_title(f"Component {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot fisherfaces
    for i in range(min(n_display, len(fisherfaces))):
        ax = plt.subplot(gs[1, i + 1])
        fisherface_img = fisherfaces[i].reshape(h, w)
        
        # Normalize for better visualization
        fisherface_img = (fisherface_img - fisherface_img.min()) / (fisherface_img.max() - fisherface_img.min())
        
        ax.imshow(fisherface_img, cmap=cmap)
        ax.set_title(f"Component {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.suptitle("Eigenfaces vs Fisherfaces: Feature Comparison", fontsize=18, y=1.02)
    
    return fig

def show_original_faces(X, h, w, faces_indices=None, n_faces=5):
    """Display original face images for reference."""
    if faces_indices is None:
        # If no specific indices provided, choose some samples
        if len(X) >= n_faces:
            step = len(X) // n_faces
            faces_indices = list(range(0, len(X), step))[:n_faces]
        else:
            faces_indices = list(range(len(X)))
    
    fig = plt.figure(figsize=(12, 4))
    for i, face_idx in enumerate(faces_indices):
        ax = fig.add_subplot(1, len(faces_indices), i + 1)
        face_img = X[face_idx].reshape(h, w)
        ax.imshow(face_img, cmap='gray')
        ax.set_title(f"Face #{face_idx}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.suptitle("Sample Original Face Images", fontsize=16)
    
    return fig

def save_reconstruction_comparison(X, pca, h, w, sample_indices=None):
    """Create a visualization showing original faces and their reconstructions."""
    if sample_indices is None:
        # Choose some sample faces
        sample_indices = [0, 15, 30, 45, 60]
    
    n_samples = len(sample_indices)
    fig = plt.figure(figsize=(12, 6))
    
    # Standardize data as done in training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for i, idx in enumerate(sample_indices):
        # Original face
        ax1 = plt.subplot(2, n_samples, i + 1)
        original = X[idx].reshape(h, w)
        ax1.imshow(original, cmap='gray')
        ax1.set_title(f"Original #{idx}")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Reconstructed face using eigenfaces
        ax2 = plt.subplot(2, n_samples, i + 1 + n_samples)
        
        # Project to PCA space and back
        x_scaled = X_scaled[idx:idx+1]
        x_pca = pca.transform(x_scaled)
        x_reconstructed = pca.inverse_transform(x_pca)
        reconstructed = scaler.inverse_transform(x_reconstructed).reshape(h, w)
        
        ax2.imshow(reconstructed, cmap='gray')
        ax2.set_title("Reconstructed")
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    plt.suptitle("Face Reconstruction with Eigenfaces", fontsize=16, y=1.02)
    return fig

def main():
    """Main function to run the eigenfaces vs fisherfaces comparison."""
    # Create output directory if it doesn't exist
    output_dir = "face_recognition_comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    faces, X, y, h, w = load_face_data()
    
    # Extract eigenfaces
    n_components_pca = 25
    pca, X_pca, variance = extract_eigenfaces(X, n_components=n_components_pca)
    eigenfaces = pca.components_.reshape((n_components_pca, h, w))
    
    # Extract fisherfaces
    lda, pca_lda, X_lda, fisherfaces_flat = extract_fisherfaces(X, y)
    n_fisherfaces = fisherfaces_flat.shape[0]
    fisherfaces = fisherfaces_flat.reshape((n_fisherfaces, h, w))
    
    # Create comparison visualization
    n_display = 5  # Number of components to display
    cmap = 'plasma'  # Color map for better visual distinction
    
    comparison_fig = create_comparison_image(eigenfaces, fisherfaces, h, w, n_display, cmap)
    comparison_fig.savefig(os.path.join(output_dir, 'eigenfaces_vs_fisherfaces.png'), dpi=300, bbox_inches='tight')
    
    # Show original faces for reference
    original_fig = show_original_faces(X, h, w)
    original_fig.savefig(os.path.join(output_dir, 'original_faces.png'), dpi=300, bbox_inches='tight')
    
    # Show reconstruction comparison
    recon_fig = save_reconstruction_comparison(X, pca, h, w)
    recon_fig.savefig(os.path.join(output_dir, 'face_reconstruction.png'), dpi=300, bbox_inches='tight')
    
    print("\nImages saved in directory:", output_dir)
    print("1. eigenfaces_vs_fisherfaces.png - Side-by-side comparison")
    print("2. original_faces.png - Sample original faces")
    print("3. face_reconstruction.png - Face reconstruction using eigenfaces")
    print("\nThese images can be used directly in your presentation slides.")

if __name__ == "__main__":
    main()