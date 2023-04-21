import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from train_VAE import VanillaVAE, CustomDataset

def apply_augmentations(img):
    # Apply cutout of size 32x32 in random location
    x = np.random.randint(0, img.shape[1] - 32)
    y = np.random.randint(0, img.shape[0] - 32)
    img[y:y+32, x:x+32] = 0

    # Apply horizontal flip with probability of 50%
    if np.random.rand() < 0.5: img = cv2.flip(img, 1)

    # Apply a rotation between -15 to 15 degrees, with uniform probability
    rotation = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), np.random.uniform(-15, 15), 1)
    img = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
    
    # Normalize image
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return img

def create_bovw_representation(latent_space, kmeans):
    # Create a histogram representation of visual vocabulary
    histogram = np.zeros(len(kmeans.cluster_centers_))
    idx = kmeans.predict([latent_space])
    histogram[idx] += 1
    return histogram

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VanillaVAE(in_channels=3, latent_dim=64).to(device)
    model.load_state_dict(torch.load("models\\trained_autoencoder_model.pt"))
    model.eval()

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = 'data'
    batch_size = 32

    dataset = CustomDataset(root_dir, transform=transform, train=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_data, Y_data, accuracies, confusion_matrices = [], [], [], []
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        # Extract latent space
        with torch.no_grad():
            mu, _ = model.encode(data)
            latent_space = mu.squeeze().cpu().numpy()
            if latent_space is not None:
                X_data.extend(latent_space)
                Y_data.extend(label)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Iterate over several k to find optimal value
    for k in [160]:
        # Train k-means with different number of clusters each time
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_data)

        # Get histograms for training data
        X_train_bovw = []
        for latent_space in X_train:
            if latent_space is not None:
                histogram = create_bovw_representation(latent_space, kmeans)
                X_train_bovw.append(histogram)

        # Get histograms for test data
        X_test_bovw = []
        for latent_space in X_test:
            if latent_space is not None:
                histogram = create_bovw_representation(latent_space, kmeans)
                X_test_bovw.append(histogram)

        # Predict classifications with support vector machine
        svm = SVC(kernel="linear", C=1, probability=True)
        svm.fit(X_train_bovw, Y_train)

        # Compute accuracies and confusion matrices
        Y_pred = svm.predict(X_test_bovw)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        confusion_matrices.append(conf_matrix)

        # Save models
        with open(f"models\\k_{k}\\kmeans_model_{k}.pkl", "wb") as f: pickle.dump(kmeans, f)
        with open(f"models\\k_{k}\\svm_model_{k}.pkl", "wb") as f: pickle.dump(svm, f)
        
    print(accuracies, confusion_matrices)