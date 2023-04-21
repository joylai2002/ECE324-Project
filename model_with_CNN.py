import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

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

def extract_features(img_path, model):
    # Ignore .webp images
    if '.jpg' in img_path or '.jpeg' in img_path:
        # Read image
        img = cv2.imread(img_path)

        if img is None: return None
        
        # Convert to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        # Apply transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_transformed = transform(img_resized)

        img_transformed = img_transformed.unsqueeze(0).to(device)

        with torch.no_grad(): features = model(img_transformed).view(1, -1).squeeze()

        return features.cpu().numpy()
    return None

def create_bovw_representation(descriptors, kmeans):
    # Create a histogram representation of visual vocabulary
    histogram = np.zeros(len(kmeans.cluster_centers_))
    for descriptor in descriptors:
        idx = kmeans.predict(descriptor.reshape(1, -1))
        histogram[idx] += 1
    return histogram

def classify_and_visualize(img_path, kmeans, svm, scaler):
    # Read image and extract keypoints and descriptors
    img = cv2.imread(img_path)
    keypoints, descriptors = extract_features(img_path)

    # Create histogram representation and pass it into SVM to predict class
    histogram = create_bovw_representation(descriptors, kmeans)
    histogram_scaled = scaler.transform([histogram])
    predicted_class = svm.predict(histogram_scaled)[0]

    # # Filter keypoints by SIFT response
    min_response = np.percentile([kp.response for kp in keypoints], 75)
    keypoints = [kp for kp in keypoints if kp.response > min_response]

    # # Filter keypoints by size
    keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)
    keypoints = keypoints[:len(keypoints)//6]
    top_keypoints = [keypoints.index(kp) for kp in keypoints]

    # Get descriptors corresponding to filtered keypoints
    descriptors = descriptors[top_keypoints]

    # Draw keypoints onto image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Plot image with keypoints on subplot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis("off")

    # Plot histogram on other subplot
    plt.subplot(1, 2, 2)
    plt.bar(range(len(histogram)), histogram)
    plt.title("Visual words histogram")
    plt.xlabel("Visual word index")
    plt.ylabel("Frequency")
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predicted_class

if __name__ == '__main__':
    # Set the device to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet50 model and apply average pooling
    base_model = models.resnet50(pretrained=True)
    model = base_model.to(device)

    # Retrieve data
    data_dir = "data"
    class_labels = os.listdir(data_dir)

    # Initialize arrays
    X_data, Y_data, descriptors_list, accuracies, confusion_matrices = [], [], [], [], []

    # Iterate though classes
    for label in class_labels:
        class_folder = os.path.join(data_dir, label)
        for img_name in os.listdir(class_folder):
            # Read image and extract features
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            features = extract_features(img_path, model)
            # Split data into X and Y values and add descriptors to list for kmeans
            if features is not None:
                X_data.append(features)
                Y_data.append(label)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)


    # Scale data to zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Predict classifications with support vector machine
    svm = SVC(kernel="linear", C=1, probability=True)
    svm.fit(X_train_scaled, Y_train)

    # Compute accuracies and confusion matrices
    Y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    print(accuracy, conf_matrix)