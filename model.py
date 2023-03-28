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

def extract_features(img_path):
    # Ignore .webp images
    if '.jpg' in img_path or '.jpeg' in img_path:
        # Read image and make it grayscale
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply augmentations
        img_gray = apply_augmentations(img_gray)

        # Create SIFT model and extract keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)

        # Filter keypoints by SIFT response
        min_response = np.percentile([kp.response for kp in keypoints], 75)
        keypoints = [kp for kp in keypoints if kp.response > min_response]

        # Filter keypoints by size
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)
        keypoints = keypoints[:len(keypoints)//6]
        top_keypoints = [keypoints.index(kp) for kp in keypoints]

        # Get descriptors corresponding to filtered keypoints
        descriptors = descriptors[top_keypoints]

        return keypoints, descriptors
    return None, None

def create_bovw_representation(descriptors, kmeans):
    # Create a histogram representation of visual vocabulary
    histogram = np.zeros(len(kmeans.cluster_centers_))
    for descriptor in descriptors:
        idx = kmeans.predict([descriptor])
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
    # Retrieve data
    data_dir = "/Users/cameronsmith/Desktop/BOVW/data"
    class_labels = os.listdir(data_dir)

    # Initialize arrays
    X_data, Y_data, descriptors_list, accuracies, confusion_matrices = [], [], [], [], []

    # Iterate though classes
    for label in class_labels:
        class_folder = os.path.join(data_dir, label)
        for img_name in os.listdir(class_folder):
            # Read image and extract keypoints and descriptors
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = extract_features(img_path)
            # Split data into X and Y values and add descriptors to list for kmeans
            if descriptors is not None:
                X_data.append(img_path)
                Y_data.append(label)
                descriptors_list.extend(descriptors)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Iterate over several k to find optimal value
    for k in [2, 4, 8, 16, 32, 64, 96, 128, 160, 192]:
        # Train k-means with different number of clusters each time
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(descriptors_list)

        # Get histograms for training data
        X_train_bovw = []
        for descriptors in X_train:
            histogram = create_bovw_representation(descriptors, kmeans)
            X_train_bovw.append(histogram)

        # Get histograms for training data
        X_test_bovw = []
        for descriptors in X_test:
            histogram = create_bovw_representation(descriptors, kmeans)
            X_test_bovw.append(histogram)

        # Scale data to zero mean and unit variance        
        scaler = StandardScaler()
        X_train_bovw = scaler.fit_transform(X_train_bovw)
        X_test_bovw = scaler.transform(X_test_bovw)

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
        with open(f"models/k_{k}/kmeans_model_{k}.pkl", "wb") as f: pickle.dump(kmeans, f)
        with open(f"models/k_{k}/svm_model_{k}.pkl", "wb") as f: pickle.dump(svm, f)
        with open(f"models/k_{k}/scaler_{k}.pkl", "wb") as f: pickle.dump(scaler, f)

    print(accuracies, confusion_matrices)