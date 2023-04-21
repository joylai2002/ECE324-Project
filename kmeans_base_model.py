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

        if img is None: return None, None
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply augmentations
        img_gray = apply_augmentations(img_gray)

        # Create SIFT model and extract keypoints and descriptors
        sift = cv2.KAZE_create()
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)

        return keypoints, descriptors
    return None, None

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
    # Retrieve data
    data_dir = "data"
    class_labels = os.listdir(data_dir)

    # Initialize arrays
    X_data, Y_data, descriptors_list, accuracies, confusion_matrices = [], [], [], [], []

    with open("models\\data\\X_data.pkl", "rb") as f:
        X_data = pickle.load(f)

    with open("models\\data\\Y_data.pkl", "rb") as f:
        Y_data = pickle.load(f)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Extract descriptors from the training data
    X_train_descriptors = [descriptors for img_path, descriptors in X_train]
    X_train_descriptors_concatenated = np.vstack(X_train_descriptors)
    
    # Iterate over several k to find optimal value
    for k in [160]:
        # Train k-means with different number of clusters each time
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        print(len(X_train_descriptors_concatenated))
        print(len(X_train_descriptors_concatenated[0]))
        kmeans.fit(X_train_descriptors_concatenated)

        # Get histograms for training data
        X_train_bovw = []
        for img_path, descriptors in X_train:
            if descriptors is not None:
                histogram = create_bovw_representation(descriptors, kmeans)
                X_train_bovw.append(histogram)

        # Get histograms for training data
        X_test_bovw = []
        for img_path, descriptors in X_test:
            if descriptors is not None:
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