import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch

class PSO:
    def __init__(self, n_particles, n_features, n_clusters, bounds, w, c1, c2, max_iter, stop_after_no_improvement=10, device='cuda'):
        self.n_particles = n_particles
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.stop_after_no_improvement = stop_after_no_improvement
        self.device = device

    def init_particles(self):
        particles = []
        for _ in range(self.n_particles):
            particle = torch.FloatTensor(self.n_clusters, self.n_features).uniform_(float(self.bounds[0].min()), float(self.bounds[1].max())).to(self.device).half()
            particles.append(particle)
        return particles

    def get_fitness(self, particle, data):
        batch_size = 1048576 // 4
        distances = torch.zeros(particle.shape[0], data.shape[0], device=self.device)
        for i in range(0, particle.shape[0], batch_size):
            particle_batch = particle[i:i + batch_size]
            distances[i:i + batch_size] = torch.cdist(particle_batch, data)
        min_distances = torch.min(distances, axis=-1)[0]
        return torch.sum(min_distances)

    def optimize(self, data):
        data_gpu = torch.tensor(data, dtype=torch.float16, device=self.device).half()
        particles = self.init_particles()
        velocities = torch.zeros((self.n_particles, self.n_clusters, self.n_features), device=self.device)
        pbest_positions = [particle.clone() for particle in particles]
        pbest_fitness = [float('inf') for _ in range(self.n_particles)]
        gbest_position = None
        gbest_fitness = float('inf')

        no_improvement_counter = 0

        for i in range(self.max_iter):
            fitness_values = torch.stack([self.get_fitness(particle, data_gpu) for particle in particles])

            pbest_update = fitness_values < torch.tensor(pbest_fitness, device=self.device)
            for idx, update in enumerate(pbest_update):
                if update:
                    pbest_positions[idx] = particles[idx]
                    pbest_fitness[idx] = fitness_values[idx]

            gbest_update = torch.argmin(fitness_values)
            if fitness_values[gbest_update] < gbest_fitness:
                gbest_fitness = fitness_values[gbest_update]
                gbest_position = particles[gbest_update].clone()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.stop_after_no_improvement:
                break

            for i in range(self.n_particles):
                r1 = torch.rand((self.n_clusters, self.n_features), device=self.device)
                r2 = torch.rand((self.n_clusters, self.n_features), device=self.device)

                cognitive_velocity = self.c1 * r1 * (pbest_positions[i] - particles[i])
                social_velocity = self.c2 * r2 * (gbest_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity
                particles[i] += velocities[i]

        return gbest_position

def pso_clustering(descriptors, n_clusters, max_iter, lb, ub, swarm_size=200, omega=0.5, phip=1.5, phig=1.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pso = PSO(n_particles=swarm_size, n_features=descriptors.shape[1], n_clusters=n_clusters, bounds=(lb, ub), w=omega, c1=phip, c2=phig, max_iter=max_iter, device=device)
    
    centroids = pso.optimize(descriptors)
    centroids = centroids.cpu().numpy()

    return centroids

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
        if img is None:
            return None, None
    
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply augmentations
        img_gray = apply_augmentations(img_gray)

        # Create SIFT model and extract keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)

        return keypoints, descriptors
    return None, None

def create_bovw_representation(descriptors, cluster_centers):
    # Create a histogram representation of visual vocabulary
    histogram = np.zeros(len(cluster_centers))
    for descriptor in descriptors:
        idx = np.argmin(np.linalg.norm(descriptor - cluster_centers, axis=1))
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
    X_data, Y_data, descriptors_list, accuracies, confusion_matrices = [], [], [], [], []

    with open("models\\data\\X_data_KAZE.pkl", "rb") as f:
        X_data = pickle.load(f)

    with open("models\\data\\Y_data_KAZE.pkl", "rb") as f:
        Y_data = pickle.load(f)


    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Extract descriptors from the training data
    X_train_descriptors = [descriptors for img_path, descriptors in X_train]
    X_train_descriptors_concatenated = np.vstack(X_train_descriptors)

    # Iterate over several k to find optimal value
    for k in [160]:
        max_iter = 200
        lb = np.tile(np.min(X_train_descriptors_concatenated, axis=0), k)
        ub = np.tile(np.max(X_train_descriptors_concatenated, axis=0), k)
        cluster_centers = pso_clustering(X_train_descriptors_concatenated, k, max_iter, lb, ub)

        # Get histograms for training data
        X_train_bovw = []
        for img_path, descriptors in X_train:
            if descriptors is not None:
                histogram = create_bovw_representation(descriptors, cluster_centers)
                X_train_bovw.append(histogram)

        # Get histograms for training data
        X_test_bovw = []
        for img_path, descriptors in X_test:
            if descriptors is not None:
                histogram = create_bovw_representation(descriptors, cluster_centers)
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
        with open(f"models/pso_{k}/pso_model_{k}.pkl", "wb") as f: pickle.dump(cluster_centers, f)
        with open(f"models/pso_{k}/svm_model_{k}.pkl", "wb") as f: pickle.dump(svm, f)
        with open(f"models/pso_{k}/scaler_{k}.pkl", "wb") as f: pickle.dump(scaler, f)

    print(accuracies, confusion_matrices)