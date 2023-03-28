from model import *

def load_models():
    with open("models/k_160/kmeans_model_160.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("models/k_160/svm_model_160.pkl", "rb") as f:
        svm = pickle.load(f)

    with open("models/k_160/scaler_160.pkl", "rb") as f:
        scaler = pickle.load(f)

    return kmeans, svm, scaler


kmeans, svm, scaler = load_models()
img_path = "/Users/cameronsmith/Desktop/BOVW/data/Greek Revival architecture/4155_800px-Murphy_House_01.jpg"
predicted_class = classify_and_visualize(img_path, kmeans, svm, scaler)
print(f"The predicted architectural style is: {predicted_class}")