import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

def load_class_names(csv_path="labels.csv"):
    """Load class names from labels CSV file"""
    df = pd.read_csv(csv_path)
    # Create a dictionary mapping ClassId to Name
    class_dict = dict(zip(df['ClassId'], df['Name']))
    # Return as list ordered by ClassId
    return [class_dict[i] for i in sorted(class_dict.keys())]

def load_dataset(data_dir="traffic_Data/DATA"):
    """Load dataset from traffic_Data/DATA folder structure"""
    print(f"Loading dataset from {data_dir}...")
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found!")
    
    X = []
    y = []
    
    # Get all class folders (0, 1, 2, ...)
    class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name))
    
    for class_folder in class_folders:
        class_id = int(class_folder.name)
        
        # Load all images from this class folder
        image_files = list(class_folder.glob("*.png")) + \
                     list(class_folder.glob("*.jpg")) + \
                     list(class_folder.glob("*.jpeg"))
        
        print(f"Loading class {class_id}: {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                X.append(img)
                y.append(class_id)
    
    print(f"Loaded {len(X)} images across {len(class_folders)} classes")
    return np.array(X), np.array(y)

def preprocess_image(img):
    """Preprocess image for SVM"""
    # Resize to 32x32
    img = cv2.resize(img, (32, 32))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Flatten
    return gray.flatten()

def train_model():
    """Train SVM classifier on traffic signs"""
    print("Starting training process...")
    
    # Load class names from CSV
    class_names = load_class_names("labels.csv")
    print(f"Loaded {len(class_names)} class names from labels.csv")
    
    # Load dataset from traffic_Data/DATA
    X, y = load_dataset("traffic_Data/DATA")
    
    # Preprocess all images
    print("Preprocessing images...")
    X_processed = np.array([preprocess_image(img) for img in X])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train SVM
    print("Training SVM classifier...")
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save model and class names
    print("Saving model...")
    joblib.dump(clf, "traffic_sign_model.pkl")
    joblib.dump(class_names, "class_names.pkl")
    
    print("Training complete! Model saved as 'traffic_sign_model.pkl'")
    return clf, accuracy

if __name__ == "__main__":
    train_model()
