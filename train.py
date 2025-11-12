import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
import zipfile
from pathlib import Path

# Simple dataset - we'll create synthetic data for common traffic signs
# In a real scenario, you'd use GTSRB dataset
class_names = [
    "Stop Sign",
    "Speed Limit",
    "Yield Sign",
    "No Entry",
    "Pedestrian Crossing"
]

def create_sample_dataset():
    """Create a simple synthetic dataset for demonstration"""
    print("Creating sample dataset...")
    data_dir = Path("dataset")
    data_dir.mkdir(exist_ok=True)
    
    X = []
    y = []
    
    # Generate synthetic images for each class
    for class_id in range(len(class_names)):
        class_dir = data_dir / str(class_id)
        class_dir.mkdir(exist_ok=True)
        
        # Create 50 sample images per class
        for i in range(50):
            # Create a simple image with different patterns per class
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            
            if class_id == 0:  # Stop sign - red octagon
                cv2.circle(img, (16, 16), 12, (0, 0, 255), -1)
            elif class_id == 1:  # Speed limit - circle
                cv2.circle(img, (16, 16), 12, (255, 255, 255), 2)
                cv2.putText(img, "50", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif class_id == 2:  # Yield - triangle
                pts = np.array([[16, 5], [28, 28], [4, 28]], np.int32)
                cv2.fillPoly(img, [pts], (0, 255, 255))
            elif class_id == 3:  # No entry - red circle with white bar
                cv2.circle(img, (16, 16), 12, (0, 0, 255), -1)
                cv2.rectangle(img, (4, 14), (28, 18), (255, 255, 255), -1)
            elif class_id == 4:  # Pedestrian
                cv2.circle(img, (16, 10), 3, (255, 255, 255), -1)
                cv2.line(img, (16, 13), (16, 22), (255, 255, 255), 2)
            
            # Add some noise
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img_path = class_dir / f"{i}.png"
            cv2.imwrite(str(img_path), img)
            
            X.append(img)
            y.append(class_id)
    
    print(f"Created {len(X)} sample images across {len(class_names)} classes")
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
    
    # Create or load dataset
    X, y = create_sample_dataset()
    
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
