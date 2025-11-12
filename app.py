import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os

# Load model and class names
@st.cache(allow_output_mutation=True)
def load_model():
    if not os.path.exists("traffic_sign_model.pkl"):
        st.error("Model not found! Please run 'python train.py' first to train the model.")
        return None, None
    
    model = joblib.load("traffic_sign_model.pkl")
    class_names = joblib.load("class_names.pkl")
    return model, class_names

def preprocess_image(img):
    """Preprocess image for prediction"""
    # Convert PIL to numpy array
    img_array = np.array(img)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to 32x32
    img_resized = cv2.resize(img_array, (32, 32))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Flatten
    return gray.flatten().reshape(1, -1)

def main():
    st.set_page_config(
        page_title="Traffic Sign Classifier",
        page_icon="🚦",
        layout="centered"
    )
    
    st.title("🚦 Traffic Sign Classifier")
    st.write("Upload an image of a traffic sign to classify it using SVM")
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            # Preprocess and predict
            with st.spinner("Classifying..."):
                try:
                    processed = preprocess_image(image)
                    prediction = model.predict(processed)[0]
                    probabilities = model.predict_proba(processed)[0]
                    
                    # Display result
                    st.success(f"**{class_names[prediction]}**")
                    st.progress(float(probabilities[prediction]))
                    st.write(f"Confidence: {probabilities[prediction] * 100:.2f}%")
                    
                    # Show all probabilities
                    st.write("---")
                    st.write("**All Predictions:**")
                    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                        st.write(f"{name}: {prob * 100:.2f}%")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write("This is a simple traffic sign classifier using:")
        st.write("- **Model**: Support Vector Machine (SVM)")
        st.write("- **UI Framework**: Streamlit")
        st.write("- **Image Processing**: OpenCV")
        
        st.write("---")
        st.header("Classes")
        if class_names:
            for i, name in enumerate(class_names):
                st.write(f"{i+1}. {name}")
        
        st.write("---")
        st.info("💡 Tip: For best results, upload clear images of traffic signs")

if __name__ == "__main__":
    main()
