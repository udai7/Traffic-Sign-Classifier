## Traffic Sign Classifier — Project Documentation

### Overview

This small project demonstrates a complete pipeline for training and serving a simple traffic sign classifier. It includes:

- `train.py` — a script that creates a small synthetic dataset, trains an SVM classifier using scikit-learn, evaluates it, and saves the trained model and metadata (class names) to disk.
- `app.py` — a Streamlit-based web UI that loads the saved model and lets users upload images of traffic signs to classify them.
- `dataset/` — directory used for storing dataset images (synthetic data created by `train.py`).
- `requirements.txt` — list of Python package dependencies used by the project.

Note: You referred to `capp.py` in your request; the repository contains `app.py` which is the Streamlit application. I'll assume `app.py` is what you meant.

---

## How the system works (high-level)

The pipeline has two main parts: offline training and online prediction.

1. Training (offline)

   - `train.py` generates a small synthetic dataset (simple shapes and colors representing traffic signs).
   - Images are preprocessed to grayscale and flattened.
   - A Support Vector Machine (SVM) classifier (RBF kernel) from scikit-learn is trained.
   - The trained model (`traffic_sign_model.pkl`) and `class_names.pkl` are saved with `joblib`.

2. Serving (online)
   - `app.py` (Streamlit) loads the model files and offers a web UI.
   - Users upload images. Images are preprocessed in the same way as during training.
   - The model predicts the traffic sign class and returns the predicted label and confidence.

### Architecture diagram

The diagram below shows the main components and how data flows between them. This is provided as a Mermaid graph (supported by many Markdown renderers) and an ASCII fallback.

```mermaid
flowchart TD
  A[Dataset (dataset/)] -->|created by| B[train.py]
  B -->|preprocess & train| C[SVM Model (scikit-learn)]
  C -->|saved to disk| D[traffic_sign_model.pkl & class_names.pkl]
  D -->|loaded by| E[app.py (Streamlit UI)]
  F[User Upload Image] -->|HTTP upload| E
  E -->|preprocess| G[Image preprocessing (OpenCV)]
  G -->|feature vector| H[Model.predict]
  H -->|class + probabilities| E
  E -->|display| I[Web UI]
```

ASCII fallback:

[dataset/] -> train.py -> (preprocess -> SVM train) -> traffic_sign_model.pkl
traffic_sign_model.pkl -> app.py (Streamlit UI) <- user uploads image

---

## Files and detailed explanation

### `requirements.txt`

The project uses the following packages (as listed in `requirements.txt`):

- scikit-learn — provides SVM implementation, train/test splitting, and evaluation metrics.
- opencv-python — image reading, resizing, conversion, drawing shapes and preprocessing.
- numpy — numeric operations, arrays and random noise used to create synthetic dataset.
- streamlit — lightweight web UI framework to host the prediction interface.
- pillow — used by Streamlit to process user-uploaded images as PIL images.
- joblib — efficient serialization for scikit-learn models and Python objects.

Rationale: these libraries are mature, widely used, and well-suited for a small demonstration classifier. The stack is minimal and supports rapid prototyping.

---

### `train.py` — detailed walkthrough

Contract (inputs / outputs):

- Inputs: none required; the script creates a synthetic dataset under `dataset/` if not present.
- Outputs: `traffic_sign_model.pkl` (trained scikit-learn model) and `class_names.pkl` (list of class names). Returns the trained classifier and accuracy when run as main.
- Error modes: filesystem write errors (permission), missing OpenCV or scikit-learn packages.

Key sections of `train.py`:

1. Imports and global values

- `os`, `pathlib.Path` — filesystem and path helpers.
- `cv2` (OpenCV) — image operations (draw shapes, save images, resize, color conversions).
- `numpy` — used to generate images and noise.
- `sklearn.svm` (SVM), `train_test_split`, `accuracy_score` — model training and evaluation.
- `joblib` — save/load trained model and metadata.

The script defines `class_names` as a small list of human-readable labels for the synthetic traffic signs.

2. create_sample_dataset()

Purpose: create a synthetic dataset of small 32x32 RGB images for demonstration.

What it does:

- Ensures `dataset/` exists and then creates one subdirectory per class (0..4).
- For each class, generates 50 images with different shapes/colors per class:
  - Stop sign: a red circle (approximate octagon visually) drawn using `cv2.circle`.
  - Speed limit: white circle outline + text "50" drawn with `cv2.putText`.
  - Yield: filled triangle using `cv2.fillPoly`.
  - No entry: red circle with a white bar (rectangle) overlay.
  - Pedestrian: a simple stick figure-like representation.
- Adds random per-pixel noise to make images less perfect and saves each as PNG.
- Returns arrays (X images, y labels).

Notes / limitations:

- This dataset is synthetic and extremely simplified; real traffic sign datasets (e.g., GTSRB) are recommended for realistic performance and transfer to real-world images.
- Images are small (32x32) to keep training fast for this demo.

3. preprocess_image(img)

Purpose: convert an RGB image to a feature vector compatible with scikit-learn SVM.

Steps:

- Resize to 32x32 using `cv2.resize` to ensure consistent shape.
- Convert to grayscale (`cv2.cvtColor`) — reduces features and simplifies the classifier.
- Flatten to a 1D vector using `.flatten()`.

4. train_model()

Purpose: orchestrate dataset creation, preprocessing, training, evaluation, and serialization.

Steps:

- Calls `create_sample_dataset()` to get images and labels.
- Preprocesses every image using `preprocess_image` into a flattened grayscale vector.
- Splits data into train and test sets with `train_test_split(test_size=0.2)`.
- Instantiates an SVM classifier: `svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)`.
  - kernel='rbf' is the standard radial basis function kernel good for moderate non-linear separation.
  - `probability=True` enables `predict_proba()` in the UI for showing confidence.
- Fits the model on the training set and evaluates accuracy on the test set.
- Uses `joblib.dump` to save both the classifier and `class_names`.

Why SVM here?

- SVMs are simple, robust, and work well on small-to-medium-sized feature vectors. For this toy dataset and flattened grayscale features, an SVM is fast to train and easy to serialize.
- For larger, more complex image tasks, a convolutional neural network (e.g., with PyTorch or TensorFlow) would be a better fit. But for demo purposes, an SVM keeps complexity low.

5. Script entry

- The bottom `if __name__ == "__main__":` block simply calls `train_model()` so the file can be executed directly: `python train.py`.

Edge cases & improvements listed:

- Use a real dataset (GTSRB) and file-based dataset loader.
- Replace the raw pixel features with more robust descriptors (HOG, SIFT, SURF) or use a small CNN.
- Add model versioning and a test suite for predictions.

---

### `app.py` (Streamlit UI) — detailed walkthrough

Contract (inputs / outputs):

- Inputs: user-uploaded image files (jpg, jpeg, png, bmp)
- Outputs: displayed predicted label and confidence on a Streamlit page. No external outputs except logs.

Key sections:

1. Imports

- `streamlit` — UI elements and layout.
- `cv2`, `numpy`, `PIL.Image` — image processing.
- `joblib` — load model and class names saved by `train.py`.
- `os` — file system checks.

2. load_model() cached helper

- Decorated with `@st.cache(allow_output_mutation=True)` to avoid reloading the model on each interaction.
- It checks whether `traffic_sign_model.pkl` exists; if not, it displays an error message instructing to run `python train.py`.
- Returns `(model, class_names)` loaded from disk.

3. preprocess_image(img)

- Converts a PIL image uploaded by the user to a numpy array.
- Converts RGB->BGR because OpenCV expects BGR order.
- Resizes to 32x32 and converts to grayscale; flattens to a single-row feature vector.
- This preprocessing mirrors `train.py`'s preprocessing exactly — crucial to avoid train/serve mismatch.

4. main() Streamlit flow

- Sets page config and title.
- Loads the model via `load_model()`; if the model isn't found, `st.stop()` prevents further execution.
- Shows a file uploader; when an image is uploaded:
  - Displays the uploaded image in the left column.
  - In the right column, performs preprocessing and prediction inside a `st.spinner`.
  - Uses `predict` and `predict_proba` to obtain class index and class probabilities.
  - Uses `st.success` to display the predicted label and `st.progress` + `st.write` to visualize confidence.
  - Lists all class probabilities for transparency.
- The sidebar provides brief info about the model and lists classes.

Why Streamlit?

- Streamlit makes creating interactive UIs for ML models very fast with minimal code.
- Good for demos and prototypes; it automatically hosts controls and images and refreshes on changes.

Notes & improvements:

- The UI does minimal validation on uploaded images — you might want to check resolution, colorspace, or add cropping/centering tools.
- Consider adding batch upload support and a small dataset statistics panel.

---

## Why each package was chosen (short justification)

- scikit-learn: provides a reliable SVM implementation and utilities (train/test split, accuracy). Great for classical ML and small to medium datasets.
- opencv-python: versatile for image operations (drawing shapes for dataset, resizing, color conversion). Efficient in C and widely used.
- numpy: fundamental for numeric arrays and random noise generation.
- streamlit: rapid prototype UI for model demonstration; very little boilerplate.
- pillow: used by Streamlit and convenient for handling uploaded image file objects.
- joblib: fast and compatible with scikit-learn objects for saving/loading models.

---

## How to run (quick start)

1. Create a virtual environment and install dependencies (Windows PowerShell commands):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model (creates `dataset/`, trains, and saves the model files):

```powershell
python train.py
```

3. Run the Streamlit app (in the activated venv):

```powershell
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) and upload an image to test.

---

## Quick suggestions for production or next steps

- Use a real labeled dataset (GTSRB) and a proper data loader.
- Replace the SVM with a small CNN (PyTorch or TensorFlow) if you want better real-world accuracy.
- Add unit tests for preprocessing and a small integration test that loads the model and runs one image through it.
- Add model input validation and standardization, e.g., histogram equalization or color normalization.
- Add Dockerfile for consistent deployments and CI to run training & test steps.

---

## Appendix — important code snippets explained (quick reference)

- train.py: the `svm.SVC(..., probability=True)` flag enables probability estimates used by `predict_proba()` in `app.py`.
- app.py: `@st.cache(allow_output_mutation=True)` prevents reloading the scikit-learn model on every UI rerun; use caution with stale caches when updating the model file.

---

## Completion notes

- I used the repository files `train.py`, `app.py`, and `requirements.txt` as the source for this documentation.
- If `capp.py` is a different file you intended, please provide it and I will add a matching section.

If you'd like, I can:

- Add a small Mermaid PNG export or inline image of the diagram.
- Add a `README.md` with a one-line summary and badges.
- Create a small `tests/test_preprocess.py` to unit-test the preprocessing functions from both files.
