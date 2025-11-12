# Traffic Sign Classifier

A small demo project that trains a Support Vector Machine (SVM) to classify simple traffic sign images and serves a live demo UI using Streamlit.

![App screenshot](./image.png)

## What this repository contains

- `train.py` — creates a small synthetic dataset, trains an SVM model, evaluates it, and saves the model (`traffic_sign_model.pkl`) and `class_names.pkl`.
- `app.py` — Streamlit web UI to upload an image and get a prediction from the trained model.
- `dataset/` — folder where synthetic images are created by `train.py`.
- `PROJECT_DOCUMENTATION.md` — detailed project documentation, architecture diagram, and rationale for used libraries.
- `requirements.txt` — Python dependencies required to run/train the project.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Train the model (creates `dataset/` and saves `traffic_sign_model.pkl`)

```powershell
python train.py
```

4. Run the Streamlit app

```powershell
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) to access the UI.

## Notes

- The dataset is synthetic and meant for demo purposes only. For production or research, use a curated dataset such as GTSRB and a CNN-based model.
- See `PROJECT_DOCUMENTATION.md` for a detailed explanation of the code and the reasons behind chosen libraries.

## License

This project is provided as-is for demonstration and educational use. Feel free to adapt it.
