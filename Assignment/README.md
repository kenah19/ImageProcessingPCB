# PCB Defect Inspection Streamlit App

This directory contains the training notebook and a Streamlit demonstration app
for comparing eight PCB defect classifiers and classifying an uploaded defect
patch.

## Features

- Select any baseline or HPO-tuned model.
- View its held-out test metrics, confusion matrix, and per-class report.
- Upload a cropped PCB defect patch and see the predicted class, confidence, and
  complete probability distribution.
- Reuses the model definitions, image normalization, and class order from the
  training notebook.

## Export the checkpoints

Run `PCB Defect Inspection System.ipynb` through the final **EXPORT TRAINED
MODELS** cell. It creates eight `.pth` files and `model_manifest.json` under
`Exported_Models/`.

The AlexNet checkpoints exceed GitHub's regular 100 MiB file limit, so `.pth`
files are configured for Git LFS. Install Git LFS before adding the exported
models:

```bash
git lfs install
git add Assignment/.gitattributes Assignment/Exported_Models
```

## Run locally

From this directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate`.

## Deploy from GitHub

1. Commit and push this directory, including the Git LFS checkpoint objects, to
   the GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from the GitHub repository and branch.
4. Set the entrypoint to `Assignment/app.py`.
5. In advanced settings, select Python 3.12, then deploy.

The `requirements.txt` file is beside the entrypoint, which Streamlit Community
Cloud detects automatically. Git LFS repositories are supported by Community
Cloud.
