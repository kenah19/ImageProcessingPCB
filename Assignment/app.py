"""Streamlit demonstration app for PCB defect classification."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model_definitions import load_exported_model


APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "Benchmark_Results"
MODELS_DIR = APP_DIR / "Exported_Models"
SUMMARY_PATH = RESULTS_DIR / "final_benchmark_summary.csv"
DEFAULT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).replace(" ", "_")


def result_path(value: str) -> Path:
    """Resolve Windows-style paths from the benchmark CSV on any platform."""
    normalized = value.replace("\\", "/")
    filename = Path(normalized).name
    return RESULTS_DIR / filename


def preprocess_image(image: Image.Image, size: tuple[int, int] = (224, 224)):
    """Apply the PCB enhancement, letterboxing, and training normalization."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    processed = ImageOps.pad(
        Image.fromarray(enhanced_rgb),
        size,
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
    )
    tensor = transforms.ToTensor()(processed)
    return processed, NORMALIZE(tensor).unsqueeze(0)


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)


@st.cache_resource(show_spinner="Loading trained model…")
def get_model(checkpoint_path: str):
    return load_exported_model(checkpoint_path)


st.set_page_config(
    page_title="PCB Defect Inspection",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
    [data-testid="stMetric"] {background: #f5f8f7; border: 1px solid #dbe5e2;
        padding: 0.8rem 1rem; border-radius: 0.75rem;}
    .eyebrow {color: #147d64; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; font-size: .78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Image Processing · Demonstration</div>', unsafe_allow_html=True)
st.title("PCB Defect Inspection System")
st.caption(
    "Compare trained classifiers, inspect their held-out test performance, "
    "and classify a cropped PCB defect image."
)

if not SUMMARY_PATH.exists():
    st.error(f"Benchmark summary is missing: {SUMMARY_PATH}")
    st.stop()

results = load_results()
model_names = results["Model Name"].tolist()

with st.sidebar:
    st.header("Model selection")
    selected_name = st.selectbox("Choose a trained model", model_names)
    st.caption("Test metrics are read from the untouched test-set benchmark.")

selected = results.loc[results["Model Name"] == selected_name].iloc[0]
checkpoint_path = MODELS_DIR / f"{safe_filename(selected_name)}.pth"

st.subheader(selected_name)
metric_columns = st.columns(5)
metric_columns[0].metric("Test accuracy", f'{selected["Accuracy (%)"]:.2f}%')
metric_columns[1].metric("Macro mAP", f'{selected["mAP (Academic)"]:.4f}')
metric_columns[2].metric("Macro F1", f'{selected["F1-Score (Macro, %)"]:.2f}%')
metric_columns[3].metric("Mean latency", f'{selected["Latency Mean (ms/image)"]:.3f} ms')
metric_columns[4].metric("Parameters", f'{selected["Params (M)"]:.2f} M')

results_tab, classify_tab = st.tabs(["Test results", "Classify an image"])

with results_tab:
    left, right = st.columns([1.05, 0.95], gap="large")
    confusion_path = result_path(str(selected["Confusion Matrix"]))
    report_path = result_path(str(selected["Classification Report"]))

    with left:
        st.markdown("#### Confusion matrix")
        if confusion_path.exists():
            st.image(str(confusion_path), use_column_width=True)
        else:
            st.warning(f"Confusion matrix not found: {confusion_path.name}")

    with right:
        st.markdown("#### Per-class report")
        if report_path.exists():
            report = pd.read_csv(report_path, index_col=0)
            st.dataframe(report.round(4), use_container_width=True)
        else:
            st.warning(f"Classification report not found: {report_path.name}")

        st.markdown("#### Complete benchmark row")
        display_columns = [
            "GFLOPs",
            "Latency Median (ms/image)",
            "Recall (Weighted, %)",
            "F1-Score (Weighted, %)",
        ]
        st.dataframe(
            selected[display_columns].rename("Value").to_frame(),
            use_container_width=True,
        )

with classify_tab:
    st.info(
        "Upload a cropped defect patch containing one defect. The classifier was "
        "trained on 224 × 224 defect patches, not complete PCB-board photographs."
    )
    uploaded_file = st.file_uploader(
        "Test image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, PNG, and BMP.",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            processed_image, input_tensor = preprocess_image(uploaded_image)
        except Exception as error:
            st.error(f"The uploaded file could not be read as an image: {error}")
            st.stop()

        preview_col, prediction_col = st.columns([0.85, 1.15], gap="large")
        with preview_col:
            st.markdown("#### Input preview")
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
            with st.expander("Show model-ready image"):
                st.image(
                    processed_image,
                    caption="Bilateral filter + CLAHE + 224 × 224 letterbox",
                    use_column_width=True,
                )

        with prediction_col:
            st.markdown("#### Prediction")
            if not checkpoint_path.exists():
                st.warning(
                    "This checkpoint has not been exported yet. Run the notebook's "
                    "‘EXPORT TRAINED MODELS’ cell, then add Exported_Models to Git LFS."
                )
            else:
                try:
                    model, checkpoint = get_model(str(checkpoint_path))
                    with torch.inference_mode():
                        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
                    class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
                    predicted_index = int(probabilities.argmax().item())
                    predicted_class = str(class_names[predicted_index])
                    confidence = float(probabilities[predicted_index].item())

                    st.success(f"Predicted defect: **{predicted_class.replace('_', ' ')}**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                    probability_table = pd.DataFrame(
                        {
                            "Defect class": [name.replace("_", " ") for name in class_names],
                            "Probability": probabilities.numpy(),
                        }
                    ).sort_values("Probability", ascending=False)
                    st.bar_chart(
                        probability_table.set_index("Defect class"),
                        horizontal=True,
                    )
                    st.dataframe(
                        probability_table.style.format({"Probability": "{:.2%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as error:
                    st.error(f"Model loading or inference failed: {error}")

st.divider()
st.caption(
    "Predictions are an educational demonstration of the trained experiment "
    "models and should be confirmed through the normal inspection workflow."
)
