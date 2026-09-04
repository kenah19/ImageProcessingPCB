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
from streamlit_drawable_canvas import st_canvas

from model_definitions import load_exported_model


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

RESULTS_DIR = APP_DIR / "Benchmark_Results"
MODELS_DIR = APP_DIR / "Exported_Models"
BENCHMARK_PCB_DIR = APP_DIR / "Benchmark_PCB"

SUMMARY_PATH = RESULTS_DIR / "final_benchmark_summary.csv"


# ============================================================
# MODEL CLASSES
# ============================================================

DEFAULT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def safe_filename(name: str) -> str:
    """Convert model name into a safe filename."""

    return re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name,
    ).replace(" ", "_")


def result_path(value: str) -> Path:
    """
    Resolve Windows-style paths stored inside the benchmark CSV
    so they work on Linux/Streamlit Cloud as well.
    """

    normalized = value.replace("\\", "/")
    filename = Path(normalized).name

    return RESULTS_DIR / filename


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image,
    size: tuple[int, int] = (224, 224),
):
    """
    Apply the same PCB enhancement and normalization used during
    model training.

    Processing:
        1. Convert image to RGB
        2. Bilateral filtering
        3. LAB conversion
        4. CLAHE enhancement
        5. Resize with letterbox
        6. Convert to tensor
        7. ImageNet normalization
    """

    rgb = np.asarray(
        image.convert("RGB")
    )

    # RGB -> BGR
    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Bilateral filtering
    filtered = cv2.bilateralFilter(
        bgr,
        d=9,
        sigmaColor=75,
        sigmaSpace=75,
    )

    # Convert into LAB colour space
    lab = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2LAB,
    )

    lightness, channel_a, channel_b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_lightness = clahe.apply(lightness)

    enhanced = cv2.merge(
        (
            enhanced_lightness,
            channel_a,
            channel_b,
        )
    )

    # LAB -> RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2RGB,
    )

    # Resize into 224 x 224 while maintaining aspect ratio
    processed = ImageOps.pad(
        Image.fromarray(enhanced_rgb),
        size,
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
    )

    tensor = transforms.ToTensor()(processed)

    tensor = NORMALIZE(tensor)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return processed, tensor


# ============================================================
# MODEL / RESULTS LOADING
# ============================================================

@st.cache_data
def load_results() -> pd.DataFrame:
    """Load benchmark results."""

    return pd.read_csv(SUMMARY_PATH)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model(checkpoint_path: str):
    """Load and cache trained model."""

    return load_exported_model(checkpoint_path)


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_image(
    model,
    checkpoint,
    input_tensor,
):
    """
    Run classification on one preprocessed PCB defect image.
    """

    # Determine model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.inference_mode():

        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    probabilities = probabilities.detach().cpu()

    class_names = checkpoint.get(
        "class_names",
        DEFAULT_CLASSES,
    )

    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = str(
        class_names[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index].item()
    )

    return (
        predicted_class,
        confidence,
        probabilities,
        class_names,
    )


# ============================================================
# SYNTHETIC DEFECT FUNCTIONS
# ============================================================

def resize_reference_for_canvas(
    image: Image.Image,
    max_width: int = 850,
):
    """
    Resize reference PCB for the drawing canvas while
    maintaining its original aspect ratio.
    """

    image = image.convert("RGB")

    original_width, original_height = image.size

    if original_width <= max_width:
        return image

    scale = max_width / original_width

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def extract_canvas_rgb(canvas_data):
    """
    Convert canvas RGBA result into standard RGB numpy array.
    """

    image = np.asarray(canvas_data)

    if image.shape[2] == 4:

        rgb = cv2.cvtColor(
            image.astype(np.uint8),
            cv2.COLOR_RGBA2RGB,
        )

    else:

        rgb = image[:, :, :3].astype(np.uint8)

    return rgb


def detect_modified_region(
    original_rgb: np.ndarray,
    modified_rgb: np.ndarray,
    threshold: int = 25,
):
    """
    Compare original PCB against user-modified PCB.

    Returns:
        mask
        bounding box (x, y, w, h)

    If no drawing is detected, bounding box is None.
    """

    # Ensure both images have same dimensions
    if original_rgb.shape != modified_rgb.shape:

        modified_rgb = cv2.resize(
            modified_rgb,
            (
                original_rgb.shape[1],
                original_rgb.shape[0],
            ),
        )

    # Absolute pixel difference
    difference = cv2.absdiff(
        original_rgb,
        modified_rgb,
    )

    # Convert difference to grayscale
    difference_gray = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY,
    )

    # Threshold changed pixels
    _, mask = cv2.threshold(
        difference_gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # Remove very small isolated differences
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Slight dilation to join nearby drawn pixels
    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    changed_pixels = cv2.countNonZero(mask)

    # Prevent tiny accidental pixel differences
    if changed_pixels < 20:
        return mask, None

    coordinates = cv2.findNonZero(mask)

    if coordinates is None:
        return mask, None

    x, y, width, height = cv2.boundingRect(
        coordinates
    )

    return mask, (
        x,
        y,
        width,
        height,
    )


def crop_defect_region(
    modified_rgb: np.ndarray,
    bounding_box,
    padding_ratio: float = 1.2,
    minimum_padding: int = 35,
):
    """
    Crop the synthetic defect together with surrounding PCB context.

    A square crop is produced because the classifier was trained on
    local PCB defect patches rather than whole-board photographs.
    """

    x, y, width, height = bounding_box

    image_height, image_width = modified_rgb.shape[:2]

    # Defect center
    center_x = x + width // 2
    center_y = y + height // 2

    # Base defect dimension
    largest_dimension = max(
        width,
        height,
    )

    padding = max(
        minimum_padding,
        int(largest_dimension * padding_ratio),
    )

    crop_size = largest_dimension + (padding * 2)

    # Keep crop square
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Shift crop if it goes outside image
    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > image_width:
        shift = x2 - image_width
        x1 -= shift
        x2 = image_width

    if y2 > image_height:
        shift = y2 - image_height
        y1 -= shift
        y2 = image_height

    # Final safety limits
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    defect_crop = modified_rgb[
        y1:y2,
        x1:x2,
    ]

    return defect_crop, (
        x1,
        y1,
        x2,
        y2,
    )


def draw_detection_box(
    image_rgb: np.ndarray,
    crop_coordinates,
):
    """
    Draw a bounding box showing the region selected for
    classification.
    """

    preview = image_rgb.copy()

    x1, y1, x2, y2 = crop_coordinates

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    return preview


# ============================================================
# PREDICTION DISPLAY
# ============================================================

def display_prediction(
    predicted_class,
    confidence,
    probabilities,
    class_names,
):
    """Display classification result."""

    st.success(
        "Predicted defect: "
        f"**{predicted_class.replace('_', ' ')}**"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    # Warn when prediction is uncertain.
    #
    # IMPORTANT:
    # 60% is a demonstration threshold, not a statistically
    # validated rejection threshold.
    if confidence < 0.60:

        st.warning(
            "Low-confidence prediction. The synthetic drawing may "
            "not closely resemble one of the six defect patterns "
            "learned by the classifier."
        )

    probability_table = pd.DataFrame(
        {
            "Defect class": [
                str(name).replace("_", " ")
                for name in class_names
            ],
            "Probability": probabilities.numpy(),
        }
    )

    probability_table = probability_table.sort_values(
        "Probability",
        ascending=False,
    )

    st.markdown("#### Classification probabilities")

    st.bar_chart(
        probability_table.set_index(
            "Defect class"
        )["Probability"],
        horizontal=True,
    )

    st.dataframe(
        probability_table.style.format(
            {
                "Probability": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Inspection",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    [data-testid="stMetric"] {
        background: #f5f8f7;
        border: 1px solid #dbe5e2;
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
    }

    .eyebrow {
        color: #147d64;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .78rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    'Image Processing · Demonstration'
    '</div>',
    unsafe_allow_html=True,
)

st.title(
    "PCB Defect Inspection System"
)

st.caption(
    "Evaluate trained PCB classifiers, create synthetic PCB "
    "defects interactively, and classify uploaded defect patches."
)


# ============================================================
# CHECK BENCHMARK SUMMARY
# ============================================================

if not SUMMARY_PATH.exists():

    st.error(
        f"Benchmark summary is missing: {SUMMARY_PATH}"
    )

    st.stop()


results = load_results()

model_names = results[
    "Model Name"
].tolist()


# ============================================================
# SIDEBAR MODEL SELECTION
# ============================================================

with st.sidebar:

    st.header(
        "Model selection"
    )

    selected_name = st.selectbox(
        "Choose a trained model",
        model_names,
    )

    st.caption(
        "The selected classifier is used for both synthetic "
        "defect testing and uploaded-image classification."
    )


selected = results.loc[
    results["Model Name"] == selected_name
].iloc[0]


checkpoint_path = (
    MODELS_DIR
    / f"{safe_filename(selected_name)}.pth"
)


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

st.subheader(
    selected_name
)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Test accuracy",
    f'{selected["Accuracy (%)"]:.2f}%',
)

metric_columns[1].metric(
    "Macro mAP",
    f'{selected["mAP (Academic)"]:.4f}',
)

metric_columns[2].metric(
    "Macro F1",
    f'{selected["F1-Score (Macro, %)"]:.2f}%',
)

metric_columns[3].metric(
    "Mean latency",
    f'{selected["Latency Mean (ms/image)"]:.3f} ms',
)

metric_columns[4].metric(
    "Parameters",
    f'{selected["Params (M)"]:.2f} M',
)


# ============================================================
# MAIN TABS
# ============================================================

results_tab, synthetic_tab, upload_tab = st.tabs(
    [
        "Test Results",
        "Synthetic PCB Test",
        "Upload Defect Image",
    ]
)


# ============================================================
# TAB 1 — BENCHMARK RESULTS
# ============================================================

with results_tab:

    left, right = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    confusion_path = result_path(
        str(
            selected[
                "Confusion Matrix"
            ]
        )
    )

    report_path = result_path(
        str(
            selected[
                "Classification Report"
            ]
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    with left:

        st.markdown(
            "#### Confusion matrix"
        )

        if confusion_path.exists():

            st.image(
                str(confusion_path),
                use_container_width=True,
            )

        else:

            st.warning(
                "Confusion matrix not found: "
                f"{confusion_path.name}"
            )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    with right:

        st.markdown(
            "#### Per-class report"
        )

        if report_path.exists():

            report = pd.read_csv(
                report_path,
                index_col=0,
            )

            st.dataframe(
                report.round(4),
                use_container_width=True,
            )

        else:

            st.warning(
                "Classification report not found: "
                f"{report_path.name}"
            )

        st.markdown(
            "#### Complete benchmark row"
        )

        display_columns = [
            "GFLOPs",
            "Latency Median (ms/image)",
            "Recall (Weighted, %)",
            "F1-Score (Weighted, %)",
        ]

        st.dataframe(
            selected[
                display_columns
            ]
            .rename("Value")
            .to_frame(),
            use_container_width=True,
        )


# ============================================================
# TAB 2 — SYNTHETIC PCB DEFECT
# ============================================================

with synthetic_tab:

    st.markdown(
        "### Create a Synthetic PCB Defect"
    )

    st.info(
        "Draw a synthetic defect directly on a normal PCB reference "
        "image. You do not need to specify the defect type. "
        "After you submit the drawing, the trained model will "
        "independently classify it as one of the six defect classes."
    )

    # --------------------------------------------------------
    # FIND REFERENCE PCB IMAGES
    # --------------------------------------------------------

    benchmark_files = []

    if BENCHMARK_PCB_DIR.exists():

        for extension in [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
        ]:

            benchmark_files.extend(
                BENCHMARK_PCB_DIR.glob(
                    extension
                )
            )

    benchmark_files = sorted(
        benchmark_files
    )

    if not benchmark_files:

        st.warning(
            "No benchmark PCB reference images were found.\n\n"
            "Create this folder:\n\n"
            "`Benchmark_PCB/`\n\n"
            "Then add at least one normal PCB image such as "
            "`pcb_01.png`."
        )

    else:

        # ----------------------------------------------------
        # REFERENCE PCB SELECTION
        # ----------------------------------------------------

        selected_reference_name = st.selectbox(
            "Normal PCB reference image",
            [
                file.name
                for file in benchmark_files
            ],
        )

        selected_reference_path = (
            BENCHMARK_PCB_DIR
            / selected_reference_name
        )

        reference_image = Image.open(
            selected_reference_path
        ).convert("RGB")

        # Resize reference for browser canvas
        canvas_background = (
            resize_reference_for_canvas(
                reference_image,
                max_width=850,
            )
        )

        canvas_width, canvas_height = (
            canvas_background.size
        )

        # ----------------------------------------------------
        # DRAWING SETTINGS
        # ----------------------------------------------------

        settings_col1, settings_col2, settings_col3 = (
            st.columns(3)
        )

        with settings_col1:

            drawing_mode = st.selectbox(
                "Drawing tool",
                [
                    "freedraw",
                    "line",
                    "circle",
                ],
                format_func=lambda value: {
                    "freedraw": "Free draw",
                    "line": "Straight line",
                    "circle": "Circle",
                }[value],
            )

        with settings_col2:

            stroke_width = st.slider(
                "Brush width",
                min_value=2,
                max_value=30,
                value=6,
            )

        with settings_col3:

            stroke_color = st.color_picker(
                "Drawing colour",
                "#1A1A1A",
            )

        st.caption(
            "The drawing controls only modify the reference PCB. "
            "They do not provide the classifier with any defect label."
        )

        # ----------------------------------------------------
        # DRAWING CANVAS
        # ----------------------------------------------------

        st.markdown(
            "#### Draw on the normal PCB"
        )

        canvas_result = st_canvas(
            fill_color="rgba(20, 20, 20, 1)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=canvas_background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            key=f"pcb_canvas_{selected_reference_name}",
        )

        # ----------------------------------------------------
        # CLASSIFICATION BUTTON
        # ----------------------------------------------------

        classify_synthetic = st.button(
            "Classify Synthetic Defect",
            type="primary",
            use_container_width=True,
        )

        if classify_synthetic:

            # ------------------------------------------------
            # CHECK MODEL
            # ------------------------------------------------

            if not checkpoint_path.exists():

                st.error(
                    "The selected model checkpoint could not be found: "
                    f"{checkpoint_path.name}"
                )

            elif canvas_result.image_data is None:

                st.warning(
                    "Please draw a defect on the PCB before "
                    "running classification."
                )

            else:

                # --------------------------------------------
                # ORIGINAL IMAGE IN CANVAS DIMENSIONS
                # --------------------------------------------

                original_rgb = np.asarray(
                    canvas_background.convert("RGB")
                ).astype(np.uint8)

                # --------------------------------------------
                # MODIFIED IMAGE FROM CANVAS
                # --------------------------------------------

                modified_rgb = extract_canvas_rgb(
                    canvas_result.image_data
                )

                # --------------------------------------------
                # DETECT DRAWING
                # --------------------------------------------

                difference_mask, bounding_box = (
                    detect_modified_region(
                        original_rgb,
                        modified_rgb,
                    )
                )

                if bounding_box is None:

                    st.warning(
                        "No significant drawing was detected. "
                        "Please draw a synthetic defect on the PCB "
                        "and try again."
                    )

                else:

                    # ----------------------------------------
                    # CROP LOCAL DEFECT PATCH
                    # ----------------------------------------

                    defect_crop, crop_coordinates = (
                        crop_defect_region(
                            modified_rgb,
                            bounding_box,
                        )
                    )

                    defect_image = Image.fromarray(
                        defect_crop
                    )

                    # ----------------------------------------
                    # MODEL PREPROCESSING
                    # ----------------------------------------

                    processed_image, input_tensor = (
                        preprocess_image(
                            defect_image
                        )
                    )

                    # ----------------------------------------
                    # PREVIEW
                    # ----------------------------------------

                    preview_col, model_col = st.columns(
                        [1, 1],
                        gap="large",
                    )

                    with preview_col:

                        st.markdown(
                            "#### Detected defect region"
                        )

                        detected_preview = (
                            draw_detection_box(
                                modified_rgb,
                                crop_coordinates,
                            )
                        )

                        st.image(
                            detected_preview,
                            caption=(
                                "Automatically detected region "
                                "submitted for classification"
                            ),
                            use_container_width=True,
                        )

                        st.markdown(
                            "#### Extracted defect patch"
                        )

                        st.image(
                            defect_image,
                            caption=(
                                "Local PCB region containing "
                                "the synthetic defect"
                            ),
                            use_container_width=True,
                        )

                        with st.expander(
                            "Show model-ready 224 × 224 image"
                        ):

                            st.image(
                                processed_image,
                                caption=(
                                    "Bilateral filter + CLAHE + "
                                    "224 × 224 letterbox"
                                ),
                                use_container_width=True,
                            )

                    # ----------------------------------------
                    # RUN MODEL
                    # ----------------------------------------

                    with model_col:

                        st.markdown(
                            "#### Model Prediction"
                        )

                        try:

                            model, checkpoint = get_model(
                                str(
                                    checkpoint_path
                                )
                            )

                            (
                                predicted_class,
                                confidence,
                                probabilities,
                                class_names,
                            ) = predict_image(
                                model,
                                checkpoint,
                                input_tensor,
                            )

                            display_prediction(
                                predicted_class,
                                confidence,
                                probabilities,
                                class_names,
                            )

                        except Exception as error:

                            st.error(
                                "Model loading or inference failed: "
                                f"{error}"
                            )


# ============================================================
# TAB 3 — UPLOAD DEFECT IMAGE
# ============================================================

with upload_tab:

    st.info(
        "Upload a cropped defect patch containing one defect. "
        "The classifier was trained on 224 × 224 defect patches, "
        "not complete PCB-board photographs."
    )

    uploaded_file = st.file_uploader(
        "Test image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ],
        help=(
            "Supported formats: JPG, PNG, and BMP."
        ),
    )

    if uploaded_file is not None:

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            processed_image, input_tensor = (
                preprocess_image(
                    uploaded_image
                )
            )

        except Exception as error:

            st.error(
                "The uploaded file could not be read "
                f"as an image: {error}"
            )

            st.stop()

        preview_col, prediction_col = st.columns(
            [0.85, 1.15],
            gap="large",
        )

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        with preview_col:

            st.markdown(
                "#### Input preview"
            )

            st.image(
                uploaded_image,
                caption="Uploaded image",
                use_container_width=True,
            )

            with st.expander(
                "Show model-ready image"
            ):

                st.image(
                    processed_image,
                    caption=(
                        "Bilateral filter + CLAHE + "
                        "224 × 224 letterbox"
                    ),
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with prediction_col:

            st.markdown(
                "#### Prediction"
            )

            if not checkpoint_path.exists():

                st.warning(
                    "This checkpoint has not been exported yet. "
                    "Run the notebook's 'EXPORT TRAINED MODELS' "
                    "cell, then add Exported_Models to Git LFS."
                )

            else:

                try:

                    model, checkpoint = get_model(
                        str(
                            checkpoint_path
                        )
                    )

                    (
                        predicted_class,
                        confidence,
                        probabilities,
                        class_names,
                    ) = predict_image(
                        model,
                        checkpoint,
                        input_tensor,
                    )

                    display_prediction(
                        predicted_class,
                        confidence,
                        probabilities,
                        class_names,
                    )

                except Exception as error:

                    st.error(
                        "Model loading or inference failed: "
                        f"{error}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictions are an educational demonstration of the trained "
    "PCB defect classification models. Synthetic user drawings may "
    "differ from the real defect distribution used during training."
)