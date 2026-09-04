# Author: TAY KE NAH

from pathlib import Path
from io import BytesIO
from datetime import datetime

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageOps
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)

from model_definitions import load_exported_model


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title = "PCB Defect Inspection System",
    page_icon = "🔍",
    layout = "wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "Benchmark_Results"
MODEL_DIR = BASE_DIR / "Exported_Models"
PCB_DIR = BASE_DIR / "Benchmark_PCB"
SUMMARY_FILE = RESULTS_DIR / "final_benchmark_summary.csv"

DEFAULT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]


# ============================================================
# Dark dashboard style
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: #ffffff;
}

.mainTitle {
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.subTitle {
    color: #9ca3af;
    margin-bottom: 25px;
}

div[data-testid="stMetric"] {
    background-color: #171c24;
    border: 1px solid #2a303b;
    border-radius: 12px;
    padding: 17px;
}

div[data-testid="stMetricValue"] {
    color: #22d3c5;
}

.resultCard {
    background-color: #171c24;
    border: 1px solid #2a303b;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 15px;
}

.resultLabel {
    color: #9ca3af;
    font-size: 14px;
}

.prediction {
    color: #4ade80;
    font-size: 30px;
    font-weight: bold;
}

.confidence {
    color: #22d3c5;
    font-size: 30px;
    font-weight: bold;
}

.probRow {
    margin-bottom: 14px;
}

.probText {
    display: flex;
    justify-content: space-between;
    color: #e5e7eb;
    margin-bottom: 5px;
}

.probBackground {
    background-color: #2c3440;
    height: 10px;
    border-radius: 10px;
    overflow: hidden;
}

.probFill {
    background: linear-gradient(90deg, #16a34a, #4ade80);
    height: 10px;
    border-radius: 10px;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

div.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html = True)


# ============================================================
# Helper functions
# ============================================================

def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def get_value(row, names, default = None):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def get_result_path(value):
    if value is None:
        return None

    filename = Path(str(value).replace("\\", "/")).name
    return RESULTS_DIR / filename


# ============================================================
# Load benchmark result
# ============================================================

@st.cache_data
def load_results():
    return pd.read_csv(SUMMARY_FILE)


# ============================================================
# Load trained model
# ============================================================

@st.cache_resource(show_spinner = "Loading trained model...")
def get_model(checkpoint_path):
    return load_exported_model(checkpoint_path)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(img):
    img = img.convert("RGB")
    img_np = np.array(img)

    # RGB to BGR
    img_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

    # Bilateral filtering
    img_filtered = cv.bilateralFilter(
        img_bgr,
        d = 9,
        sigmaColor = 75,
        sigmaSpace = 75
    )

    # Convert to LAB
    img_lab = cv.cvtColor(img_filtered, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(img_lab)

    # CLAHE enhancement
    clahe = cv.createCLAHE(
        clipLimit = 2.0,
        tileGridSize = (8, 8)
    )

    l = clahe.apply(l)
    img_lab = cv.merge((l, a, b))

    # LAB back to RGB
    img_rgb = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    img_enhanced = Image.fromarray(img_rgb)

    # Resize to model input size
    img_processed = ImageOps.pad(
        img_enhanced,
        (224, 224),
        method = Image.Resampling.LANCZOS,
        color = (0, 0, 0)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img_processed).unsqueeze(0)

    return tensor, img_processed


# ============================================================
# Run model prediction
# ============================================================

def predict_image(model, checkpoint, input_tensor):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    input_tensor = input_tensor.to(device)
    model.eval()

    start_time = datetime.now()

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim = 1)[0]

    end_time = datetime.now()

    inference_time = (end_time - start_time).total_seconds() * 1000

    probabilities = probabilities.detach().cpu()

    class_names = checkpoint.get("class_names", DEFAULT_CLASSES)

    predicted_index = int(probabilities.argmax().item())
    predicted_class = str(class_names[predicted_index])
    confidence = float(probabilities[predicted_index].item())

    return (
        predicted_class,
        confidence,
        probabilities,
        class_names,
        inference_time
    )


# ============================================================
# Resize PCB for drawing canvas
# ============================================================

def resize_for_canvas(img, max_width = 780):
    img = img.convert("RGB")

    width, height = img.size

    if width <= max_width:
        return img

    scale = max_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    return img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


# ============================================================
# Convert drawing canvas to RGB image
# ============================================================

def get_canvas_image(canvas_data):
    img = np.asarray(canvas_data).astype(np.uint8)

    if img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    else:
        img = img[:, :, :3]

    return img


# ============================================================
# Detect where user changed the PCB
# ============================================================

def detect_defect(original, modified, threshold = 25):
    if original.shape != modified.shape:
        modified = cv.resize(
            modified,
            (original.shape[1], original.shape[0])
        )

    diff = cv.absdiff(original, modified)
    gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)

    _, mask = cv.threshold(
        gray,
        threshold,
        255,
        cv.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel
    )

    mask = cv.dilate(
        mask,
        kernel,
        iterations = 1
    )

    if cv.countNonZero(mask) < 20:
        return None

    points = cv.findNonZero(mask)

    if points is None:
        return None

    return cv.boundingRect(points)


# ============================================================
# Crop defect region with surrounding PCB
# ============================================================

def crop_defect(img, box, padding_ratio = 1.2, min_padding = 35):
    x, y, width, height = box

    img_height, img_width = img.shape[:2]

    centre_x = x + width // 2
    centre_y = y + height // 2

    largest_side = max(width, height)
    padding = max(min_padding, int(largest_side * padding_ratio))

    crop_size = largest_side + padding * 2

    x1 = centre_x - crop_size // 2
    y1 = centre_y - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > img_width:
        x1 -= x2 - img_width
        x2 = img_width

    if y2 > img_height:
        y1 -= y2 - img_height
        y2 = img_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    crop = img[y1:y2, x1:x2]

    return crop


# ============================================================
# Circle defect location
# ============================================================

def mark_defect(img, box):
    result = img.copy()

    x, y, width, height = box

    centre_x = x + width // 2
    centre_y = y + height // 2

    radius = max(width, height) // 2
    radius = radius + max(15, int(radius * 0.5))

    cv.circle(
        result,
        (centre_x, centre_y),
        radius,
        (255, 0, 0),
        4
    )

    return result


# ============================================================
# Get Top-3 probabilities
# ============================================================

def get_top3(probabilities, class_names):
    values = probabilities.numpy()
    indexes = np.argsort(values)[::-1][:3]

    top3 = []

    for index in indexes:
        top3.append({
            "class": str(class_names[index]),
            "probability": float(values[index])
        })

    return top3


# ============================================================
# Display Top-3 probabilities
# ============================================================

def show_top3(top3):
    st.markdown("#### Top-3 Probabilities")

    for i, item in enumerate(top3, start = 1):
        percentage = item["probability"] * 100

        st.markdown(
            f"""
            <div class="probRow">
                <div class="probText">
                    <span>{i}. {item["class"]}</span>
                    <span>{percentage:.2f}%</span>
                </div>

                <div class="probBackground">
                    <div class="probFill"
                         style="width:{percentage:.2f}%;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html = True
        )


# ============================================================
# Convert image to PNG bytes
# ============================================================

def image_to_bytes(img):
    buffer = BytesIO()

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img.save(buffer, format = "PNG")
    buffer.seek(0)

    return buffer


# ============================================================
# Generate PDF inspection report
# ============================================================

def create_pdf_report(
    model_name,
    pcb_name,
    predicted_class,
    confidence,
    inference_time,
    top3,
    highlighted_img,
    defect_img
):
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize = A4,
        rightMargin = 1.5 * cm,
        leftMargin = 1.5 * cm,
        topMargin = 1.5 * cm,
        bottomMargin = 1.5 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent = styles["Title"],
        alignment = TA_CENTER,
        fontSize = 20,
        spaceAfter = 15
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent = styles["Heading2"],
        fontSize = 13,
        spaceBefore = 10,
        spaceAfter = 8
    )

    story = []

    story.append(
        Paragraph(
            "PCB Defect Inspection Report",
            title_style
        )
    )

    report_time = datetime.now().strftime("%d %B %Y, %I:%M %p")

    information = [
        ["Inspection Information", ""],
        ["Report Generated", report_time],
        ["PCB Reference", pcb_name],
        ["Model", model_name],
        ["Predicted Defect", predicted_class],
        ["Prediction Confidence", f"{confidence * 100:.2f}%"],
        ["Inference Time", f"{inference_time:.2f} ms"]
    ]

    info_table = Table(
        information,
        colWidths = [5.5 * cm, 11 * cm]
    )

    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("SPAN", (0, 0), (1, 0)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F3F4F6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(info_table)
    story.append(Spacer(1, 12))

    # Top-3 probability table
    story.append(
        Paragraph(
            "Top-3 Class Probabilities",
            heading_style
        )
    )

    probability_data = [
        ["Rank", "Defect Class", "Probability"]
    ]

    for i, item in enumerate(top3, start = 1):
        probability_data.append([
            str(i),
            item["class"],
            f'{item["probability"] * 100:.2f}%'
        ])

    probability_table = Table(
        probability_data,
        colWidths = [2 * cm, 9.5 * cm, 5 * cm]
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(probability_table)
    story.append(Spacer(1, 12))

    # Highlighted PCB
    story.append(
        Paragraph(
            "PCB with Defect Location Highlighted",
            heading_style
        )
    )

    highlighted_buffer = image_to_bytes(highlighted_img)

    highlighted_pdf_img = PDFImage(
        highlighted_buffer,
        width = 16 * cm,
        height = 9 * cm
    )

    highlighted_pdf_img.hAlign = "CENTER"

    story.append(highlighted_pdf_img)
    story.append(Spacer(1, 12))

    # Cropped defect
    story.append(
        Paragraph(
            "Detected Defect Region",
            heading_style
        )
    )

    defect_buffer = image_to_bytes(defect_img)

    defect_pdf_img = PDFImage(
        defect_buffer,
        width = 7 * cm,
        height = 7 * cm
    )

    defect_pdf_img.hAlign = "CENTER"

    story.append(defect_pdf_img)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "This report was automatically generated by the "
            "PCB Defect Inspection System.",
            styles["Normal"]
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# Find PCB reference images
# ============================================================

def get_pcb_images():
    if not PCB_DIR.exists():
        return []

    files = []

    for extension in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        files.extend(PCB_DIR.glob(extension))

    return sorted(files)


# ============================================================
# App header
# ============================================================

st.markdown(
    '<div class="mainTitle">PCB Defect Inspection System</div>',
    unsafe_allow_html = True
)

st.markdown(
    '<div class="subTitle">'
    'Evaluate trained PCB classifiers, draw defects on benchmark '
    'PCB images and obtain classification results instantly.'
    '</div>',
    unsafe_allow_html = True
)


# ============================================================
# Read benchmark result CSV
# ============================================================

if not SUMMARY_FILE.exists():
    st.error("final_benchmark_summary.csv was not found.")
    st.stop()

results = load_results()

model_col = find_column(
    results,
    ["Model", "Model Name", "model_name"]
)

if model_col is None:
    st.error("Model column was not found in the benchmark result file.")
    st.stop()


# ============================================================
# Select trained model
# ============================================================

model_names = results[model_col].astype(str).tolist()

selected_model = st.selectbox(
    "Select Model",
    model_names
)

selected_row = results[
    results[model_col].astype(str) == selected_model
].iloc[0]

st.subheader(selected_model)


# ============================================================
# Benchmark metrics
# ============================================================

accuracy = get_value(
    selected_row,
    ["Test Accuracy (%)", "Test Accuracy", "Accuracy (%)", "Accuracy"],
    0
)

map_score = get_value(
    selected_row,
    ["Macro mAP", "mAP", "Macro Avg F1", "Macro F1"],
    0
)

f1_score = get_value(
    selected_row,
    ["F1-Score (Weighted, %)", "Macro F1 (%)", "Macro F1", "F1 Score (%)"],
    0
)

latency = get_value(
    selected_row,
    ["Latency Mean (ms/image)", "Latency Median (ms/image)", "Mean Latency (ms/image)", "Latency"],
    0
)

parameters = get_value(
    selected_row,
    ["Parameters (M)", "Parameters", "Params (M)"],
    0
)

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:
    try:
        st.metric("Test Accuracy", f"{float(accuracy):.2f}%")
    except:
        st.metric("Test Accuracy", str(accuracy))

with metric2:
    try:
        st.metric("Macro mAP", f"{float(map_score):.4f}")
    except:
        st.metric("Macro mAP", str(map_score))

with metric3:
    try:
        st.metric("F1 Score", f"{float(f1_score):.2f}%")
    except:
        st.metric("F1 Score", str(f1_score))

with metric4:
    try:
        st.metric("Mean Latency", f"{float(latency):.3f} ms")
    except:
        st.metric("Mean Latency", str(latency))

with metric5:
    try:
        st.metric("Parameters", f"{float(parameters):.2f} M")
    except:
        st.metric("Parameters", str(parameters))


# ============================================================
# Get model checkpoint
# ============================================================

checkpoint_value = get_value(
    selected_row,
    [
        "Checkpoint",
        "Checkpoint Path",
        "Model Path",
        "Exported Model",
        "Exported Model Path"
    ]
)

checkpoint_path = None

if checkpoint_value is not None:
    checkpoint_name = Path(
        str(checkpoint_value).replace("\\", "/")
    ).name

    checkpoint_path = MODEL_DIR / checkpoint_name


# ============================================================
# Tabs
# ============================================================

test_tab, pcb_tab, upload_tab = st.tabs([
    "Test Results",
    "PCB Defect Test",
    "Upload Defect Image"
])


# ============================================================
# Test Results tab
# ============================================================

with test_tab:
    left, right = st.columns([1.1, 1])

    confusion_value = get_value(
        selected_row,
        [
            "Confusion Matrix",
            "Confusion Matrix Path",
            "Confusion Matrix File"
        ]
    )

    report_value = get_value(
        selected_row,
        [
            "Classification Report",
            "Classification Report Path",
            "Report Path"
        ]
    )

    with left:
        st.subheader("Confusion Matrix")

        confusion_path = get_result_path(confusion_value)

        if confusion_path is not None and confusion_path.exists():
            st.image(
                str(confusion_path),
                use_container_width = True
            )
        else:
            st.info("Confusion matrix was not found.")

    with right:
        st.subheader("Per-Class Report")

        report_path = get_result_path(report_value)

        if report_path is not None and report_path.exists():
            try:
                report_df = pd.read_csv(report_path, index_col = 0)

                st.dataframe(
                    report_df,
                    use_container_width = True
                )
            except:
                st.info("Classification report could not be loaded.")
        else:
            st.info("Classification report was not found.")

    st.subheader("Complete Benchmark Row")

    st.dataframe(
        selected_row.to_frame(name = "Value"),
        use_container_width = True
    )


# ============================================================
# PCB Defect Test tab
# ============================================================

with pcb_tab:
    st.subheader("PCB Defect Classification")

    st.info(
        "Select a normal PCB reference image and draw a defect directly "
        "on the PCB. The trained model will classify the drawn region "
        "into one of the six PCB defect classes."
    )

    pcb_images = get_pcb_images()

    if len(pcb_images) == 0:
        st.warning(
            "No benchmark PCB reference images were found.\n\n"
            "Create the folder `Benchmark_PCB/` inside the Assignment "
            "folder and add the PCB reference images."
        )

    else:
        # ----------------------------------------------------
        # Select benchmark PCB
        # ----------------------------------------------------

        st.markdown("### 1. Select a Normal PCB Reference Image")

        pcb_names = [img.name for img in pcb_images]

        selected_pcb = st.selectbox(
            "PCB Reference",
            pcb_names,
            label_visibility = "collapsed"
        )

        selected_pcb_path = PCB_DIR / selected_pcb

        pcb_img = Image.open(selected_pcb_path).convert("RGB")
        canvas_img = resize_for_canvas(pcb_img)

        # Show available benchmark images
        preview_cols = st.columns(min(len(pcb_images), 5))

        for i, img_path in enumerate(pcb_images[:5]):
            with preview_cols[i]:
                preview_img = Image.open(img_path).convert("RGB")

                st.image(
                    preview_img,
                    caption = img_path.name,
                    use_container_width = True
                )

        # ----------------------------------------------------
        # Drawing controls
        # ----------------------------------------------------

        st.markdown("### 2. Draw the Defect Region")

        tool_col, width_col, colour_col = st.columns([1.5, 1, 1])

        with tool_col:
            drawing_mode = st.selectbox(
                "Drawing Tool",
                ["freedraw", "line", "circle"],
                format_func = lambda x: {
                    "freedraw": "Brush",
                    "line": "Straight Line",
                    "circle": "Circle"
                }[x]
            )

        with width_col:
            stroke_width = st.slider(
                "Brush Width",
                2,
                30,
                6
            )

        with colour_col:
            stroke_colour = st.color_picker(
                "Drawing Colour",
                "#D91E18"
            )

        canvas_result = st_canvas(
            fill_color = "rgba(217, 30, 24, 0.30)",
            stroke_width = stroke_width,
            stroke_color = stroke_colour,
            background_image = canvas_img,
            update_streamlit = True,
            height = canvas_img.height,
            width = canvas_img.width,
            drawing_mode = drawing_mode,
            display_toolbar = True,
            key = f"canvas_{selected_pcb}"
        )

        classify_btn = st.button(
            "Submit & Classify",
            type = "primary",
            use_container_width = True
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if classify_btn:
            if checkpoint_path is None:
                st.error("Checkpoint path was not found for the selected model.")

            elif not checkpoint_path.exists():
                st.error(f"Model checkpoint was not found: {checkpoint_path}")

            elif canvas_result.image_data is None:
                st.warning("Please draw a defect on the PCB first.")

            else:
                original = np.array(canvas_img).astype(np.uint8)
                modified = get_canvas_image(canvas_result.image_data)

                box = detect_defect(original, modified)

                if box is None:
                    st.warning(
                        "No clear defect drawing was detected. "
                        "Please draw a visible defect on the PCB."
                    )

                else:
                    defect_crop = crop_defect(modified, box)
                    defect_img = Image.fromarray(defect_crop)

                    input_tensor, processed_img = preprocess_image(defect_img)

                    with st.spinner("Classifying PCB defect..."):
                        model, checkpoint = get_model(str(checkpoint_path))

                        (
                            predicted_class,
                            confidence,
                            probabilities,
                            class_names,
                            inference_time
                        ) = predict_image(
                            model,
                            checkpoint,
                            input_tensor
                        )

                    highlighted_img = mark_defect(modified, box)
                    top3 = get_top3(probabilities, class_names)

                    st.divider()
                    st.markdown("### 3. Classification Result")

                    result_col, image_col, probability_col = st.columns(
                        [0.9, 1.25, 0.9]
                    )

                    # Classification result
                    with result_col:
                        st.markdown(
                            f"""
                            <div class="resultCard">
                                <div class="resultLabel">
                                    Predicted Class
                                </div>

                                <div class="prediction">
                                    {predicted_class}
                                </div>

                                <hr>

                                <div class="resultLabel">
                                    Prediction Confidence
                                </div>

                                <div class="confidence">
                                    {confidence * 100:.2f}%
                                </div>

                                <br>

                                <div class="resultLabel">
                                    Inference Time
                                </div>

                                <div style="
                                    color:white;
                                    font-size:18px;
                                    font-weight:600;
                                ">
                                    {inference_time:.2f} ms
                                </div>
                            </div>
                            """,
                            unsafe_allow_html = True
                        )

                        if confidence < 0.60:
                            st.warning(
                                "The model has relatively low confidence "
                                "for this prediction."
                            )

                    # Highlighted PCB
                    with image_col:
                        st.markdown("#### Defect Location")

                        st.image(
                            highlighted_img,
                            caption = "Detected defect location",
                            use_container_width = True
                        )

                    # Top 3
                    with probability_col:
                        show_top3(top3)

                    # ------------------------------------------------
                    # Model input details
                    # ------------------------------------------------

                    with st.expander("View Model Input Details"):
                        crop_col, input_col = st.columns(2)

                        with crop_col:
                            st.markdown("#### Cropped Defect Region")

                            st.image(
                                defect_img,
                                use_container_width = True
                            )

                        with input_col:
                            st.markdown("#### Model Input - 224 x 224")

                            st.image(
                                processed_img,
                                use_container_width = True
                            )

                    # ------------------------------------------------
                    # Generate PDF report
                    # ------------------------------------------------

                    pdf_data = create_pdf_report(
                        selected_model,
                        selected_pcb,
                        predicted_class,
                        confidence,
                        inference_time,
                        top3,
                        highlighted_img,
                        defect_img
                    )

                    report_filename = (
                        "PCB_Inspection_Report_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S")
                        + ".pdf"
                    )

                    st.download_button(
                        label = "Download PDF Report",
                        data = pdf_data,
                        file_name = report_filename,
                        mime = "application/pdf",
                        use_container_width = True
                    )


# ============================================================
# Upload defect image tab
# ============================================================

with upload_tab:
    st.info(
        "Upload a cropped PCB defect image containing one defect. "
        "The classifier was trained using 224 x 224 defect patches."
    )

    uploaded_file = st.file_uploader(
        "Test Image",
        type = ["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        uploaded_img = Image.open(uploaded_file).convert("RGB")

        image_col, result_col = st.columns(2)

        with image_col:
            st.image(
                uploaded_img,
                caption = "Uploaded Defect Image",
                use_container_width = True
            )

        if checkpoint_path is None:
            st.error("Checkpoint path was not found.")

        elif not checkpoint_path.exists():
            st.error(f"Model checkpoint was not found: {checkpoint_path}")

        else:
            input_tensor, processed_img = preprocess_image(uploaded_img)

            with st.spinner("Classifying PCB defect..."):
                model, checkpoint = get_model(str(checkpoint_path))

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    class_names,
                    inference_time
                ) = predict_image(
                    model,
                    checkpoint,
                    input_tensor
                )

            top3 = get_top3(probabilities, class_names)

            with result_col:
                st.markdown(
                    f"""
                    <div class="resultCard">
                        <div class="resultLabel">
                            Predicted Class
                        </div>

                        <div class="prediction">
                            {predicted_class}
                        </div>

                        <hr>

                        <div class="resultLabel">
                            Prediction Confidence
                        </div>

                        <div class="confidence">
                            {confidence * 100:.2f}%
                        </div>

                        <br>

                        <div class="resultLabel">
                            Inference Time
                        </div>

                        <div style="
                            color:white;
                            font-size:18px;
                            font-weight:600;
                        ">
                            {inference_time:.2f} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

                show_top3(top3)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "The PCB Defect Inspection System is developed for educational "
    "and image-processing demonstration purposes."
)