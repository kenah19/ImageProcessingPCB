from pathlib import Path
from io import BytesIO
from datetime import datetime
import time

import cv2 as cv
import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageOps
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from streamlit_image_select import image_select

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)

from model_definitions import load_exported_model


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title = "PCB Defect Inspection System",
    page_icon = "🔍",
    layout = "wide"
)


# ============================================================
# File Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Exported_Models"

PCB_DIR = (
    BASE_DIR
    / "dataset"
    / "PCB_DATASET"
    / "PCB_USED"
)


# Change this when your best trained model is ready
BEST_MODEL_FILE = "CS-ResNet_-_Baseline.pth"

BEST_MODEL_PATH = MODEL_DIR / BEST_MODEL_FILE


DEFAULT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]


# ============================================================
# Dashboard Style
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0d1117;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .mainTitle {
        color: white;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subTitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .stepDescription {
        color: #9ca3af;
        font-size: 15px;
        margin-top: -10px;
        margin-bottom: 15px;
    }

    .selectedBar {
        background-color: #111b2e;
        border: 1px solid #2f81f7;
        border-radius: 10px;
        padding: 14px 18px;
        color: #e6edf3;
        font-size: 16px;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .selectedPCBName {
        color: #58a6ff;
        font-weight: bold;
    }

    .resultCard {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px;
    }

    .resultLabel {
        color: #8b949e;
        font-size: 14px;
    }

    .predictedClass {
        color: #3fb950;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 18px;
    }

    .confidenceValue {
        color: #2dd4bf;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 18px;
    }

    .inferenceValue {
        color: white;
        font-size: 18px;
        font-weight: 600;
    }

    .probabilityRow {
        margin-bottom: 15px;
    }

    .probabilityText {
        display: flex;
        justify-content: space-between;
        color: #e6edf3;
        margin-bottom: 5px;
    }

    .probabilityBackground {
        background-color: #30363d;
        border-radius: 10px;
        height: 10px;
    }

    .probabilityFill {
        background: linear-gradient(
            90deg,
            #238636,
            #3fb950
        );
        height: 10px;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 8px;
        font-weight: 600;
    }

    div.stDownloadButton > button {
        width: 100%;
        height: 46px;
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html = True
)


# ============================================================
# Load Best Trained Model
# ============================================================

@st.cache_resource(
    show_spinner = "Loading trained PCB model..."
)
def load_best_model():
    return load_exported_model(
        BEST_MODEL_PATH
    )


# ============================================================
# Get Benchmark PCB Images
# ============================================================

def get_pcb_images():
    if not PCB_DIR.exists():
        return []

    pcb_images = []

    for file in PCB_DIR.iterdir():
        if (
            file.is_file()
            and file.suffix.lower()
            in [".png", ".jpg", ".jpeg", ".bmp"]
        ):
            pcb_images.append(file)

    return sorted(pcb_images)


# ============================================================
# Resize PCB for Drawing Canvas
# ============================================================

def resize_for_canvas(img, max_width = 850):
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
# Convert Canvas Image
# ============================================================

def get_canvas_image(canvas_data):
    img = np.asarray(
        canvas_data
    ).astype(np.uint8)

    if img.shape[2] == 4:
        img = cv.cvtColor(
            img,
            cv.COLOR_RGBA2RGB
        )

    return img[:, :, :3]


# ============================================================
# Detect Defect Region
# ============================================================

def detect_defect_region(original, modified):
    if original.shape != modified.shape:
        modified = cv.resize(
            modified,
            (
                original.shape[1],
                original.shape[0]
            )
        )

    difference = cv.absdiff(
        original,
        modified
    )

    difference_gray = cv.cvtColor(
        difference,
        cv.COLOR_RGB2GRAY
    )

    _, mask = cv.threshold(
        difference_gray,
        25,
        255,
        cv.THRESH_BINARY
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

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
# Crop Defect Region
# ============================================================

def crop_defect_region(img, box):
    x, y, width, height = box

    img_height, img_width = img.shape[:2]

    centre_x = x + width // 2
    centre_y = y + height // 2

    largest_side = max(
        width,
        height
    )

    padding = max(
        35,
        int(largest_side * 1.2)
    )

    crop_size = (
        largest_side
        + padding * 2
    )

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

    return img[
        y1:y2,
        x1:x2
    ]


# ============================================================
# Preprocess Defect Image
# ============================================================

def preprocess_image(img):
    img = img.convert("RGB")

    img_np = np.array(img)

    # RGB to BGR
    img_bgr = cv.cvtColor(
        img_np,
        cv.COLOR_RGB2BGR
    )

    # Bilateral filtering
    img_filtered = cv.bilateralFilter(
        img_bgr,
        9,
        75,
        75
    )

    # Convert to LAB
    img_lab = cv.cvtColor(
        img_filtered,
        cv.COLOR_BGR2LAB
    )

    l, a, b = cv.split(
        img_lab
    )

    # CLAHE
    clahe = cv.createCLAHE(
        clipLimit = 2.0,
        tileGridSize = (8, 8)
    )

    l = clahe.apply(l)

    img_lab = cv.merge(
        (l, a, b)
    )

    # LAB to RGB
    img_rgb = cv.cvtColor(
        img_lab,
        cv.COLOR_LAB2RGB
    )

    img_enhanced = Image.fromarray(
        img_rgb
    )

    # Resize to 224 x 224
    img_processed = ImageOps.pad(
        img_enhanced,
        (224, 224),
        method = Image.Resampling.LANCZOS,
        color = (0, 0, 0)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean = [
                0.485,
                0.456,
                0.406
            ],
            std = [
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    input_tensor = transform(
        img_processed
    ).unsqueeze(0)

    return input_tensor, img_processed


# ============================================================
# Classify Defect
# ============================================================

def classify_defect(
    model,
    checkpoint,
    input_tensor
):
    model.eval()

    device = next(
        model.parameters()
    ).device

    input_tensor = input_tensor.to(
        device
    )

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim = 1
        )[0]

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    ) * 1000

    probabilities = (
        probabilities
        .detach()
        .cpu()
    )

    class_names = checkpoint.get(
        "class_names",
        DEFAULT_CLASSES
    )

    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = str(
        class_names[predicted_index]
    )

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    return (
        predicted_class,
        confidence,
        probabilities,
        class_names,
        inference_time
    )


# ============================================================
# Highlight Defect Location
# ============================================================

def highlight_defect(img, box):
    result = img.copy()

    x, y, width, height = box

    centre_x = x + width // 2
    centre_y = y + height // 2

    radius = max(
        width,
        height
    ) // 2

    radius += max(
        15,
        int(radius * 0.5)
    )

    cv.circle(
        result,
        (
            centre_x,
            centre_y
        ),
        radius,
        (255, 0, 0),
        4
    )

    return result


# ============================================================
# Get Top 3 Probabilities
# ============================================================

def get_top3(
    probabilities,
    class_names
):
    values = probabilities.numpy()

    indexes = np.argsort(
        values
    )[::-1][:3]

    top3 = []

    for index in indexes:
        top3.append({
            "class":
                str(class_names[index]),

            "probability":
                float(values[index])
        })

    return top3


# ============================================================
# Display Top 3 Probabilities
# ============================================================

def show_top3(top3):
    st.markdown(
        "#### Top-3 Class Probabilities"
    )

    for i, result in enumerate(
        top3,
        start = 1
    ):
        percentage = (
            result["probability"]
            * 100
        )

        st.markdown(
            f'<div class="probabilityRow">'
            f'<div class="probabilityText">'
            f'<span>{i}. {result["class"]}</span>'
            f'<span>{percentage:.2f}%</span>'
            f'</div>'
            f'<div class="probabilityBackground">'
            f'<div class="probabilityFill" '
            f'style="width:{percentage:.2f}%;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html = True
        )


# ============================================================
# Convert Image to Bytes
# ============================================================

def image_to_bytes(img):
    buffer = BytesIO()

    if isinstance(
        img,
        np.ndarray
    ):
        img = Image.fromarray(img)

    img.save(
        buffer,
        format = "PNG"
    )

    buffer.seek(0)

    return buffer


# ============================================================
# Calculate PDF Image Size
# ============================================================

def get_pdf_image_size(
    img,
    max_width,
    max_height
):
    if isinstance(
        img,
        np.ndarray
    ):
        height, width = img.shape[:2]

    else:
        width, height = img.size

    ratio = min(
        max_width / width,
        max_height / height
    )

    return (
        width * ratio,
        height * ratio
    )


# ============================================================
# Generate PDF Report
# ============================================================

def create_pdf_report(
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

    content = []

    content.append(
        Paragraph(
            "PCB Defect Inspection Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    report_time = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    info_data = [
        [
            "PCB Reference",
            pcb_name
        ],
        [
            "Predicted Defect",
            predicted_class
        ],
        [
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        ],
        [
            "Inference Time",
            f"{inference_time:.2f} ms"
        ],
        [
            "Report Generated",
            report_time
        ]
    ]

    info_table = Table(
        info_data,
        colWidths = [
            6 * cm,
            10 * cm
        ]
    )

    info_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#E5E7EB"
                )
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    content.append(
        info_table
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Top-3 Class Probabilities",
            styles["Heading2"]
        )
    )

    probability_data = [
        [
            "Rank",
            "Class",
            "Probability"
        ]
    ]

    for i, result in enumerate(
        top3,
        start = 1
    ):
        probability_data.append([
            str(i),
            result["class"],
            f'{result["probability"] * 100:.2f}%'
        ])

    probability_table = Table(
        probability_data,
        colWidths = [
            2 * cm,
            9 * cm,
            5 * cm
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#1F2937"
                )
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            )
        ])
    )

    content.append(
        probability_table
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Defect Location",
            styles["Heading2"]
        )
    )

    highlighted_buffer = image_to_bytes(
        highlighted_img
    )

    (
        highlighted_width,
        highlighted_height
    ) = get_pdf_image_size(
        highlighted_img,
        16 * cm,
        9 * cm
    )

    highlighted_pdf = PDFImage(
        highlighted_buffer,
        width = highlighted_width,
        height = highlighted_height
    )

    highlighted_pdf.hAlign = "CENTER"

    content.append(
        highlighted_pdf
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Detected Defect Region",
            styles["Heading2"]
        )
    )

    defect_buffer = image_to_bytes(
        defect_img
    )

    (
        defect_width,
        defect_height
    ) = get_pdf_image_size(
        defect_img,
        7 * cm,
        7 * cm
    )

    defect_pdf = PDFImage(
        defect_buffer,
        width = defect_width,
        height = defect_height
    )

    defect_pdf.hAlign = "CENTER"

    content.append(
        defect_pdf
    )

    document.build(
        content
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="mainTitle">'
    'PCB Defect Inspection System'
    '</div>',
    unsafe_allow_html = True
)

st.markdown(
    '<div class="subTitle">'
    'Select a PCB reference image, draw a defect directly '
    'on the board and allow the trained model to classify '
    'the defect automatically.'
    '</div>',
    unsafe_allow_html = True
)


# ============================================================
# Check Model Availability
# ============================================================

model_available = (
    BEST_MODEL_PATH.exists()
)

if not model_available:
    st.info(
        "The trained model has not been added yet. "
        "The PCB selection and drawing interface "
        "can still be tested."
    )


# ============================================================
# Load Benchmark PCB Images
# ============================================================

pcb_images = get_pcb_images()

if len(pcb_images) == 0:
    st.warning(
        "No benchmark PCB images were found."
    )

    st.stop()


# ============================================================
# Step 1 - Select Benchmark PCB
# ============================================================

st.markdown(
    "## 1. Select Benchmark PCB"
)

st.markdown(
    '<div class="stepDescription">'
    'Click on a PCB image to select it. '
    'The selected PCB will be used for defect drawing.'
    '</div>',
    unsafe_allow_html = True
)


pcb_path_list = [
    str(pcb)
    for pcb in pcb_images
]

pcb_caption_list = [
    pcb.name
    for pcb in pcb_images
]


selected_index = image_select(
    label = "",
    images = pcb_path_list,
    captions = pcb_caption_list,
    index = 0,
    return_value = "index",
    use_container_width = True
)


selected_path = pcb_images[
    selected_index
]

selected_pcb = selected_path.name


st.markdown(
    f'<div class="selectedBar">'
    f'✓ &nbsp; Selected PCB: '
    f'<span class="selectedPCBName">'
    f'{selected_pcb}'
    f'</span>'
    f'</div>',
    unsafe_allow_html = True
)


pcb_img = Image.open(
    selected_path
).convert("RGB")


# ============================================================
# Selected PCB Preview
# ============================================================

with st.expander(
    "View Selected PCB"
):
    st.image(
        pcb_img,
        caption = selected_pcb,
        use_container_width = True
    )


# ============================================================
# Step 2 - Draw Defect
# ============================================================

st.markdown(
    "## 2. Draw the Defect Region"
)

st.markdown(
    '<div class="stepDescription">'
    'Draw directly on the selected PCB. '
    'The modified region will be detected automatically '
    'before classification.'
    '</div>',
    unsafe_allow_html = True
)


canvas_img = resize_for_canvas(
    pcb_img
)


tool_col, width_col, colour_col = st.columns(
    [1.3, 1, 1]
)


with tool_col:
    drawing_mode = st.selectbox(
        "Drawing Tool",
        [
            "freedraw",
            "line",
            "circle"
        ],
        format_func = lambda x: {
            "freedraw": "Brush",
            "line": "Straight Line",
            "circle": "Circle"
        }[x]
    )


with width_col:
    stroke_width = st.slider(
        "Drawing Width",
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


# ============================================================
# Classification Button
# ============================================================

classify_button = st.button(
    "Classify Defect",
    type = "primary",
    use_container_width = True
)


# ============================================================
# Classification Process
# ============================================================

if classify_button:

    if not model_available:
        st.warning(
            "The trained model is not available yet. "
            "Please add the best trained model before "
            "running classification."
        )

    elif canvas_result.image_data is None:
        st.warning(
            "Please draw a defect on the PCB "
            "before classification."
        )

    else:
        original = np.array(
            canvas_img
        ).astype(np.uint8)

        modified = get_canvas_image(
            canvas_result.image_data
        )

        defect_box = detect_defect_region(
            original,
            modified
        )

        if defect_box is None:
            st.warning(
                "No clear defect region was detected. "
                "Please draw a visible defect on the PCB."
            )

        else:
            defect_crop = crop_defect_region(
                modified,
                defect_box
            )

            defect_img = Image.fromarray(
                defect_crop
            )

            (
                input_tensor,
                processed_img
            ) = preprocess_image(
                defect_img
            )


            with st.spinner(
                "Running PCB defect classification..."
            ):
                model, checkpoint = (
                    load_best_model()
                )

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    class_names,
                    inference_time
                ) = classify_defect(
                    model,
                    checkpoint,
                    input_tensor
                )


            highlighted_img = highlight_defect(
                modified,
                defect_box
            )

            top3 = get_top3(
                probabilities,
                class_names
            )


            # ====================================================
            # Step 3 - Classification Result
            # ====================================================

            st.divider()

            st.markdown(
                "## 3. Classification Result"
            )


            (
                result_col,
                image_col,
                probability_col
            ) = st.columns(
                [0.8, 1.4, 0.9]
            )


            with result_col:
                st.markdown(
                    f'<div class="resultCard">'
                    f'<div class="resultLabel">'
                    f'Predicted Defect'
                    f'</div>'
                    f'<div class="predictedClass">'
                    f'{predicted_class}'
                    f'</div>'
                    f'<div class="resultLabel">'
                    f'Prediction Confidence'
                    f'</div>'
                    f'<div class="confidenceValue">'
                    f'{confidence * 100:.2f}%'
                    f'</div>'
                    f'<div class="resultLabel">'
                    f'Inference Time'
                    f'</div>'
                    f'<div class="inferenceValue">'
                    f'{inference_time:.2f} ms'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html = True
                )

                if confidence < 0.60:
                    st.warning(
                        "The model has relatively low "
                        "confidence for this prediction."
                    )


            with image_col:
                st.markdown(
                    "#### Defect Location"
                )

                st.image(
                    highlighted_img,
                    caption = "Detected defect location",
                    use_container_width = True
                )


            with probability_col:
                show_top3(
                    top3
                )


            # ====================================================
            # Classification Input
            # ====================================================

            with st.expander(
                "View Classification Input"
            ):
                crop_col, input_col = st.columns(
                    2
                )

                with crop_col:
                    st.markdown(
                        "#### Detected Defect Region"
                    )

                    st.image(
                        defect_img,
                        use_container_width = True
                    )

                with input_col:
                    st.markdown(
                        "#### Model Input 224 × 224"
                    )

                    st.image(
                        processed_img,
                        use_container_width = True
                    )


            # ====================================================
            # PDF Report
            # ====================================================

            pdf_report = create_pdf_report(
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
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".pdf"
            )


            st.download_button(
                label = "Download PDF Report",
                data = pdf_report,
                file_name = report_filename,
                mime = "application/pdf",
                use_container_width = True
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "PCB Defect Inspection System - "
    "Image Processing Assignment"
)