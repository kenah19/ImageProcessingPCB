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
PCB_DIR = BASE_DIR / "dataset" / "PCB_DATASET" / "PCB_USED"

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
        max-width: 1450px;
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
        margin-top: -8px;
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

    .metricCard {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        min-height: 120px;
    }

    .metricLabel {
        color: #8b949e;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .metricValue {
        color: white;
        font-size: 27px;
        font-weight: 700;
    }

    .defectClass {
        color: #3fb950;
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
# Session State
# ============================================================

if "selected_pcb" not in st.session_state:
    st.session_state.selected_pcb = None

if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None


# ============================================================
# Load Trained Model
# ============================================================

@st.cache_resource(show_spinner = "Loading trained PCB model...")
def load_best_model():
    return load_exported_model(BEST_MODEL_PATH)


# ============================================================
# Benchmark PCB Images
# ============================================================

@st.cache_data
def get_pcb_images():
    if not PCB_DIR.exists():
        return []

    pcb_images = []

    for file in PCB_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            pcb_images.append(file)

    return sorted(pcb_images)


def resize_for_canvas(img, max_width = 850):
    img = img.convert("RGB")
    width, height = img.size

    if width <= max_width:
        return img

    scale = max_width / width

    return img.resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS
    )


def get_canvas_image(canvas_data):
    img = np.asarray(canvas_data).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

    return img[:, :, :3]


# ============================================================
# Defect Localisation
# ============================================================

def create_difference_mask(benchmark, inspection):
    if benchmark.shape != inspection.shape:
        inspection = cv.resize(
            inspection,
            (benchmark.shape[1], benchmark.shape[0]),
            interpolation = cv.INTER_LINEAR
        )

    difference = cv.absdiff(benchmark, inspection)
    difference_gray = cv.cvtColor(difference, cv.COLOR_RGB2GRAY)

    _, mask = cv.threshold(
        difference_gray,
        25,
        255,
        cv.THRESH_BINARY
    )

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        open_kernel
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        close_kernel
    )

    mask = cv.dilate(
        mask,
        open_kernel,
        iterations = 1
    )

    return mask


def boxes_are_close(box1, box2, gap = 12):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left1 = x1 - gap
    top1 = y1 - gap
    right1 = x1 + w1 + gap
    bottom1 = y1 + h1 + gap

    left2 = x2
    top2 = y2
    right2 = x2 + w2
    bottom2 = y2 + h2

    return not (
        right1 < left2
        or right2 < left1
        or bottom1 < top2
        or bottom2 < top1
    )


def combine_boxes(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])

    x2 = max(
        box1[0] + box1[2],
        box2[0] + box2[2]
    )

    y2 = max(
        box1[1] + box1[3],
        box2[1] + box2[3]
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


def merge_nearby_boxes(boxes, gap = 12):
    boxes = boxes.copy()
    changed = True

    while changed:
        changed = False
        merged = []

        while boxes:
            current = boxes.pop(0)
            i = 0

            while i < len(boxes):
                if boxes_are_close(current, boxes[i], gap):
                    current = combine_boxes(current, boxes.pop(i))
                    changed = True
                else:
                    i += 1

            merged.append(current)

        boxes = merged

    return boxes


def detect_defect_regions(benchmark, inspection):
    mask = create_difference_mask(
        benchmark,
        inspection
    )

    count, labels, stats, centres = cv.connectedComponentsWithStats(
        mask,
        connectivity = 8
    )

    min_area = max(
        20,
        int(mask.shape[0] * mask.shape[1] * 0.00003)
    )

    boxes = []

    for i in range(1, count):
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        width = int(stats[i, cv.CC_STAT_WIDTH])
        height = int(stats[i, cv.CC_STAT_HEIGHT])
        area = int(stats[i, cv.CC_STAT_AREA])

        if area >= min_area:
            boxes.append(
                (x, y, width, height)
            )

    boxes = merge_nearby_boxes(
        boxes,
        gap = 12
    )

    boxes = sorted(
        boxes,
        key = lambda box: (box[1], box[0])
    )

    return boxes, mask


def map_box_to_original(box, canvas_size, original_size):
    x, y, width, height = box

    canvas_width, canvas_height = canvas_size
    original_width, original_height = original_size

    scale_x = original_width / canvas_width
    scale_y = original_height / canvas_height

    return (
        int(x * scale_x),
        int(y * scale_y),
        max(1, int(width * scale_x)),
        max(1, int(height * scale_y))
    )


def crop_defect_region(img, box):
    x, y, width, height = box
    img_height, img_width = img.shape[:2]

    centre_x = x + width // 2
    centre_y = y + height // 2

    largest_side = max(width, height)
    padding = max(35, int(largest_side * 1.2))
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

    return img[y1:y2, x1:x2]


# ============================================================
# Model Preprocessing
# ============================================================

def preprocess_image(img):
    img = img.convert("RGB")
    img_np = np.array(img)

    img_bgr = cv.cvtColor(
        img_np,
        cv.COLOR_RGB2BGR
    )

    img_filtered = cv.bilateralFilter(
        img_bgr,
        9,
        75,
        75
    )

    img_lab = cv.cvtColor(
        img_filtered,
        cv.COLOR_BGR2LAB
    )

    l, a, b = cv.split(img_lab)

    clahe = cv.createCLAHE(
        clipLimit = 2.0,
        tileGridSize = (8, 8)
    )

    l = clahe.apply(l)

    img_lab = cv.merge(
        (l, a, b)
    )

    img_rgb = cv.cvtColor(
        img_lab,
        cv.COLOR_LAB2RGB
    )

    img_enhanced = Image.fromarray(
        img_rgb
    )

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

    input_tensor = transform(
        img_processed
    ).unsqueeze(0)

    return input_tensor, img_processed


# ============================================================
# Model Classification
# ============================================================

def classify_defect(model, checkpoint, input_tensor):
    model.eval()

    device = next(
        model.parameters()
    ).device

    input_tensor = input_tensor.to(
        device
    )

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim = 1
        )[0]

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    probabilities = probabilities.detach().cpu()

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
        probabilities[predicted_index].item()
    )

    top_indexes = torch.argsort(
        probabilities,
        descending = True
    )[:3]

    top3 = []

    for index in top_indexes:
        index = int(index.item())

        top3.append({
            "class": str(class_names[index]),
            "probability": float(probabilities[index].item())
        })

    return (
        predicted_class,
        confidence,
        inference_time,
        top3
    )


# ============================================================
# Final PCB Annotation
# ============================================================

def annotate_defects(img, results):
    annotated = img.copy()

    for i, result in enumerate(results, start = 1):
        x, y, width, height = result["box"]

        cv.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            max(2, int(annotated.shape[1] / 450))
        )

        label = (
            f'{i}. {result["predicted_class"]} '
            f'{result["confidence"] * 100:.1f}%'
        )

        font_scale = max(
            0.5,
            annotated.shape[1] / 1600
        )

        thickness = max(
            1,
            int(annotated.shape[1] / 700)
        )

        (text_width, text_height), baseline = cv.getTextSize(
            label,
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        label_y = max(
            text_height + 8,
            y - 8
        )

        cv.rectangle(
            annotated,
            (x, label_y - text_height - 8),
            (x + text_width + 10, label_y + baseline),
            (255, 0, 0),
            -1
        )

        cv.putText(
            annotated,
            label,
            (x + 5, label_y - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv.LINE_AA
        )

    return annotated


# ============================================================
# PDF Report
# ============================================================

def image_to_buffer(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    buffer = BytesIO()
    img.save(buffer, format = "PNG")
    buffer.seek(0)

    return buffer


def get_pdf_image_size(img, max_width, max_height):
    if isinstance(img, np.ndarray):
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


def create_pdf_report(pcb_name, results, annotated_img):
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

    summary_data = [
        ["PCB Reference", pcb_name],
        ["Detected Defect Regions", str(len(results))],
        ["Model", "CS-ResNet - Baseline"],
        [
            "Report Generated",
            datetime.now().strftime("%d %B %Y, %I:%M %p")
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths = [6 * cm, 10 * cm]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    content.append(summary_table)
    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Detected Defects",
            styles["Heading2"]
        )
    )

    defect_data = [
        [
            "Region",
            "Predicted Class",
            "Confidence",
            "Inference Time"
        ]
    ]

    for i, result in enumerate(results, start = 1):
        defect_data.append([
            str(i),
            result["predicted_class"],
            f'{result["confidence"] * 100:.2f}%',
            f'{result["inference_time"]:.2f} ms'
        ])

    defect_table = Table(
        defect_data,
        colWidths = [2 * cm, 7 * cm, 3.5 * cm, 3.5 * cm]
    )

    defect_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    content.append(defect_table)
    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Final Inspection Result",
            styles["Heading2"]
        )
    )

    annotated_buffer = image_to_buffer(
        annotated_img
    )

    width, height = get_pdf_image_size(
        annotated_img,
        17 * cm,
        10 * cm
    )

    pdf_img = PDFImage(
        annotated_buffer,
        width = width,
        height = height
    )

    pdf_img.hAlign = "CENTER"
    content.append(pdf_img)

    for i, result in enumerate(results, start = 1):
        content.append(Spacer(1, 12))

        content.append(
            Paragraph(
                f"Region {i}: {result['predicted_class']}",
                styles["Heading3"]
            )
        )

        top3_data = [
            ["Rank", "Class", "Probability"]
        ]

        for rank, item in enumerate(result["top3"], start = 1):
            top3_data.append([
                str(rank),
                item["class"],
                f'{item["probability"] * 100:.2f}%'
            ])

        top3_table = Table(
            top3_data,
            colWidths = [2 * cm, 9 * cm, 5 * cm]
        )

        top3_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ])
        )

        content.append(top3_table)

    document.build(content)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="mainTitle">PCB Defect Inspection System</div>',
    unsafe_allow_html = True
)

st.markdown(
    '<div class="subTitle">'
    'Select a benchmark PCB, create one or more defects, then inspect the '
    'modified PCB. The system localises each changed region and classifies '
    'each detected defect independently.'
    '</div>',
    unsafe_allow_html = True
)


# ============================================================
# Model Availability
# ============================================================

model_available = BEST_MODEL_PATH.exists()

if not model_available:
    st.info(
        "The trained model is not available yet. "
        "PCB selection and drawing can still be tested."
    )


# ============================================================
# Step 1 - Select Benchmark PCB
# ============================================================

pcb_images = get_pcb_images()

if len(pcb_images) == 0:
    st.warning(
        "No benchmark PCB images were found."
    )
    st.stop()

st.markdown(
    "## 1. Select Benchmark PCB"
)

st.markdown(
    '<div class="stepDescription">'
    'Click a benchmark PCB image to use it as the reference board.'
    '</div>',
    unsafe_allow_html = True
)

selected_index = image_select(
    label = "",
    images = [str(pcb) for pcb in pcb_images],
    captions = [pcb.name for pcb in pcb_images],
    index = 0,
    return_value = "index",
    use_container_width = True
)

selected_path = pcb_images[selected_index]
selected_pcb = selected_path.name

if st.session_state.selected_pcb != selected_pcb:
    st.session_state.selected_pcb = selected_pcb
    st.session_state.inspection_result = None

st.markdown(
    f'<div class="selectedBar">'
    f'✓ &nbsp; Selected PCB: '
    f'<span class="selectedPCBName">{selected_pcb}</span>'
    f'</div>',
    unsafe_allow_html = True
)

pcb_img = Image.open(
    selected_path
).convert("RGB")

with st.expander("View Original Benchmark PCB"):
    st.image(
        pcb_img,
        caption = selected_pcb,
        use_container_width = True
    )


# ============================================================
# Step 2 - Create PCB Defect
# ============================================================

st.markdown(
    "## 2. Create PCB Defect"
)

st.markdown(
    '<div class="stepDescription">'
    'Draw one or more defects directly on the PCB. '
    'Different defect regions can be drawn at different locations.'
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
        ["freedraw", "line", "circle"],
        format_func = lambda value: {
            "freedraw": "Brush",
            "line": "Straight Line",
            "circle": "Circle"
        }[value]
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
# Step 3 - Inspect PCB
# ============================================================

st.markdown(
    "## 3. Inspect PCB"
)

st.markdown(
    '<div class="stepDescription">'
    'The benchmark PCB and modified PCB are compared first. '
    'Each detected region is then cropped and classified separately.'
    '</div>',
    unsafe_allow_html = True
)

inspect_button = st.button(
    "Inspect and Classify Defects",
    type = "primary",
    use_container_width = True
)

if inspect_button:
    st.session_state.inspection_result = None

    if not model_available:
        st.warning(
            "The trained model is not available. "
            "Please add the best trained model before classification."
        )

    elif canvas_result.image_data is None:
        st.warning(
            "Please draw at least one defect before inspection."
        )

    else:
        benchmark_canvas = np.array(
            canvas_img
        ).astype(np.uint8)

        inspection_canvas = get_canvas_image(
            canvas_result.image_data
        )

        defect_boxes, difference_mask = detect_defect_regions(
            benchmark_canvas,
            inspection_canvas
        )

        if len(defect_boxes) == 0:
            st.warning(
                "No clear defect region was detected. "
                "Please draw a visible defect on the PCB."
            )

        else:
            try:
                model, checkpoint = load_best_model()

            except Exception as error:
                st.error(
                    "The trained model could not be loaded. "
                    "Please check model_definitions.py and the exported checkpoint."
                )

                st.code(
                    str(error)
                )

            else:
                original_width, original_height = pcb_img.size

                inspection_full = cv.resize(
                    inspection_canvas,
                    (original_width, original_height),
                    interpolation = cv.INTER_LINEAR
                )

                results = []

                with st.spinner(
                    f"Inspecting {len(defect_boxes)} detected region(s)..."
                ):
                    for box in defect_boxes:
                        original_box = map_box_to_original(
                            box,
                            (canvas_img.width, canvas_img.height),
                            pcb_img.size
                        )

                        defect_crop = crop_defect_region(
                            inspection_full,
                            original_box
                        )

                        defect_img = Image.fromarray(
                            defect_crop
                        )

                        input_tensor, processed_img = preprocess_image(
                            defect_img
                        )

                        (
                            predicted_class,
                            confidence,
                            inference_time,
                            top3
                        ) = classify_defect(
                            model,
                            checkpoint,
                            input_tensor
                        )

                        results.append({
                            "box": original_box,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "inference_time": inference_time,
                            "top3": top3,
                            "crop": defect_img,
                            "processed": processed_img
                        })

                annotated_img = annotate_defects(
                    inspection_full,
                    results
                )

                st.session_state.inspection_result = {
                    "pcb_name": selected_pcb,
                    "results": results,
                    "annotated_img": annotated_img,
                    "difference_mask": difference_mask
                }


# ============================================================
# Inspection Result
# ============================================================

inspection_result = st.session_state.inspection_result

if inspection_result is not None:
    st.divider()

    st.markdown(
        "## 4. Inspection Result"
    )

    results = inspection_result["results"]
    annotated_img = inspection_result["annotated_img"]

    total_inference = sum(
        result["inference_time"]
        for result in results
    )

    unique_classes = sorted(
        set(
            result["predicted_class"]
            for result in results
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metricCard">'
            f'<div class="metricLabel">Detected Defect Regions</div>'
            f'<div class="metricValue">{len(results)}</div>'
            f'</div>',
            unsafe_allow_html = True
        )

    with col2:
        class_text = ", ".join(unique_classes)

        st.markdown(
            f'<div class="metricCard">'
            f'<div class="metricLabel">Detected Defect Classes</div>'
            f'<div class="metricValue defectClass">{class_text}</div>'
            f'</div>',
            unsafe_allow_html = True
        )

    with col3:
        st.markdown(
            f'<div class="metricCard">'
            f'<div class="metricLabel">Total Model Inference Time</div>'
            f'<div class="metricValue">{total_inference:.2f} ms</div>'
            f'</div>',
            unsafe_allow_html = True
        )

    st.markdown(
        "### Final Annotated PCB"
    )

    st.image(
        annotated_img,
        caption = "Detected defect regions and classification results",
        use_container_width = True
    )

    summary_rows = []

    for i, result in enumerate(results, start = 1):
        summary_rows.append({
            "Region": i,
            "Predicted Class": result["predicted_class"],
            "Confidence": f'{result["confidence"] * 100:.2f}%',
            "Inference Time": f'{result["inference_time"]:.2f} ms'
        })

    st.markdown(
        "### Detected Defect Summary"
    )

    st.dataframe(
        summary_rows,
        use_container_width = True,
        hide_index = True
    )

    st.markdown(
        "### Region Details"
    )

    for i, result in enumerate(results, start = 1):
        with st.expander(
            f'Region {i} - {result["predicted_class"]} '
            f'({result["confidence"] * 100:.2f}%)'
        ):
            crop_col, input_col, prob_col = st.columns(
                [1, 1, 1.2]
            )

            with crop_col:
                st.markdown(
                    "#### Detected Region"
                )

                st.image(
                    result["crop"],
                    use_container_width = True
                )

            with input_col:
                st.markdown(
                    "#### Model Input 224 × 224"
                )

                st.image(
                    result["processed"],
                    use_container_width = True
                )

            with prob_col:
                st.markdown(
                    "#### Top-3 Probabilities"
                )

                for rank, item in enumerate(
                    result["top3"],
                    start = 1
                ):
                    st.write(
                        f'{rank}. {item["class"]}: '
                        f'{item["probability"] * 100:.2f}%'
                    )

    with st.expander(
        "View Defect Difference Mask"
    ):
        st.image(
            inspection_result["difference_mask"],
            caption = "Difference mask used to localise changed regions",
            use_container_width = True
        )

    pdf_report = create_pdf_report(
        inspection_result["pcb_name"],
        results,
        annotated_img
    )

    report_filename = (
        "PCB_Inspection_Report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pdf"
    )

    st.download_button(
        label = "Download PDF Inspection Report",
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
    "PCB Defect Inspection System - Image Processing Assignment"
)
