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

UNCERTAIN_THRESHOLD = 0.50


# ============================================================
# Dashboard Style
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1118;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 3.5rem;
        padding-bottom: 3rem;
    }

    /* Horizontal benchmark PCB gallery */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        gap: 12px !important;
        padding-bottom: 12px !important;
        scrollbar-width: thin;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 0 0 220px !important;
        width: 220px !important;
        min-width: 220px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"] img {
        width: 100% !important;
        object-fit: contain !important;
        background-color: #161b22;
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
        height: 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {
        background: #111821;
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pcbScrollMarker) div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
        background: #44566c;
        border-radius: 8px;
    }

    .mainHeader {
        background: linear-gradient(90deg, #0f2238, #0b1118);
        border: 1px solid #1f3b5d;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    .mainTitle {
        color: #f8fafc;
        font-size: 34px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .mainSubtitle {
        color: #a8b4c4;
        font-size: 16px;
    }

    .stepDescription {
        color: #9aa9ba;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 14px;
    }

    .selectedBar {
        background-color: #101d2b;
        border: 1px solid #2b4a6b;
        border-radius: 12px;
        padding: 14px 16px;
        color: #e6edf3;
        font-size: 16px;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    .selectedPCBName {
        color: #58a6ff;
        font-weight: 700;
    }

    .comparisonTitleBlue {
        background-color: #1769c2;
        color: white;
        font-weight: 700;
        padding: 10px 12px;
        border-radius: 8px 8px 0 0;
    }

    .comparisonTitleGreen {
        background-color: #138a4a;
        color: white;
        font-weight: 700;
        padding: 10px 12px;
        border-radius: 8px 8px 0 0;
    }

    .metricCard {
        background-color: #121c28;
        border: 1px solid #27384b;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
    }

    .metricLabel {
        color: #8f9dac;
        font-size: 14px;
        margin-bottom: 6px;
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
        min-height: 44px;
        border-radius: 9px;
        font-weight: 650;
    }

    div.stDownloadButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 9px;
        font-weight: 650;
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

if "uploaded_pcb_bytes" not in st.session_state:
    st.session_state.uploaded_pcb_bytes = None

if "uploaded_pcb_name" not in st.session_state:
    st.session_state.uploaded_pcb_name = None

if "show_selected_pcb" not in st.session_state:
    st.session_state.show_selected_pcb = False


def select_benchmark_pcb(pcb_name):
    if st.session_state.selected_pcb != pcb_name:
        st.session_state.selected_pcb = pcb_name
        st.session_state.inspection_result = None
        st.session_state.show_selected_pcb = False
        st.session_state.uploaded_pcb_bytes = None
        st.session_state.uploaded_pcb_name = None


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


def make_gallery_thumbnail(img, size = (320, 220)):
    img = img.convert("RGB")

    canvas_width, canvas_height = size

    # Resize the PCB without changing its aspect ratio
    img.thumbnail(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS
    )

    # Place every PCB on an identical fixed-size canvas
    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (22, 27, 34)
    )

    x = (canvas_width - img.width) // 2
    y = (canvas_height - img.height) // 2

    canvas.paste(
        img,
        (x, y)
    )

    return canvas


def image_to_bytes(img, image_format = "PNG"):
    buffer = BytesIO()
    img.save(buffer, format = image_format)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data
def get_gallery_thumbnails(pcb_path_list):
    thumbnails = []

    for path_text in pcb_path_list:
        pcb = Image.open(Path(path_text)).convert("RGB")
        thumbnails.append(
            make_gallery_thumbnail(pcb)
        )

    return thumbnails


def align_inspection_image(benchmark, inspection):
    benchmark_height, benchmark_width = benchmark.shape[:2]

    inspection = cv.resize(
        inspection,
        (benchmark_width, benchmark_height),
        interpolation = cv.INTER_AREA
    )

    benchmark_gray = cv.cvtColor(benchmark, cv.COLOR_RGB2GRAY)
    inspection_gray = cv.cvtColor(inspection, cv.COLOR_RGB2GRAY)

    orb = cv.ORB_create(nfeatures = 3000)

    benchmark_keypoints, benchmark_descriptors = orb.detectAndCompute(
        benchmark_gray,
        None
    )

    inspection_keypoints, inspection_descriptors = orb.detectAndCompute(
        inspection_gray,
        None
    )

    if benchmark_descriptors is None or inspection_descriptors is None:
        return inspection, False

    if len(benchmark_keypoints) < 8 or len(inspection_keypoints) < 8:
        return inspection, False

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(inspection_descriptors, benchmark_descriptors)
    matches = sorted(matches, key = lambda match: match.distance)
    matches = matches[:min(300, len(matches))]

    if len(matches) < 8:
        return inspection, False

    inspection_points = np.float32([
        inspection_keypoints[match.queryIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    benchmark_points = np.float32([
        benchmark_keypoints[match.trainIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv.findHomography(
        inspection_points,
        benchmark_points,
        cv.RANSAC,
        5.0
    )

    if homography is None:
        return inspection, False

    aligned = cv.warpPerspective(
        inspection,
        homography,
        (benchmark_width, benchmark_height)
    )

    return aligned, True


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

    # Convert both images to grayscale
    benchmark_gray = cv.cvtColor(
        benchmark,
        cv.COLOR_RGB2GRAY
    )

    inspection_gray = cv.cvtColor(
        inspection,
        cv.COLOR_RGB2GRAY
    )

    # Compare benchmark and inspection PCB
    difference_gray = cv.absdiff(
        benchmark_gray,
        inspection_gray
    )

    # Reduce small image noise
    difference_gray = cv.GaussianBlur(
        difference_gray,
        (5, 5),
        0
    )

    # Detect changed pixels
    _, mask = cv.threshold(
        difference_gray,
        10,
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
        8,
        int(mask.shape[0] * mask.shape[1] * 0.00001)
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


def crop_defect_region(img, box):
    x, y, width, height = box
    img_height, img_width = img.shape[:2]

    centre_x = x + width // 2
    centre_y = y + height // 2

    largest_side = max(width, height)
    padding = max(15, int(largest_side * 0.3))
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

        if result["is_uncertain"]:
            label = (
                f'{i}. Uncertain - {result["predicted_class"]} '
                f'{result["confidence"] * 100:.1f}%'
            )
        else:
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


def create_pdf_report(
    pcb_name,
    benchmark_img,
    uploaded_img,
    results,
    annotated_img,
    alignment_used
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

    uncertain_count = sum(
        result["is_uncertain"]
        for result in results
    )

    summary_data = [
        ["PCB Reference", pcb_name],
        ["Detected Defect Regions", str(len(results))],
        ["Uncertain Regions", str(uncertain_count)],
        ["Model", "CS-ResNet - Baseline"],
        ["Uncertainty Threshold", f"{UNCERTAIN_THRESHOLD * 100:.0f}%"],
        [
            "Image Alignment",
            "Automatic ORB alignment applied"
            if alignment_used
            else "Automatic alignment not applied"
        ],
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
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    content.append(summary_table)
    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            "Predictions below the confidence threshold are flagged as "
            "Uncertain. This is a low-confidence warning and does not "
            "guarantee detection of unknown defect types.",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 15))

    # Show the benchmark and original uploaded PCB for traceability
    content.append(
        Paragraph(
            "Inspection Images",
            styles["Heading2"]
        )
    )

    benchmark_buffer = image_to_buffer(
        benchmark_img
    )

    benchmark_width, benchmark_height = get_pdf_image_size(
        benchmark_img,
        7.8 * cm,
        5.5 * cm
    )

    benchmark_pdf = PDFImage(
        benchmark_buffer,
        width = benchmark_width,
        height = benchmark_height
    )

    uploaded_buffer = image_to_buffer(
        uploaded_img
    )

    uploaded_width, uploaded_height = get_pdf_image_size(
        uploaded_img,
        7.8 * cm,
        5.5 * cm
    )

    uploaded_pdf = PDFImage(
        uploaded_buffer,
        width = uploaded_width,
        height = uploaded_height
    )

    image_table = Table(
        [
            ["Benchmark PCB", "Uploaded PCB for Inspection"],
            [benchmark_pdf, uploaded_pdf]
        ],
        colWidths = [8.2 * cm, 8.2 * cm]
    )

    image_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    content.append(image_table)
    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Detected Defect Summary",
            styles["Heading2"]
        )
    )

    defect_data = [
        [
            "Region",
            "Result",
            "Top Prediction",
            "Confidence",
            "Inference"
        ]
    ]

    for i, result in enumerate(results, start = 1):
        defect_data.append([
            str(i),
            result["display_class"],
            result["predicted_class"],
            f'{result["confidence"] * 100:.2f}%',
            f'{result["inference_time"]:.2f} ms'
        ])

    defect_table = Table(
        defect_data,
        colWidths = [1.4 * cm, 3.3 * cm, 4.2 * cm, 3.2 * cm, 3.5 * cm]
    )

    defect_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    content.append(defect_table)
    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Final Annotated PCB",
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
        content.append(Spacer(1, 15))

        if result["is_uncertain"]:
            region_title = (
                f"Region {i}: Uncertain - Highest prediction "
                f"{result['predicted_class']} "
                f"({result['confidence'] * 100:.2f}%)"
            )
        else:
            region_title = (
                f"Region {i}: {result['predicted_class']} "
                f"({result['confidence'] * 100:.2f}%)"
            )

        content.append(
            Paragraph(
                region_title,
                styles["Heading3"]
            )
        )

        crop_buffer = image_to_buffer(
            result["crop"]
        )

        crop_width, crop_height = get_pdf_image_size(
            result["crop"],
            7 * cm,
            5 * cm
        )

        crop_pdf = PDFImage(
            crop_buffer,
            width = crop_width,
            height = crop_height
        )

        processed_buffer = image_to_buffer(
            result["processed"]
        )

        processed_width, processed_height = get_pdf_image_size(
            result["processed"],
            7 * cm,
            5 * cm
        )

        processed_pdf = PDFImage(
            processed_buffer,
            width = processed_width,
            height = processed_height
        )

        region_image_table = Table(
            [
                ["Detected Region", "Model Input 224 x 224"],
                [crop_pdf, processed_pdf]
            ],
            colWidths = [8 * cm, 8 * cm]
        )

        region_image_table.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
            ])
        )

        content.append(region_image_table)
        content.append(Spacer(1, 8))

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
    """
    <div class="mainHeader">
        <div class="mainTitle">PCB Defect Inspection System</div>
        <div class="mainSubtitle">
            Detect and classify PCB defects using image processing and deep learning.
        </div>
    </div>
    """,
    unsafe_allow_html = True
)


# ============================================================
# Model Availability
# ============================================================

model_available = BEST_MODEL_PATH.exists()

if not model_available:
    st.info(
        "The trained model is not available yet. "
        "PCB selection and image upload can still be tested."
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
    'Click a PCB image to select it. All benchmark thumbnails are displayed at the same size.'
    '</div>',
    unsafe_allow_html = True
)

pcb_path_list = [
    str(path)
    for path in pcb_images
]

gallery_images = get_gallery_thumbnails(
    tuple(pcb_path_list)
)

if st.session_state.selected_pcb is None:
    st.session_state.selected_pcb = pcb_images[0].name

with st.container(border = True):
    st.markdown(
        '<div class="pcbScrollMarker"></div>',
        unsafe_allow_html = True
    )

    gallery_cols = st.columns(
        len(gallery_images),
        gap = "small"
    )

    for i, col in enumerate(gallery_cols):
        pcb_name = pcb_images[i].name
        is_selected = st.session_state.selected_pcb == pcb_name

        with col:
            st.image(
                gallery_images[i],
                use_container_width = True
            )

            st.button(
                f'✓ {pcb_name}' if is_selected else pcb_name,
                key = f'pcb_select_{i}',
                type = "primary" if is_selected else "secondary",
                use_container_width = True,
                on_click = select_benchmark_pcb,
                args = (pcb_name,)
            )

selected_pcb = st.session_state.selected_pcb
selected_path = next(
    path
    for path in pcb_images
    if path.name == selected_pcb
)

pcb_img = Image.open(
    selected_path
).convert("RGB")

st.markdown(
    f'<div class="selectedBar">'
    f'✅ &nbsp; Selected PCB: '
    f'<span class="selectedPCBName">{selected_pcb}</span>'
    f'</div>',
    unsafe_allow_html = True
)

view_col, download_col = st.columns(2)

with view_col:
    if st.button(
        "👁 View Selected PCB",
        use_container_width = True
    ):
        st.session_state.show_selected_pcb = (
            not st.session_state.show_selected_pcb
        )

with download_col:
    suffix = selected_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        image_format = "JPEG"
        mime_type = "image/jpeg"
    else:
        image_format = "PNG"
        mime_type = "image/png"

    st.download_button(
        "⬇ Download Benchmark",
        data = image_to_bytes(
            pcb_img,
            image_format
        ),
        file_name = selected_pcb,
        mime = mime_type,
        use_container_width = True
    )

# Drawing colour hint
st.markdown(
    """
<div style="background-color: #101d2b; border: 1px solid #2b4a6b; border-radius: 10px; padding: 14px 16px; margin-top: 10px; margin-bottom: 10px;">

<div style="color: #f8fafc; font-weight: 700; margin-bottom: 10px;">
🎨 Suggested PCB Drawing Colours
</div>

<div style="color: #9aa9ba; font-size: 14px; margin-bottom: 12px;">
When editing the downloaded benchmark PCB, use one of these dark green colours to simulate PCB defects.
</div>

<div style="display: flex; gap: 24px; flex-wrap: wrap;">

<div style="display: flex; align-items: center; gap: 10px;">
<div style="width: 38px; height: 38px; background-color: #183E0C; border: 1px solid #64748b; border-radius: 6px;"></div>
<div style="color: #e6edf3; font-size: 14px;">
<b>Dark Green</b><br>
HEX: #183E0C<br>
RGB: (24, 62, 12)
</div>
</div>

<div style="display: flex; align-items: center; gap: 10px;">
<div style="width: 38px; height: 38px; background-color: #306E1E; border: 1px solid #64748b; border-radius: 6px;"></div>
<div style="color: #e6edf3; font-size: 14px;">
<b>Light Green</b><br>
HEX: #306E1E<br>
RGB: (48, 110, 30)
</div>
</div>

</div>
</div>
    """,
    unsafe_allow_html = True
)

if st.session_state.show_selected_pcb:
    st.markdown(
        "### Selected Benchmark PCB"
    )

    preview_left, preview_middle, preview_right = st.columns(
        [0.05, 0.9, 0.05]
    )

    with preview_middle:
        st.image(
            pcb_img,
            caption = selected_pcb,
            use_container_width = True
        )


# ============================================================
# Step 2 and Step 3
# ============================================================

upload_col, preview_col = st.columns(
    [0.8, 1.2],
    gap = "large"
)


# ============================================================
# Step 2 - Upload PCB for Inspection
# ============================================================

with upload_col:
    st.markdown(
        "## 2. Upload PCB for Inspection"
    )

    st.markdown(
        '<div class="stepDescription">'
        'Upload a PCB image with visible defects to compare with the selected benchmark.'
        '</div>',
        unsafe_allow_html = True
    )

    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type = [
            "jpg",
            "jpeg",
            "png",
            "bmp"
        ],
        accept_multiple_files = False,
        key = "inspection_uploader"
    )

    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.getvalue()

            if len(uploaded_bytes) == 0:
                st.error(
                    "The selected image file is empty."
                )
            else:
                Image.open(
                    BytesIO(uploaded_bytes)
                ).convert("RGB")

                st.session_state.uploaded_pcb_bytes = uploaded_bytes
                st.session_state.uploaded_pcb_name = uploaded_file.name

        except Exception as error:
            st.error(
                "The uploaded file could not be read as an image."
            )
            st.code(
                str(error)
            )

    inspection_img = None

    if st.session_state.uploaded_pcb_bytes is not None:
        try:
            inspection_img = Image.open(
                BytesIO(
                    st.session_state.uploaded_pcb_bytes
                )
            ).convert("RGB")

            st.success(
                f'Uploaded PCB: {st.session_state.uploaded_pcb_name}'
            )

        except Exception:
            inspection_img = None
            st.session_state.uploaded_pcb_bytes = None
            st.session_state.uploaded_pcb_name = None


# ============================================================
# Step 3 - PCB Comparison Preview
# ============================================================

with preview_col:
    st.markdown(
        "## 3. PCB Comparison Preview"
    )

    st.markdown(
        '<div class="stepDescription">'
        'Preview the selected benchmark and uploaded PCB before inspection.'
        '</div>',
        unsafe_allow_html = True
    )

    compare_col1, compare_col2 = st.columns(2)

    with compare_col1:
        st.markdown(
            f'<div class="comparisonTitleBlue">'
            f'Benchmark PCB ({selected_pcb})'
            f'</div>',
            unsafe_allow_html = True
        )

        st.image(
            pcb_img,
            use_container_width = True
        )

    with compare_col2:
        upload_name = (
            st.session_state.uploaded_pcb_name
            if st.session_state.uploaded_pcb_name
            else "No image uploaded"
        )

        st.markdown(
            f'<div class="comparisonTitleGreen">'
            f'Uploaded PCB ({upload_name})'
            f'</div>',
            unsafe_allow_html = True
        )

        if inspection_img is not None:
            st.image(
                inspection_img,
                use_container_width = True
            )
        else:
            placeholder = Image.new(
                "RGB",
                pcb_img.size,
                (17, 24, 33)
            )

            st.image(
                placeholder,
                use_container_width = True
            )


# ============================================================
# Step 4 - Start Inspection
# ============================================================

st.markdown(
    "## 4. Start Inspection"
)

st.markdown(
    '<div class="stepDescription">'
    'Perform image alignment, defect localisation and CS-ResNet classification.'
    '</div>',
    unsafe_allow_html = True
)

inspect_col1, inspect_col2, inspect_col3 = st.columns([1, 1.5, 1])

with inspect_col2:
    inspect_button = st.button(
        "⚙ Inspect and Classify Defects",
        type = "primary",
        use_container_width = True,
        disabled = inspection_img is None
    )


# ============================================================
# Run Inspection
# ============================================================

if inspect_button:
    st.session_state.inspection_result = None

    if not model_available:
        st.warning(
            "The trained model is not available. "
            "Please add the best trained model before classification."
        )

    elif inspection_img is None:
        st.warning(
            "Please upload a PCB image before inspection."
        )

    else:
        benchmark_img = np.array(
            pcb_img
        ).astype(np.uint8)

        uploaded_img = np.array(
            inspection_img
        ).astype(np.uint8)

        with st.spinner(
            "Aligning and comparing PCB images..."
        ):
            aligned_img, alignment_used = align_inspection_image(
                benchmark_img,
                uploaded_img
            )

            defect_boxes, difference_mask = detect_defect_regions(
                benchmark_img,
                aligned_img
            )

        if len(defect_boxes) == 0:
            st.warning(
                "No clear defect region was detected. "
                "Please make sure the uploaded PCB matches the selected "
                "benchmark and contains a visible defect."
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
                results = []

                with st.spinner(
                    f'Classifying {len(defect_boxes)} detected region(s)...'
                ):
                    for box in defect_boxes:
                        defect_crop = crop_defect_region(
                            aligned_img,
                            box
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

                        is_uncertain = confidence < UNCERTAIN_THRESHOLD

                        display_class = (
                            "Uncertain"
                            if is_uncertain
                            else predicted_class
                        )

                        results.append({
                            "box": box,
                            "predicted_class": predicted_class,
                            "display_class": display_class,
                            "is_uncertain": is_uncertain,
                            "confidence": confidence,
                            "inference_time": inference_time,
                            "top3": top3,
                            "crop": defect_img,
                            "processed": processed_img
                        })

                annotated_img = annotate_defects(
                    aligned_img,
                    results
                )

                pdf_report = create_pdf_report(
                    selected_pcb,
                    benchmark_img,
                    uploaded_img,
                    results,
                    annotated_img,
                    alignment_used
                )

                st.session_state.inspection_result = {
                    "pcb_name": selected_pcb,
                    "results": results,
                    "annotated_img": annotated_img,
                    "difference_mask": difference_mask,
                    "aligned_img": aligned_img,
                    "alignment_used": alignment_used,
                    "pdf_report": pdf_report,
                    "report_filename": (
                        "PCB_Inspection_Report_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S")
                        + ".pdf"
                    )
                }


# ============================================================
# Inspection Result
# ============================================================

inspection_result = st.session_state.inspection_result

if inspection_result is not None:
    st.divider()

    st.markdown(
        "## 5. Inspection Result"
    )

    st.info(
        f'Predictions below {UNCERTAIN_THRESHOLD * 100:.0f}% confidence '
        'are flagged as Uncertain.'
    )

    results = inspection_result["results"]
    annotated_img = inspection_result["annotated_img"]

    total_inference = sum(
        result["inference_time"]
        for result in results
    )

    unique_classes = sorted(
        set(
            result["display_class"]
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

    if inspection_result["alignment_used"]:
        st.success(
            "Automatic image alignment was applied before comparison."
        )
    else:
        st.info(
            "The uploaded PCB was resized to the benchmark size. "
            "Automatic feature alignment was not applied."
        )

    with st.expander(
        "View Aligned Uploaded PCB"
    ):
        st.image(
            inspection_result["aligned_img"],
            use_container_width = True
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
            "Result": result["display_class"],
            "Top Prediction": result["predicted_class"],
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
        if result["is_uncertain"]:
            region_title = (
                f'Region {i} - Uncertain | Highest: '
                f'{result["predicted_class"]} '
                f'({result["confidence"] * 100:.2f}%)'
            )
        else:
            region_title = (
                f'Region {i} - {result["predicted_class"]} '
                f'({result["confidence"] * 100:.2f}%)'
            )

        with st.expander(
            region_title
        ):
            crop_col, input_col, prob_col = st.columns(
                [1, 1, 1.2]
            )

            with crop_col:
                st.markdown(
                    "#### Detected Region"
                )

                display_crop = result["crop"].resize(
                    (224, 224),
                    Image.Resampling.NEAREST
                )

                st.image(
                    display_crop,
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

                if result["is_uncertain"]:
                    st.warning(
                        "Low-confidence classification. "
                        "The highest model prediction is shown below."
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

    st.download_button(
        label = "Download PDF Inspection Report",
        data = inspection_result["pdf_report"],
        file_name = inspection_result["report_filename"],
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
