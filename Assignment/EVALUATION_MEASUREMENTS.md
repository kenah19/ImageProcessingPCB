# PCB inspection evaluation measurements

This guide describes the containment-based evaluation in `PCB Defect Inspection System - Dual Evaluation.ipynb`.

## 1. What is evaluated?

| Evaluation | Input | Question answered |
|---|---|---|
| XML-crop classification | Existing crops extracted using XML annotations | Can the model classify a defect when its location is already supplied? |
| Mask localisation | Full-board image, clean reference and XML annotations | Does the difference mask locate the annotated defects? |
| Full pipeline | Mask-derived crops passed to each model | Does the system both locate and correctly classify defects? |

XML annotations are reference answers, not model detections. The mask has no defect-class predictions. Its per-category results use the XML labels to identify which kinds of annotated defects were localised.

Validation uses board IDs **09 and 10**. Testing uses **11 and 12**. The `boards` result counts full-board image files, not distinct board designs. The `samples` result counts processed XML crops; multiple defects can occur in one board image.

## 2. Containment and coverage

Let:

- **R** be the actual white pixels belonging to one detected mask region (mask rectangle).
- **B** be an XML bounding rectangle after alignment-coordinate transformation.
- **I** be the number of pixels from R inside B (mask-pixels).
- **A(R)** be the total pixel count of R (mask-region pixels).
- **A(B)** be the area of the XML rectangle.

### Containment

`Containment = I / A(R)`

Containment answers: **How much of this detected region lies inside the annotation?**

A containment of 0.80 means 80% of the region's pixels are inside the XML box. A small region can have high containment, even if it covers very little of the annotation.

### Coverage

`Coverage = I / A(B)`

Coverage answers: **How much of the annotated rectangle contains pixels from this detected region?**

A coverage of 0.10 means the region occupies 10% of the XML rectangle. The denominator includes background within that rectangle. Requiring high coverage may reject valid, tightly localised defects when annotations include surrounding context.

### Intersection pixels

`Intersection pixels = I`

This is the absolute number of region pixels inside the XML box. It is saved for inspection but **has no minimum threshold in the current matching rule**. It remains necessary to calculate containment and coverage.

### Current matching thresholds

```python
CONTAINMENT_THRESHOLD = 0.80
COVERAGE_THRESHOLD = 0.10
```

A region–annotation pair qualifies only when both conditions pass:

```python
qualifies = containment >= 0.80 and coverage >= 0.10
```

For example, suppose an XML box has area 5,000 pixels, a region contains 1,000 pixels, and 950 lie inside the box:

| Measurement | Calculation | Result |
|---|---|---|
| Containment | 950 / 1,000 | 95% — passes |
| Coverage | 950 / 5,000 | 19% — passes |
| Intersection pixels | 950 | Reported only |

### How this differs from IoU

`IoU = intersection area / union area`

For a mask region and XML box, the union is `A(R) + A(B) - I`. In the example above, mask-region/box IoU is `950 / (1,000 + 5,000 - 950)`, or about 18.81%.

Coverage is not IoU. They are equal only when the entire mask region lies inside the XML box. The earlier notebook evaluation used **bounding-box IoU**, which also counts background inside the detected rectangle. The current evaluator does not use IoU to decide matches.

## 3. Region construction and matching

The notebook forms connected components from the processed binary mask, filters small components and groups nearby accepted components using the existing box-merging rule. One evaluated region can therefore contain several nearby fragments.

Matching uses these regions' actual pixels. The notebook's **20% bounding-box expansion affects display and classifier crops only**; it does not add white pixels or improve containment/coverage scores.

XML boxes are transformed when the image is aligned, then represented as axis-aligned rectangles. Pixel centres determine whether pixels lie inside each rectangle. Coverage retains the full rectangle area, including any portion outside the aligned image, so alignment failures are not silently excluded.

Among qualifying pairs, the matcher first maximises the number of **one-to-one matches**, then favours higher total mean containment and coverage. Predicted classes do not influence matching.

- A region can match at most one annotation.
- An annotation can match at most one region.
- A qualifying candidate may remain unmatched when another pairing is selected.
- Extra fragments can become false positives; a merged region covering multiple annotations can leave annotations unmatched.

## 4. Mask localisation measurements

| Measurement | Definition | Interpretation |
|---|---|---|
| True positive (TP) | A selected region–annotation match | An annotated defect was successfully localised under the matching rule. |
| False positive (FP) | A detected region with no selected match | An extra or insufficiently overlapping detection. |
| False negative (FN) | An annotation with no selected match | An expected defect that was not successfully localised. |
| Precision | `TP / (TP + FP)` | Fraction of detected regions that successfully match annotations. |
| Recall | `TP / (TP + FN)` | Fraction of annotated defects successfully localised. |
| F1 | `2*TP / (2*TP + FP + FN)` | Balance between precision and recall. |
| Support | `TP + FN` | Number of annotated defects. |

An FP does not always mean noise: a real defect region can fail the overlap conditions. One failed region–annotation pair can create **both one FP and one FN**.

Example: 80 matches, 20 extra detections and 40 missed annotations give:

- Precision: `80 / 100 = 80%`.
- Recall: `80 / 120 = 66.67%`.
- F1: `160 / 220 = 72.73%`.

The mask-only category table reports **support, localised count, missed count and recall** for each XML category. Per-category mask precision/F1 are omitted because unmatched regions have no known predicted defect class.

### Mask confusion matrix

| Actual / Predicted | Detected region | Missed |
|---|---:|---:|
| Defect | TP | FN |
| No matched annotation | FP | N/A |

This is a detection count matrix. **True negatives are undefined** because the evaluation does not enumerate defect-free background units. The bottom-right cell is N/A, not zero. Mask accuracy, specificity and a background-class score are consequently not reported.

## 5. Model classification measurements

### XML crops

The model receives an existing XML-derived crop. The ordinary six-by-six confusion matrix uses actual classes as rows and predicted classes as columns:

- Diagonal cells: correct classifications.
- Off-diagonal cells: one defect class mistaken for another.
- Accuracy: correctly classified crops divided by all evaluated XML crops.
- Per-class precision: correct predictions of a class divided by all predictions of that class.
- Per-class recall: correctly classified crops of a class divided by all actual crops of that class.
- Per-class F1: balance between that class's precision and recall.
- Support: number of actual crops in that class.

Strong XML-crop accuracy does not establish successful full-board localisation.

### Full pipeline

Per-class counts require both successful localisation and the correct class:

| Outcome | Effect |
|---|---|
| Matched region, correct class | One TP for that class |
| Matched region, wrong class | One FP for the predicted class and one FN for the actual class |
| Unmatched detected region | One FP for its predicted class |
| Unmatched annotation | One FN for its actual class |

A wrong-class match remains a **localisation TP**. This explains why mask detection scores can be high while class-aware scores are lower.

The augmented confusion matrix has six actual-class rows plus a **No matched annotation** row, and six predicted-class columns plus a **Missed** column:

- Main diagonal: correctly localised and classified defects.
- Off-diagonal class cells: wrong classifications of matched regions.
- Last row: extra detections, grouped by predicted class.
- Last column: missed annotations, grouped by actual class.
- Bottom-right: N/A; there is no measured true-negative count.

### Micro, macro and weighted averages

| Average | Calculation | Use |
|---|---|---|
| Micro | Sum TP, FP and FN across classes, then calculate precision/recall/F1 | Overall performance with more influence from frequent classes |
| Macro | Arithmetic mean of the six per-class scores | Gives every defect class equal importance |
| Weighted | Mean of per-class scores weighted by actual support | Reflects the class distribution of the evaluated data |

Support means annotation count for the full pipeline and crop count for XML classification. Macro F1 averages the per-class F1 values; it is not calculated from macro precision and macro recall.

### Matched-only accuracy

`Matched-only accuracy = correctly classified matched regions / all matched regions`

If nine regions matched and all nine were classified correctly, this is 100%. It excludes all unmatched detections and missed annotations, so it **is not overall pipeline accuracy**. Always interpret it alongside detection recall and class-aware precision/recall/F1.

## 6. Reading the saved outputs

Results are saved separately for validation and test:

```text
Dual_Evaluation_Results/containment/<val-or-test>/
    mask_detection/
        classification_report.csv
        confusion_matrix.csv
        confusion_matrix.png
        per_category_localisation.csv
    models/<model>/
        xml_classification_report.csv
        xml_confusion_matrix.csv
        xml_confusion_matrix.png
        pipeline_classification_report.csv
        pipeline_confusion_matrix.csv
        pipeline_confusion_matrix.png
    mask_overlap_pairs.csv
    summary.csv
    final_benchmark_summary.csv
    benchmark_log.txt
    METRICS.md
    settings.json
```

- `mask_overlap_pairs.csv`: containment, coverage and pixel intersections for all candidate region–annotation pairs, including failures. Positive overlap alone does not prove a selected match.
- Per-model `*_pipeline_events.csv` files: selected matches, extra detections, missed annotations and predicted classes.
- Per-model `*_pipeline_boards.csv` files: image-level annotation/detection/match counts and alignment status.
- `benchmark_log.txt`: completed-run summary, thresholds and metric explanations.
- `settings.json`: matching source, thresholds and checkpoint hashes used for the run.

Summary column mapping:

| Column | Meaning |
|---|---|
| `accuracy` | XML-crop classification accuracy; not populated for detection rows |
| `macro_f1` | XML classification macro F1 or full-pipeline class-aware macro F1, depending on the row |
| `detection_tp`, `detection_fp`, `detection_fn` | Class-agnostic localisation counts |
| `detection_precision`, `detection_recall`, `detection_f1` | Class-agnostic localisation scores |
| `class_micro_precision`, `class_micro_recall`, `class_micro_f1` | Full-pipeline scores pooled across defect classes |
| `matched_only_accuracy` | Classification accuracy restricted to selected localisation matches |

Detection counts repeat across model rows because the models share the same mask regions. They are not separate mask experiments for each model.

Proportions are stored on a **0–1 scale**: 0.95 means 95%. Undefined precision/recall/F1 denominators return 0, consistent with `zero_division=0`. Matched-only accuracy is NaN when no regions match. Other blank/NaN cells generally mean the measurement does not apply to that evaluation row.

## 7. Practical interpretation and limits

- Low detection precision: inspect extra regions, alignment artefacts and overlap failures.
- Low detection recall: inspect missed annotations, removed small defects and merged regions.
- High detection recall but low class-aware F1: inspect crop content and class confusions.
- High matched-only accuracy but low detection recall: the model classifies the successfully matched subset well, while the localisation stage remains a bottleneck.

These are fixed-threshold localisation and classification measurements. They are **not detection mAP or pixel-level segmentation accuracy**. XML rectangles cannot establish exact defect outlines.

Change thresholds using validation examples, then freeze them before testing. Lower thresholds can accept more valid regions but can also accept irrelevant changes inside annotated boxes. Compare saved settings when comparing runs, and do not directly treat the previous IoU scores and current containment scores as the same measurement. The test cell requires the matching configuration and checkpoints to agree with the completed validation run.
