# Interpreting the detection benchmark

All proportions are 0-1 (multiply by 100 for percent). Counts are regions or
annotations, not pixels or distinct board designs. `boards` counts image files.

## Mask localisation only
The mask has no defect-class predictions. A TP is a one-to-one region/annotation
match passing both containment and coverage thresholds.
FP is an unmatched detected region. FN is an unmatched annotation. A poorly
located region can create both an FP and an FN.
- Containment = intersecting mask pixels / all pixels in that detected region.
- Coverage = intersecting mask pixels / transformed XML rectangle area.
- Intersection pixels = actual white region pixels inside the XML rectangle;
  reported for inspection only, with no minimum-pixel matching condition.
- Precision = TP/(TP+FP): fraction of detected regions that qualify as real defects.
- Recall = TP/(TP+FN): fraction of annotated defects successfully localised.
- F1 = 2TP/(2TP+FP+FN): balance of precision and recall.
- Support = TP+FN: number of annotated defects.
- Per-category localisation recall uses the XML label, not a predicted class.
  Per-category mask precision/F1 are not defined: unmatched regions have no known
  class. They are deliberately omitted rather than assigned a class from a folder.
The 2x2 detection matrix has actual rows [Defect, No matched annotation] and
prediction columns [Detected region, Missed]. Its bottom-right entry is N/A.
True negatives cannot be counted without defining background evaluation units;
therefore mask accuracy, specificity and background classification scores are N/A.

## XML-crop classification (each model)
The six-by-six confusion matrix has actual classes on rows and predictions on
columns. Diagonal entries are correct classifications. Accuracy is correct crops
/ all XML crops. Precision, recall and F1 use the usual class-specific counts.
Support counts labelled crops. This assumes the defect location is already known.

## Full pipeline (each model)
The augmented matrix uses six actual-class rows plus a No matched annotation row,
and six predicted-class columns plus a Missed column. Off-diagonal class cells are
wrong classifications of matched regions. The last row contains extra detections
by predicted class; the last column contains missed annotations by actual class.
The last-row/last-column cell is N/A, not a true-negative count.
Per-class TP requires both a qualifying location and the correct class. A wrong
class is FP for its predicted class and FN for its actual class. Extra detections
are class FP; missed annotations are class FN. Support is actual annotation count.
- Micro precision/recall/F1 pool TP, FP and FN across the six classes.
- Macro averages give each of the six classes equal weight.
- Weighted averages weight each class by annotation support (or crop support for XML).
- Matched-only accuracy = correctly classified matched regions / matched regions.
  It excludes misses and extra detections and is not overall system accuracy.
- Detection metrics in model rows repeat the shared mask localisation result;
  classification does not change the mask's detected regions.
Undefined precision/recall/F1 denominators produce 0, matching zero_division=0.
Matched-only accuracy is NaN if no regions matched. Other NaN/blank entries mean
not applicable. These are fixed-threshold localisation scores, not detection mAP
or pixel-level segmentation accuracy. No model latency/GFLOPs is claimed here.

## Files and selection
mask_detection/ contains one class-agnostic report/matrix and category recall table.
Each models/<model>/ folder contains xml and pipeline report/matrix CSVs and PNGs.
benchmark_log.txt and final_benchmark_summary.csv summarise this completed split.
settings.json freezes thresholds, matching source and checkpoints. Tune validation
only and keep the settings fixed for test. New runs overwrite this split's files.
