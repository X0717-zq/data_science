# Six-stage workflow: content and layout review

The latest six-part user specification is the content authority. The previously approved pastel horizontal bands, dashed boundaries and rectangular modules are retained.

## Coverage

| Stage | Included content |
| --- | --- |
| 1. Dataset construction and preparation | Raw transaction cleaning and feature organization; all normal transactions retained; a random subset of original fraud transactions; fraud prevalence below 1%; existing 9,943 observations and eight raw fields. |
| 2. Dataset split | Five repeated stratified random holdouts; 70/15/15 proportions; shared frozen indices; training-fitted encoding/scaling applied to validation and test. |
| 3. Imbalance handling | Training-only plain, class-weight, SMOTE, ADASYN, NearMiss, SMOTE+Tomek and SMOTE+ENN treatments. |
| 4. Model construction | RF traditional ensemble baseline and MLP neural classifier; both crossed with the same seven strategies. Existing EasyEnsemble and Isolation Forest comparisons retained as separate controls. |
| 5. Selection and thresholds | Validation PR-AUC for model comparison and MLP best-epoch selection; validation F2 for thresholds, larger-threshold tie-break; frozen models and thresholds before testing. |
| 6. Evaluation and errors | Independent-test PR-AUC/AP/ROC-AUC and operating-point metrics; PR/ROC curves, confusion matrices, learning curves and transaction-level TP/FP/TN/FN cases; five-round mean and sample SD. Learning curves explicitly use only training and validation data. |

## Method correction

The phrase “PR-AUC as a threshold-selection criterion” was corrected. PR-AUC is an aggregate of a precision-recall curve across thresholds; selecting one decision threshold needs an operating-point criterion. This diagram uses the established validation F2 criterion. The existing `nn_comparison_config.json` and `run_nn_rf_comparison.py` were read without modification and confirm validation PR-AUC checkpoint selection, validation F2 thresholds, and the larger-threshold tie-break. The figure does not claim that a fixed-parameter RF has per-epoch checkpoint selection.

The previous 225-fit label describes only the original RF nine-method learning-curve analysis. It is omitted from this broader RF/MLP workflow to avoid attributing that count to the combined experiment. The earlier figures remain intact.

## Verification

- All six requested stages and 26 required content terms were checked in the exported SVG and PDF.
- All 61 text items were found in both vector exports.
- Rendered visual review: all six stages readable; no clipping or text/connector overlap.
- PDF collision audit: PASS, 0 failures and 0 warnings. The 97 contained-fill overlays are intentional labels within diagram backgrounds.
- PDF minimum font size: 9.1 pt. Minimum measured text-to-box clearance: 2.96 pt.
- Equal-row boxes were checked for matching sizes, alignment and gutters. This is a single-axis schematic, so the plot-area alignment audit is not applicable.
- PNG: 3,733 x 3,933 pixels, 400 dpi. Editable-text PDF and SVG included.
- Source preflight: 0 failures. Font-size, DPI and dimensions warnings arise because those values are in the separate JSON configuration; they were checked in the output. The PNG-without-TIFF warning is accepted because this is a report illustration with vector originals, not a journal TIFF submission.
- No model was rerun, no original observation was changed, and no report prose was overwritten.
