# Figure 6-1 restyling verification

- Content authority: user reference image 1 and the original `flowchart()` function.
- Style authority: user reference image 2; pastel horizontal bands, dashed group boundaries, rectangular nodes and bold section labels.
- Scientific purpose: describe the existing frozen five-round fraud detection workflow; this is not a new experiment or a performance claim.
- Backend: Python/matplotlib. No data, fitted models, experimental results or original figure files were changed.
- Content: all 10 nodes plus the original protocol footer retained. Exact content comparison against the original source and exported SVG passed (ignoring layout whitespace only).
- Connectors: all original directed relationships retained; no test-to-training or test-to-selection feedback introduced.
- Exports: 612 x 452 pt; PNG at 400 dpi; editable-text SVG and PDF.

## Rendered review

| Group | Review |
| --- | --- |
| Data preparation | Dataset counts, eight original features, five repeated stratified holdouts, split proportions and frozen indices retained. |
| Dataset split | Equal-width training/validation/test boxes; sample and positive counts retained; aligned branch connectors. |
| Training and evaluation | Resampling scope, alternative fitting options, continuous scores, F2 tie-break rule, frozen threshold, test metrics and transaction outcomes retained. |
| Summary | Learning-curve percentages, nested-training/fixed-validation restriction, 225 fits, sample SD and FP/FN case review retained. |
| Protocol footer | Entire original constraint retained and readable. |

Rendered PDF collision audit: PASS, 0 FAIL and 0 WARN. The 56 contained-fill overlays are the intended labels inside node and group backgrounds. PDF minimum font: 9.2 pt. All node text clears its enclosing border by at least 3.49 pt. The single-axis panel-alignment check is not applicable; additional direct geometry assertions confirm equal repeated-node widths, heights and gutters.

Source preflight: 0 FAIL. Four static warnings were reviewed:

1. Font sizes come from the separate JSON configuration; actual PDF sizes are 9.2 pt or greater.
2. PNG rather than TIFF is intentional for a report illustration. PDF and SVG provide vector originals; no journal TIFF submission was requested.
3. PNG resolution comes from configuration and is 400 dpi.
4. Dimensions come from configuration, 612 x 452 pt. This is a report diagram rather than a journal-specific submission.

The original figure remains available as `fig6_1_workflow.*`. The new files can replace Figure 6-1 in the user's report; no chapter prose was edited.
