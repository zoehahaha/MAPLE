# Paper results

The PREP manuscript defines the results reported in this directory.
`tables/` contains Tables I-IV with the paper's displayed precision, signs,
units, and footnotes. In Table II, `F` denotes full oracle evaluation on
Snips and arXiv. Table III includes the reported mean Miss-PR percentages.

`figures/` contains the original Figures 3-6 exported from the paper.
The accompanying CSV files contain numbers explicitly printed in the figures
or the corresponding evaluation text:

- Figure 3: the matched PREP certification comparison described in Section III-C.
- Figure 4: the approximate values stated in Section III-E.
- Figure 5: the four percentage annotations printed above the boxplots, together
  with the eight runs per workload and positive savings reported in Section III-E.
- Figure 6: the values stated in Section III-F. `value_kind` distinguishes
  reported numbers, approximate numbers, and strict lower bounds.

Use the original figure images for the complete plotted results.
`prep report` exports these paper results to the requested output directory.
`prep replay` runs the implementation on a supplied offline workload and
records the resulting measurements separately.
