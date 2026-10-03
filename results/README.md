# Reported results

These CSV files transcribe the latest manuscript supplied on 2 October 2026, **Structured Full-Page Transcription and Deed Segmentation in Historical Notarial Manuscripts**. No experiments were rerun or results recalculated for this repository.

| File | Source in the paper |
|---|---|
| [corpus.csv](corpus.csv) | Table 1: corpus statistics and FasterDAN page allocations |
| [recognition.csv](recognition.csv) | Tables 2, 3 and 5: recognition, transfer and adaptation |
| [segmentation.csv](segmentation.csv) | Section 4.3 and Table 4: line accuracy, validation F1 and full-register output counts |
| [figure5.csv](figure5.csv) | Figure 5: recognition within and across registers |

CER, WER, mAP-CER and LOER are percentages. Accuracy and F1 are fractions. Blank metric cells mean unreported, not zero. The READ-2016 CER is marked as approximate.

Validation scores and test scores are identified separately. The Figure 5 R352 HTR+ value is validation CER; the other five values are test CER. R335 partition names in FasterDAN transfer describe the evaluation material, not data used to train the R352 source model.

Table 1 contains FasterDAN allocations, not BERT partition counts. In the segmentation file, full-register predicted deed counts are separate from the validation F1 values and do not measure correct deed recovery. The input used for those aggregate counts is left unspecified where the table does not identify it.
