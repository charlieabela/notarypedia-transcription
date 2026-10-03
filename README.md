# Notarypedia transcription

Companion code and reported results for **Structured Full-Page Transcription and Deed Segmentation in Historical Notarial Manuscripts**.

**Authors:** Chukwuma Sidney Anih, Charlene Ellul, Colin Layfield, Vanessa Buhagiar, Joan Abela and Charlie Abela. University of Malta and the Notarial Archives Foundation.

The study investigates transcription and deed segmentation using 410 annotated pages from two seventeenth-century Maltese notarial registers, R352 and R335. FasterDAN produces full-page text with body–margin structure. A multilingual BERT classifier assigns line labels, and a deterministic decoder groups labelled lines into deeds.

## Materials

| Location | Contents |
|---|---|
| [fasterdan/](fasterdan/README.md) | Recognition code, notarial PAGE-XML formatter, synthetic generation, training and inference |
| [deed_segmenter/](deed_segmenter/README.md) | BERT training, inference and deed reconstruction |
| [results/](results/README.md) | Machine-readable values reported in the paper, with table references and evaluation partitions |

The component READMEs provide the original installation and usage instructions. Their references to `notarypedia-placements` identify the source repository. The training configurations remain in the original scripts.

## Relationship to the paper

The reported experiments cover recognition, body–margin structure, cross-register transfer and deed segmentation. The combined FasterDAN model achieved 18.15% test CER. The R352 model applied to R335 without fine-tuning achieved 85.74% test CER.

The results files distinguish validation from test scores. Reported BERT test-line classification used PAGE-XML reference transcriptions. Table 4 reports validation F1 and separate full-register reconstruction counts; these counts are not numbers of correctly recovered test deeds.

The CSV files transcribe the accompanying manuscript. They are not newly computed evaluation outputs. The copied code is supplied as companion implementation material; no experiments were rerun when preparing this repository.

## Source and access

The source files and existing documentation were copied **without modification** from:

- Repository: `charlieabela/notarypedia-placements`
- Branch: `fasterdan-htr`
- Commit: `e83e73afff2ba27c1451dc5f3fc0f90305d8d74c`

Only the two relevant components were selected. Unrelated applications, font binaries, an empty file and a duplicate ignore file were omitted. Existing code, requirements, citation information and licence files were preserved. Synthetic training requires the original font collection in addition to the data and checkpoints.

Manuscript images, PAGE-XML transcriptions, trained weights and underlying evaluation outputs are not bundled. Enquiries about access should be directed to [Charlie Abela](mailto:charlie.abela@um.edu.mt) or [Chukwuma Sidney Anih](mailto:chukwuma.anih.24@um.edu.mt).

See [CITATION.cff](CITATION.cff) for citation information. FasterDAN retains its [CeCILL-C licence](fasterdan/LICENSE_CECILL-C.md) and upstream attribution. Other materials retain the terms supplied with them; contact the authors where no licence is specified.
