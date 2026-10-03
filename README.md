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

The component READMEs provide the installation and usage instructions.

## Relationship to the paper

The reported experiments cover recognition, body–margin structure, cross-register transfer and deed segmentation. The combined FasterDAN model achieved 18.15% test CER. The R352 model applied to R335 without fine-tuning achieved 85.74% test CER.

The results files distinguish validation from test scores. Reported BERT test-line classification used PAGE-XML reference transcriptions.

The CSV files transcribe the accompanying manuscript. The code is supplied as companion implementation material.

Synthetic training requires the original font collection in addition to the data and checkpoints.

Manuscript images, PAGE-XML transcriptions, trained weights and underlying evaluation outputs are not bundled. Enquiries should be directed to [Charlie Abela](mailto:charlie.abela@um.edu.mt) or [Chukwuma Sidney Anih](mailto:chukwuma.anih.24@um.edu.mt).

See [CITATION.cff](CITATION.cff) for citation information. FasterDAN retains its [CeCILL-C licence](fasterdan/LICENSE_CECILL-C.md) and upstream attribution. Other materials retain the terms supplied with them; contact the authors where no licence is specified.
