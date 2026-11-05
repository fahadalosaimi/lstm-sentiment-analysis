# Original experiment: historical outputs only

Source: `DLProject_ipynb2 (1).ipynb`, preserved without modification.
No original training was rerun. Cell numbers are one-based positions in the notebook.

| Item | Saved evidence |
|---|---|
| Raw data (cells 4, 10) | 50,000 reviews; 25,000 negative and 25,000 positive; no missing values |
| Split (cell 15) | 35,000 train / 7,500 validation / 7,500 test |
| Vocabulary / embedding | 214,620 words + padding; 100-dimensional embeddings |
| LSTM / dropout | 128 hidden units; one direction/layer; dropout 0.5 after LSTM |
| Training | Adam, learning rate 0.001, batch 64, 10 epochs |
| Selection (cell 24) | Epoch 10 by validation accuracy: 0.8581 |
| Test (cell 25) | Accuracy 0.8629; loss 0.5232 |
| Confusion matrix (cell 28) | `[[3188, 562], [466, 3284]]`; rows=true, columns=predicted; negative then positive |

Recalculation from saved counts: 6,472 / 7,500 = 0.8629333333 accuracy;
macro-F1 = 0.8629108727. These calculations validate internal consistency only.

At epoch 10 train accuracy was 0.9822 and validation accuracy 0.8581. Validation
loss reached 0.3858 at epoch 6 then rose to 0.5354 by epoch 10, indicating
overfitting. The original vocabulary used all texts before splitting, and its
LSTM consumed padded steps without explicit true lengths. Notebook cell execution
counts are null, and package versions, exact download provenance and trained
weights are unavailable. Consequently this historical score is not an unbiased,
independently reproduced benchmark or a result for the improved pipeline.

The improved implementation is a separate experiment: deduplication, vocabulary,
preprocessing, packed sequences and checkpoint selection differ. Its performance
must be measured in a future authorized run. No improved score is reported here.
