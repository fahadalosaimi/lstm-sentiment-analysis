# Local IMDb data

Place the downloaded CSV at `Data/IMDB Dataset.csv`. The original file is read-only
input; the improved pipeline never rewrites it. It expects UTF-8 CSV columns
`review,sentiment`, with `positive` and `negative` labels.

The local file was inspected: 50,000 records, 25,000 per class, and no missing
values in either column. Its SHA-256 is:

```text
dfc447764f82be365fa9c2beef4e8df89d3919e3da95f5088004797d79695aa2
```

Original benchmark: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/index.html),
introduced by Maas et al., *Learning Word Vectors for Sentiment Analysis* (ACL 2011).
The benchmark has an official 25,000/25,000 train/test split.

The project owner confirmed downloading this CSV from
[IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The same inspected SHA-256 was verified in the returned Colab run, and the local
file remains unchanged. Attribution is owner-confirmed; no publisher checksum was
available to independently authenticate the downloaded bytes.

Download manually from that page, extract IMDB Dataset.csv and place it at the
path above. Then run `python -B -m sentiment.data` for a read-only preparation
check, or use the Colab upload/validation stage. Do not commit the CSV or Kaggle
credentials. The Stanford archive is not a drop-in CSV; document conversion if
using a different source/version.

The publisher page lists **Other (specified in description)** as its license.
Redistribution permission is unresolved; the CSV remains excluded by default.
See [licensing review](../LICENSING.md). No permissive data license is asserted.

The CSV is excluded by `.gitignore`. Dataset redistribution rights are separate
from rights to this project's code. No download or credential setup is automatic.
