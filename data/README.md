# IMDb Dataset Directory

This directory is intended to store the raw IMDb 50k Movie Reviews dataset file: `IMDB Dataset.csv`.

## Dataset Provenance & Attribution
- **Dataset Name**: Large Movie Review Dataset (v1.0)
- **Creators / Citation**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). *Learning Word Vectors for Sentiment Analysis*. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- **Kaggle Mirror**: [Kaggle IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **License**: Dataset is released for educational and research use.

## Git Tracking Policy
The raw CSV file (`IMDB Dataset.csv`) is ~66.2 MB. Because GitHub recommends keeping individual files under 50 MB (and strictly limits files to 100 MB), the raw dataset is excluded from version control via `.gitignore` to prevent repository bloat.

## Setup Instructions
1. Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place the uncompressed CSV into this directory:
   ```text
   data/
   └── IMDB Dataset.csv
   ```
3. Verify that the file contains exactly 50,000 rows across two columns (`review`, `sentiment`).
