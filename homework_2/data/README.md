# Supplied HW2 datasets

These five files are the unmodified data supplied with the corrected course
Homework 2 materials (posted October 28; provided here November 6, 2026):

- `taxi_trips.parquet`: 523,554 trips; seven standardized features and duration in minutes.
- `retail.csv`: 397,924 rows; `StockCode`, `CustomerID`, and positive `Quantity`.
- `train.csv`: 600 labeled toy observations.
- `val.csv`: 200 labeled toy observations.
- `test.csv`: 200 unlabeled toy observations, with columns `x1,x2` only.

`manifest.json` records the original file sizes and SHA-256 hashes. The experiment
runner checks every hash before fitting models. The assignment PDF embeds no
external dataset URLs; all required data are bundled here for account-free
reproduction. The files are attributed to the course-provided HW2 materials;
no unverified claim about their upstream preprocessing or provenance is made.
