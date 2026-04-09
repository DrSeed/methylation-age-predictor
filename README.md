# DNA Methylation Age Predictor

Implementation of epigenetic age prediction from Illumina 450K/EPIC methylation arrays.

## Features
- Beta-value preprocessing and normalisation
- Horvath clock CpG site extraction
- Elastic net training for age prediction
- Biological age acceleration calculation
- Correlation plots and residual analysis

## Usage
```bash
python predict_age.py --betas data/beta_values.csv --metadata data/ages.csv
```
