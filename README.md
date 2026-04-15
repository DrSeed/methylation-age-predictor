# DNA Methylation Age Predictor

> Your DNA has a clock. Specific CpG sites gain or lose methylation as you age in a remarkably predictable pattern. This pipeline builds an epigenetic clock and uses it to predict biological age, which can diverge from chronological age in ways that predict disease and mortality.

## Why Biological Age Matters

Two people can both be 50 chronologically but have very different biological ages. Accelerated epigenetic aging (biological > chronological) predicts higher risk of cardiovascular disease, cancer, and all-cause mortality.

## How It Works

1. Take beta values from Illumina 450K or EPIC arrays
2. Select the ~350 age-predictive CpG sites
3. Train elastic net regression (L1 + L2 regularisation)
4. The difference between predicted and actual age = age acceleration

## Interpreting Age Acceleration

Positive = epigenome looks older than expected (associated with obesity, smoking, pollution). Negative = looks younger (linked to fitness, caloric restriction). The beauty is that epigenetic age is modifiable, unlike your genome.

## Usage
```bash
python predict_age.py --betas data/beta_values.csv --metadata data/ages.csv
```
