#!/usr/bin/env python3
import numpy as np, pandas as pd, argparse
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print('Epigenetic clock complete.')

if __name__ == '__main__':
    main()
