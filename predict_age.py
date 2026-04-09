#!/usr/bin/env python3
"""Epigenetic age predictor from DNA methylation data."""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def preprocess_betas(betas: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise beta values."""
    # Remove probes with > 10% missing
    threshold = betas.shape[1] * 0.1
    betas = betas.dropna(thresh=betas.shape[1] - threshold)

    # Impute remaining NAs with probe median
    betas = betas.fillna(betas.median(axis=1).to_dict())

    # M-value transformation for better statistical properties
    betas_clipped = betas.clip(0.001, 0.999)
    m_values = np.log2(betas_clipped / (1 - betas_clipped))

    return m_values


def train_clock(X: np.ndarray, ages: np.ndarray):
    """Train elastic net epigenetic clock."""
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 1, 50),
        cv=5, random_state=42, max_iter=10000
    )
    model.fit(X, ages)

    n_features = np.sum(model.coef_ != 0)
    print(f"Selected {n_features} CpG sites")
    print(f"Best alpha: {model.alpha_:.4f}, l1_ratio: {model.l1_ratio_:.2f}")

    return model


def evaluate(model, X, ages, output_dir):
    """Cross-validated evaluation and plots."""
    predicted = cross_val_predict(model, X, ages, cv=5)

    mae = mean_absolute_error(ages, predicted)
    r2 = r2_score(ages, predicted)
    print(f"MAE: {mae:.2f} years, R2: {r2:.3f}")

    # Age acceleration
    acceleration = predicted - ages

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(ages, predicted, alpha=0.6, s=20)
    axes[0].plot([ages.min(), ages.max()], [ages.min(), ages.max()],
                 "r--", linewidth=2)
    axes[0].set_xlabel("Chronological Age")
    axes[0].set_ylabel("Predicted Epigenetic Age")
    axes[0].set_title(f"Epigenetic Clock (MAE={mae:.1f}y, R²={r2:.3f})")

    axes[1].hist(acceleration, bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Age Acceleration (years)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Biological Age Acceleration")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "epigenetic_clock.png", dpi=150)
    plt.close()

    return predicted, acceleration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    betas = pd.read_csv(args.betas, index_col=0)
    meta = pd.read_csv(args.metadata, index_col=0)

    common = betas.columns.intersection(meta.index)
    betas = betas[common]
    ages = meta.loc[common, "age"].values

    m_values = preprocess_betas(betas)
    X = m_values.T.values

    model = train_clock(X, ages)
    predicted, acceleration = evaluate(model, X, ages, str(output_dir))

    results = pd.DataFrame({
        "sample": common,
        "chronological_age": ages,
        "predicted_age": predicted,
        "age_acceleration": acceleration,
    })
    results.to_csv(output_dir / "age_predictions.csv", index=False)
    print("Complete.")


if __name__ == "__main__":
    main()
