#!/usr/bin/env python3
"""
predict.py
----------
Reads a CSV, performs EDA (stats, missingness, distributions, correlations, etc.),
creates charts, and exports a single PDF report with proper headings.

Usage:
    python predict.py --input data.csv --output report.pdf
"""

import argparse
import textwrap
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import scatter_matrix
import sys
print(sys.executable)

plt.switch_backend("Agg")  # For headless environments


def load_csv_to_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include=[np.number]).T
    desc["missing"] = df[desc.index].isna().sum()
    return desc


def detect_categoricals(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    cats = []
    for col in df.columns:
        if df[col].dtype == "object":
            cats.append(col)
        else:
            if df[col].nunique(dropna=True) <= max_unique:
                cats.append(col)
    return cats


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    wrapped = textwrap.fill(body, width=110)
    ax.text(0.02, 0.95, title, fontsize=18, fontweight="bold", va="top")
    ax.text(0.02, 0.90, wrapped, fontsize=11, va="top")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def save_stats_table(desc: pd.DataFrame, pdf: PdfPages, title: str) -> None:
    if desc.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    display_df = desc.head(12).round(4)
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     rowLabels=display_df.index,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# --------------------- PLOTS --------------------- #

def plot_missingness(df: pd.DataFrame, pdf: PdfPages) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count of NaNs")
    ax.set_xlabel("Columns")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, pdf: PdfPages, bins: int = 30, max_cols: int = 12) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"Histogram: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_categorical_bars(df: pd.DataFrame, pdf: PdfPages, top_k: int = 15, max_cols: int = 8) -> None:
    cats = detect_categoricals(df)[:max_cols]
    for col in cats:
        counts = df[col].astype(str).value_counts().head(top_k)
        fig, ax = plt.subplots(figsize=(10, 5))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top {top_k} Values: {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xlabel(col)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, pdf: PdfPages) -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, aspect='auto', interpolation='nearest')
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 8) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f"Boxplot: {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel(col)
        pdf.savefig(fig)
        plt.close(fig)


def plot_violinplots(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 6) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.violinplot(df[col].dropna(), showmeans=True)
        ax.set_title(f"Violin Plot: {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel(col)
        pdf.savefig(fig)
        plt.close(fig)


def plot_density_plots(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 8) -> None:
    # Density plots without requiring scipy
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=30, density=True, alpha=0.5, label="Histogram")
        data.plot(kind="hist", bins=30, density=True, alpha=0.3, ax=ax)  # smooth histogram
        ax.set_title(f"Approx Density Plot: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def plot_scatter_matrix(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 5) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if len(num_cols) > 1:
        fig = scatter_matrix(df[num_cols], figsize=(10, 10), diagonal="kde")
        plt.suptitle("Scatter Matrix", y=1.02, fontsize=14, fontweight="bold")
        pdf.savefig(fig[0][0].figure)
        plt.close(fig[0][0].figure)


def plot_line_charts(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 6) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if num_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[num_cols].plot(ax=ax)
        ax.set_title("Line Chart (first few numeric cols)", fontsize=12, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)


def plot_pie_charts(df: pd.DataFrame, pdf: PdfPages, max_cols: int = 4) -> None:
    cats = detect_categoricals(df)[:max_cols]
    for col in cats:
        counts = df[col].astype(str).value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 6))
        counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Pie Chart: {col}", fontsize=12, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)


# --------------------- MAIN PIPELINE --------------------- #

def summary_text(df: pd.DataFrame, desc: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    lines.append(f"Numeric columns: {len(numeric_cols)} | Categorical/object columns: {len(object_cols)}")
    missing_total = int(df.isna().sum().sum())
    lines.append(f"Total missing values: {missing_total}")
    if not desc.empty:
        means = desc['mean'].dropna().to_dict()
        if means:
            sample = list(means.items())[:8]
            lines.append("Sample means: " + "; ".join(f"{k}={v:.4g}" for k, v in sample))
    return "\n".join(lines)


def analyze_to_pdf(csv_path: str, out_pdf: str) -> None:
    df = load_csv_to_df(csv_path)
    desc = compute_basic_stats(df)

    with PdfPages(out_pdf) as pdf:
        # Summary page
        add_text_page(pdf, "Dataset Summary", summary_text(df, desc))

        # Stats table
        save_stats_table(desc, pdf, "Descriptive Statistics (Numeric)")

        # Visualizations
        plot_missingness(df, pdf)
        plot_histograms(df, pdf)
        plot_categorical_bars(df, pdf)
        plot_correlation_heatmap(df, pdf)
        plot_boxplots(df, pdf)
        plot_violinplots(df, pdf)
        plot_density_plots(df, pdf)
        plot_scatter_matrix(df, pdf)
        plot_line_charts(df, pdf)
        plot_pie_charts(df, pdf)

        # Closing notes
        add_text_page(pdf, "Notes",
                      "This report was auto-generated. Graphs are limited in number for readability. "
                      "Consider domain-specific EDA for deeper insights.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to input CSV")
    p.add_argument("--output", "-o", required=True, help="Path to output PDF")
    return p.parse_args()


def main():
    args = parse_args()
    analyze_to_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
