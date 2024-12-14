#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 3, 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

# Plot a heatmap of memory vs kernel size.

def main():
    sns.set_context("paper")
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs='*', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    if ("lcnn.kernel-height" in df.columns and "lcnn.kernel-width" in df.columns):
        df["lcnn.kernel-area"] = df["lcnn.kernel-height"] * df["lcnn.kernel-width"]
        df["lcnn.kernel-size"] = df["lcnn.kernel-height"].astype(str) + "x" + df["lcnn.kernel-width"].astype(str)
    cols = ["lcnn.kernel-size", "lcnn.kernel-area", "lcnn.memory-length", "f-value"]
    df = df[cols].groupby(["lcnn.kernel-size", "lcnn.memory-length"]).mean().reset_index()
    df.sort_values(by=["lcnn.kernel-area", "lcnn.memory-length"], inplace=True)
    df = df[["lcnn.kernel-size", "lcnn.memory-length", "f-value"]]
    print(df)
    df = df.pivot_table(index="lcnn.kernel-size", columns="lcnn.memory-length", values="f-value", sort=False)

    # Draw the heatmap with the mask and correct aspect ratio
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(df, square=True, cmap=cmap, linewidths=.5, cbar_kws={"shrink": .5})

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("./log/memory_vs_kernel_plot.pdf")
    plt.show()

if __name__ == "__main__": main()
