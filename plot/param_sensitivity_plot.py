#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 4, 3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import argparse
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be plotted.")
    args = parser.parse_args()

    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    # legend_handles = []
    for i, csv in enumerate(args.csvs):
        df = pd.read_csv(csv)
        df = df.replace([np.inf, -np.inf], np.nan)
        # TODO this is dirty, how to get rid of the extreme values at the source?
        df.loc[df["f-value"] > 1e1, "f-value"] = np.nan
        df = df.iloc[:200:max(1, int(len(df) / 200)), :]

        # label = None
        # if i == 0: label = "113x141"
        # if i == 1: label = "57x70"
        # legend_handles.append(mpatches.Patch(color=sns.color_palette("deep")[i], label=label))

        sns.scatterplot(data=df, x=args.param, y="f-value", marker="+", linewidth=1)
    # plt.legend(title="state size", handles=legend_handles)

    plt.yscale('log')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"{args.param}_sensitivity.pdf")
    plt.show()
