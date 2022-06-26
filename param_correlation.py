#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 4, 3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse


if __name__ == "__main__":
    sns.set_context("paper")
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    # TODO this is dirty, the stats object should not replace nans by infs.
    # df.loc[df["f-value"] > 1e1, "f-value"] = np.nan
    # df = df.replace([np.inf, -np.inf], np.nan)

    sns.set_context("paper")
    sns.set_style("whitegrid")

    fig, ax = plt.subplots()
    sns.violinplot(ax=ax, data=df, x=args.param, y="f-value", palette="deep", bw=0.1, scale="count")
    ax.set_yscale('log')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
    fig.tight_layout()
    fig.savefig(f"{args.param}_correlation.pdf")
    plt.show()

