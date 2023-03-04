#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 4, 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import scipy
from pprint import pprint
from collections import defaultdict


def best_run(df, param):
    # Get the mean f-value for each net of each run.
    groupers = [param, "run"]
    fv_for_run = df.groupby(groupers).mean()["f-value"].reset_index()
    # Choose the best net for each parameter value.
    fv_best_run = fv_for_run.loc[fv_for_run.groupby([param])["f-value"].idxmin()]
    best_selector = False
    for r, p in zip(fv_best_run["run"], fv_best_run[param]):
        best_selector = (df["run"] == r) & (df[param] == p) | best_selector
    return df[best_selector]


if __name__ == "__main__":
    sns.set_context("paper")
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    # TODO this is dirty, the stats object should not replace nans by infs.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = best_run(df, args.param)

    # Print some useful statistics.
    best_df = df.sort_values(args.param)
    stats_df = best_df.groupby(["run", args.param]).agg({'f-value': ['mean','std']})
    stats_df = stats_df.reset_index().sort_values(args.param)
    print(stats_df)

    # Print p-values.
    pvals = defaultdict(lambda: {})
    for param1 in stats_df[args.param]:
        for param2 in stats_df[args.param]:
            if param1 == param2: continue
            a = best_df[best_df[args.param] == param1]["f-value"]
            b = best_df[best_df[args.param] == param2]["f-value"]
            pvals[param1][param2] = scipy.stats.ttest_ind(a, b, equal_var=False, alternative='less')
    pprint(pvals)

    # Plot the boxplot for the best runs.
    sns.violinplot(data=df, x=args.param, y="f-value", palette="deep", bw=0.1, scale="count")
    plt.yscale('log')
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("compare_plot.pdf")
    plt.show()