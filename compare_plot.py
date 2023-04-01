#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 4, 3
import matplotlib.pyplot as plt
from matplotlib import ticker as pltticker
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
    fv_for_run = df.groupby(groupers).mean(numeric_only=True)["f-value"].reset_index()
    # Choose the best net for each parameter value.
    fv_best_run = fv_for_run.loc[fv_for_run.groupby([param])["f-value"].idxmin()]
    best_selector = False
    for r, p in zip(fv_best_run["run"], fv_best_run[param]):
        best_selector = (df["run"] == r) & (df[param] == p) | best_selector
    return df[best_selector]

# Seaborn's violinplot computes kernel before log transformation so the
# violins have various artifacts. Use this function if the
# data are transformed to logspace before plotting.
def set_log_y(ax):
    ax.yaxis.set_major_formatter(pltticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ymin, ymax = ax.get_ylim()
    major_ticks = np.arange(np.floor(ymin), ymax)
    ax.yaxis.set_ticks(major_ticks)
    minor_ticks = [np.log10(y) for t in major_ticks for y in np.linspace(10 ** t, 10 ** (t + 1), 10)]
    minor_ticks = [y for y in minor_ticks if y < ymax]
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.grid(which='minor', color='whitesmoke')
    return ax

if __name__ == "__main__":
    sns.set_context("paper")
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("--sort-by", type=str, help="The parameter by which should the plot be sorted.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    if ("lcnn.state-height" in df.columns and "lcnn.state-width" in df.columns):
        df["lcnn.state-area"] = df["lcnn.state-height"] * df["lcnn.state-width"]
        df["lcnn.state-size"] = df["lcnn.state-height"].astype(str) + "x" + df["lcnn.state-width"].astype(str)
    if ("lcnn.kernel-height" in df.columns and "lcnn.kernel-width" in df.columns):
        df["lcnn.kernel-area"] = df["lcnn.kernel-height"] * df["lcnn.kernel-width"]
        df["lcnn.kernel-size"] = df["lcnn.kernel-height"].astype(str) + "x" + df["lcnn.kernel-width"].astype(str)
    # TODO this is dirty, the stats object should not replace nans by infs.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = best_run(df, args.param)

    if args.sort_by:
        df = df.sort_values(args.sort_by)

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
    log_y = np.log10(df["f-value"])
    ax = sns.violinplot(data=df, x=args.param, y=log_y, palette="deep")
    set_log_y(ax)

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("compare_plot.pdf")
    plt.show()
