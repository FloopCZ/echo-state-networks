#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 6, 2
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
    major_ticks = np.arange(np.ceil(ymin), ymax)
    ax.yaxis.set_ticks(major_ticks)
    minor_ticks = [np.log10(y) for t in major_ticks for y in np.linspace(10 ** t, 10 ** (t + 1), 10)]
    minor_ticks = [y for y in minor_ticks if y < ymax]
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.grid(which='minor', color='whitesmoke')
    return ax

def plot(df, param, plot_type, logscale, color=None, connect_label=None, order=None):
    # Print some useful statistics.
    stats_df = df.groupby(["run", param]).agg({'f-value': ['mean','std']})
    stats_df = stats_df.reset_index().sort_values(param)
    print(stats_df)

    # Print p-values.
    pvals = defaultdict(lambda: {})
    for param1 in stats_df[param]:
        for param2 in stats_df[param]:
            if param1 == param2: continue
            a = df[df[param] == param1]["f-value"]
            b = df[df[param] == param2]["f-value"]
            pvals[param1][param2] = scipy.stats.ttest_ind(a, b, equal_var=False, alternative='less')
    pprint(pvals)

    # Plot the boxplot for the best runs.
    if logscale:
        df["f-value"] = np.log10(df["f-value"])

    palette = None if color else "deep"
    if plot_type == "violin":
        ax = sns.violinplot(data=df, x=param, y="f-value", color=color, palette=palette, hue=param, order=order, zorder=2)
    elif plot_type == "box":
        ax = sns.boxplot(data=df, x=param, y="f-value", color=color, palette=palette, hue=param, order=order, zorder=2)
    else:
        raise ValueError(f"Unknown plot type `{plot_type}`.")

    if connect_label:
        sns.lineplot(data=df, x=param, y="f-value", color=color, palette=palette,
                     zorder=1, label=connect_label, errorbar=None)

    if logscale:
        set_log_y(ax)

    return ax

def main():
    sns.set_context("paper")
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("--sort-by", type=str, help="The parameter by which should the plot be sorted.")
    parser.add_argument("--connect", type=str, help="The parameter by which should the individual violins "
                                                    "be connected using a line.")
    parser.add_argument("--rotate-labels", type=int, help="The rotation of the x-axis labels.")
    parser.add_argument("--all-runs", action=argparse.BooleanOptionalAction, help="Select all runs, not only the best")
    parser.add_argument("--no-legend", action='store_true', help="Disable legend.")
    parser.add_argument("--plot-type", type=str, help="Plot type (e.g., violin, box.", default="violin")
    parser.add_argument("--logscale", action=argparse.BooleanOptionalAction, help="Plot in logarithmic scale")
    parser.add_argument("--order", nargs='*', type=str, help="Ordered labels. Has to cover all labels.")
    parser.add_argument("csvs", nargs='*', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()
    print(args.order)

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    if ("lcnn.state-height" in df.columns and "lcnn.state-width" in df.columns):
        df["lcnn.state-area"] = df["lcnn.state-height"] * df["lcnn.state-width"]
        df["lcnn.state-size"] = df["lcnn.state-height"].astype(str) + "x" + df["lcnn.state-width"].astype(str)
    if ("lcnn.kernel-height" in df.columns and "lcnn.kernel-width" in df.columns):
        df["lcnn.kernel-area"] = df["lcnn.kernel-height"] * df["lcnn.kernel-width"]
        df["lcnn.kernel-size"] = df["lcnn.kernel-height"].astype(str) + "x" + df["lcnn.kernel-width"].astype(str)
    if ("lcnn.memory-length" in df.columns and "lcnn.topology" in df.columns):
        df["lcnn.memory-length-topo"] = df["lcnn.memory-length"].astype(str) + "-" + df["lcnn.topology"].astype(str)
    # TODO this is dirty, the stats object should not replace nans by infs.
    df = df.replace([np.inf, -np.inf], np.nan)
    if args.sort_by:
        df = df.sort_values(args.sort_by)
    if args.connect:
        connect_unique = df[args.connect].unique()
        for connect_value, color in zip(connect_unique, sns.color_palette("deep", len(connect_unique))):
            df_exp_set = df[df[args.connect] == connect_value]
            df_best = best_run(df_exp_set, args.param).copy() if not args.all_runs else df_exp_set
            plot(df_best, param=args.param, plot_type=args.plot_type, logscale=args.logscale,
                 color=color, connect_label=connect_value, order=args.order)
            plt.legend(title=args.connect)
    else:
        df_best = best_run(df, args.param).copy() if not args.all_runs else df
        plot(df_best, param=args.param, plot_type=args.plot_type, logscale=args.logscale, order=args.order)

    if args.no_legend and plt.gca().get_legend():
        plt.gca().get_legend().remove()
    plt.gca().tick_params(axis='x', labelrotation=args.rotate_labels)

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("compare_plot.pdf")
    plt.show()

if __name__ == "__main__": main()
