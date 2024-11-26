#!/usr/bin/env python

# Simple pretty plotting of data from csv files.

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 4, 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, nargs='+', help="The parameter on the y-axis.")
    parser.add_argument("--csvs", nargs='+', help="The csvs to be plotted.")
    args = parser.parse_args()

    sns.set_context("paper")
    sns.set_style("whitegrid")

    for csv in args.csvs:
        df = pd.read_csv(csv)
        df = df[:500]
        df['step'] = range(len(df))
        for param in args.params:
            sns.lineplot(data=df, x='step', y=param, linewidth=1)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"./log/{'_'.join(args.params)}_func.pdf")
    plt.show()
