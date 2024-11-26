#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 6, 3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

def main():
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate-labels", type=int, help="The rotation of the x-axis labels.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.barplot(df, x="state size", y="time [s]", hue="model", ax=ax1)
    sns.barplot(df, x="state size", y="max memory [MiB]", hue="model", ax=ax2)
    ax1.tick_params(axis='x', labelrotation=args.rotate_labels)
    ax2.tick_params(axis='x', labelrotation=args.rotate_labels)

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("./log/timings_plot.pdf")
    plt.show()

if __name__ == "__main__": main()
