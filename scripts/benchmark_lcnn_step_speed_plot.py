#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 6, 2
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
    args = parser.parse_args()

    y_label = "duration [μs]"
    df = None
    for state_height, state_width in (20, 25), (28, 36), (40, 50), (57, 70), (80, 100), (113, 141):
        for kernel_height, kernel_width in (5, 5), (7, 7):
            csv_path = f"./log/benchmark_lcnn_step_speed/benchmark_lcnn_step_speed_{state_height}_{state_width}_{kernel_height}_{kernel_width}.csv"
            csv_df = pd.read_csv(csv_path)
            csv_df["state size"] = f"{state_height}x{state_width}"
            csv_df["kernel size"] = f"{kernel_height}x{kernel_width}"
            csv_df[y_label] = csv_df["average"] * 1e6
            csv_df["method"] = csv_df["label"]
            df = pd.concat((df, csv_df))
    assert df is not None

    # Filter out unneeded data.
    df = df[df["method"] != "dense"]
    df = df[df["kernel size"] == "7x7"]

    ax = sns.barplot(df, x="state size", y=y_label, hue="method")
    ax.tick_params(axis='x', labelrotation=args.rotate_labels)

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("./log/benchmark_lcnn_step_speed.pdf")
    plt.show()

if __name__ == "__main__": main()
