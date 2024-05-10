#!/usr/bin/env python

# Print the run with the best mean f-value.

import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The parameter on the X-axis.")
    parser.add_argument("csvs", nargs='+', help="The csvs to be concatenated and plotted.")
    args = parser.parse_args()

    df = pd.concat(pd.read_csv(csv) for csv in args.csvs)
    # TODO this is dirty, the stats object should not replace nans by infs.
    df = df.replace([np.inf, -np.inf], np.nan)
    df_best = df.groupby("run")["f-value"].mean(numeric_only=True).argmin()
    print(df_best)

if __name__ == "__main__": main()
