#!/usr/bin/env python
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("csv", type=str)
    args = argparse.parse_args()

    df = pd.read_csv(args.csv)
    sns.lineplot(df)
    plt.show()
