#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 6, 2
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Data for the lyapunos.
    df_all = None
    for H in [0, 25, 50, 100, 200]:
        for run in [0, 1, 2, 3, 4]:
            run_root = Path(f"./log/optimize-lcnn-40-50-k7-ettm1-ahead192-loop-memlens-seed50/optimize-lcnn-40-50-k7-ettm1-ahead192-loop-memlen{H}-seed50/run{run}")
            path = run_root / f"lyapunov/results.csv"
            df = pd.read_csv(path)
            df["H"] = H
            df["run"] = run
            if df_all is None:
                df_all = df 
            else:
                df_all = pd.concat([df_all, df])
    assert df_all is not None
    print(df_all)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.boxplot(data=df_all, x="H", y="lyap", palette="deep", hue="H", legend=False)
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("memory_length_lyapunov.pdf")
    plt.show()