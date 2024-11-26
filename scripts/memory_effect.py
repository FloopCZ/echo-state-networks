#!/usr/bin/env python

# Plot the effect of forced memory.
# This script is deprecated.

import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = 5, 2
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Data for the memory effect table.
    df_all = None
    for topo in ["lcnn", "sparse"]:
        for memory in ["", "-memlen0"]:
            for ds in ["ettm1", "ettm2"]:
                dirname = Path(f"./log/optimize-{topo}-40-50-k7-{ds}-ahead192-loop{memory}-seed50")
                for h in [96, 192, 336, 720]:
                    path = dirname / f"evaluate-{ds}-loop-test-lms0-retrain0-ahead{h}-stride1/results.csv"
                    if path.exists():
                        df = pd.read_csv(path)
                        df["dataset"] = ds
                        df["horizon"] = h
                        df["model"] = topo + memory
                        if df_all is None:
                            df_all = df 
                        else:
                            df_all = pd.concat([df_all, df])
    # The dataframe now contains the model, dataset, horizon, mse, and mae columns.
    # Let's pivot the models to have the mse and mae columns for each model.
    assert df_all is not None
    df_all = df_all.pivot(index=["dataset", "horizon"], columns=["model"], values=["mse", "mae"])
    df_all = df_all.swaplevel(axis=1)
    print(df_all)
    df_all.to_csv("memory-effect-full.csv")
    df_all = df_all.groupby(by="dataset").mean()
    print(df_all)
    df_all.to_csv("memory-effect-mean.csv")

    # Memory effect on the hyperoptimizer.
    df_all = None
    for topo in ["lcnn", "sparse"]:
        for memory in ["", "-memlen0"]:
            for ds in ["ettm1"]:
                for run in range(5):
                    csv = Path(f"./log/optimize-{topo}-40-50-k7-{ds}-ahead192-loop{memory}-seed50/optimization_results_{run}_1.csv")
                    if not csv.exists():
                        continue
                    df = pd.read_csv(csv)
                    df["dataset"] = ds
                    df["model"] = topo + memory
                    if df_all is None:
                        df_all = df 
                    else:
                        df_all = pd.concat([df_all, df])
    assert df_all is not None
    print(df_all)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    df_all["MSE"] = df_all["f-value"]
    sns.boxplot(data=df_all, x="model", y="MSE", palette="deep", hue="model")
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("memory-effect.pdf")
    plt.show()