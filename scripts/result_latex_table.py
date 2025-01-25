#!/usr/bin/env python
# This is a dirty script that generates the result LaTex tables.

import pandas as pd
from io import StringIO
import textwrap
import math
import numpy as np
from collections import defaultdict
import os

RESULTS="""
Dataset       , Horizon , iTransformer-mse , iTransformer-mae , RLinear-mse , RLinear-mae , PatchTST-mse , PatchTST-mae , TimesNet-mse , TimesNet-mae , DLinear-mse , DLinear-mae , TSMixer-mse , TSMixer-mae , FEDformer-mse , FEDformer-mae , Autoformer-mse , Autoformer-mae
ETTm1         , 96      , 0.334            , 0.368            , 0.355       , 0.376       , 0.329        , 0.367        , 0.338        , 0.375        , 0.345       , 0.372       , 0.285       , 0.339       , 0.326         , 0.390         , 0.510          , 0.492
ETTm1         , 192     , 0.377            , 0.391            , 0.391       , 0.392       , 0.367        , 0.385        , 0.374        , 0.387        , 0.380       , 0.389       , 0.327       , 0.365       , 0.365         , 0.415         , 0.514          , 0.495
ETTm1         , 336     , 0.426            , 0.420            , 0.424       , 0.415       , 0.399        , 0.410        , 0.410        , 0.411        , 0.413       , 0.413       , 0.356       , 0.382       , 0.392         , 0.425         , 0.510          , 0.492
ETTm1         , 720     , 0.491            , 0.459            , 0.487       , 0.450       , 0.454        , 0.439        , 0.478        , 0.450        , 0.474       , 0.453       , 0.419       , 0.414       , 0.446         , 0.458         , 0.527          , 0.493
ETTm2         , 96      , 0.180            , 0.264            , 0.182       , 0.265       , 0.175        , 0.259        , 0.187        , 0.267        , 0.193       , 0.292       , 0.163       , 0.252       , 0.180         , 0.271         , 0.205          , 0.293
ETTm2         , 192     , 0.250            , 0.309            , 0.246       , 0.304       , 0.241        , 0.302        , 0.249        , 0.309        , 0.284       , 0.362       , 0.216       , 0.290       , 0.252         , 0.318         , 0.278          , 0.336
ETTm2         , 336     , 0.311            , 0.348            , 0.307       , 0.342       , 0.305        , 0.343        , 0.321        , 0.351        , 0.369       , 0.427       , 0.268       , 0.324       , 0.324         , 0.364         , 0.343          , 0.379
ETTm2         , 720     , 0.412            , 0.407            , 0.407       , 0.398       , 0.402        , 0.400        , 0.408        , 0.403        , 0.554       , 0.522       , 0.420       , 0.422       , 0.410         , 0.420         , 0.414          , 0.419
ETTh1         , 96      , 0.386            , 0.405            , 0.386       , 0.395       , 0.414        , 0.419        , 0.384        , 0.402        , 0.386       , 0.400       , 0.361       , 0.392       , 0.376         , 0.415         , 0.435          , 0.446
ETTh1         , 192     , 0.441            , 0.436            , 0.437       , 0.424       , 0.460        , 0.445        , 0.436        , 0.429        , 0.437       , 0.432       , 0.404       , 0.418       , 0.423         , 0.446         , 0.456          , 0.457
ETTh1         , 336     , 0.487            , 0.458            , 0.479       , 0.446       , 0.501        , 0.466        , 0.491        , 0.469        , 0.481       , 0.459       , 0.420       , 0.431       , 0.444         , 0.462         , 0.486          , 0.487
ETTh1         , 720     , 0.503            , 0.491            , 0.481       , 0.470       , 0.500        , 0.488        , 0.521        , 0.500        , 0.519       , 0.516       , 0.463       , 0.472       , 0.469         , 0.492         , 0.515          , 0.517
ETTh2         , 96      , 0.297            , 0.349            , 0.288       , 0.338       , 0.302        , 0.348        , 0.340        , 0.374        , 0.333       , 0.387       , 0.274       , 0.341       , 0.332         , 0.374         , 0.332          , 0.368
ETTh2         , 192     , 0.380            , 0.400            , 0.374       , 0.390       , 0.388        , 0.400        , 0.402        , 0.414        , 0.477       , 0.476       , 0.339       , 0.385       , 0.407         , 0.446         , 0.426          , 0.434
ETTh2         , 336     , 0.428            , 0.432            , 0.415       , 0.426       , 0.426        , 0.433        , 0.452        , 0.452        , 0.594       , 0.541       , 0.361       , 0.406       , 0.400         , 0.447         , 0.477          , 0.479
ETTh2         , 720     , 0.427            , 0.445            , 0.420       , 0.440       , 0.431        , 0.446        , 0.462        , 0.468        , 0.831       , 0.657       , 0.445       , 0.470       , 0.412         , 0.469         , 0.453          , 0.490
Weather       , 96      , 0.174            , 0.214            , 0.192       , 0.232       , 0.177        , 0.218        , 0.172        , 0.220        , 0.196       , 0.255       , 0.145       , 0.198       , 0.238         , 0.314         , 0.249          , 0.329
Weather       , 192     , 0.221            , 0.254            , 0.240       , 0.271       , 0.225        , 0.259        , 0.219        , 0.261        , 0.237       , 0.296       , 0.191       , 0.242       , 0.275         , 0.329         , 0.325          , 0.370
Weather       , 336     , 0.278            , 0.296            , 0.292       , 0.307       , 0.278        , 0.297        , 0.280        , 0.306        , 0.283       , 0.335       , 0.242       , 0.280       , 0.339         , 0.377         , 0.351          , 0.391
Weather       , 720     , 0.358            , 0.347            , 0.364       , 0.353       , 0.354        , 0.348        , 0.365        , 0.359        , 0.345       , 0.381       , 0.320       , 0.336       , 0.389         , 0.409         , 0.415          , 0.426
Electricity   , 96      , 0.148            , 0.240            , 0.201       , 0.281       , 0.181        , 0.270        , 0.168        , 0.272        , 0.197       , 0.282       , 0.131       , 0.229       , 0.186         , 0.302         , 0.196          , 0.313
Electricity   , 192     , 0.162            , 0.253            , 0.201       , 0.283       , 0.188        , 0.274        , 0.184        , 0.289        , 0.196       , 0.285       , 0.151       , 0.246       , 0.197         , 0.311         , 0.211          , 0.324
Electricity   , 336     , 0.178            , 0.269            , 0.215       , 0.298       , 0.204        , 0.293        , 0.198        , 0.300        , 0.209       , 0.301       , 0.161       , 0.261       , 0.213         , 0.328         , 0.214          , 0.327
Electricity   , 720     , 0.225            , 0.317            , 0.257       , 0.331       , 0.246        , 0.324        , 0.220        , 0.320        , 0.245       , 0.333       , 0.197       , 0.293       , 0.233         , 0.344         , 0.236          , 0.342
Traffic       , 96      , 0.395            , 0.268            , 0.649       , 0.389       , 0.462        , 0.295        , 0.593        , 0.321        , 0.650       , 0.396       , 0.376       , 0.264       , 0.576         , 0.359         , 0.597          , 0.371
Traffic       , 192     , 0.417            , 0.276            , 0.601       , 0.366       , 0.466        , 0.296        , 0.617        , 0.336        , 0.598       , 0.370       , 0.397       , 0.277       , 0.610         , 0.380         , 0.607          , 0.382
Traffic       , 336     , 0.433            , 0.283            , 0.609       , 0.369       , 0.482        , 0.304        , 0.629        , 0.336        , 0.605       , 0.373       , 0.413       , 0.290       , 0.608         , 0.375         , 0.623          , 0.387
Traffic       , 720     , 0.467            , 0.302            , 0.647       , 0.387       , 0.514        , 0.322        , 0.640        , 0.350        , 0.645       , 0.394       , 0.444       , 0.306       , 0.621         , 0.375         , 0.639          , 0.395
Exchange      , 96      , 0.086            , 0.206            , 0.093       , 0.217       , 0.088        , 0.205        , 0.107        , 0.234        , 0.088       , 0.218       , N/A         , N/A         , 0.148         , 0.278         , 0.197          , 0.323
Exchange      , 192     , 0.177            , 0.299            , 0.184       , 0.307       , 0.176        , 0.299        , 0.226        , 0.344        , 0.176       , 0.315       , N/A         , N/A         , 0.271         , 0.315         , 0.300          , 0.369
Exchange      , 336     , 0.331            , 0.417            , 0.351       , 0.432       , 0.301        , 0.397        , 0.367        , 0.448        , 0.313       , 0.427       , N/A         , N/A         , 0.460         , 0.427         , 0.509          , 0.524
Exchange      , 720     , 0.847            , 0.691            , 0.886       , 0.714       , 0.901        , 0.714        , 0.964        , 0.746        , 0.839       , 0.695       , N/A         , N/A         , 1.195         , 0.695         , 1.447          , 0.941
Solar         , 96      , 0.203            , 0.237            , 0.322       , 0.339       , 0.234        , 0.286        , 0.250        , 0.292        , 0.290       , 0.378       , N/A         , N/A         , 0.242         , 0.342         , 0.884          , 0.711
Solar         , 192     , 0.233            , 0.261            , 0.359       , 0.356       , 0.267        , 0.310        , 0.296        , 0.318        , 0.320       , 0.398       , N/A         , N/A         , 0.285         , 0.380         , 0.834          , 0.692
Solar         , 336     , 0.248            , 0.273            , 0.397       , 0.369       , 0.290        , 0.315        , 0.319        , 0.330        , 0.353       , 0.415       , N/A         , N/A         , 0.282         , 0.376         , 0.941          , 0.723
Solar         , 720     , 0.249            , 0.275            , 0.397       , 0.356       , 0.289        , 0.317        , 0.338        , 0.337        , 0.356       , 0.413       , N/A         , N/A         , 0.357         , 0.427         , 0.882          , 0.717
"""
THEIR_MODELS=[]
# DATASETS=("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "Electricity", "Traffic", "Exchange", "Solar")
DATASETS=("ETTm1", "ETTm2", "Weather", "Solar", "Electricity", "Traffic", "ETTh1", "ETTh2", "Exchange")
PRED_LENS=(96, 192, 336, 720)
OUR_MODEL_DIRS={
    "ESN": lambda ds, ahead: f"./log/optimize-sparse-40-50-k7-{ds.lower()}-ahead192-loop-memlen0-seed50/evaluate-{ds.lower()}-loop-test-lms0-retrain0-ahead{ahead}-stride1",
    "LCESN": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead192-loop-seed50/evaluate-{ds.lower()}-loop-test-lms0-retrain0-ahead{ahead}-stride1",
    "LCESN-LMS": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead192-loop-seed50/evaluate-{ds.lower()}-loop-test-lms1-retrain0-ahead{ahead}-stride1",
    "LCESN-LR100": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead192-loop-seed50/evaluate-{ds.lower()}-loop-test-lms1-retrain100-ahead{ahead}-stride1",
    "LCESN-LR1": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead192-loop-seed50/evaluate-{ds.lower()}-loop-test-lms0-retrain1-ahead{ahead}-stride1"}
OUR_MODELS = list(OUR_MODEL_DIRS.keys())

MODEL_TO_TITLE = {
    "SotA": r"SotA",
    "SUM": r"\textbf{SUM}",
    "ESN": r"ESN",
    "LCESN": r"\textbf{LCESN}",
    "LCESN-LMS": r"\textbf{LCESN-LMS}",
    "LCESN-LR100": r"\textbf{LCESN-LR100}",
    "LCESN-LR1": r"\textbf{LCESN-LR1}",
    "TSMixer": "TSMixer",
    "iTransformer": "iTransformer",
    "PatchTST": "PatchTST",
    "DLinear": "DLinear",
    "FEDformer": "FEDformer",
    "RLinear": "RLinear",
    "Autoformer": "Autoformer",
    "TimesNet": "TimesNet"
}

MODEL_TO_CITE = {
    "SotA": r"",
    "ESN": r"{\citeyearpar{jaeger2001echo}}",
    "LCESN": r"{\textbf{Ours}}",
    "LCESN-LMS": r"{\textbf{Ours}}",
    "LCESN-LR100": r"{\textbf{Ours}}",
    "LCESN-LR1": r"{\textbf{Ours}}",
    "TSMixer": r"{\citeyearpar{chen2023tsmixer}}",
    "iTransformer": r"{\citeyearpar{liu2024itransformer}}",
    "PatchTST": r"{\citeyearpar{nie2023patchtst}}",
    "DLinear": r"{\citeyearpar{zeng2022dlinear}}",
    "FEDformer": r"{\citeyearpar{zhou2022fedformer}}",
    "RLinear": r"{\citeyearpar{li2023rlinear}}",
    "Autoformer": r"{\citeyearpar{wu2021autoformer}}",
    "TimesNet": r"{\citeyearpar{wu2023timesnet}}"
}

def metric_selector(models, metric, extra_models=()):
    return [f"{model}-{metric}" for model in models + list(extra_models)]

def add_our_best(df) -> pd.DataFrame:
    df2 = df.copy()
    df2["Ours-mse"] = df2[metric_selector(OUR_MODELS, "mse")].min(axis=1, numeric_only=True)
    df2["Ours-mae"] = df2[metric_selector(OUR_MODELS, "mae")].min(axis=1, numeric_only=True)
    return df2

def result(df, model, ds, pred_len, metric) -> float:
    df_ds = df[(df["Dataset"] == ds) & (df["Horizon"] == pred_len)]
    return np.round(df_ds[model+'-'+metric].iloc[0], 3)

def model_place(df, model, ds, pred_len, metric) -> int:
    if model not in OUR_MODELS:
        df2 = add_our_best(df)
        selector = metric_selector(THEIR_MODELS, metric, ["Ours"])
    else:
        df2 = df.copy()
        selector = metric_selector(THEIR_MODELS, metric, [model])
    this_result = result(df2, model, ds, pred_len, metric)
    if np.isnan(this_result): return 100
    df2_ds = df2[(df2["Dataset"] == ds) & (df2["Horizon"] == pred_len)][selector]
    results = df2_ds.sort_values(df2_ds.index[0], axis=1).values.flatten()
    results = np.unique(np.round(results, 3))
    return np.nonzero(results == this_result)[0][0]

def num_first(df, model, metric) -> int:
    return sum(model_place(df, model, ds, pred_len, metric) == 0 for ds in DATASETS for pred_len in PRED_LENS)

def num_compete(df, model, metric) -> int:
    df2 = add_our_best(df)
    return pd.notna(df2[model+"-"+metric]).sum()

def score(df, model, metric) -> float:
    if num_compete(df, model, metric) == 0:
        return -1
    return np.round(num_first(df, model, metric) / num_compete(df, model, metric), 3)

def score_place(df, model, metric) -> int:
    if model not in OUR_MODELS:
        order = [score(df, m, metric) for m in THEIR_MODELS + ["Ours"]]
    else:
        order = [score(df, m, metric) for m in THEIR_MODELS + [model]]
    order = np.flip(np.sort(np.unique(np.round(order, 3))))
    return np.where(order == score(df, model, metric))[0][0]

def score_latex(df, model, metric) -> str:
    score_str = f"{num_first(df, model, metric)}/{num_compete(df, model, metric)}"
    place = score_place(df, model, metric) 
    if place == 0:
        return f"\\firstres{{{score_str}}}"
    if place == 1:
        return f"\\secondres{{{score_str}}}"
    return score_str

def result_latex(df, model, ds, pred_len, metric) -> str:
    this_result = result(df, model, ds, pred_len, metric)
    if math.isnan(this_result):
        return f"N/A"
    if this_result >= 10:
        return f"$\\infty$"
    place = model_place(df, model, ds, pred_len, metric) 
    if place == 0:
        return f"\\firstres{{{this_result:.3f}}}"
    if place == 1:
        return f"\\secondres{{{this_result:.3f}}}"
    return f"{this_result:.3f}"

def avg_result(df, model, ds, metric) -> float:
    df_mean = df[df["Dataset"] == ds].mean(numeric_only=True)
    return np.round(df_mean[model+'-'+metric], 3)

def avg_result_place(df, model, ds, metric) -> int:
    if model not in OUR_MODELS:
        df2 = add_our_best(df)
        selector = metric_selector(THEIR_MODELS, metric, ["Ours"])
    else:
        df2 = df.copy()
        selector = metric_selector(THEIR_MODELS, metric, [model])
    this_result = avg_result(df2, model, ds, metric)
    if np.isnan(this_result): return 100
    df2_mean = df2[df2["Dataset"] == ds].mean(numeric_only=True)[selector]
    results = df2_mean.sort_values().values
    results = np.unique(np.round(results, 3))
    return np.nonzero(results == this_result)[0][0]

def avg_result_latex(df, model, ds, metric) -> str:
    this_result = avg_result(df, model, ds, metric)
    if math.isnan(this_result):
        return f"N/A"
    if this_result >= 10:
        return f"$\\infty$"
    place = avg_result_place(df, model, ds, metric) 
    if place == 0:
        return f"\\firstres{{{this_result:.3f}}}"
    if place == 1:
        return f"\\secondres{{{this_result:.3f}}}"
    return f"{this_result:.3f}"

def avg_num_first(df, model, metric) -> int:
    return sum(avg_result_place(df, model, ds, metric) == 0 for ds in DATASETS)

def avg_num_compete(df, model, metric) -> int:
    df2 = add_our_best(df)
    df2 = df2.groupby("Dataset").mean(numeric_only=True)
    return pd.notna(df2[model+"-"+metric]).sum()

def avg_score(df, model, metric) -> float:
    if avg_num_compete(df, model, metric) == 0:
        return -1
    return np.round(avg_num_first(df, model, metric) / avg_num_compete(df, model, metric), 3)

def avg_score_place(df, model, metric) -> int:
    if model not in OUR_MODELS:
        order = [avg_score(df, m, metric) for m in THEIR_MODELS + ["Ours"]]
    else:
        order = [avg_score(df, m, metric) for m in THEIR_MODELS + [model]]
    order = np.flip(np.sort(np.unique(np.round(order, 3))))
    return np.where(order == avg_score(df, model, metric))[0][0]

def avg_score_latex(df, model, metric) -> str:
    score_str = f"{avg_num_first(df, model, metric)}/{avg_num_compete(df, model, metric)}"
    place = avg_score_place(df, model, metric) 
    if place == 0:
        return f"\\firstres{{{score_str}}}"
    if place == 1:
        return f"\\secondres{{{score_str}}}"
    return score_str

def relative_ranking_matrix(df, metric='mse'):
    models = OUR_MODELS + THEIR_MODELS
    n = len(models)
    ranking_matrix = np.zeros((n, n), dtype=int)
    
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            results = {
                model: result(df, model, ds, pred_len, metric) 
                for model in models if not pd.isna(result(df, model, ds, pred_len, metric))
            }
            
            # Iterate over all pairs of models to update ranking matrix
            for i, model_i in enumerate(models):
                for j, model_j in enumerate(models):
                    if i != j and model_i in results and model_j in results:
                        if np.round(results[model_i], 3) < np.round(results[model_j], 3):
                            ranking_matrix[i, j] += 1
    
    # Create DataFrame for better visualization
    heade = [MODEL_TO_TITLE[m] for m in models]
    ranking_df = pd.DataFrame(ranking_matrix, index=heade, columns=heade)
    
    # Add sums and use them to sort
    ranking_df['SUM'] = ranking_df.sum(axis=1)
    ranking_df.sort_values('SUM', inplace=True, ascending=False)
    
    return ranking_df

def main():
    df = pd.read_csv(StringIO(RESULTS.replace(" ", "")))
    global THEIR_MODELS
    THEIR_MODELS=list(set(c.removesuffix("-mse").removesuffix("-mae") for c in df.columns if c not in ("Dataset", "Horizon")))

    model_data = []
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            model_data.append({'Dataset': ds, 'Horizon': pred_len})
            for model in OUR_MODELS:
                model_data[-1][model+'-mse'] = pd.NA
                model_data[-1][model+'-mae'] = pd.NA
                csv_file = OUR_MODEL_DIRS[model](ds, pred_len) + "/results.csv"
                try:
                    if os.path.exists(csv_file):
                        results = pd.read_csv(csv_file, index_col=0)
                        model_data[-1][model+'-mse'] = results["mse"].values[0]
                        model_data[-1][model+'-mae'] = results["mae"].values[0]
                except pd.errors.EmptyDataError:
                    pass
    model_df = pd.DataFrame(model_data)
    df = pd.merge(df, model_df, on=["Dataset", "Horizon"], how="outer")

    THEIR_MODELS = sorted(THEIR_MODELS, key=lambda model: score(df, model, "mse"), reverse=True)

    # Relative ranking matrix.

    ranking_matrix = relative_ranking_matrix(df, metric='mse')

    # Rotate the headers by 90 degrees
    latex_str = ranking_matrix.to_latex(
        header=True, column_format='c' * len(ranking_matrix.columns) + '|c'
    )
    lines = latex_str.splitlines()
    header = lines[2].replace('\\\\', '').split('&')[1:]
    rotated_header_line = ' & ' + ' & '.join(f"\\rotatebox{{90}}{{{c.strip()}}}" for c in header)
    lines[2] = rotated_header_line + ' \\\\'
    latex_str = '\n'.join(lines)

    print(textwrap.dedent(r"""
        \begin{table}[htbp]
        \caption{}
        \label{tab:relative-ranking}
        \centering
    """))
    print(latex_str)
    print(r"""\end{table}""")
    print()

    # All results.

    print(textwrap.dedent(r"""
        \newcommand{\firstres}[1]{{\textbf{\textcolor{red}{#1}}}}
        \newcommand{\secondres}[1]{{\underline{\textcolor{blue}{#1}}}}
        \newcolumntype{?}{!{\vrule width 1pt}}

        \newcommand{\resulttitlescale}{0.8}
        \newcommand{\resultdsscale}{0.95}
        \newcommand{\resultscale}{0.78}

        \begin{table}[htbp]
        \label{tab:real-world-results}
        \vskip -0.0in
        \vspace{3pt}
        \renewcommand{\arraystretch}{0.9} 
        \centering
        \resizebox{1\columnwidth}{!}{
        \begin{threeparttable}
        \begin{small}
        \renewcommand{\multirowsetup}{\centering}
        \setlength{\tabcolsep}{1pt}
        """))

    print(textwrap.dedent(r"""\begin{tabular}{c|c|"""), end="")
    print("|".join(["cc"] * len(OUR_MODELS)) + "?cc|", end="")
    print("|".join(["cc"] * (len(THEIR_MODELS) - 1)) + "}")

    print(textwrap.dedent(r"""
        \toprule
        \multicolumn{2}{c}{\multirow{2}{*}{Models}}
        """))
    for model in OUR_MODELS + THEIR_MODELS:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{MODEL_TO_TITLE[model]}}}}} ")
    print(r"""\\""")

    print(r"""\multicolumn{2}{c}{} """)
    for model in OUR_MODELS + THEIR_MODELS:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{MODEL_TO_CITE[model]}}}}} ")
    print(r"""\\""")

    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        print(f"\\cmidrule(lr){{{2*i+3}-{2*i+4}}}")
    print(r"""\multicolumn{2}{c}{Metric} """)
    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        print(r"""& \scalebox{\resultscale}{MSE} & \scalebox{\resultscale}{MAE}""")
    print(r"""\\""")
    print(r"""\midrule""")

    for ds in DATASETS:
        print(f"""\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{\\scalebox{{\\resultdsscale}}{{{ds}}}}}}}""")
        for pred_len in PRED_LENS:
            print(f"& \\scalebox{{\\resultscale}}{{{pred_len}}} ", end="")
            for model in OUR_MODELS + THEIR_MODELS:
                mse_result = result_latex(df, model, ds, pred_len, "mse")
                mae_result = result_latex(df, model, ds, pred_len, "mae")
                print(f"& \\scalebox{{\\resultscale}}{{{mse_result}}} ", end="")
                print(f"& \\scalebox{{\\resultscale}}{{{mae_result}}} ")
            print(r"""\\""")
            
        print(r"""\cmidrule(lr){2-26}""")

        print(f"& \\scalebox{{\\resultscale}}{{Avg}} ", end="")
        for model in OUR_MODELS + THEIR_MODELS:
            mse_result = avg_result_latex(df, model, ds, "mse")
            mae_result = avg_result_latex(df, model, ds, "mae")
            print(f"& \\scalebox{{\\resultscale}}{{{mse_result}}} ", end="")
            print(f"& \\scalebox{{\\resultscale}}{{{mae_result}}} ")
        print(r"""\\""")

        print(r"""\midrule""")

    print(r"""\multicolumn{2}{c|}{\scalebox{\resultscale}{{\# $1^{\text{st}}$}}}""")
    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        score_mse_str = score_latex(df, model, "mse")
        score_mae_str = score_latex(df, model, "mae")
        print(f"& \\scalebox{{\\resultscale}}{{{score_mse_str}}}", end="")
        print(f"& \\scalebox{{\\resultscale}}{{{score_mae_str}}}")
    print(r"""\\""")
    print(r"""\bottomrule""")

    print(textwrap.dedent(r"""
    \end{tabular}
    \end{small}
    \end{threeparttable}
    }
    \end{table}
    """))

    # Average only.

    THEIR_MODELS = sorted(THEIR_MODELS, key=lambda model: avg_score(df, model, "mse"), reverse=True)

    print(textwrap.dedent(r"""
        \begin{table}[htbp]
        \label{tab:real-world-average-results}
        \vskip -0.0in
        \vspace{3pt}
        \renewcommand{\arraystretch}{0.9} 
        \centering
        \resizebox{1\columnwidth}{!}{
        \begin{threeparttable}
        \begin{small}
        \renewcommand{\multirowsetup}{\centering}
        \setlength{\tabcolsep}{1pt}
        """))

    print(textwrap.dedent(r"""\begin{tabular}{c|"""), end="")
    print("|".join(["cc"] * len(OUR_MODELS)) + "?cc|", end="")
    print("|".join(["cc"] * (len(THEIR_MODELS) - 1)) + "}")

    print(textwrap.dedent(r"""
        \toprule
        {\multirow{2}{*}{Models}}
        """))
    for model in OUR_MODELS + THEIR_MODELS:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{MODEL_TO_TITLE[model]}}}}} ")
    print(r"""\\""")

    print(r"""{} """)
    for model in OUR_MODELS + THEIR_MODELS:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{MODEL_TO_CITE[model]}}}}} ")
    print(r"""\\""")

    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        print(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}")
    print(r"""{Metric} """)
    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        print(r"""& \scalebox{\resultscale}{MSE} & \scalebox{\resultscale}{MAE}""")
    print(r"""\\""")
    print(r"""\midrule""")

    for ds in DATASETS:
        print(f"""{{\\rotatebox{{0}}{{\\scalebox{{\\resultdsscale}}{{{ds}}}}}}}""")

        for model in OUR_MODELS + THEIR_MODELS:
            mse_result = avg_result_latex(df, model, ds, "mse")
            mae_result = avg_result_latex(df, model, ds, "mae")
            print(f"& \\scalebox{{\\resultscale}}{{{mse_result}}} ", end="")
            print(f"& \\scalebox{{\\resultscale}}{{{mae_result}}} ")
        print(r"""\\""")

        print(r"""\midrule""")

    print(r"""{\scalebox{\resultscale}{{\# $1^{\text{st}}$}}}""")
    for i, model in enumerate(OUR_MODELS + THEIR_MODELS):
        score_mse_str = avg_score_latex(df, model, "mse")
        score_mae_str = avg_score_latex(df, model, "mae")
        print(f"& \\scalebox{{\\resultscale}}{{{score_mse_str}}}", end="")
        print(f"& \\scalebox{{\\resultscale}}{{{score_mae_str}}}")
    print(r"""\\""")
    print(r"""\bottomrule""")

    print(textwrap.dedent(r"""
    \end{tabular}
    \end{small}
    \end{threeparttable}
    }
    \end{table}
    """))

if __name__ == "__main__":
    main()
