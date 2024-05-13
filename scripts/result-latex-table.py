#!/usr/bin/env/python
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
DATASETS=("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "Electricity", "Traffic", "Exchange", "Solar")
PRED_LENS=(96, 192, 336, 720)
OUR_MODELS={
    "LCESN": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead{ahead}-loop-seed50/evaluate-{ds.lower()}-loop-test-lms0-retrain0-ahead{ahead}-stride1",
    "LCESN-LMS": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead{ahead}-loop-seed50/evaluate-{ds.lower()}-loop-test-lms1-retrain0-ahead{ahead}-stride1",
    "LCESN-LR100": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead{ahead}-loop-seed50/evaluate-{ds.lower()}-loop-test-lms1-retrain100-ahead{ahead}-stride1",
    "LCESN-LR1": lambda ds, ahead: f"./log/optimize-lcnn-40-50-k7-{ds.lower()}-ahead{ahead}-loop-seed50/evaluate-{ds.lower()}-loop-test-lms1-retrain1-ahead{ahead}-stride1"}

if __name__ == "__main__":
    df = pd.read_csv(StringIO(RESULTS.replace(" ", "")))
    models=list(set(c.removesuffix("-mse").removesuffix("-mae") for c in df.columns if c not in ("Dataset", "Horizon")))
    mse_model_selector=[f"{model}-mse" for model in models]
    mae_model_selector=[f"{model}-mae" for model in models]

    num_first=defaultdict(lambda: 0)
    num_compete=defaultdict(lambda: 0)
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            df_ds = df[(df["Dataset"] == ds) & (df["Horizon"] == pred_len)]
            for model in models:
                mse_order = np.unique(df_ds[mse_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten())
                mae_order = np.unique(df_ds[mae_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten())

                mse_result = df_ds[model+'-mse'].iloc[0]
                if mse_result == mse_order[0]:
                    num_first[model+'-mse'] += 1
                if not math.isnan(mse_result):
                    num_compete[model+'-mse'] += 1

                mae_result = df_ds[model+'-mae'].iloc[0]
                if mae_result == mae_order[0]:
                    num_first[model+'-mae'] += 1
                if not math.isnan(mae_result):
                    num_compete[model+'-mae'] += 1
    models.sort(key=lambda model: num_first[model+'-mse']/num_compete[model+'-mse'], reverse=True)
    models_1st_mse = []
    models_1st_mae = []
    for model in models:
        models_1st_mse.append(num_first[model+'-mse']/num_compete[model+'-mse'])
        models_1st_mae.append(num_first[model+'-mae']/num_compete[model+'-mae'])
    models_1st_mse = np.flip(np.unique(models_1st_mse))
    models_1st_mae = np.flip(np.unique(models_1st_mae))

    # Third party results.

    print(textwrap.dedent(r"""
        \newcommand{\firstres}[1]{{\textbf{\textcolor{red}{#1}}}}
        \newcommand{\secondres}[1]{{\underline{\textcolor{blue}{#1}}}}

        \newcommand{\resulttitlescale}{0.8}
        \newcommand{\resultdsscale}{0.95}
        \newcommand{\resultscale}{0.78}

        \begin{table}[htbp]
        \label{tab:third-party-results}
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
    print("|".join(["cc"] * len(models)) + "}")

    print(textwrap.dedent(r"""
        \toprule
        \multicolumn{2}{c}{\multirow{2}{*}{Models}}
        """))
    for model in models:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{model}}}}} ")
    print(r"""\\""")

    print(r"""\multicolumn{2}{c}{} """)
    for model in models:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{\\citeyearpar{{{model}}}}}}} ")
    print(r"""\\""")

    for i, model in enumerate(models):
        print(f"\\cmidrule(lr){{{2*i+3}-{2*i+4}}}")
    print(r"""\multicolumn{2}{c}{Metric} """)
    for i, model in enumerate(models):
        print(r"""& \scalebox{\resultscale}{MSE} & \scalebox{\resultscale}{MAE}""")
    print(r"""\\""")
    print(r"""\midrule""")

    for ds in DATASETS:
        print(f"""\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{\\scalebox{{\\resultdsscale}}{{{ds}}}}}}}""")
        for pred_len in PRED_LENS:
            df_ds = df[(df["Dataset"] == ds) & (df["Horizon"] == pred_len)]
            print(f"& \\scalebox{{\\resultscale}}{{{pred_len}}} ", end="")
            for model in models:
                mse_order = np.unique(df_ds[mse_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten())
                mae_order = np.unique(df_ds[mae_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten())

                mse_result = df_ds[model+'-mse'].iloc[0]
                if math.isnan(mse_result):
                    mse_result = f"N/A"
                elif mse_result == mse_order[0]:
                    mse_result = f"\\firstres{{{mse_result:.3f}}}"
                elif mse_result == mse_order[1]:
                    mse_result = f"\\secondres{{{mse_result:.3f}}}"
                else:
                    mse_result = f"{mse_result:.3f}"

                mae_result = df_ds[model+'-mae'].iloc[0]
                if math.isnan(mae_result):
                    mae_result = f"N/A"
                elif mae_result == mae_order[0]:
                    mae_result = f"\\firstres{{{mae_result:.3f}}}"
                elif mae_result == mae_order[1]:
                    mae_result = f"\\secondres{{{mae_result:.3f}}}"
                else:
                    mae_result = f"{mae_result:.3f}"

                print(f"& \\scalebox{{\\resultscale}}{{{mse_result}}} ", end="")
                print(f"& \\scalebox{{\\resultscale}}{{{mae_result}}} ")
            print(r"""\\""")
            
        # print(r"""\cmidrule(lr){2-26}""")
        print(r"""\midrule""")

    print(r"""\multicolumn{2}{c|}{\scalebox{\resultscale}{{\# $1^{\text{st}}$}}}""")
    for i, model in enumerate(models):
        model_1st_mse = num_first[model+'-mse']/num_compete[model+'-mse']
        model_1st_mse_str = f"{num_first[model+'-mse']}/{num_compete[model+'-mse']}"
        if model_1st_mse == models_1st_mse[0]:
            model_1st_mse_str = f"\\firstres{{{model_1st_mse_str}}}"
        elif model_1st_mse == models_1st_mse[1]:
            model_1st_mse_str = f"\\secondres{{{model_1st_mse_str}}}"
        print(f"& \\scalebox{{\\resultscale}}{{{model_1st_mse_str}}}", end="")
        model_1st_mae = num_first[model+'-mae']/num_compete[model+'-mae']
        model_1st_mae_str = f"{num_first[model+'-mae']}/{num_compete[model+'-mae']}"
        if model_1st_mae == models_1st_mae[0]:
            model_1st_mae_str = f"\\firstres{{{model_1st_mae_str}}}"
        elif model_1st_mae == models_1st_mae[1]:
            model_1st_mae_str = f"\\secondres{{{model_1st_mae_str}}}"
        print(f"& \\scalebox{{\\resultscale}}{{{model_1st_mae_str}}}", end="")
    print(r"""\\""")
    print(r"""\bottomrule""")

    print(textwrap.dedent(r"""
        \end{tabular}
        \end{small}
        \end{threeparttable}
    }
    \end{table}
    """))

    # Our results.

    our_df = df[["Dataset", "Horizon"]].copy()
    our_df["SotA-mse"] = df[mse_model_selector].min(axis=1, skipna=True, numeric_only=True)
    our_df["SotA-mae"] = df[mae_model_selector].min(axis=1, skipna=True, numeric_only=True)

    for model in OUR_MODELS:
        model_results_mse = []
        model_results_mae = []
        for ds in DATASETS:
            for pred_len in PRED_LENS:
                csv_file = OUR_MODELS[model](ds, pred_len) + "/results.csv"
                if os.path.exists(csv_file):
                    results = pd.read_csv(OUR_MODELS[model](ds, pred_len) + "/results.csv", index_col=0)
                    model_results_mse.append(results["mse"].values[0])
                    model_results_mae.append(results["mae"].values[0])
                else:
                    model_results_mse.append(np.nan)
                    model_results_mae.append(np.nan)
        our_df[model+'-mse'] = model_results_mse
        our_df[model+'-mae'] = model_results_mae

    our_models=list(set(c.removesuffix("-mse").removesuffix("-mae") for c in our_df.columns if c not in ("Dataset", "Horizon")))
    our_mse_model_selector=[f"{model}-mse" for model in our_models]
    our_mae_model_selector=[f"{model}-mae" for model in our_models]

    our_num_first=defaultdict(lambda: 0)
    our_num_compete=defaultdict(lambda: 0)
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            df_ds = df[(df["Dataset"] == ds) & (df["Horizon"] == pred_len)]
            our_df_ds = our_df[(our_df["Dataset"] == ds) & (our_df["Horizon"] == pred_len)]
            for model in our_models:
                mse_order = np.unique(np.hstack([df_ds[mse_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten(), 
                                      our_df_ds[our_mse_model_selector].sort_values(our_df_ds.index[0], axis=1).values.flatten()]))
                mae_order = np.unique(np.hstack([df_ds[mae_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten(), 
                                      our_df_ds[our_mae_model_selector].sort_values(our_df_ds.index[0], axis=1).values.flatten()]))

                mse_result = our_df_ds[model+'-mse'].iloc[0]
                if mse_result == mse_order[0]:
                    our_num_first[model+'-mse'] += 1
                if not math.isnan(mse_result):
                    our_num_compete[model+'-mse'] += 1

                mae_result = our_df_ds[model+'-mae'].iloc[0]
                if mae_result == mae_order[0]:
                    our_num_first[model+'-mae'] += 1
                if not math.isnan(mae_result):
                    our_num_compete[model+'-mae'] += 1
    def model_score(model, suffix):
        if our_num_compete[model+suffix] == 0:
            return -1
        return our_num_first[model+suffix]/our_num_compete[model+suffix]
    our_models.sort(key=lambda model: model_score(model, "-mse"), reverse=True)
    our_models_1st_mse = []
    our_models_1st_mae = []
    for model in our_models:
        our_models_1st_mse.append(model_score(model, "-mse"))
        our_models_1st_mae.append(model_score(model, "-mae"))
    our_models_1st_mse = np.flip(np.unique(np.hstack((models_1st_mse, our_models_1st_mse))))
    our_models_1st_mae = np.flip(np.unique(np.hstack((models_1st_mae, our_models_1st_mae))))

    print(textwrap.dedent(r"""
        \renewcommand{\resulttitlescale}{0.8}
        \renewcommand{\resultdsscale}{0.95}
        \renewcommand{\resultscale}{0.78}

        \begin{table}[htbp]
        \label{tab:our-results}
        \vskip -0.0in
        \vspace{3pt}
        \renewcommand{\arraystretch}{0.9}
        \centering
        \resizebox{0.7\columnwidth}{!}{
        \begin{threeparttable}
        \begin{small}
        \renewcommand{\multirowsetup}{\centering}
        \setlength{\tabcolsep}{1pt}
        """))

    print(textwrap.dedent(r"""\begin{tabular}{c|c|"""), end="")
    print("|".join(["cc"] * len(our_models)) + "}")

    print(textwrap.dedent(r"""
        \toprule
        \multicolumn{2}{c}{\multirow{2}{*}{Models}}
        """))
    for model in our_models:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{{model}}}}} ")
    print(r"""\\""")

    print(r"""\multicolumn{2}{c}{} """)
    for model in our_models:
        print(f"& \\multicolumn{{2}}{{c}}{{\\scalebox{{\\resulttitlescale}}{{\\citeyearpar{{{model}}}}}}} ")
    print(r"""\\""")

    for i, model in enumerate(our_models):
        print(f"\\cmidrule(lr){{{2*i+3}-{2*i+4}}}")
    print(r"""\multicolumn{2}{c}{Metric} """)
    for i, model in enumerate(our_models):
        print(r"""& \scalebox{\resultscale}{MSE} & \scalebox{\resultscale}{MAE}""")
    print(r"""\\""")
    print(r"""\midrule""")

    for ds in DATASETS:
        print(f"""\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{\\scalebox{{\\resultdsscale}}{{{ds}}}}}}}""")
        for pred_len in PRED_LENS:
            df_ds = df[(df["Dataset"] == ds) & (df["Horizon"] == pred_len)]
            our_df_ds = our_df[(our_df["Dataset"] == ds) & (our_df["Horizon"] == pred_len)]
            print(f"& \\scalebox{{\\resultscale}}{{{pred_len}}} ", end="")
            for model in our_models:
                mse_order = np.unique(np.hstack([df_ds[mse_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten(), 
                                      our_df_ds[our_mse_model_selector].sort_values(our_df_ds.index[0], axis=1).values.flatten()]))
                mae_order = np.unique(np.hstack([df_ds[mae_model_selector].sort_values(df_ds.index[0], axis=1).values.flatten(), 
                                      our_df_ds[our_mae_model_selector].sort_values(our_df_ds.index[0], axis=1).values.flatten()]))

                mse_result = our_df_ds[model+'-mse'].iloc[0]
                if math.isnan(mse_result):
                    mse_result = f"N/A"
                elif mse_result == mse_order[0]:
                    mse_result = f"\\firstres{{{mse_result:.3f}}}"
                elif mse_result == mse_order[1]:
                    mse_result = f"\\secondres{{{mse_result:.3f}}}"
                else:
                    mse_result = f"{mse_result:.3f}"

                mae_result = our_df_ds[model+'-mae'].iloc[0]
                if math.isnan(mae_result):
                    mae_result = f"N/A"
                elif mae_result == mae_order[0]:
                    mae_result = f"\\firstres{{{mae_result:.3f}}}"
                elif mae_result == mae_order[1]:
                    mae_result = f"\\secondres{{{mae_result:.3f}}}"
                else:
                    mae_result = f"{mae_result:.3f}"

                print(f"& \\scalebox{{\\resultscale}}{{{mse_result}}} ", end="")
                print(f"& \\scalebox{{\\resultscale}}{{{mae_result}}} ")
            print(r"""\\""")
            
        # print(r"""\cmidrule(lr){2-26}""")
        print(r"""\midrule""")

    print(r"""\multicolumn{2}{c|}{\scalebox{\resultscale}{{\# $1^{\text{st}}$}}}""")
    for i, model in enumerate(our_models):
        if our_num_compete[model+'-mse'] == 0:
            print(r"& \scalebox{\resultscale}{N/A} & \scalebox{\resultscale}{N/A}")
            continue
        model_1st_mse = our_num_first[model+'-mse']/our_num_compete[model+'-mse']
        model_1st_mse_str = f"{our_num_first[model+'-mse']}/{our_num_compete[model+'-mse']}"
        if model_1st_mse == our_models_1st_mse[0]:
            model_1st_mse_str = f"\\firstres{{{model_1st_mse_str}}}"
        elif model_1st_mse == our_models_1st_mse[1]:
            model_1st_mse_str = f"\\secondres{{{model_1st_mse_str}}}"
        print(f"& \\scalebox{{\\resultscale}}{{{model_1st_mse_str}}}", end="")
        model_1st_mae = our_num_first[model+'-mae']/our_num_compete[model+'-mae']
        model_1st_mae_str = f"{our_num_first[model+'-mae']}/{our_num_compete[model+'-mae']}"
        if model_1st_mae == our_models_1st_mae[0]:
            model_1st_mae_str = f"\\firstres{{{model_1st_mae_str}}}"
        elif model_1st_mae == our_models_1st_mae[1]:
            model_1st_mae_str = f"\\secondres{{{model_1st_mae_str}}}"
        print(f"& \\scalebox{{\\resultscale}}{{{model_1st_mae_str}}}")
    print(r"""\\""")
    print(r"""\bottomrule""")

    print(textwrap.dedent(r"""
        \end{tabular}
        \end{small}
        \end{threeparttable}
    }
    \end{table}
    """))