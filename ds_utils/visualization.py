import warnings

import pandas as pd
import plotnine as pn
from typing import Optional, Dict
import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import patchworklib as pw
from IPython.core.display import display, HTML


def print_categorical_frequencies(df):
    for y in df.columns:
        if not (df[y].dtype == np.float64 or df[y].dtype == np.int64):
            display(HTML(f"<h2> {y}"))
            display(df[y].value_counts())
            print("\n")


def sns_pairplot(df):
    """
    Scales with ncols^2!
    """
    # pick
    # g = sns.PairGrid(df.iloc[:,[1,2,3,4,5,6,80]])
    # g.map_lower(sns.histplot)
    # g.map_diag(sns.histplot, kde=True)
    # g.map_upper(sns.histplot)
    # return g
    pass


def sns_clustermap(df):
    return sns.clustermap(
        df.fillna(df.mean()).select_dtypes(["int", "float"]), z_score=1
    )


def sns_corrplot(df):
    # add , annot=True for correlation values
    return sns.clustermap(
        df.select_dtypes(["int", "float"]).corr(), cmap="vlag", vmin=-1, vmax=1
    )


def plot_multiple_histograms(df, lib="seaborn", columns=None):
    if columns is None:
        columns = df.columns.tolist()
    if lib == "ggplot":  # not very reliable
        return gg_multiple_histograms(df[columns])
    elif lib == "seaborn":
        return sns_multiple_histograms(df[columns])
    else:
        raise ValueError(
            f"Parameter 'lib' must libe 'ggplot' or 'seaborn'. Got {lib} instead"
        )


def gg_multiple_histograms(df):
    num_plots = df.shape[1]
    num_cols = min(math.ceil(np.sqrt(num_plots)), 9)

    # create plots
    plots = []
    for y in df.columns:
        p = gg_histogram(df, var=y, bins=10, labs={"title": y})
        plots.append(p)

    p = [pw.load_ggplot(i, figsize=(3, 2)) for i in plots]

    s = []
    for i in range(num_plots // num_cols + (num_plots % num_cols != 0)):
        s += [[]]
        for j in range(min(num_cols, num_plots - i * num_cols)):
            s[-1] += [f"p[{i * num_cols + j}]"]

    return eval("(" + "/".join([f"({'|'.join(i)})" for i in s]) + ").savefig()")


def sns_multiple_histograms(df):
    num_plots = df.shape[1]
    num_cols = min(math.ceil(np.sqrt(num_plots)), 9)
    num_rows = math.ceil(num_plots / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    for ind, col in enumerate(df.columns):
        i = math.floor(ind / num_cols)
        j = ind - i * num_cols

        if num_rows == 1:
            if num_cols == 1:
                sns.histplot(df[col], kde=True, ax=axs, color="#5F9E6E")
            else:
                sns.histplot(df[col], kde=True, ax=axs[j], color="#5F9E6E")
        else:
            sns.histplot(df[col], kde=True, ax=axs[i, j], color="#5F9E6E")


def gg_histogram(
    df: pd.DataFrame,
    var: str,
    bins: Optional[int] = None,
    labs: Optional[Dict] = None,
    axis_text_x: bool = True,
    axis_text_x_angle: str = 15,
):
    labs = {} if labs is None else labs

    if axis_text_x:
        axis_text_x = pn.element_text(angle=axis_text_x_angle)
    else:
        axis_text_x = pn.element_blank()

    return (
        pn.ggplot(df, pn.aes(var))
        + pn.geom_histogram(bins=bins, fill="#64345F", alpha=0.9)
        + pn.theme_538()
        + pn.theme(figure_size=[5, 3], axis_text_x=axis_text_x)
        + pn.labs(**labs)
    )


def sns_bivariate_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    min_freq: int = 30,
    max_categories: int = 10,
    ascending: bool = False,
    agg_fun: str = "median",
    sort: str = "categorical",  # yes, no
    target_type: str = "numeric",  # or binary
    qcut: bool = True,
    cut_nbins: int = 10,
):
    df = df.copy()

    if (
        pd.api.types.is_numeric_dtype(df[x])
        and df[x].value_counts().shape[0] > df.shape[0] / 3
    ):
        if qcut:
            df[x] = pd.qcut(df[x], q=cut_nbins)
        else:
            df[x] = pd.cut(df[x], bins=cut_nbins)
    else:
        valid_values = (
            df[x].value_counts().loc[lambda x: x > min_freq].head(max_categories).index
        )
        df = df.loc[df[x].isin(valid_values), [x, y]]
        if sort == "yes" or (
            sort == "categorical" and not pd.api.types.is_numeric_dtype(df[x])
        ):
            sorting = (
                df.groupby(x)[y].agg(agg_fun).sort_values(ascending=ascending).index
            )
            df[x] = pd.Categorical(df[x], categories=sorting)

    if not len(df):
        raise Exception(
            "Empty data frame after filters! Check if min_freq is too high."
        )

    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5), gridspec_kw={"height_ratios": [1, 4]}
    )
    sns.countplot(ax=axes[0], data=df, x=x, color="#5F9E6E").set(title=x)
    axes[1].tick_params(axis="x", rotation=30)
    if target_type == "numeric":
        sns.boxplot(ax=axes[1], data=df, x=x, y=y, color="#5F9E6E")
    elif target_type == "binary":
        df = df.groupby(x)[y].mean().reset_index()
        sns.barplot(ax=axes[1], data=df, x=x, y=y, color="#5F9E6E")
        axes[1].set_ylim(0, 1)


def gg_boxplot_bar_combined(
    df: pd.DataFrame,
    x: str,
    y: str,
    min_freq: int = 30,
    max_categories: int = 10,
    agg_fun: str = "median",
    sort: bool = True,
):
    """
    Plots a bar plot and a box plot below
    """
    gg_bar(
        df=df,
        x=x,
        y=y,
        min_freq=min_freq,
        max_categories=max_categories,
        agg_fun=agg_fun,
        sort=sort,
    ).draw()
    gg_boxplot(
        df=df,
        x=x,
        y=y,
        labs={"title": x},
        min_freq=min_freq,
        max_categories=max_categories,
        agg_fun=agg_fun,
        sort=sort,
    ).draw()


def gg_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    min_freq: int = 30,
    max_categories: int = 10,
    ascending: bool = False,
    agg_fun: str = "median",
    sort: bool = True,
):

    valid_values = (
        df[x].value_counts().loc[lambda x: x > min_freq].head(max_categories).index
    )
    df_plot = df.loc[df[x].isin(valid_values), [x, y]]
    if sort:
        sorting = df.groupby(x)[y].agg(agg_fun).sort_values(ascending=ascending).index
        df_plot[x] = pd.Categorical(df_plot[x], categories=sorting)

    if not len(df_plot):
        raise Exception(
            "Empty data frame after filters! Check if min_freq is too high."
        )

    return (
        pn.ggplot(df_plot, pn.aes(x))
        + pn.geom_histogram(binwidth=1, fill="#64345F", alpha=0.9)
        + pn.theme_538()
        + pn.theme(figure_size=[5, 1], axis_text_x=pn.element_blank())
        + pn.labs(x=None, title=x)
    )


def gg_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    labs: Optional[Dict] = None,
    min_freq: int = 30,
    max_categories: int = 10,
    ascending: bool = False,
    agg_fun: str = "median",
    sort: bool = True,
):

    valid_values = (
        df[x].value_counts().loc[lambda l: l > min_freq].head(max_categories).index
    )
    df_plot = df.loc[df[x].isin(valid_values), [x, y]]
    if sort:
        sorting = (
            df_plot.groupby(x)[y].agg(agg_fun).sort_values(ascending=ascending).index
        )
        df_plot[x] = pd.Categorical(df_plot[x], categories=sorting)
    if not len(df_plot):
        raise Exception(
            "Empty data frame after filters! Check if min_freq is too high."
        )

    return (
        pn.ggplot(df_plot, pn.aes(x=x, y=y))
        + pn.geom_boxplot(alpha=0.9)
        +
        #             pn.scale_y_log10() +
        pn.theme_538()
        + pn.theme(figure_size=[5, 3], axis_text_x=pn.element_text(angle=90))
        + pn.labs()
    )

