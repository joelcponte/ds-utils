import warnings

import pandas as pd
import plotnine as pn
from typing import Optional, Dict


def gg_boxplot_bar_combined(
    df: pd.DataFrame,
    x: str,
    y: str,
    min_freq: int = 30,
    max_categories: int = 10,
    agg_fun: str = "median",
    sort: bool = True,
):
    gg_bar(df=df, x=x, y=y, min_freq=min_freq, max_categories=max_categories, agg_fun=agg_fun, sort=sort).draw()
    gg_boxplot(df=df, x=x, y=y, labs={"title": x}, min_freq=min_freq, max_categories=max_categories, agg_fun=agg_fun, sort=sort).draw()


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

    valid_values = df[x].value_counts().loc[lambda x: x > min_freq].head(max_categories).index
    df_plot = df.loc[df[x].isin(valid_values), [x, y]]
    if sort:
        sorting = df.groupby(x)[y].agg(agg_fun).sort_values(ascending=ascending).index
        df_plot[x] = pd.Categorical(df_plot[x], categories=sorting)

    if not len(df_plot):
        warnings.warn("Empty data frame after filters!")

    return (pn.ggplot(df_plot,  pn.aes(x)) +
            pn.geom_histogram(binwidth=1, fill="#64345F", alpha=0.9) +
            pn.theme_538() +
            pn.theme(figure_size=[5,1],
                     axis_text_x=pn.element_blank()) +
            pn.labs(x=None, title=x)
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

    valid_values = df[x].value_counts().loc[lambda l: l > min_freq].head(max_categories).index
    df_plot = df.loc[df[x].isin(valid_values), [x, y]]
    if sort:
        sorting = df.groupby(x)[y].agg(agg_fun).sort_values(ascending=ascending).index
        df_plot[x] = pd.Categorical(df_plot[x], categories=sorting)
    if not len(df_plot):
        warnings.warn("Empty data frame after filters!")
    return (pn.ggplot(df_plot,  pn.aes(x=x, y=y)) +
            pn.geom_boxplot(alpha=0.9) +
            #             pn.scale_y_log10() +
            pn.theme_538() +
            pn.theme(figure_size=[5,3],
                     axis_text_x=pn.element_text(angle=90)) +
            pn.labs()
            )
