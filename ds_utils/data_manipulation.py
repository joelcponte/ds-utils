def cap_freq(df, field, min_freq, new_label):
    df = df.copy()
    to_keep = df[field].value_counts().loc[lambda x: x > min_freq].index
    df.loc[~df[field].isin(to_keep), field] = new_label
    return df


def cap_n_categories(df, field, max_cats, new_label):
    df = df.copy()
    to_keep = df[field].value_counts().head(max_cats).index
    df.loc[~df[field].isin(to_keep), field] = new_label
    return df


def freq_table(df, x, y):
    t = df.groupby([x, y]).size()
    return t.reset_index().pivot(x, y, 0)

