
def describe_dataset(df):
    display(df.head())
    display(df.describe(include="all", datetime_is_numeric=True))