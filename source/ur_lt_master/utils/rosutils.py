import pandas as pd


def load_rosbag_csv(path):
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    return df


def convert_timestamp(df, columns, new_names):
    if len(columns) != len(new_names):
        raise ValueError('a new name is needed for every timestamp column')

    for i, c in enumerate(columns):
        t = pd.to_datetime(df[c], unit='ns')
        df[new_names[i]] = t
        df[c] /= 1e9

    return df
