import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(method='ffill')

    return df