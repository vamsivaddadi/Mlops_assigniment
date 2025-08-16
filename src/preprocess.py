import pandas as pd
from pathlib import Path

def load_data(file_path):
    """Load Iris dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    df = df.drop_duplicates()

    if df.isnull().sum().any():
        df = df.dropna()

    if df['variety'].dtype == 'object':
        df.loc[:, 'variety'] = df['variety'].astype('category').cat.codes

    return df

if __name__ == "__main__":
    raw_path = Path("data/raw/iris.csv")
    processed_path = Path("data/processed/iris_processed.csv")

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(raw_path)
    df_processed = preprocess_data(df_raw)

    df_processed.to_csv(processed_path, index=False)
