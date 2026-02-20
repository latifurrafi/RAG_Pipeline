# csv_loader.py

import pandas as pd
from config import DATA_PATH


def load_csv_documents():
    df = pd.read_csv(DATA_PATH)

    documents = []

    for _, row in df.iterrows():

        text = "\n".join(
            f"{col}: {row[col]}"
            for col in df.columns
            if pd.notna(row[col])
        )

        documents.append(text)

    return documents