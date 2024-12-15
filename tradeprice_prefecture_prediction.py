import os
from pathlib import Path

import joblib
import pandas as pd

ROOT_MODELS = "best_models"
ROOT_IMPUTERS = "imputer_models"


def load_models_from_directory(directory: str) -> dict[str, object]:
    models = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            prefecture = Path(filename).stem.split("_")[-1]
            models[prefecture] = joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model {filename}: {e}")
    return models


def load_imputers_from_directory(directory: str) -> dict[str, object]:
    imputers = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            prefecture = Path(filename).stem
            imputers[prefecture] = joblib.load(filepath)
        except Exception as e:
            print(f"Error loading imputer {filename}: {e}")
    return imputers


def predict_for_dataframe(
    df: pd.DataFrame,
    models_dict: dict[str, object],
    imputers_dict: dict[str, object] = None,
) -> pd.Series:
    predictions = []
    for _, row in df.iterrows():
        prefecture = row["Prefecture"]

        imputer = imputers_dict[prefecture]
        row = imputer.transform(row)

        model = models_dict.get(prefecture)
        if model is None:
            raise ValueError(f"No model found for prefecture: {prefecture}")

        prediction = model.predict(row.to_frame().T)[0]
        predictions.append(prediction)

    return pd.Series(predictions)


def main():
    models_dict = load_models_from_directory(ROOT_MODELS)
    imputers_dict = load_imputers_from_directory(ROOT_IMPUTERS)

    df_test = pd.read_csv("japan_housing_data/test.csv")

    predictions = predict_for_dataframe(df_test, models_dict, imputers_dict)

    print(predictions)


if __name__ == "__main__":
    main()
