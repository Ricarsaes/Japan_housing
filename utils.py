import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class LabelEncoderTransformer:
    def __init__(self):
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders_[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            le = self.label_encoders_[col]

            # Handle unseen labels
            test_values = X[col].unique()
            train_values = le.classes_
            unseen_labels = set(test_values) - set(train_values)
            le.classes_ = np.append(le.classes_, list(unseen_labels))

            X_transformed[col] = le.transform(X[col])
        return X_transformed


def train_random_forest_old(X_train, y_train, X_test, y_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    label_encoders = {}

    if not categorical_cols.empty:
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            label_encoders[col] = le

            test_values = X_test[col].unique()
            train_values = le.classes_

            unseen_labels = set(test_values) - set(train_values)
            le.classes_ = np.append(le.classes_, list(unseen_labels))
            X_test[col] = le.transform(X_test[col])

    assert X_train.select_dtypes(
        include=["object"]
    ).empty, "X_train still has non-numeric columns!"
    assert X_test.select_dtypes(
        include=["object"]
    ).empty, "X_test still has non-numeric columns!"

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred)

    print(f"Random Forest R-squared: {r2_rf:.4f}")

    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    top_12_features = importance_df["Feature"].head(12).tolist()

    return {"model": rf_model, "r2": r2_rf, "top_12_features": top_12_features}


def train_random_forest(X_train, y_train, X_test, y_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "label_encoder",
                Pipeline(
                    [
                        ("label_enc", LabelEncoderTransformer()),
                    ]
                ),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2_rf = r2_score(y_test, y_pred)

    print(f"Random Forest R-squared: {r2_rf:.4f}")

    feature_names = list(categorical_cols) + list(
        X_train.select_dtypes(include=["number"]).columns
    )
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    top_12_features = importance_df["Feature"].head(12).tolist()

    return {"model": pipeline, "r2": r2_rf, "top_12_features": top_12_features}


def train_xgboost(X_train, y_train, X_test, y_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    numerical_columns = X_train.select_dtypes(include=["number"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred_xgb = pipeline.predict(X_test)

    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"XGB Regressor R-squared: {r2_xgb:.4f}")

    return {"model": pipeline, "r2": r2_xgb}


def train_stacker(X_train, y_train, X_test, y_test, top_features):

    X_train = X_train[top_features]
    X_test = X_test[top_features]

    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    base_models = [
        ("linear", LinearRegression(n_jobs=-1)),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
        ),
        ("xgboost", XGBRegressor(n_estimators=70, learning_rate=0.1, random_state=42)),
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        n_jobs=-1,
        passthrough=False,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("stacking", stacking_model)]
    )

    pipeline.fit(X_train, y_train)
    y_pred_ensemble = pipeline.predict(X_test)

    r2_stacker = r2_score(y_test, y_pred_ensemble)
    print(f"Stacking Regressor R-squared: {r2_stacker:.4f}")

    return {"model": pipeline, "r2": r2_stacker}


def cap_outliers(series, lower_percentile=0.01, upper_percentile=0.99):
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return np.clip(series, lower, upper)


def plot_scores(random_forest_dict, xgboost_dict, stacking_dict):
    performance_data = []

    for prefecture in random_forest_dict.keys():
        performance_data.append(
            {
                "Prefecture": prefecture,
                "Model": "Random Forest",
                "R2_Score": random_forest_dict[prefecture]["r2"],
            }
        )
        performance_data.append(
            {
                "Prefecture": prefecture,
                "Model": "XGBoost",
                "R2_Score": xgboost_dict[prefecture]["r2"],
            }
        )
        performance_data.append(
            {
                "Prefecture": prefecture,
                "Model": "Stacking Regressor",
                "R2_Score": stacking_dict[prefecture]["r2"],
            }
        )

    performance_df = pd.DataFrame(performance_data)

    plt.figure(figsize=(16, 10))
    sns.barplot(
        data=performance_df,
        x="Prefecture",
        y="R2_Score",
        hue="Model",
        palette="viridis",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Performance by Prefecture (R² Score)")
    plt.ylabel("R² Score")
    plt.xlabel("Prefecture")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
