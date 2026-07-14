import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]


def load_and_preprocess(data_path):
    """
    Load the wholesale customer dataset and prepare the spending
    variables for clustering.
    """
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].copy()

    # Reduce the effect of highly skewed spending values
    X_log = np.log1p(X)

    # Standardise the transformed features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    return df, X_scaled, scaler
