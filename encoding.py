import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

def encode_categorical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None,
    *,
    ohe_max_cardinality: int = 3,
    high_card_strategy: str = "ordinal",  # "ordinal" | "target"
    drop_first: bool = False,
    dtype: type = float,
    keep_original: bool = False,          # <— NEW
    encoded_suffix: str = "__enc",        # <— NEW: avoids name clashes
):
    """
    Encode categorical features with:
      - OneHotEncoder for low-cardinality features
      - OrdinalEncoder or TargetEncoder for high-cardinality features

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train : pd.Series, required if high_card_strategy="target"
    ohe_max_cardinality : int
        Threshold for deciding OHE vs high-card strategy.
    high_card_strategy : {"ordinal", "target"}
        Encoding method for high-cardinality features.
    drop_first : bool
        Drop first level in one-hot to reduce collinearity.
    dtype : type
        Output dtype for encoded columns.
    keep_original : bool
        If True, keep the original categorical columns beside their encoded versions.
    encoded_suffix : str
        Suffix for high-cardinality encoded column names to avoid collisions.

    Returns
    -------
    X_train_enc, X_test_enc : pd.DataFrame
    artifacts : dict of fitted encoders
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Identify categorical features
    cat_cols = [
        c for c in X_train.columns
        if X_train[c].dtype.name in ("object", "category")
    ]

    nuniques = {c: X_train[c].nunique(dropna=True) for c in cat_cols}
    ohe_cols = [c for c in cat_cols if nuniques[c] <= ohe_max_cardinality]
    high_cols = [c for c in cat_cols if nuniques[c] > ohe_max_cardinality]

    # Containers
    Xtr_parts, Xte_parts = [], []
    artifacts = {}

    # Pass through numeric / non-categorical columns
    passthrough = [c for c in X_train.columns if c not in cat_cols]
    Xtr_parts.append(X_train[passthrough])
    Xte_parts.append(X_test[passthrough])

    # Optionally keep original categoricals
    if keep_original and cat_cols:
        Xtr_parts.append(X_train[cat_cols])
        Xte_parts.append(X_test[cat_cols])

    # --- One-hot encoding (low-cardinality) ---
    if ohe_cols:
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first" if drop_first else None,
            dtype=dtype
        )
        tr = ohe.fit_transform(X_train[ohe_cols])
        te = ohe.transform(X_test[ohe_cols])

        tr_df = pd.DataFrame(tr, index=X_train.index, columns=ohe.get_feature_names_out(ohe_cols))
        te_df = pd.DataFrame(te, index=X_test.index, columns=ohe.get_feature_names_out(ohe_cols))

        Xtr_parts.append(tr_df)
        Xte_parts.append(te_df)
        artifacts["ohe"] = ohe

    # --- High-cardinality encoding (ordinal/target) ---
    if high_cols:
        if high_card_strategy == "ordinal":
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan
            )
            tr = ord_enc.fit_transform(X_train[high_cols])
            te = ord_enc.transform(X_test[high_cols])

            # rename to avoid collision with originals
            enc_cols = [f"{c}{encoded_suffix}" for c in high_cols]
            tr_df = pd.DataFrame(tr, index=X_train.index, columns=enc_cols).astype(dtype)
            te_df = pd.DataFrame(te, index=X_test.index, columns=enc_cols).astype(dtype)

            Xtr_parts.append(tr_df)
            Xte_parts.append(te_df)
            artifacts["ordinal"] = ord_enc

        elif high_card_strategy == "target":
            if y_train is None:
                raise ValueError("y_train must be provided when using target encoding.")
            te_enc = TargetEncoder()
            tr = te_enc.fit_transform(X_train[high_cols], y_train)
            te = te_enc.transform(X_test[high_cols])

            enc_cols = [f"{c}{encoded_suffix}" for c in high_cols]
            tr_df = pd.DataFrame(tr, index=X_train.index, columns=enc_cols).astype(dtype)
            te_df = pd.DataFrame(te, index=X_test.index, columns=enc_cols).astype(dtype)

            Xtr_parts.append(tr_df)
            Xte_parts.append(te_df)
            artifacts["target"] = te_enc

        else:
            raise ValueError("high_card_strategy must be 'ordinal' or 'target'.")

    # --- Combine all parts ---
    X_train_enc = pd.concat(Xtr_parts, axis=1)
    X_test_enc = pd.concat(Xte_parts, axis=1)

    return X_train_enc, X_test_enc, artifacts
