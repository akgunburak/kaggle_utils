import pandas as pd
import numpy as np
from typing import List, Iterable, Tuple, Dict, Any, Optional, Callable
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from boruta import BorutaPy
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import clone, is_classifier, is_regressor
from category_encoders import TargetEncoder


def drop_high_null_columns(df, threshold=50, show=False):
    null_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = null_percent[null_percent > threshold].index
    if show:
        print(f"Columns to drop: {list(cols_to_drop)}")
    return cols_to_drop.to_list()


def drop_correlated_vars(X, y, corr_thr=0.9, matrix_to_excel=False, verbose=True):
    numeric_cols = X.select_dtypes(include=['number']).columns
    corr_matrix = X[numeric_cols].corr()
    if matrix_to_excel:
        corr_matrix.to_excel("correlation_matrix.xlsx")
    vars_to_del = []

    for var1 in corr_matrix.columns:
        for var2 in corr_matrix.columns:
            if var1 != var2:
                corr_coef = corr_matrix.loc[var1, var2]
                if abs(corr_coef) > corr_thr:
                    r_sq_var1 = y.corr(X[var1])#r2_score(y, X[var1])
                    r_sq_var2 = y.corr(X[var2])#r2_score(y, X[var2])
                    if verbose:
                        print(var1, "---->", var2, ' r:', round(corr_coef, 4), ' r2:', round(r_sq_var1, 4), round(r_sq_var2, 4))
                    if r_sq_var1 < r_sq_var2:
                        vars_to_del.append(var1)
                    else:
                        vars_to_del.append(var2)

    vars_to_del = list(set(vars_to_del))
    print("*********Variables dropped by correlation(each other):")
    for variable in vars_to_del:
        print(variable)
    print("*********Number of variables dropped by correlation(each other): ", len(vars_to_del), "\n")
    return vars_to_del


def drop_by_pps(
    X: pd.DataFrame,
    y: Iterable,
    thr: float = 0.6,
    target_name: str = "target",
    show=False
) -> List[str]:
    """
    Finds redundant feature pairs using PPS and determines which features to drop
    based on weaker PPS-to-target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Iterable
        Target vector aligned with X.
    thr : float, default=0.6
        If PPS(feature_i -> feature_j) > threshold, treat (i, j) as redundant.
    target_name : str, default="target"
        Temporary column name used when computing PPS to the target.

    Returns
    -------
    to_drop : List[str]
        List of features to drop due to redundancy.
    """
    # Compute PPS to target for all features
    df_target = X.assign(**{target_name: y})
    pps_to_target = {
        f: pps.score(df_target, f, target_name)["ppscore"] for f in X.columns
    }

    to_drop = []
    cols = list(X.columns)

    # Find redundant pairs and decide which one to drop
    for i, f1 in enumerate(cols):
        for f2 in cols[i+1:]:
            score = pps.score(X, f1, f2)["ppscore"]
            if score > thr:
                if show:
                    print(f"Redundant pair: ({f1}, {f2}), PPS={score:.3f}")
                if pps_to_target[f1] >= pps_to_target[f2]:
                    to_drop.append(f2)
                else:
                    to_drop.append(f1)
                    
    print("*********Number of variables dropped by PPS: ", len(set(to_drop).to_list()), "\n")    
    return sorted(set(to_drop))


def boruta_feature_selection(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    task: str,                              # must be "classification" or "regression"
    rf_n_estimators: int = 600,
    rf_max_depth: Optional[int] = None,
    rf_random_state: int = 42,
    rf_class_weight: Optional[str] = None,  # only for classification
    boruta_verbose: int = 0,
    boruta_n_estimators: str | int = 'auto'
) -> Tuple[list[str], Dict[str, Any]] | list[str]:
    """Boruta feature selection with TargetEncoder for categoricals.
    
    Parameters
    ----------
    task : str
        Must be "classification" or "regression".
    """

    X = X.copy()

    # --- split & impute
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median") if num_cols else None
    if num_imputer:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent") if cat_cols else None
    if cat_imputer:
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # --- Target encoding
    enc = None
    if cat_cols:
        enc = TargetEncoder(cols=cat_cols, smoothing=0.3)
        X = enc.fit_transform(X, y)

    # --- choose estimator
    if task == "classification":
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            n_jobs=-1,
            class_weight=rf_class_weight,
            random_state=rf_random_state,
        )
    elif task == "regression":
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            n_jobs=-1,
            random_state=rf_random_state,
        )
    else:
        raise ValueError("task must be either 'classification' or 'regression'.")

    # --- Boruta
    boruta = BorutaPy(
        estimator=rf,
        n_estimators=boruta_n_estimators,
        verbose=boruta_verbose,
        random_state=rf_random_state
    )

    Xb = X.values
    yb = np.asarray(y).ravel()
    boruta.fit(Xb, yb)

    feature_names = X.columns.tolist()
    support_mask = boruta.support_.astype(bool)
    selected_features = [f for f, keep in zip(feature_names, support_mask) if keep]

    return selected_features


def adversarial_validation(
    df: pd.DataFrame,
    folds: Iterable[Tuple[np.ndarray, np.ndarray]],
    target_col: str = "__is_test__",
    lgb_params: Dict[str, Any] = None,
    num_boost_round: int = 5000
):
    """
    Adversarial Validation using precomputed folds.

    Parameters
    ----------
    df : DataFrame
        Combined dataset containing both train & test rows and a binary flag column `target_col`.
        0 = train, 1 = test.
    folds : iterable of (train_idx, valid_idx)
        Precomputed folds (e.g., from StratifiedKFold.split). Indices must align with `df`.
    target_col : str
        Column name of the 0/1 adversarial target flag.
    lgb_params : dict
        LightGBM parameters (defaults provided if None).
    num_boost_round : int

    Returns
    -------
    results : dict
        {
          'auc': float,
          'oof_pred': np.ndarray,
          'fold_aucs': List[float],
          'models': List[lgb.Booster],
          'features': List[str],
          'importances': pd.DataFrame,          # long form
          'avg_importance': pd.DataFrame        # aggregated
        }
    """
    if target_col not in df.columns:
        raise ValueError(f"`{target_col}` not found in df columns.")

    # infer features automatically (everything except target)
    features: List[str] = [c for c in df.columns if c != target_col]

    X = df[features]
    y = df[target_col].astype(int).values

    if lgb_params is None:
        lgb_params = dict(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=64,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.0,
            verbosity=-1,
            seed=42,
            feature_fraction_seed=42,
            bagging_seed=42,
            n_jobs=-1,
        )

    oof = np.zeros(len(df))
    fold_aucs: List[float] = []
    models: List[lgb.Booster] = []
    imp_rows = []

    for i, (train_idx, valid_idx) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=features, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, feature_name=features, free_raw_data=False)

        model = lgb.train(
            lgb_params,
            dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=num_boost_round,
        )

        models.append(model)
        preds = model.predict(X_va, num_iteration=model.best_iteration)
        oof[valid_idx] = preds
        auc = roc_auc_score(y_va, preds)
        fold_aucs.append(auc)
        print(f"[Fold {i}] best_iter={model.best_iteration} | AUC={auc:.6f}")

        # importances
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
        for f, g, s in zip(features, gain, split):
            imp_rows.append({"fold": i, "feature": f, "gain": float(g), "split": int(s)})

    overall_auc = roc_auc_score(y, oof)
    print("\n=== Adversarial Validation Report ===")
    print(f"OOF AUC: {overall_auc:.6f}")
    print("Per-fold:", [f"{a:.6f}" for a in fold_aucs])

    importances = pd.DataFrame(imp_rows)
    avg_importance = (
        importances.groupby("feature", as_index=False)
        .agg(gain_mean=("gain", "mean"),
             gain_std=("gain", "std"),
             split_mean=("split", "mean"),
             split_std=("split", "std"))
        .sort_values("gain_mean", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "auc": float(overall_auc),
        "oof_pred": oof,
        "fold_aucs": fold_aucs,
        "models": models,
        "features": features,
        "importances": importances,
        "avg_importance": avg_importance,
    }


def custom_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    estimator,
    n_features_to_drop: int = 5,
    n_features_to_stop: int = 20,
    early_stopping_rounds: int = 50,
    scorer: Optional[Callable[..., float]] = None,
) -> Dict[str, Any]:
    """
    Iterative feature elimination using user-provided estimator and folds.
    Works for classification (binary/multiclass) and regression.

    Args
    ----
    X, y          : Data and targets. X must be a DataFrame.
    folds         : List of (train_idx, val_idx) numpy arrays.
    estimator     : Unfitted sklearn-style estimator. Must expose feature_importances_ after fit.
    n_features_to_drop : Features to remove per iteration.
    n_features_to_stop : Stop when this many features remain.
    early_stopping_rounds : If >0 and estimator is LightGBM, applies early stopping with eval_set.
    scorer        : Optional callable to compute a *higher-is-better* score.
                    Signature options:
                      - scorer(y_true, y_pred)                # generic
                      - scorer(y_true, y_pred, y_proba=None)  # if you want probs too
                    If None:
                      * classification -> F1-macro on hard predictions
                      * regression     -> negative RMSE (-sqrt(MSE))

    Returns
    -------
    {
      "best_features": list[str],       # feature subset with max mean CV score
      "best_cv_score": float,           # mean CV score at that subset
      "history": list[dict],            # per-step logs (n_features, cv_mean, cv_std, etc.)
      "final_model": fitted_estimator   # trained on full data with best subset
    }
    """
    cols = list(X.columns)
    if n_features_to_stop >= len(cols):
        # Nothing to drop; just train and return
        final_model = clone(estimator).fit(X, y)
        return {
            "best_features": cols,
            "best_cv_score": np.nan,
            "history": [],
            "final_model": final_model,
        }

    # default scorers
    def default_classification_scorer(y_true, y_pred, y_proba=None):
        return float(f1_score(y_true, y_pred, average="macro"))

    def default_regression_scorer(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return -rmse  # higher is better

    # wrapper to call user scorer or defaults
    def compute_score(y_true, y_pred, y_proba=None):
        if scorer is not None:
            try:
                # try flexible 3-arg signature (y_true, y_pred, y_proba)
                return float(scorer(y_true, y_pred, y_proba))
            except TypeError:
                # fallback to 2-arg signature (y_true, y_pred)
                return float(scorer(y_true, y_pred))
        # defaults:
        if is_classifier(estimator):
            return default_classification_scorer(y_true, y_pred, y_proba)
        elif is_regressor(estimator):
            return default_regression_scorer(y_true, y_pred)
        else:
            raise ValueError("Estimator must be a classifier or regressor.")

    history = []
    dropped_cum = []
    best_cv_score = -np.inf
    best_features = cols.copy()

    step = 0
    while len(cols) > n_features_to_stop:
        imp_sum = pd.Series(0.0, index=cols)
        fold_scores = []

        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx][cols], X.iloc[va_idx][cols]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            model = clone(estimator)

            fit_kwargs = {}
            # Optional LightGBM early stopping (classification or regression)
            if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)) and early_stopping_rounds and early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(Xva, yva)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                # LightGBM will infer categoricals from dtype='category'

            model.fit(Xtr, ytr, **fit_kwargs)

            if not hasattr(model, "feature_importances_"):
                raise ValueError("Estimator must expose feature_importances_ for RFE dropping logic.")
            imp_sum += pd.Series(model.feature_importances_, index=cols)

            # predictions for scoring
            if is_classifier(model):
                # Prefer hard predictions for multiclass; works for binary too
                y_pred = model.predict(Xva)
                y_proba = None
                # If user scorer wants proba and model provides it, hand it over
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(Xva)
                    except Exception:
                        y_proba = None
                score = compute_score(yva, y_pred, y_proba)
            else:
                # regression
                y_pred = model.predict(Xva)
                score = compute_score(yva, y_pred)

            fold_scores.append(float(score))

        imp_mean = imp_sum / len(folds)
        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        history.append({
            "step": step,
            "n_features": len(cols),
            "features_used": cols.copy(),
            "dropped_cumulative": dropped_cum.copy(),
            "cv_score_mean": cv_mean,
            "cv_score_std": cv_std,
        })

        # update best subset (tie-breaker: fewer features)
        if (cv_mean > best_cv_score) or (np.isclose(cv_mean, best_cv_score) and len(cols) < len(best_features)):
            best_cv_score = cv_mean
            best_features = cols.copy()

        # decide how many to drop without overshooting
        k_max = len(cols) - n_features_to_stop
        k = int(min(n_features_to_drop, k_max))
        dropped_this = list(imp_mean.sort_values(ascending=True).head(k).index)

        dropped_cum.extend(dropped_this)
        cols = [c for c in cols if c not in dropped_this]
        print(f"step {step}: kept {len(cols)} features --> Score: {best_cv_score}")
        step += 1

    final_model = clone(estimator).fit(X[best_features], y)

    return {
        "best_features": best_features,
        "best_cv_score": best_cv_score,
        "history": history,
        "final_model": final_model,
    }
