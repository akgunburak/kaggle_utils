import pandas as pd
import numpy as np
from typing import List, Iterable, Tuple, Dict, Any
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
from boruta import BorutaPy
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import clone


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


def drop_high_null_columns(df, threshold=50, show=False):
    null_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = null_percent[null_percent > threshold].index
    if show:
        print(f"Columns to drop: {list(cols_to_drop)}")
    return cols_to_drop.to_list()


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

    return sorted(set(to_drop))


def custom_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    estimator,  # preconfigured, unfitted estimator (e.g., lgb.LGBMClassifier(...))
    n_features_to_drop: int = 5,
    n_features_to_stop: int = 20,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    """
    Iterative feature elimination using a user-provided estimator and user-provided folds.

    Args:
        X, y: data and labels (X must be a DataFrame so we can track columns)
        folds: list of (train_idx, val_idx) numpy arrays (your CV definition)
        estimator: an unfitted estimator with .fit(), .predict_proba(), and .feature_importances_
                   (e.g., lgb.LGBMClassifier preconfigured with your preferred params/importance_type)
        n_features_to_drop: features to remove per iteration
        n_features_to_stop: stop when this many features remain
        early_stopping_rounds: if >0 and estimator supports LightGBM callbacks, apply early stopping

    Returns:
        {
          "best_features": list[str],
          "best_cv_f1": float,
          "history": list[dict],
          "final_model": fitted_estimator_on_best_subset
        }
    """
    cols = list(X.columns)
    history = []
    dropped_cum = []
    best_cv_f1 = -1.0
    best_features = cols.copy()

    # detect categorical features by dtype if useful for your estimator
    categorical_cols = [c for c in cols if getattr(X[c].dtype, "name", "") == "category"]

    step = 0
    while len(cols) > n_features_to_stop:
        # 1) CV training on current feature set
        imp_sum = pd.Series(0.0, index=cols)
        f1_scores = []

        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx][cols], X.iloc[va_idx][cols]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            model = clone(estimator)

            # LightGBM-specific early stopping via callbacks (silently skip if not LGBM)
            fit_kwargs = {}
            if isinstance(model, lgb.LGBMClassifier) and early_stopping_rounds and early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(Xva, yva)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                # If you want categorical handling, ensure model is configured accordingly.
                # LightGBM will infer categoricals from dtype='category'.

            model.fit(Xtr, ytr, **fit_kwargs)

            # accumulate importances
            if hasattr(model, "feature_importances_"):
                imp_sum += pd.Series(model.feature_importances_, index=cols)
            else:
                raise ValueError("Estimator must expose feature_importances_ for RFE dropping logic.")

            # fold F1
            proba = model.predict(Xva)
            f1_scores.append(f1_score(yva, proba))

        imp_mean = imp_sum / len(folds)
        cv_f1_mean = float(np.mean(f1_scores))
        cv_f1_std = float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0

        # 2) log before dropping
        history.append({
            "step": step,
            "n_features": len(cols),
            "features_used": cols.copy(),
            "dropped_cumulative": dropped_cum.copy(),
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
        })

        # 3) update best subset if improved (tie-breaker: fewer features)
        if (cv_f1_mean > best_cv_f1) or (np.isclose(cv_f1_mean, best_cv_f1) and len(cols) < len(best_features)):
            best_cv_f1 = cv_f1_mean
            best_features = cols.copy()

        # 4) decide how many to drop and drop least important
        k_max = len(cols) - n_features_to_stop
        k = int(min(n_features_to_drop, k_max))
        dropped_this = list(imp_mean.sort_values(ascending=True).head(k).index)

        dropped_cum.extend(dropped_this)
        cols = [c for c in cols if c not in dropped_this]
        print(f"step {step}: kept {len(cols)} features --> cv_f1_mean: {cv_f1_mean}")
        step += 1

    # Final model on full data with the best subset
    final_model = clone(estimator).fit(X[best_features], y)

    return {
        "best_features": best_features,
        "best_cv_f1": best_cv_f1,
        "history": history,
        "final_model": final_model,
    }


def boruta_feature_selection(X_train, X_test, y_train):
    # Find numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # Fill the numerical data with median
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])

    # Fill the categorical data with mode
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
            
    X_train, X_test, encoders = encode_categorical(
    X_train, X_test, y_train,
    ohe_max_cardinality=0,
    high_card_strategy="target"  # or "ordinal"
    )
            
    # define model and boruta objects
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42)

    # training
    boruta_selector.fit(X_train, y_train)

    # selected features
    selected_features = X_train.columns[boruta_selector.support_]
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
