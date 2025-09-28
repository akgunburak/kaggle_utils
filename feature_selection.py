import pandas as pd
import numpy as np
from typing import List, Iterable
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
from boruta import BorutaPy
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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


def custom_rfe(X_train, X_test, y_train, y_test, path, n_features_to_drop=5, n_features_to_stop=20):
    log_dict = {}
    step = 0
    cols_to_drop = []
    while len(X_train.columns) > n_features_to_stop:
        # Train LightGBM
        #train_data = lgb.Dataset(X_train, label=y_train)
        #test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        lgbm_params = {
            "categorical_feature": [col for col in X_train if X_train[col].dtype == 'category'],
            "objective": "binary",
            "metric": "binary_logloss",
            'verbose': -1
        }
        with open(path, 'rb') as file:
            lgbm_best_params = pickle.load(file)
        lgbm_params.update(lgbm_best_params)
        #lgbm_num_round = 1000
        lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        lgbm_model.fit(
                X_train,
                y_train)
        #model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data])
        lgbm_y_pred_proba = lgbm_model.predict_proba(X_test)
        #lgbm_y_pred = lgbm_model.predict(X_test)
        lgbm_y_pred = (lgbm_y_pred_proba[:, 1] >= 0.5).astype(int)
        importance_df = pd.DataFrame({'Feature': lgbm_model.feature_names_in_, 'Importance': lgbm_model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        lgbm_f1 = f1_score(y_test, lgbm_y_pred)
    
        log_dict_detail = {
            "features_used": X_train.columns.to_list(),
            "features_dropped": cols_to_drop,
            "lgbm_f1": lgbm_f1
        }
        log_dict[step] = log_dict_detail
    
        cols_to_drop = importance_df.sort_values("Importance").head(n_features_to_drop)["Feature"].to_list()
                
        X_train = X_train.drop(cols_to_drop, axis=1)
        X_test = X_test.drop(cols_to_drop, axis=1)
        print("step: ", step, "n_features: ", len(X_train.columns))
        step += 1
        
    return log_dict


def boruta_feature_selection(X_train, X_test, y_train):
    # Sütunları ayır
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # Sayısal için median doldurma
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])

    # Kategorik için en sık görüleni doldurma
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
            
    X_train, X_test, encoders = encode_categorical(
    X_train, X_test, y_train,
    ohe_max_cardinality=0,
    high_card_strategy="target"  # or "ordinal"
    )
            
    # model ve Boruta nesnesi
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42)

    # eğitimi
    boruta_selector.fit(X_train, y_train)

    # seçilen özellikler
    selected_features = X_train.columns[boruta_selector.support_]
    return selected_features
