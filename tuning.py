import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, roc_auc_score
)
import os
import optuna
import pickle
import torch
import torch.nn as nn
from encoding import *
from train import *


def objective_xgb_cv(trial, task, cross_val_splits, X, y, path,
                     metric="f1", pos_weight=None, n_classes=None, enable_categorical=True):
    """
    Optuna objective for XGBoost with CV + categorical encoding per fold.
    Tunes encoding params:
      - ohe_max_cardinality (int)
      - high_card_strategy ('ordinal'|'target')
      - drop_first (bool)
    """
    
    # ----- XGBoost parameter space -----
    param = {
        'verbosity': 0,
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
        'subsample': trial.suggest_float('subsample', 0.3, 1),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
        'max_depth': trial.suggest_int('max_depth', 5, 100),
        'random_state': 2020,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }

    if task == 'binary':
        param['objective'] = 'binary:logistic'
        param['eval_metric'] = 'logloss'
        if pos_weight is not None:
            param['scale_pos_weight'] = trial.suggest_int('scale_pos_weight', 1, pos_weight + 100)

    elif task == 'multiclass':
        param['objective'] = 'multi:softprob'
        param['num_class'] = n_classes
        param['eval_metric'] = 'mlogloss'

    else:  # regression
        param['objective'] = 'reg:squarederror'
        param['eval_metric'] = 'rmse'

    fold_scores = []

    for fold, (train_idx_cv, val_idx_cv) in enumerate(cross_val_splits, start=1):
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]

        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, enable_categorical=enable_categorical)
        dtest  = xgb.DMatrix(X_val,   label=y_val,     enable_categorical=enable_categorical)

        bst = xgb.train(param, dtrain, evals=[(dtest, "test")], verbose_eval=False)
        preds = bst.predict(dtest)

        # ----- compute metric -----
        if task == 'binary':
            pred_labels = (preds > 0.5).astype(int)

            if metric == "f1":
                fold_score = f1_score(y_val, pred_labels, zero_division=0)
            elif metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            elif metric == "auc":
                fold_score = roc_auc_score(y_val, preds)
            elif metric == "precision":
                fold_score = precision_score(y_val, pred_labels, zero_division=0)
            elif metric == "recall":
                fold_score = recall_score(y_val, pred_labels, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric {metric} for binary")

            # for logs
            y_true_1 = y_val.value_counts(normalize=True).get(1, 0)
            y_true_0 = y_val.value_counts(normalize=True).get(0, 0)
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True).get(1, 0)
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True).get(0, 0)

        elif task == 'multiclass':
            pred_labels = np.argmax(preds, axis=1)
            if metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            else:
                raise ValueError(f"Only 'accuracy' supported for multiclass by default")

            y_true_dist = y_val.value_counts(normalize=True).to_dict()
            y_pred_dist = pd.Series(pred_labels).value_counts(normalize=True).to_dict()

        else:  # regression
            if metric == "rmse":
                fold_score = np.sqrt(mean_squared_error(y_val, preds))
            elif metric == "mae":
                fold_score = mean_absolute_error(y_val, preds)
            else:
                raise ValueError(f"Unsupported metric {metric} for regression")

        # ----- top features (after encoding) -----
        feature_importances = bst.get_score(importance_type='weight')
        importance_df = pd.DataFrame(list(feature_importances.items()),
                                     columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = ', '.join(importance_df['Feature'].head(5).tolist())

        # ----- logging -----
        log_row = {
            'fold': fold,
            'params': str(param),
            'metric_name': metric,
            'metric_value': fold_score,
            'top_5_features': top_5_features
        }

        if task == 'binary':
            log_row.update({
                'y_true_1': y_true_1, 'y_true_0': y_true_0,
                'y_pred_1': y_pred_1, 'y_pred_0': y_pred_0
            })
        elif task == 'multiclass':
            log_row.update({
                'true_dist': str(y_true_dist),
                'pred_dist': str(y_pred_dist)
            })

        log_df = pd.DataFrame([log_row])
        header_needed = not os.path.exists(path)
        log_df.to_csv(path, mode='a', header=header_needed, index=False)

        fold_scores.append(fold_score)

    avg_score = np.mean(fold_scores)
    print(f"[{metric}] Fold scores: {fold_scores} | Average: {avg_score:.4f}")
    return avg_score


def objective_lgbm_cv(trial, task, cross_val_splits, X, y, path,
                      metric="f1", n_classes=None, device="cpu"):
    """
    Optuna objective for LightGBM with CV.

    Parameters
    ----------
    task : 'binary', 'multiclass', or 'regression'
    metric : evaluation metric to optimize ('f1','accuracy','auc','rmse','mae',...)
    n_classes : required for multiclass classification
    path : CSV file path where logs will be appended
    """

    # ----- parameter space -----
    param = {
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
        'random_state': 2020,
        'verbose': False,
        "device": device
    }

    if task == 'binary':
        param['objective'] = 'binary'
        param['is_unbalance'] = trial.suggest_categorical('is_unbalance', [True, False])

    elif task == 'multiclass':
        param['is_unbalance'] = trial.suggest_categorical('is_unbalance', [True, False])
        param['objective'] = 'multiclass'
        param['num_class'] = n_classes

    else:  # regression
        param['objective'] = 'regression'

    fold_scores = []

    for fold, (train_idx_cv, val_idx_cv) in enumerate(cross_val_splits, start=1):
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]
        
        if task in ['binary', 'multiclass']:
            model = lgb.LGBMClassifier(**param) if task != 'regression' else lgb.LGBMRegressor(**param)
        else:
            model = lgb.LGBMRegressor(**param)

        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val, y_val)]
        )

        preds = model.predict(X_val)
        if task == 'binary':
            pred_proba = model.predict_proba(X_val)[:, 1]
            pred_labels = (pred_proba > 0.5).astype(int)

            if metric == "f1":
                fold_score = f1_score(y_val, pred_labels, zero_division=0)
            elif metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            elif metric == "auc":
                fold_score = roc_auc_score(y_val, pred_proba)
            elif metric == "precision":
                fold_score = precision_score(y_val, pred_labels, zero_division=0)
            elif metric == "recall":
                fold_score = recall_score(y_val, pred_labels, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric {metric} for binary")

            # logs
            y_true_1 = y_val.value_counts(normalize=True).get(1, 0)
            y_true_0 = y_val.value_counts(normalize=True).get(0, 0)
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True).get(1, 0)
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True).get(0, 0)

        elif task == 'multiclass':
            pred_labels = np.argmax(model.predict_proba(X_val), axis=1)
            if metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            else:
                raise ValueError(f"Only 'accuracy' supported for multiclass by default")

            y_true_dist = y_val.value_counts(normalize=True).to_dict()
            y_pred_dist = pd.Series(pred_labels).value_counts(normalize=True).to_dict()

        else:  # regression
            if metric == "rmse":
                fold_score = np.sqrt(mean_squared_error(y_val, preds))
            elif metric == "mae":
                fold_score = mean_absolute_error(y_val, preds)
            else:
                raise ValueError(f"Unsupported metric {metric} for regression")

        # --- top features ---
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = ', '.join(importance_df['Feature'].head(5).tolist())

        # --- logging ---
        log_row = {
            'fold': fold,
            'params': str(param),
            'metric_name': metric,
            'metric_value': fold_score,
            'top_5_features': top_5_features
        }

        if task == 'binary':
            log_row.update({
                'y_true_1': y_true_1, 'y_true_0': y_true_0,
                'y_pred_1': y_pred_1, 'y_pred_0': y_pred_0
            })
        elif task == 'multiclass':
            log_row.update({
                'true_dist': str(y_true_dist),
                'pred_dist': str(y_pred_dist)
            })

        log_df = pd.DataFrame([log_row])
        header_needed = not os.path.exists(path)
        log_df.to_csv(path, mode='a', header=header_needed, index=False)

        fold_scores.append(fold_score)

    avg_score = np.mean(fold_scores)
    print(f"[{metric}] Fold scores: {fold_scores} | Average: {avg_score:.4f}")
    return avg_score


def objective_cb_cv(trial, task, cross_val_splits, X, y, path,
                          metric="f1", n_classes=None, task_type):
    """
    Optuna objective for CatBoost with CV.

    Parameters
    ----------
    task : 'binary', 'multiclass', or 'regression'
    metric : evaluation metric to optimize ('f1','accuracy','auc','rmse','mae',...)
    n_classes : required for multiclass classification
    path : CSV file path where logs will be appended
    """
                              
    for col in X.columns:
        if X[col].dtype == 'category':
            X[col] = X[col].astype(str).fillna('Unknown').astype('category')

    # ----- parameter space -----
    param = {
        'iterations': trial.suggest_int('iterations', 300, 1000),  # reduce max iterations
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # focus on realistic LRs
        'depth': trial.suggest_int('depth', 4, 10),  # typical effective depths
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),  # fewer splits
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 300),
        'random_seed': 2020,
        'verbose': False,
        'cat_features': [col for col in X if X[col].dtype == 'category'],
        "task_type": task_type
    }

    # choose a bootstrap type compatible with subsample or not
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
    param['bootstrap_type'] = bootstrap_type
    
    if bootstrap_type == 'Bayesian':
        # subsample is NOT allowed; use bagging_temperature instead (0 => no sampling, higher => more)
        param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    else:
        # subsample IS allowed for Bernoulli/Poisson
        param['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

    if task == 'binary':
        param['loss_function'] = 'Logloss'
        param['auto_class_weights'] = trial.suggest_categorical('auto_class_weights', ["Balanced", "SqrtBalanced"])

    elif task == 'multiclass':
        param['loss_function'] = 'MultiClass'
        param['classes_count'] = n_classes
        param['auto_class_weights'] = trial.suggest_categorical('auto_class_weights', ["Balanced", "SqrtBalanced"])

    else:  # regression
        param['loss_function'] = 'RMSE'

    fold_scores = []

    for fold, (train_idx_cv, val_idx_cv) in enumerate(cross_val_splits, start=1):
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]

        if task in ['binary', 'multiclass']:
            model = CatBoostClassifier(**param) if task != 'regression' else CatBoostRegressor(**param)
        else:
            model = CatBoostRegressor(**param)

        model.fit(X_train_cv, y_train_cv, eval_set=(X_val, y_val), verbose=False)

        if task == 'binary':
            pred_proba = model.predict_proba(X_val)[:, 1]
            pred_labels = (pred_proba > 0.5).astype(int)

            if metric == "f1":
                fold_score = f1_score(y_val, pred_labels, zero_division=0)
            elif metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            elif metric == "auc":
                fold_score = roc_auc_score(y_val, pred_proba)
            elif metric == "precision":
                fold_score = precision_score(y_val, pred_labels, zero_division=0)
            elif metric == "recall":
                fold_score = recall_score(y_val, pred_labels, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric {metric} for binary")

            y_true_1 = y_val.value_counts(normalize=True).get(1, 0)
            y_true_0 = y_val.value_counts(normalize=True).get(0, 0)
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True).get(1, 0)
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True).get(0, 0)

        elif task == 'multiclass':
            pred_labels = np.argmax(model.predict_proba(X_val), axis=1)
            if metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            else:
                raise ValueError(f"Only 'accuracy' supported for multiclass by default")

            y_true_dist = y_val.value_counts(normalize=True).to_dict()
            y_pred_dist = pd.Series(pred_labels).value_counts(normalize=True).to_dict()

        else:  # regression
            preds = model.predict(X_val)
            if metric == "rmse":
                fold_score = np.sqrt(mean_squared_error(y_val, preds))
            elif metric == "mae":
                fold_score = mean_absolute_error(y_val, preds)
            else:
                raise ValueError(f"Unsupported metric {metric} for regression")

        # --- top features ---
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.get_feature_importance()
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = ', '.join(importance_df['Feature'].head(5).tolist())

        # --- logging ---
        log_row = {
            'fold': fold,
            'params': str(param),
            'metric_name': metric,
            'metric_value': fold_score,
            'top_5_features': top_5_features
        }

        if task == 'binary':
            log_row.update({
                'y_true_1': y_true_1, 'y_true_0': y_true_0,
                'y_pred_1': y_pred_1, 'y_pred_0': y_pred_0
            })
        elif task == 'multiclass':
            log_row.update({
                'true_dist': str(y_true_dist),
                'pred_dist': str(y_pred_dist)
            })

        log_df = pd.DataFrame([log_row])
        header_needed = not os.path.exists(path)
        log_df.to_csv(path, mode='a', header=header_needed, index=False)

        fold_scores.append(fold_score)

    avg_score = np.mean(fold_scores)
    print(f"[{metric}] Fold scores: {fold_scores} | Average: {avg_score:.4f}")
    return avg_score


def objective_rf_cv(trial, task, cross_val_splits, X, y, path,
                              metric="f1", pos_weight=None, n_classes=None):
    """
    Optuna objective for RandomForest with CV.

    Parameters
    ----------
    task : 'binary', 'multiclass', or 'regression'
    metric : evaluation metric to optimize ('f1','accuracy','auc','rmse','mae',...)
    n_classes : required for multiclass classification
    path : CSV file path where logs will be appended
    """
                          
    # ----- encoding params to optimize -----
    enc_ohe_max_cardinality = trial.suggest_int("ohe_max_cardinality", 2, 20)
    enc_high_card_strategy  = trial.suggest_categorical("high_card_strategy", ["ordinal", "target"])
    enc_drop_first          = trial.suggest_categorical("drop_first", [False, True])
                                
    # ----- parameter space -----
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 2020,
        'n_jobs': -1
    }

    # Handle class imbalance for binary classification
    if task == 'binary' and pos_weight is not None:
        param['class_weight'] = 'balanced'  # or you can compute {0:1, 1:pos_weight}

    fold_scores = []

    for fold, (train_idx_cv, val_idx_cv) in enumerate(cross_val_splits, start=1):
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]

        # ----- per-fold categorical encoding (fit on train only) -----
        X_train_cv, X_val, _artifacts = encode_categorical(
            X_train_cv, X_val,
            y_train=y_train_cv if enc_high_card_strategy == "target" else None,
            ohe_max_cardinality=enc_ohe_max_cardinality,
            high_card_strategy=enc_high_card_strategy,
            drop_first=enc_drop_first,
            dtype=float
        )

        if task in ['binary', 'multiclass']:
            model = RandomForestClassifier(**param)
        else:  # regression
            model = RandomForestRegressor(**param)

        model.fit(X_train_cv, y_train_cv)
        
        # --- predictions & metrics ---
        if task == 'binary':
            pred_proba = model.predict_proba(X_val)[:, 1]
            pred_labels = (pred_proba > 0.5).astype(int)

            if metric == "f1":
                fold_score = f1_score(y_val, pred_labels, zero_division=0)
            elif metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            elif metric == "auc":
                fold_score = roc_auc_score(y_val, pred_proba)
            elif metric == "precision":
                fold_score = precision_score(y_val, pred_labels, zero_division=0)
            elif metric == "recall":
                fold_score = recall_score(y_val, pred_labels, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric {metric} for binary")

            y_true_1 = y_val.value_counts(normalize=True).get(1, 0)
            y_true_0 = y_val.value_counts(normalize=True).get(0, 0)
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True).get(1, 0)
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True).get(0, 0)

        elif task == 'multiclass':
            pred_labels = model.predict(X_val)
            if metric == "accuracy":
                fold_score = accuracy_score(y_val, pred_labels)
            else:
                raise ValueError(f"Only 'accuracy' supported for multiclass by default")

            y_true_dist = y_val.value_counts(normalize=True).to_dict()
            y_pred_dist = pd.Series(pred_labels).value_counts(normalize=True).to_dict()

        else:  # regression
            preds = model.predict(X_val)
            if metric == "rmse":
                fold_score = np.sqrt(mean_squared_error(y_val, preds))
            elif metric == "mae":
                fold_score = mean_absolute_error(y_val, preds)
            else:
                raise ValueError(f"Unsupported metric {metric} for regression")

        # --- top features ---
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = ', '.join(importance_df['Feature'].head(5).tolist())

        # --- logging ---
        log_row = {
            'fold': fold,
            'params': str(param),
            'metric_name': metric,
            'metric_value': fold_score,
            'top_5_features': top_5_features
        }

        if task == 'binary':
            log_row.update({
                'y_true_1': y_true_1, 'y_true_0': y_true_0,
                'y_pred_1': y_pred_1, 'y_pred_0': y_pred_0
            })
        elif task == 'multiclass':
            log_row.update({
                'true_dist': str(y_true_dist),
                'pred_dist': str(y_pred_dist)
            })

        log_df = pd.DataFrame([log_row])
        header_needed = not os.path.exists(path)
        log_df.to_csv(path, mode='a', header=header_needed, index=False)

        fold_scores.append(fold_score)

    avg_score = np.mean(fold_scores)
    print(f"[{metric}] Fold scores: {fold_scores} | Average: {avg_score:.4f}")
    return avg_score


def objective_torchmlp_cv(trial, task, cross_val_splits, X, y, path,
                          metric="f1", pos_weight=None, n_classes=None):
    """
    Optuna objective for TorchMLPClassifier / TorchMLPRegressor with CV.
    Keeps the model's internal scaler enabled and applies per-fold categorical encoding.
    """

    # ----------------- hyperparameter space (NN) -----------------
    hidden_dim    = trial.suggest_int('hidden_dim', 64, 1024, log=True)
    n_layers      = trial.suggest_int('n_layers', 1, 6)
    dropout       = trial.suggest_float('dropout', 0.0, 0.6)
    batchnorm     = trial.suggest_categorical('batchnorm', [True, False])
    activation    = trial.suggest_categorical('activation', ['relu', 'gelu', 'leaky_relu'])
    optimizer     = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    epochs        = trial.suggest_int('epochs', 30, 400)
    patience      = trial.suggest_int('patience', 10, 50)
    use_grad_clip = trial.suggest_categorical('use_grad_clip', [True, False])
    grad_clip     = trial.suggest_float('grad_clip', 0.5, 5.0) if use_grad_clip else None
    val_size      = trial.suggest_float('val_size', 0.05, 0.2)

    # estimator options that must stay ON (you asked not to disable scaler)
    scale_features = True
    optimize_threshold = (task == 'binary') and (metric in {'f1','accuracy','precision','recall'})
    threshold_metric   = 'f1' if metric == 'f1' else 'accuracy'

    # pack params for logging
    param = {
        'hidden_dim': hidden_dim, 'n_layers': n_layers, 'dropout': dropout, 'batchnorm': batchnorm,
        'activation': activation, 'optimizer': optimizer, 'learning_rate': learning_rate,
        'weight_decay': weight_decay, 'batch_size': batch_size, 'epochs': epochs,
        'patience': patience, 'grad_clip': grad_clip, 'val_size': val_size,
        'scale_features': scale_features, 'optimize_threshold': optimize_threshold,
        'threshold_metric': threshold_metric, 'random_state': 2020, 'device': 'auto'
    }

    # ensure pandas (the encoder expects DataFrame/Series)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.Series(y)

    fold_scores = []

    def _ix(obj, idx):
        return obj.loc[idx] if hasattr(obj, "loc") else obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]

    for fold, (train_idx_cv, val_idx_cv) in enumerate(cross_val_splits, start=1):
        X_train_cv, X_val = _ix(X, train_idx_cv), _ix(X, val_idx_cv)
        y_train_cv, y_val = _ix(y, train_idx_cv), _ix(y, val_idx_cv)

        # ---- encode categoricals per fold (NO external scaling; model will scale internally) ----
        Xtr_cv, Xva_cv, enc_artifacts = encode_categorical(
            X_train_cv, X_val, y_train=y_train_cv,
            ohe_max_cardinality=0,
            high_card_strategy="target",
            drop_first=False,
            dtype=float
        )

        feature_names = list(Xtr_cv.columns)

        # ---- build and fit estimator (internal scaler kept ON) ----
        if task in ('binary', 'multiclass'):
            clf = TorchMLPClassifier(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                batchnorm=batchnorm,
                activation=activation,
                optimizer=optimizer,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                grad_clip=grad_clip,
                val_size=val_size,
                scale_features=scale_features,          # keep ON
                optimize_threshold=optimize_threshold,  # tunes cutoff on tiny val
                threshold_metric=threshold_metric,
                device="auto",
                random_state=2020
            )
            clf.fit(Xtr_cv, y_train_cv)

            # predictions & metric
            if task == 'binary':
                if metric == "auc":
                    proba = clf.predict_proba(Xva_cv)[:, 1]
                    fold_score = roc_auc_score(y_val, proba)
                else:
                    pred_labels = clf.predict(Xva_cv)
                    if metric == "f1":
                        fold_score = f1_score(y_val, pred_labels, zero_division=0)
                    elif metric == "accuracy":
                        fold_score = accuracy_score(y_val, pred_labels)
                    elif metric == "precision":
                        fold_score = precision_score(y_val, pred_labels, zero_division=0)
                    elif metric == "recall":
                        fold_score = recall_score(y_val, pred_labels, zero_division=0)
                    else:
                        raise ValueError(f"Unsupported metric {metric} for binary")

                # logs
                y_true_1 = pd.Series(y_val).value_counts(normalize=True).get(1, 0.0)
                y_true_0 = pd.Series(y_val).value_counts(normalize=True).get(0, 0.0)
                pred_labels_for_log = clf.predict(Xva_cv)
                y_pred_1 = pd.Series(pred_labels_for_log).value_counts(normalize=True).get(1, 0.0)
                y_pred_0 = pd.Series(pred_labels_for_log).value_counts(normalize=True).get(0, 0.0)

            else:  # multiclass
                if metric != "accuracy":
                    raise ValueError("Only 'accuracy' supported for multiclass by default")
                pred_labels = clf.predict(Xva_cv)
                fold_score = accuracy_score(y_val, pred_labels)

                y_true_dist = pd.Series(y_val).value_counts(normalize=True).to_dict()
                y_pred_dist = pd.Series(pred_labels).value_counts(normalize=True).to_dict()

            # --- top features (heuristic from first Linear layer) ---
            top_5_features = ""
            try:
                first_linear = None
                for m in clf.model_.net:
                    if isinstance(m, nn.Linear):
                        first_linear = m
                        break
                if first_linear is not None:
                    w = first_linear.weight.detach().cpu().numpy()    # (hidden, n_features)
                    importances = np.abs(w).sum(axis=0)               # per-feature
                    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
                    top_5_features = ", ".join(imp_df["Feature"].head(5).tolist())
            except Exception:
                pass

        else:  # --------------------------- regression ---------------------------
            reg = TorchMLPRegressor(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                batchnorm=batchnorm,
                activation=activation,
                optimizer=optimizer,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                grad_clip=grad_clip,
                val_size=val_size,
                scale_features=scale_features,  # keep ON
                device="auto",
                random_state=2020,
                loss="mse" if metric == "rmse" else "mae"
            )
            reg.fit(Xtr_cv, y_train_cv)
            preds = reg.predict(Xva_cv)

            if metric == "rmse":
                fold_score = float(np.sqrt(mean_squared_error(y_val, preds)))
            elif metric == "mae":
                fold_score = float(mean_absolute_error(y_val, preds))
            else:
                raise ValueError(f"Unsupported metric {metric} for regression")

            # "top features" heuristic
            top_5_features = ""
            try:
                first_linear = None
                for m in reg.model_.net:
                    if isinstance(m, nn.Linear):
                        first_linear = m
                        break
                if first_linear is not None:
                    w = first_linear.weight.detach().cpu().numpy()
                    importances = np.abs(w).sum(axis=0)
                    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
                    top_5_features = ", ".join(imp_df["Feature"].head(5).tolist())
            except Exception:
                pass

        # -------------- logging --------------
        log_row = {
            'fold': fold,
            'params': str(param),
            'metric_name': metric,
            'metric_value': float(fold_score),
            'top_5_features': top_5_features
        }
        if task == 'binary':
            log_row.update({
                'y_true_1': y_true_1, 'y_true_0': y_true_0,
                'y_pred_1': y_pred_1, 'y_pred_0': y_pred_0
            })
        elif task == 'multiclass':
            log_row.update({
                'true_dist': str(y_true_dist),
                'pred_dist': str(y_pred_dist)
            })

        log_df = pd.DataFrame([log_row])
        header_needed = not os.path.exists(path)
        log_df.to_csv(path, mode='a', header=header_needed, index=False)

        fold_scores.append(fold_score)

    avg_score = float(np.mean(fold_scores))
    print(f"[{metric}] Fold scores: {fold_scores} | Average: {avg_score:.4f}")
    return avg_score


# example usage
"""
splits_bin = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                  .split(X_bin, y_bin))

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective_rf_cv(
    trial=trial,
    task='binary', #multiclass, regression
    cross_val_splits=splits_bin,
    X=X_bin, y=y_bin,
    path='cv_log_binary.csv',
    metric="f1"
),
              n_trials=3)

print("Best hyperparameters:", study.best_params)

with open("xgb_params.pkl", "wb") as file:
    pickle.dump(study.best_params, file)
"""
