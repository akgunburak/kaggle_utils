import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def get_oof_predictions(models, X, y, X_test, folds=None, scorer=None, task='classification'):
    """
    Generate OOF predictions for stacking (classification or regression).
    Returns OOF train/test as pandas DataFrames.
    """
    if folds is None:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task=='classification' else KFold(n_splits=5, shuffle=True, random_state=42)
    
    n_train = X.shape[0]
    n_test = X_test.shape[0]
    n_models = len(models)

    # Prepare OOF matrices
    if task == 'classification':
        n_classes = len(np.unique(y))
        oof_train = np.zeros((n_train, n_models * n_classes))
        oof_test = np.zeros((n_test, n_models * n_classes))
        col_names = []
        for j, model in enumerate(models):
            for c in range(n_classes):
                col_names.append(f"{model.__class__.__name__}_class{c}")
    else:  # regression
        oof_train = np.zeros((n_train, n_models))
        oof_test = np.zeros((n_test, n_models))
        col_names = [f"{model.__class__.__name__}" for model in models]
    
    scores = {}
    trained_models = []

    for j, model in enumerate(models):
        print(f"\nTraining base model {j+1}/{n_models}: {model.__class__.__name__}")            
        
        if task == 'classification':
            oof_test_fold = np.zeros((n_test, folds.get_n_splits(), n_classes))
        else:
            oof_test_fold = np.zeros((n_test, folds.get_n_splits()))

        scores[model.__class__.__name__] = []

        for i, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            print(f"  Fold {i+1}/{folds.get_n_splits()}")
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

            model.fit(X_train, y_train)

            # Predict
            if task == 'classification':
                y_valid_pred = model.predict(X_valid)
                y_valid_proba = model.predict_proba(X_valid)
                oof_train[valid_idx, j*n_classes:(j+1)*n_classes] = y_valid_proba

                y_test_pred = model.predict_proba(X_test)
                oof_test_fold[:, i, :] = y_test_pred
            else:  # regression
                y_valid_pred = model.predict(X_valid).reshape(-1,1)
                oof_train[valid_idx, j] = y_valid_pred.ravel()

                y_test_pred = model.predict(X_test).reshape(-1,1)
                oof_test_fold[:, i] = y_test_pred.ravel()

            # Score if provided
            if scorer is not None:
                score = scorer(y_valid, y_valid_pred)
                scores[model.__class__.__name__].append(score)
                print(f"    Fold {i+1} score: {score:.4f}")

            # Save trained model
            trained_models.append(model)

        print(f" ---- Avg of folds: {np.mean(scores[model.__class__.__name__]):.4f} ----")

        # Average test preds across folds
        if task == 'classification':
            oof_test[:, j*n_classes:(j+1)*n_classes] = oof_test_fold.mean(axis=1)
        else:
            oof_test[:, j] = oof_test_fold.mean(axis=1)

    # Convert to DataFrames
    oof_train_df = pd.DataFrame(oof_train, columns=col_names, index=X.index)
    oof_test_df = pd.DataFrame(oof_test, columns=col_names, index=X_test.index)

    return oof_train_df, oof_test_df, scores, trained_models


def get_mean_predictions(models, X_test, task='classification'):
    """
    Make ensemble predictions from a list of fitted models.

    Parameters
    ----------
    models : list
        List of fitted models.
    X_test : array-like or pd.DataFrame
        Test data to predict on.
    task : str, 'classification' or 'regression' (default='classification')
        Type of task. For classification, averages predict_proba.
        For regression, averages predict outputs.

    Returns
    -------
    avg_preds : np.ndarray
        Averaged probabilities (classification) or averaged predictions (regression).
    final_preds : np.ndarray
        Final predictions: argmax of averaged probabilities (classification)
        or averaged predictions directly (regression).
    """

    preds_list = []

    if task == 'classification':
        for model in models:
            preds_list.append(model.predict_proba(X_test))
        preds = np.array(preds_list)  # (n_models, n_samples, n_classes)
        avg_preds = preds.mean(axis=0)  # (n_samples, n_classes)
        final_preds = np.argmax(avg_preds, axis=1)  # predicted classes

    elif task == 'regression':
        for model in models:
            preds_list.append(model.predict(X_test))
        preds = np.array(preds_list)  # (n_models, n_samples)
        avg_preds = preds.mean(axis=0)  # (n_samples,)
        final_preds = avg_preds  # directly averaged prediction for regression

    else:
        raise ValueError("task must be 'classification' or 'regression'")

    return final_preds
