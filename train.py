import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def get_oof_predictions(
    models, X, y, X_test,
    folds=None,
    scorer=None,
    task='classification',
    weight_mode='none',                 # 'none' | 'manual' | 'auto'
    model_weights=None,                 # list[float] when weight_mode='manual'
    scorer_greater_is_better=True       # set False if your scorer is a loss (lower is better)
):
    """
    Generate OOF predictions for stacking (classification or regression).

    Classification features:
        - Per-model probability features
        - SoftVoting (mean probabilities)
        - HardVoting (vote proportions)
        - WeightedSoftVoting (weighted mean probabilities)
        - WeightedHardVoting (weighted vote proportions)

    Regression features:
        - Per-model predictions
        - MeanPred (uniform mean of base models)
        - WeightedMeanPred (weighted mean of base models)

    Returns:
        oof_train_df, oof_test_df, scores, trained_models
    """
    def _normalize_weights(w):
        w = np.array(w, dtype=float)
        w = np.clip(w, a_min=0.0, a_max=None)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return np.ones_like(w) / len(w)
        return w / s

    if folds is None:
        folds = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                 if task == 'classification'
                 else KFold(n_splits=5, shuffle=True, random_state=42))

    n_train = X.shape[0]
    n_test  = X_test.shape[0]
    n_models = len(models)

    # ---------- Prepare OOF matrices & column names ----------
    if task == 'classification':
        n_classes = len(np.unique(y))
        add_weighted = (weight_mode in ('manual', 'auto'))

        base_cols = n_models * n_classes
        soft_cols = n_classes
        hard_cols = n_classes
        weighted_cols = (2 * n_classes) if add_weighted else 0

        total_cols = base_cols + soft_cols + hard_cols + weighted_cols
        oof_train = np.zeros((n_train, total_cols))
        oof_test  = np.zeros((n_test,  total_cols))

        col_names = []
        for j, model in enumerate(models):
            for c in range(n_classes):
                col_names.append(f"{model.__class__.__name__}_class{c}")
        for c in range(n_classes):
            col_names.append(f"SoftVote_class{c}")
        for c in range(n_classes):
            col_names.append(f"HardVote_class{c}")
        if add_weighted:
            for c in range(n_classes):
                col_names.append(f"WeightedSoftVote_class{c}")
            for c in range(n_classes):
                col_names.append(f"WeightedHardVote_class{c}")

        def model_slice(j):
            return slice(j * n_classes, (j + 1) * n_classes)

        soft_slice = slice(base_cols, base_cols + n_classes)
        hard_slice = slice(base_cols + n_classes, base_cols + 2 * n_classes)
        if add_weighted:
            wsoft_slice = slice(base_cols + 2 * n_classes, base_cols + 3 * n_classes)
            whard_slice = slice(base_cols + 3 * n_classes, base_cols + 4 * n_classes)

        scores = {model.__class__.__name__: [] for model in models}
        scores["SoftVoting"] = []
        scores["HardVoting"] = []
        if add_weighted:
            scores["WeightedSoftVoting"] = []
            scores["WeightedHardVoting"] = []

    else:  # regression
        add_weighted = (weight_mode in ('manual', 'auto'))
        extra = 1 if add_weighted else 0  # WeightedMeanPred
        oof_train = np.zeros((n_train, n_models + 1 + extra))
        oof_test  = np.zeros((n_test,  n_models + 1 + extra))

        col_names = [f"{model.__class__.__name__}" for model in models] + ["MeanPred"]
        mean_col = n_models
        if add_weighted:
            wmean_col = n_models + 1
            col_names.append("WeightedMeanPred")

        scores = {model.__class__.__name__: [] for model in models}
        scores["MeanPred"] = []
        if add_weighted:
            scores["WeightedMeanPred"] = []

    trained_models = []

    # ================= Cross-validation =================
    for i, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print(f"\nFold {i+1}/{folds.get_n_splits()}")

        # Support pandas indexers
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

        if task == 'classification':
            valid_probas_list, valid_hard_preds_list = [], []
            test_probas_list,  test_hard_preds_list  = [], []
            per_model_fold_scores = []  # for auto weights
        else:
            valid_preds_list = []
            test_preds_list  = []

        # ---------- Train each model ----------
        for j, model in enumerate(models):
            print(f"  Training model {j+1}/{n_models}: {model.__class__.__name__}")
            model.fit(X_train, y_train)

            if task == 'classification':
                y_valid_pred  = model.predict(X_valid)
                y_valid_proba = model.predict_proba(X_valid)

                # per-model OOF train probas
                oof_train[valid_idx, model_slice(j)] = y_valid_proba

                # accumulate test probas (avg over folds later)
                y_test_proba = model.predict_proba(X_test)
                oof_test[:, model_slice(j)] += y_test_proba

                # collect for ensemble computations
                valid_probas_list.append(y_valid_proba)
                valid_hard_preds_list.append(y_valid_pred)
                test_probas_list.append(y_test_proba)
                test_hard_preds_list.append(model.predict(X_test))

                # per-model scoring (hard labels by default)
                if scorer is not None:
                    score = scorer(y_valid, y_valid_pred)
                    scores[model.__class__.__name__].append(score)
                    per_model_fold_scores.append(score)
                    print(f"    Score: {score:.4f}")

            else:  # regression
                y_valid_pred = model.predict(X_valid).reshape(-1)
                oof_train[valid_idx, j] = y_valid_pred
                valid_preds_list.append(y_valid_pred)

                y_test_pred = model.predict(X_test).reshape(-1)
                oof_test[:, j] += y_test_pred  # accumulate; avg later
                test_preds_list.append(y_test_pred)

                if scorer is not None:
                    score = scorer(y_valid, y_valid_pred)
                    scores[model.__class__.__name__].append(score)
                    print(f"    Score: {score:.4f}")

            trained_models.append(model)

        # ---------- After all models in this fold ----------
        if task == 'classification':
            # Soft voting (uniform mean of probas)
            valid_soft = np.mean(valid_probas_list, axis=0)
            oof_train[valid_idx, soft_slice] = valid_soft

            test_soft = np.mean(test_probas_list, axis=0)
            oof_test[:, soft_slice] += test_soft

            # Hard voting (uniform proportions)
            n_valid = len(valid_idx)
            valid_hard_votes = np.zeros((n_valid, n_classes))
            for preds in valid_hard_preds_list:
                for c in range(n_classes):
                    valid_hard_votes[:, c] += (preds == c).astype(float)
            valid_hard_prop = valid_hard_votes / len(valid_hard_preds_list)
            oof_train[valid_idx, hard_slice] = valid_hard_prop

            test_hard_votes = np.zeros((n_test, n_classes))
            for preds in test_hard_preds_list:
                for c in range(n_classes):
                    test_hard_votes[:, c] += (preds == c).astype(float)
            test_hard_prop = test_hard_votes / len(test_hard_preds_list)
            oof_test[:, hard_slice] += test_hard_prop

            # Weighted voting (optional)
            if weight_mode in ('manual', 'auto'):
                if weight_mode == 'manual':
                    if model_weights is None or len(model_weights) != n_models:
                        raise ValueError("Provide model_weights (len == n_models) for weight_mode='manual'.")
                    w = _normalize_weights(model_weights)
                else:  # auto
                    if scorer is None:
                        raise ValueError("weight_mode='auto' requires a scorer.")
                    raw = np.array(per_model_fold_scores, dtype=float)
                    if not scorer_greater_is_better:
                        raw = raw.max() - raw + 1e-12  # invert for losses
                    w = _normalize_weights(raw)

                # Weighted Soft Voting (weighted mean of probas)
                valid_stack = np.stack(valid_probas_list, axis=2)      # (n_valid, n_classes, n_models)
                valid_wsoft = np.tensordot(valid_stack, w, axes=([2],[0]))  # (n_valid, n_classes)
                oof_train[valid_idx, wsoft_slice] = valid_wsoft

                test_stack = np.stack(test_probas_list, axis=2)        # (n_test, n_classes, n_models)
                test_wsoft = np.tensordot(test_stack, w, axes=([2],[0]))
                oof_test[:, wsoft_slice] += test_wsoft

                # Weighted Hard Voting (weighted vote proportions)
                valid_wvotes = np.zeros((n_valid, n_classes))
                for wj, preds in zip(w, valid_hard_preds_list):
                    for c in range(n_classes):
                        valid_wvotes[:, c] += wj * (preds == c).astype(float)
                oof_train[valid_idx, whard_slice] = valid_wvotes  # weights normalized already

                test_wvotes = np.zeros((n_test, n_classes))
                for wj, preds in zip(w, test_hard_preds_list):
                    for c in range(n_classes):
                        test_wvotes[:, c] += wj * (preds == c).astype(float)
                oof_test[:, whard_slice] += test_wvotes

            # Ensemble feature scores (based on hard labels)
            if scorer is not None:
                sv = scorer(y_valid, np.argmax(valid_soft, axis=1))
                hv = scorer(y_valid, np.argmax(valid_hard_prop, axis=1))
                scores["SoftVoting"].append(sv)
                scores["HardVoting"].append(hv)
                print(f"    SoftVoting score: {sv:.4f}")
                print(f"    HardVoting score: {hv:.4f}")

                if weight_mode in ('manual', 'auto'):
                    wsv = scorer(y_valid, np.argmax(valid_wsoft, axis=1))
                    whv = scorer(y_valid, np.argmax(valid_wvotes, axis=1))
                    scores["WeightedSoftVoting"].append(wsv)
                    scores["WeightedHardVoting"].append(whv)
                    print(f"    WeightedSoftVoting score: {wsv:.4f}")
                    print(f"    WeightedHardVoting score: {whv:.4f}")

        else:  # regression
            # Uniform mean (VotingRegressor-style without weights)
            valid_stack = np.column_stack(valid_preds_list)  # (len(valid_idx), n_models)
            test_stack  = np.column_stack(test_preds_list)   # (n_test, n_models)
            valid_mean = valid_stack.mean(axis=1)
            test_mean  = test_stack.mean(axis=1)

            oof_train[valid_idx, mean_col] = valid_mean
            oof_test[:, mean_col] += test_mean

            if scorer is not None:
                ms = scorer(y_valid, valid_mean)
                scores["MeanPred"].append(ms)
                print(f"    MeanPred score: {ms:.4f}")

            # Weighted mean (VotingRegressor-style with weights)
            if add_weighted:
                if weight_mode == 'manual':
                    if model_weights is None or len(model_weights) != n_models:
                        raise ValueError("Provide model_weights (len == n_models) for weight_mode='manual'.")
                    w = _normalize_weights(model_weights)
                else:  # auto (derive from per-model validation scores)
                    if scorer is None:
                        raise ValueError("weight_mode='auto' requires a scorer.")
                    per_model_scores = []
                    for j in range(n_models):
                        per_model_scores.append(scorer(y_valid, valid_stack[:, j]))
                    w = np.asarray(per_model_scores, float)
                    if not scorer_greater_is_better:
                        w = w.max() - w + 1e-12  # invert for losses
                    w = _normalize_weights(w)

                valid_wmean = (valid_stack * w).sum(axis=1)
                test_wmean  = (test_stack  * w).sum(axis=1)

                oof_train[valid_idx, wmean_col] = valid_wmean
                oof_test[:, wmean_col] += test_wmean

                if scorer is not None:
                    wms = scorer(y_valid, valid_wmean)
                    scores["WeightedMeanPred"].append(wms)
                    print(f"    WeightedMeanPred score: {wms:.4f}")

    # ---------- Average test predictions across folds ----------
    n_folds = folds.get_n_splits()
    oof_test /= n_folds

    # ---------- Print average scores ----------
    print("*"*50)
    for name, scs in scores.items():
        if len(scs) > 0:
            try:
                import numpy as _np
                if all(_np.isscalar(s) for s in scs):
                    print(f"{name}: mean score = {float(_np.mean(scs)):.4f}")
            except Exception:
                pass

    # ---------- Return DataFrames ----------
    import pandas as pd
    oof_train_df = pd.DataFrame(oof_train, columns=col_names, index=X.index)
    oof_test_df  = pd.DataFrame(oof_test,  columns=col_names, index=X_test.index)

    return oof_train_df, oof_test_df, scores, trained_models
