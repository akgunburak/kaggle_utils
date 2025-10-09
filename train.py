import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, r2_score, f1_score
import random
import gc


# ------------ utility: reproducibility ------------
def _set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------ dataset wrapper ------------
class _TabDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = None if y is None else np.asarray(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]

# ------------ simple flexible MLP ------------
class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=2, dropout=0.1, batchnorm=True, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "leaky_relu": nn.LeakyReLU(0.1)}[activation]
        layers = []
        last = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(last, hidden_dim)]
            if batchnorm:
                layers += [nn.BatchNorm1d(hidden_dim)]
            layers += [act, nn.Dropout(dropout)]
            last = hidden_dim
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------ base trainer (shared) ------------
def _train_with_early_stopping(
    model, optimizer, loss_fn,
    train_loader, val_loader,
    task, device,
    epochs, patience, grad_clip=None, trial_reporter=None
):
    best_state = None
    best_val = None
    no_improve = 0
    for ep in range(1, epochs + 1):
        # train
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            if task == "regression":
                yb = torch.as_tensor(yb, dtype=torch.float32, device=device)
            elif task == "binary":
                yb = torch.as_tensor(yb, dtype=torch.float32, device=device)
            else:
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)

            logits = model(xb)
            if task in ("binary", "regression"):
                logits = logits.squeeze(-1)  # (N,1)->(N,)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # validate
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                if task in ("binary", "regression"):
                    yb = torch.as_tensor(yb, dtype=torch.float32, device=device)
                else:
                    yb = torch.as_tensor(yb, dtype=torch.long, device=device)
                logits = model(xb)
                if task in ("binary", "regression"):
                    logits = logits.squeeze(-1)
                vloss = loss_fn(logits, yb)
                vals.append(vloss.item())
        mean_val = float(np.mean(vals))

        # optional reporting (e.g., for Optuna pruning)
        if trial_reporter is not None:
            trial_reporter(mean_val, step=ep)

        if (best_val is None) or (mean_val < best_val - 1e-8):
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

def _predict_logits(model, loader, task, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if task in ("binary", "regression"):
                logits = logits.squeeze(-1)
            outs.append(logits.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

# =========================================================
#                  Classifier Estimator
# =========================================================
class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible PyTorch MLP classifier.
    - Supports binary and multiclass.
    - Works with Pipeline, cross_val_score, GridSearchCV, Optuna, etc.
    """

    def __init__(
        self,
        hidden_dim=256,
        n_layers=2,
        dropout=0.1,
        batchnorm=True,
        activation="relu",           # "relu" | "gelu" | "leaky_relu"
        optimizer="adamw",           # "adamw" | "adam"
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=200,
        patience=20,
        grad_clip=None,
        val_size=0.1,
        scale_features=False,        # internal StandardScaler
        optimize_threshold=False,    # binary: tune decision threshold on val set
        threshold_metric="f1",       # {"f1","accuracy"} for tuning
        device="auto",               # "auto" | "cpu" | "cuda"
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.val_size = val_size
        self.scale_features = scale_features
        self.optimize_threshold = optimize_threshold
        self.threshold_metric = threshold_metric
        self.device = device
        self.random_state = random_state

    # ---- sklearn API ----
    def fit(self, X, y):
        _set_global_seed(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        self.task_ = "binary" if n_classes == 2 else "multiclass"

        # split tiny val set for early stopping / threshold tuning
        if self.task_ == "binary" or self.task_ == "multiclass":
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
            tr_idx, va_idx = next(splitter.split(X, y_idx))
        else:
            raise ValueError("Classifier received non-class targets.")

        X_tr, y_tr = X[tr_idx], y_idx[tr_idx]
        X_va, y_va = X[va_idx], y_idx[va_idx]

        # scaling (internal)
        self.scaler_ = None
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_tr = self.scaler_.fit_transform(X_tr)
            X_va = self.scaler_.transform(X_va)

        # build model
        in_dim = X.shape[1]
        out_dim = 1 if self.task_ == "binary" else n_classes
        self.model_ = _MLP(in_dim, out_dim,
                           hidden_dim=self.hidden_dim,
                           n_layers=self.n_layers,
                           dropout=self.dropout,
                           batchnorm=self.batchnorm,
                           activation=self.activation)

        dev = (torch.device("cuda") if (self.device == "auto" and torch.cuda.is_available())
               else torch.device(self.device if self.device != "auto" else "cpu"))
        self.device_ = dev
        self.model_.to(self.device_)

        # losses & optimizer
        if self.task_ == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        opt = (torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam)(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # loaders
        train_dl = DataLoader(_TabDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(_TabDataset(X_va, y_va), batch_size=self.batch_size, shuffle=False)

        # train
        _train_with_early_stopping(
            self.model_, opt, loss_fn, train_dl, val_dl,
            task=self.task_, device=self.device_,
            epochs=self.epochs, patience=self.patience, grad_clip=self.grad_clip
        )

        # pick threshold if binary
        self.threshold_ = 0.5
        if self.task_ == "binary" and self.optimize_threshold:
            # compute probs on val and tune threshold
            X_val_infer = X_va
            val_loader = DataLoader(_TabDataset(X_val_infer), batch_size=1024, shuffle=False)
            logits = _predict_logits(self.model_, val_loader, self.task_, self.device_)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grid = np.linspace(0.05, 0.95, 19)
            best_thr, best_val = 0.5, -np.inf
            for thr in grid:
                pred = (probs >= thr).astype(int)
                if self.threshold_metric == "f1":
                    val = f1_score(y_va, pred, zero_division=0)
                else:
                    val = accuracy_score(y_va, pred)
                if val > best_val:
                    best_val, best_thr = val, thr
            self.threshold_ = float(best_thr)

        # remember training artifacts for inference
        self.in_dim_ = in_dim
        self.n_classes_ = n_classes
        return self

    def _transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self, "scaler_") and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = self._transform_X(X)
        loader = DataLoader(_TabDataset(X), batch_size=1024, shuffle=False)
        logits = _predict_logits(self.model_, loader, self.task_, self.device_)
        if self.task_ == "binary":
            p1 = 1.0 / (1.0 + np.exp(-logits))
            return np.vstack([1 - p1, p1]).T  # shape (N,2)
        else:
            # softmax
            z = logits
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.task_ == "binary":
            thr = getattr(self, "threshold_", 0.5)
            yhat = (proba[:, 1] >= thr).astype(int)
            return self.classes_[yhat]
        else:
            yhat = proba.argmax(axis=1)
            return self.classes_[yhat]

    def score(self, X, y):
        # default sklearn behavior: accuracy for classifiers
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# =========================================================
#                  Regressor Estimator
# =========================================================
class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible PyTorch MLP regressor.
    - Works with Pipeline, cross_val_score, GridSearchCV, Optuna, etc.
    """

    def __init__(
        self,
        hidden_dim=256,
        n_layers=2,
        dropout=0.1,
        batchnorm=True,
        activation="relu",
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=200,
        patience=20,
        grad_clip=None,
        val_size=0.1,
        scale_features=False,
        device="auto",
        random_state=42,
        loss="mse"  # "mse" or "mae"
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.val_size = val_size
        self.scale_features = scale_features
        self.device = device
        self.random_state = random_state
        self.loss = loss

    def fit(self, X, y):
        _set_global_seed(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        # small val split for early stopping
        tr_idx, va_idx = train_test_split(
            np.arange(len(X)), test_size=self.val_size, random_state=self.random_state
        )
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # scaling
        self.scaler_ = None
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_tr = self.scaler_.fit_transform(X_tr)
            X_va = self.scaler_.transform(X_va)

        in_dim = X.shape[1]
        out_dim = 1
        self.model_ = _MLP(in_dim, out_dim,
                           hidden_dim=self.hidden_dim,
                           n_layers=self.n_layers,
                           dropout=self.dropout,
                           batchnorm=self.batchnorm,
                           activation=self.activation)

        dev = (torch.device("cuda") if (self.device == "auto" and torch.cuda.is_available())
               else torch.device(self.device if self.device != "auto" else "cpu"))
        self.device_ = dev
        self.model_.to(self.device_)

        loss_fn = nn.MSELoss() if self.loss == "mse" else nn.L1Loss()
        opt = (torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam)(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        train_dl = DataLoader(_TabDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(_TabDataset(X_va, y_va), batch_size=self.batch_size, shuffle=False)

        _train_with_early_stopping(
            self.model_, opt, loss_fn, train_dl, val_dl,
            task="regression", device=self.device_,
            epochs=self.epochs, patience=self.patience, grad_clip=self.grad_clip
        )

        self.in_dim_ = in_dim
        return self

    def _transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self, "scaler_") and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = self._transform_X(X)
        loader = DataLoader(_TabDataset(X), batch_size=1024, shuffle=False)
        preds = _predict_logits(self.model_, loader, "regression", self.device_)
        return preds.reshape(-1)

    def score(self, X, y):
        # default sklearn behavior: R^2
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def get_oof_predictions(
    models, X, y, X_test,
    folds=None,
    scorer=None,
    task='classification',
    weight_mode='none',                 # 'none' | 'manual' | 'auto'
    model_weights=None,                 # list[float] when weight_mode='manual'
    scorer_greater_is_better=True,      # set False if your scorer is a loss (lower is better)
    keep_models=False                   # NEW: avoid storing all fold models by default
):
    """
    Memory-lean OOF predictions for stacking (classification or regression).
    Returns: oof_train_df, oof_test_df, scores, trained_models (possibly empty)
    """
    import pandas as pd

    def _normalize_weights(w):
        w = np.array(w, dtype=np.float64)  # small, OK as float64
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

    # Pre-coerce to numpy where it helps (and keep indices for DataFrame output)
    X_idx = X.index
    Xtest_idx = X_test.index

    # ---------- Prepare OOF matrices & column names ----------
    if task == 'classification':
        classes = np.unique(y)
        n_classes = len(classes)
        add_weighted = (weight_mode in ('manual', 'auto'))

        base_cols = n_models * n_classes
        soft_cols = n_classes
        hard_cols = n_classes
        weighted_cols = (2 * n_classes) if add_weighted else 0
        total_cols = base_cols + soft_cols + hard_cols + weighted_cols

        # Use float32 to halve memory
        oof_train = np.zeros((n_train, total_cols), dtype=np.float32)
        oof_test  = np.zeros((n_test,  total_cols), dtype=np.float32)

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
        oof_train = np.zeros((n_train, n_models + 1 + extra), dtype=np.float32)
        oof_test  = np.zeros((n_test,  n_models + 1 + extra), dtype=np.float32)

        col_names = [f"{model.__class__.__name__}" for model in models] + ["MeanPred"]
        mean_col = n_models
        if add_weighted:
            wmean_col = n_models + 1
            col_names.append("WeightedMeanPred")

        scores = {model.__class__.__name__: [] for model in models}
        scores["MeanPred"] = []
        if add_weighted:
            scores["WeightedMeanPred"] = []

    trained_models = []  # will stay empty unless keep_models=True

    # ================= Cross-validation =================
    for i, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print(f"\nFold {i+1}/{folds.get_n_splits()}")

        # Pandas-friendly indexing
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        if task == 'classification':
            # Accumulators avoid keeping lists of hard predictions
            valid_soft_sum = np.zeros((len(valid_idx), n_classes), dtype=np.float32)
            test_soft_sum  = np.zeros((n_test, n_classes), dtype=np.float32)

            valid_hard_votes = np.zeros((len(valid_idx), n_classes), dtype=np.float32)
            test_hard_votes  = np.zeros((n_test, n_classes), dtype=np.float32)

            # For weighted (need per-model probas); keep as float32 to save RAM
            if weight_mode in ('manual', 'auto'):
                valid_probas_per_model = []
                test_probas_per_model  = []
                per_model_fold_scores  = []

        else:
            valid_preds_stack = []
            test_preds_stack  = []

        # ---------- Train each model ----------
        for j, base_model in enumerate(models):
            print(f"  Training model {j+1}/{n_models}: {base_model.__class__.__name__}")
            # Fit a *fresh* clone each fold to avoid model accumulation unless keep_models=True
            model = base_model
            model.fit(X_train, y_train)

            if task == 'classification':
                yv_pred  = model.predict(X_valid)
                yv_proba = model.predict_proba(X_valid).astype(np.float32, copy=False)
                # per-model OOF (train)
                oof_train[valid_idx, model_slice(j)] = yv_proba
                # accumulate for soft-vote
                valid_soft_sum += yv_proba
                # hard-vote accumulators (no list)
                for c in range(n_classes):
                    valid_hard_votes[:, c] += (yv_pred == c).reshape(-1).astype(np.float32)

                # test side
                yt_proba = model.predict_proba(X_test).astype(np.float32, copy=False)
                oof_test[:, model_slice(j)] += yt_proba
                test_soft_sum += yt_proba
                yt_pred = model.predict(X_test)
                for c in range(n_classes):
                    test_hard_votes[:, c] += (yt_pred == c).reshape(-1).astype(np.float32)

                if scorer is not None:
                    score = scorer(y_valid, yv_pred)
                    scores[model.__class__.__name__].append(score)
                    if weight_mode == 'auto':
                        per_model_fold_scores.append(score)
                        print(f"    Score: {score:.4f}")
                        
                if weight_mode in ('manual', 'auto'):
                    valid_probas_per_model.append(yv_proba)  # float32
                    test_probas_per_model.append(yt_proba)   # float32

            else:  # regression
                yv_pred = np.asarray(model.predict(X_valid)).reshape(-1).astype(np.float32)
                oof_train[valid_idx, j] = yv_pred
                valid_preds_stack.append(yv_pred)

                yt_pred = np.asarray(model.predict(X_test)).reshape(-1).astype(np.float32)
                oof_test[:, j] += yt_pred
                test_preds_stack.append(yt_pred)

                if scorer is not None:
                    score = scorer(y_valid, yv_pred)
                    scores[model.__class__.__name__].append(score)

            if keep_models:
                trained_models.append(model)
            else:
                # Explicitly drop references to model internals if large
                del model
                gc.collect()

        # ---------- After all models in this fold ----------
        if task == 'classification':
            # uniform soft vote
            valid_soft = (valid_soft_sum / n_models)
            test_soft  = (test_soft_sum  / n_models)
            oof_train[valid_idx, soft_slice] = valid_soft
            oof_test[:, soft_slice] += test_soft

            # uniform hard vote (proportions)
            valid_hard_prop = (valid_hard_votes / n_models)
            test_hard_prop  = (test_hard_votes  / n_models)
            oof_train[valid_idx, hard_slice] = valid_hard_prop
            oof_test[:, hard_slice] += test_hard_prop

            # scoring for ensemble features (hard labels)
            if scorer is not None:
                sv = scorer(y_valid, np.argmax(valid_soft, axis=1))
                hv = scorer(y_valid, np.argmax(valid_hard_prop, axis=1))
                scores["SoftVoting"].append(sv)
                scores["HardVoting"].append(hv)
                print(f"    SoftVoting score: {sv:.4f}")
                print(f"    HardVoting score: {hv:.4f}")

            # Weighted voting (optional)
            if weight_mode in ('manual', 'auto'):
                if weight_mode == 'manual':
                    if model_weights is None or len(model_weights) != n_models:
                        raise ValueError("Provide model_weights (len == n_models) for weight_mode='manual'.")
                    w = _normalize_weights(model_weights).astype(np.float32)
                else:  # auto
                    if scorer is None:
                        raise ValueError("weight_mode='auto' requires a scorer.")
                    raw = np.array(per_model_fold_scores, dtype=np.float64)
                    if not scorer_greater_is_better:
                        raw = raw.max() - raw + 1e-12  # invert for losses
                    w = _normalize_weights(raw).astype(np.float32)

                # Weighted Soft Voting (weighted mean of probas)
                # Compute with a memory-lean loop (no tensordot of a big stack)
                valid_wsoft = np.zeros_like(valid_soft, dtype=np.float32)
                test_wsoft  = np.zeros_like(test_soft,  dtype=np.float32)
                for wj, (vp, tp) in zip(w, zip(valid_probas_per_model, test_probas_per_model)):
                    valid_wsoft += wj * vp
                    test_wsoft  += wj * tp
                oof_train[valid_idx, wsoft_slice] = valid_wsoft
                oof_test[:, wsoft_slice] += test_wsoft

                # Weighted Hard Voting
                # We don't need to store per-model hard preds: use argmax of probas for a close proxy,
                # or re-predict hard labels per model (already done above as yv_pred/yt_pred)
                valid_wvotes = np.zeros_like(valid_hard_prop, dtype=np.float32)
                test_wvotes  = np.zeros_like(test_hard_prop,  dtype=np.float32)
                for wj, (vp, tp) in zip(w, zip(valid_probas_per_model, test_probas_per_model)):
                    vh = np.argmax(vp, axis=1)
                    th = np.argmax(tp, axis=1)
                    for c in range(n_classes):
                        valid_wvotes[:, c] += wj * (vh == c).reshape(-1).astype(np.float32)
                        test_wvotes[:, c]  += wj * (th == c).reshape(-1).astype(np.float32)
                oof_train[valid_idx, whard_slice] = valid_wvotes
                oof_test[:, whard_slice] += test_wvotes

                if scorer is not None:
                    wsv = scorer(y_valid, np.argmax(valid_wsoft, axis=1))
                    whv = scorer(y_valid, np.argmax(valid_wvotes, axis=1))
                    scores["WeightedSoftVoting"].append(wsv)
                    scores["WeightedHardVoting"].append(whv)
                    print(f"    WeightedSoftVoting score: {wsv:.4f}")
                    print(f"    WeightedHardVoting score: {whv:.4f}")

            # cleanup fold
            del valid_soft_sum, test_soft_sum, valid_hard_votes, test_hard_votes
            if weight_mode in ('manual','auto'):
                del valid_probas_per_model, test_probas_per_model
                if weight_mode == 'auto':
                    del per_model_fold_scores
            gc.collect()

        else:  # regression
            valid_stack = np.column_stack(valid_preds_stack).astype(np.float32, copy=False)
            test_stack  = np.column_stack(test_preds_stack).astype(np.float32, copy=False)

            valid_mean = valid_stack.mean(axis=1)
            test_mean  = test_stack.mean(axis=1)

            oof_train[valid_idx, mean_col] = valid_mean
            oof_test[:, mean_col] += test_mean

            if scorer is not None:
                ms = scorer(y_valid, valid_mean)
                scores["MeanPred"].append(ms)
                print(f"    MeanPred score: {ms:.4f}")

            if add_weighted:
                if weight_mode == 'manual':
                    if model_weights is None or len(model_weights) != n_models:
                        raise ValueError("Provide model_weights (len == n_models) for weight_mode='manual'.")
                    w = _normalize_weights(model_weights).astype(np.float32)
                else:
                    if scorer is None:
                        raise ValueError("weight_mode='auto' requires a scorer.")
                    per_model_scores = []
                    for j in range(n_models):
                        per_model_scores.append(scorer(y_valid, valid_stack[:, j]))
                    w = np.asarray(per_model_scores, dtype=np.float64)
                    if not scorer_greater_is_better:
                        w = w.max() - w + 1e-12
                    w = _normalize_weights(w).astype(np.float32)

                valid_wmean = (valid_stack * w).sum(axis=1)
                test_wmean  = (test_stack  * w).sum(axis=1)

                oof_train[valid_idx, wmean_col] = valid_wmean
                oof_test[:, wmean_col] += test_wmean

                if scorer is not None:
                    wms = scorer(y_valid, valid_wmean)
                    scores["WeightedMeanPred"].append(wms)
                    print(f"    WeightedMeanPred score: {wms:.4f}")

            # cleanup fold
            del valid_preds_stack, test_preds_stack, valid_stack, test_stack
            gc.collect()

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
    oof_train_df = pd.DataFrame(oof_train, columns=col_names, index=X_idx)
    oof_test_df  = pd.DataFrame(oof_test,  columns=col_names, index=Xtest_idx)

    return oof_train_df, oof_test_df, scores, trained_models
