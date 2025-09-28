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


def objective_xgb_cv(trial, pos_weight, cross_val_splits, X, y, path):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17]),
        'random_state': trial.suggest_categorical('random_state', [2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'scale_pos_weight': trial.suggest_int('min_child_weight', 1, pos_weight + 100)
    }
    
    scores = []
    for train_idx_cv, val_idx_cv in cross_val_splits:
        #print(len(train_idx), len(val_idx))
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]
        
        # Drop duplicates from training set and align labels
        X_train_cv = X_train_cv.copy()  # avoid SettingWithCopyWarning
        X_train_cv["__target__"] = y_train_cv.values
        X_train_cv = X_train_cv.drop_duplicates()
        y_train_cv = X_train_cv["__target__"]
        X_train_cv = X_train_cv.drop(columns="__target__")
        
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, enable_categorical=True)
        dtest = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        bst = xgb.train(param, dtrain, evals=[(dtest, "test")], verbose_eval=False)
        preds = bst.predict(dtest)
        pred_labels = [1 if pred > 0.5 else 0 for pred in preds]    

        train_idx_first = train_idx_cv[0]
        train_idx_last = train_idx_cv[-1]
        train_len = len(train_idx_cv)
        val_idx_first = val_idx_cv[0]
        val_idx_last = val_idx_cv[-1]
        val_len = len(val_idx_cv)
        y_true_1 = y_val.value_counts(normalize=True)[1]
        y_true_0 = y_val.value_counts(normalize=True)[0]
        try:
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True)[1]
        except:
            y_pred_1 = 0
        try:
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True)[0]
        except:
            y_pred_0 = 0
        params = str(param)
        precision = precision_score(y_val, pred_labels)
        recall = recall_score(y_val, pred_labels)
        f1 = f1_score(y_val, pred_labels)        
        feature_importances = bst.get_score(importance_type='weight')
        importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = importance_df['Feature'].head(5).tolist()
        top_5_features = ', '.join(top_5_features)
        log_cv_scores(train_idx_first, train_idx_last, train_len, val_idx_first, val_idx_last, val_len, y_true_1, y_true_0, y_pred_1, y_pred_0, params, precision, recall, f1, top_5_features, path)
    
        scores.append(f1)
        
    avg_f1 = sum(scores) / len(scores)
    print("Scores: ", scores)
    return avg_f1


def objective_lgbm_cv(trial, cross_val_splits, X, y, path):
    param = {
        "categorical_feature": [col for col in X if X[col].dtype == 'category'],
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        #'verbose_eval': False,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
        'feature_fraction': trial.suggest_categorical('feature_fraction', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127, 255]),
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'random_state': trial.suggest_categorical('random_state', [2020]),
        "is_unbalance": trial.suggest_categorical('is_unbalance', [True, False]),
        'num_iterations': trial.suggest_int('num_iterations', 100, 1000)
    }

    scores = []
    for train_idx_cv, val_idx_cv in cross_val_splits:
        #print(len(train_idx), len(val_idx))
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]
        
        # Drop duplicates from training set and align labels
        X_train_cv = X_train_cv.copy()  # avoid SettingWithCopyWarning
        X_train_cv["__target__"] = y_train_cv.values
        X_train_cv = X_train_cv.drop_duplicates()
        y_train_cv = X_train_cv["__target__"]
        X_train_cv = X_train_cv.drop(columns="__target__")
        
        dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
        dtest = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        bst = lgb.train(param, dtrain, valid_sets=[dtest])
        preds = bst.predict(X_val)
        pred_labels = [1 if pred > 0.5 else 0 for pred in preds]    
        
        train_idx_first = train_idx_cv[0]
        train_idx_last = train_idx_cv[-1]
        train_len = len(train_idx_cv)
        val_idx_first = val_idx_cv[0]
        val_idx_last = val_idx_cv[-1]
        val_len = len(val_idx_cv)
        y_true_1 = y_val.value_counts(normalize=True)[1]
        y_true_0 = y_val.value_counts(normalize=True)[0]
        try:
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True)[1]
        except:
            y_pred_1 = 0
        try:
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True)[0]
        except:
            y_pred_0 = 0
        params = str(param)
        precision = precision_score(y_val, pred_labels)
        recall = recall_score(y_val, pred_labels)
        f1 = f1_score(y_val, pred_labels)        
        importance_df = pd.DataFrame({
            'Feature': bst.feature_name(),
            'Importance': bst.feature_importance()
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        top_5_features = importance_df['Feature'].head(5).tolist()
        top_5_features = ', '.join(top_5_features)
        log_cv_scores(train_idx_first, train_idx_last, train_len, val_idx_first, val_idx_last, val_len, y_true_1, y_true_0, y_pred_1, y_pred_0, params, precision, recall, f1, top_5_features, path)
    
        scores.append(f1)
        
    avg_f1 = sum(scores) / len(scores)
    print("Scores: ", scores)
    return avg_f1


def objective_cb_cv(trial, cross_val_splits, X, y, path):
    param = {
        'objective': 'Logloss',
        'eval_metric': 'Logloss',
        'verbose': 0,
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'colsample_bylevel': trial.suggest_categorical('colsample_bylevel', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'depth': trial.suggest_categorical('depth', [5,7,9,11,13]),
        'random_seed': trial.suggest_categorical('random_seed', [2020]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced']),
        'iterations': trial.suggest_int('iterations', 500, 1500)
    }
    
    for col in X.columns:
        if X[col].dtype == 'category':
            X[col] = X[col].astype(str).fillna('Unknown').astype('category')
    
    scores = []
    for train_idx_cv, val_idx_cv in cross_val_splits:
        #print(len(train_idx), len(val_idx))
        X_train_cv, X_val = X.iloc[train_idx_cv], X.iloc[val_idx_cv]
        y_train_cv, y_val = y.iloc[train_idx_cv], y.iloc[val_idx_cv]
        
        # Drop duplicates from training set and align labels
        X_train_cv = X_train_cv.copy()  # avoid SettingWithCopyWarning
        X_train_cv["__target__"] = y_train_cv.values
        X_train_cv = X_train_cv.drop_duplicates()
        y_train_cv = X_train_cv["__target__"]
        X_train_cv = X_train_cv.drop(columns="__target__")
        
        train_pool = cb.Pool(X_train_cv, y_train_cv, cat_features=[col for col in X_train_cv if X_train_cv[col].dtype == 'category'])
        test_pool = cb.Pool(X_val, y_val, cat_features=[col for col in X_train_cv if X_train_cv[col].dtype == 'category'])

        model = cb.CatBoostClassifier(**param)
        model.fit(train_pool, eval_set=test_pool, verbose=False)
        pred_labels = model.predict(X_val)   
        
        train_idx_first = train_idx_cv[0]
        train_idx_last = train_idx_cv[-1]
        train_len = len(train_idx_cv)
        val_idx_first = val_idx_cv[0]
        val_idx_last = val_idx_cv[-1]
        val_len = len(val_idx_cv)
        y_true_1 = y_val.value_counts(normalize=True)[1]
        y_true_0 = y_val.value_counts(normalize=True)[0]
        try:
            y_pred_1 = pd.Series(pred_labels).value_counts(normalize=True)[1]
        except:
            y_pred_1 = 0
        try:
            y_pred_0 = pd.Series(pred_labels).value_counts(normalize=True)[0]
        except:
            y_pred_0 = 0
        params = str(param)
        precision = precision_score(y_val, pred_labels)
        recall = recall_score(y_val, pred_labels)
        f1 = f1_score(y_val, pred_labels)        
        importance_df = pd.DataFrame({'Feature': model.feature_names_, 'Importance': model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_5_features = importance_df['Feature'].head(5).tolist()
        top_5_features = ', '.join(top_5_features)
        log_cv_scores(train_idx_first, train_idx_last, train_len, val_idx_first, val_idx_last, val_len, y_true_1, y_true_0, y_pred_1, y_pred_0, params, precision, recall, f1, top_5_features, path)
    
        scores.append(f1)
        
    avg_f1 = sum(scores) / len(scores)
    print("Scores: ", scores)
    return avg_f1


def objective_rf_cv(trial, cross_val_splits, X, y, path):
    # Random Forest hyperparameters to tune via Optuna
    param = {
        'model': 'RandomForestClassifier',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, None]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': trial.suggest_categorical('random_state', [2020]),
        'n_jobs': -1
    }

    scores = []
    for train_idx_cv, val_idx_cv in cross_val_splits:
        # Split
        X_train_cv, X_val = X.loc[train_idx_cv], X.loc[val_idx_cv]
        y_train_cv, y_val = y.loc[train_idx_cv], y.loc[val_idx_cv]

        # Drop duplicates from training set and align labels (same behavior as your LGBM fn)
        X_train_cv = X_train_cv.copy()
        X_train_cv["__target__"] = y_train_cv.values
        X_train_cv = X_train_cv.drop_duplicates()
        y_train_cv = X_train_cv["__target__"]
        X_train_cv = X_train_cv.drop(columns="__target__")

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=param['n_estimators'],
            max_depth=param['max_depth'],
            max_features=param['max_features'],
            min_samples_split=param['min_samples_split'],
            min_samples_leaf=param['min_samples_leaf'],
            bootstrap=param['bootstrap'],
            class_weight=param['class_weight'],
            random_state=param['random_state'],
            n_jobs=param['n_jobs']
        )
        rf.fit(X_train_cv, y_train_cv)

        # Predict (use proba threshold 0.5 to mirror the LGBM flow)
        if hasattr(rf, "predict_proba"):
            preds_proba = rf.predict_proba(X_val)[:, 1]
            pred_labels = (preds_proba > 0.5).astype(int)
        else:
            pred_labels = rf.predict(X_val)

        # Bookkeeping for logging
        train_idx_first = train_idx_cv[0]
        train_idx_last = train_idx_cv[-1]
        train_len = len(train_idx_cv)
        val_idx_first = val_idx_cv[0]
        val_idx_last = val_idx_cv[-1]
        val_len = len(val_idx_cv)

        vc_true = y_val.value_counts(normalize=True)
        y_true_1 = float(vc_true.get(1, 0.0))
        y_true_0 = float(vc_true.get(0, 0.0))

        vc_pred = pd.Series(pred_labels).value_counts(normalize=True)
        y_pred_1 = float(vc_pred.get(1, 0.0))
        y_pred_0 = float(vc_pred.get(0, 0.0))

        params_str = str(param)
        precision = precision_score(y_val, pred_labels, zero_division=0)
        recall = recall_score(y_val, pred_labels, zero_division=0)
        f1 = f1_score(y_val, pred_labels, zero_division=0)

        # Feature importances
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None:
            rf_importance_df = pd.DataFrame({
                'Feature': X_train_cv.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            top_5_features = ', '.join(rf_importance_df['Feature'].head(5).tolist())
        else:
            top_5_features = ''

        # Your existing logger
        log_cv_scores(
            train_idx_first, train_idx_last, train_len,
            val_idx_first, val_idx_last, val_len,
            y_true_1, y_true_0, y_pred_1, y_pred_0,
            params_str, precision, recall, f1, top_5_features, path
        )

        scores.append(f1)

    avg_f1 = float(np.mean(scores)) if len(scores) > 0 else 0.0
    print("Scores: ", scores)
    return avg_f1
    

def encode_categorical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None,
    *,
    ohe_max_cardinality: int = 3,
    high_card_strategy: str = "ordinal",  # "ordinal" | "target"
    drop_first: bool = False,
    dtype: type = float,
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

    # --- One-hot encoding ---
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

    # --- High-cardinality encoding ---
    if high_cols:
        if high_card_strategy == "ordinal":
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan
            )
            tr = ord_enc.fit_transform(X_train[high_cols])
            te = ord_enc.transform(X_test[high_cols])

            tr_df = pd.DataFrame(tr, index=X_train.index, columns=high_cols).astype(dtype)
            te_df = pd.DataFrame(te, index=X_test.index, columns=high_cols).astype(dtype)

            Xtr_parts.append(tr_df)
            Xte_parts.append(te_df)
            artifacts["ordinal"] = ord_enc

        elif high_card_strategy == "target":
            if y_train is None:
                raise ValueError("y_train must be provided when using target encoding.")
            te_enc = TargetEncoder()
            tr = te_enc.fit_transform(X_train[high_cols], y_train)
            te = te_enc.transform(X_test[high_cols])

            tr_df = pd.DataFrame(tr, index=X_train.index, columns=high_cols).astype(dtype)
            te_df = pd.DataFrame(te, index=X_test.index, columns=high_cols).astype(dtype)

            Xtr_parts.append(tr_df)
            Xte_parts.append(te_df)
            artifacts["target"] = te_enc

        else:
            raise ValueError("high_card_strategy must be 'ordinal' or 'target'.")

    # --- Combine all parts ---
    X_train_enc = pd.concat(Xtr_parts, axis=1)
    X_test_enc = pd.concat(Xte_parts, axis=1)

    return X_train_enc, X_test_enc, artifacts

def log_cv_scores(train_idx_first, train_idx_last, train_len, val_idx_first, val_idx_last, val_len, y_true_1, y_true_0, y_pred_1, y_pred_0, params, precision, recall, f1, top_5_features, path):
    row = {
        "train_idx_first": train_idx_first, 
        "train_idx_last": train_idx_last,
        "train_len": train_len,
        "val_idx_first": val_idx_first, 
        "val_idx_last": val_idx_last,
        "val_len": val_len,
        "params": params,
        "y_true_1": y_true_1,
        "y_true_0": y_true_0,
        "y_pred_1": y_pred_1,
        "y_pred_0": y_pred_0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top_5_features": top_5_features
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(path):
        df_existing = pd.read_excel(path)
        df_all = pd.concat([df_existing, df_row], ignore_index=True)
        # Optional: drop duplicates by parameters to avoid resaving the same trial
        df_all.drop_duplicates(keep="first", inplace=True)
        df_all.to_excel(path, index=False)
    else:
        df_row.to_excel(path, index=False)







