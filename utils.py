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
