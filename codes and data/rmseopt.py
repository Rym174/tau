# optimize_rmse_q2.py

import pandas as pd
import numpy as np
import optuna

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcHallKierAlpha
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# 0) Silence RDKit deprecation warnings
RDLogger.DisableLog('rdApp.*')

# 1) Load & featurize full dataset
df = pd.read_csv("tau_top289_with_MorganFP.csv")
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

def morgan_fp(mol, radius):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, 2048)
    arr = np.zeros((bv.GetNumBits(),), dtype=int)
    ConvertToNumpyArray(bv, arr)
    return arr

fps2   = np.stack([morgan_fp(m, 2) for m in df["mol"]])
fps3   = np.stack([morgan_fp(m, 3) for m in df["mol"]])
maccs  = np.stack([np.array(MACCSkeys.GenMACCSKeys(m), dtype=int) for m in df["mol"]])
extra  = np.vstack([
    df["mol"].apply(Descriptors.BertzCT),
    df["mol"].apply(Descriptors.BalabanJ),
    df["mol"].apply(CalcHallKierAlpha)
]).T
phys   = np.vstack([
    df["mol"].apply(Descriptors.MolWt),
    df["mol"].apply(Descriptors.TPSA),
    df["mol"].apply(Descriptors.MolLogP)
]).T

X_full = np.hstack([fps2, fps3, maccs, extra, phys])
y_full = df["pIC50"].values

# 2) Train/Test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, shuffle=True
)

# 3) Define a pipeline factory
def build_pipeline(select_k, ridge_alpha):
    svr = SVR(kernel="linear",
              C=0.015099038552218166,
              epsilon=0.20443241251689925,
              gamma="scale")
    xgb = XGBRegressor(objective="reg:squarederror",
                       n_estimators=600, max_depth=5,
                       learning_rate=0.02, subsample=0.8,
                       colsample_bytree=0.85, seed=42,
                       verbosity=0)
    rf  = RandomForestRegressor(n_estimators=200, random_state=42)
    kr  = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)

    stack = StackingRegressor(
        estimators=[("svr", svr), ("xgb", xgb), ("rf", rf), ("kr", kr)],
        final_estimator=Ridge(alpha=ridge_alpha),
        passthrough=False, n_jobs=-1
    )

    return Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.01)),
        ("select_k",   SelectKBest(f_regression, k=select_k)),
        ("scale",      StandardScaler()),
        ("stack",      stack)
    ])

# 4) Set up CV object
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 5) Optuna objective minimizing CV-RMSE
def objective(trial):
    k     = trial.suggest_int("select_k", 300, X_tr.shape[1], step=100)
    alpha = trial.suggest_loguniform("ridge_alpha", 1e-2, 10.0)
    pipe  = build_pipeline(select_k=k, ridge_alpha=alpha)

    # Use neg_root_mean_squared_error for CV
    rmse_scores = cross_val_score(
        pipe, X_tr, y_tr,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    return float(-rmse_scores.mean())

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)

best_k     = study.best_params["select_k"]
best_alpha = study.best_params["ridge_alpha"]
best_cv_rmse = study.best_value

print(f"🔍 Best CV RMSE = {best_cv_rmse:.3f} log-units")
print(f"   Hyperparams → select_k: {best_k}, ridge_alpha: {best_alpha:.3f}")

# 6) Rebuild & evaluate best pipeline
best_pipe = build_pipeline(select_k=best_k, ridge_alpha=best_alpha)

# 6a) CV Q² (R² mean over folds)
q2_vals = cross_val_score(
    best_pipe, X_tr, y_tr,
    cv=cv, scoring="r2", n_jobs=-1
)
cv_q2, cv_q2_std = q2_vals.mean(), q2_vals.std()

# 6b) Fit on full training split and test
best_pipe.fit(X_tr, y_tr)
y_pred = best_pipe.predict(X_te)
test_r2   = r2_score(y_te, y_pred)
test_rmse = mean_squared_error(y_te, y_pred, squared=False)

# 7) Save and print results
metrics = pd.DataFrame({
    "metric": ["CV_RMSE", "CV_Q2_mean", "CV_Q2_std", "Test_R2", "Test_RMSE"],
    "value":  [best_cv_rmse, cv_q2, cv_q2_std, test_r2, test_rmse]
})
metrics.to_csv("optimized_metrics.csv", index=False)

print("✅ Results saved to optimized_metrics.csv")
print(f"CV Q² = {cv_q2:.3f} ± {cv_q2_std:.3f}")
print(f"Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}")
