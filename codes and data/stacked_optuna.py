# stacked_optuna_full_pipeline.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcHallKierAlpha
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("tau_top289_with_MorganFP.csv")
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

# 2. Feature generation
def morgan_fp(mol, radius):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, 2048)
    arr = np.zeros((2048,), dtype=int)
    ConvertToNumpyArray(bv, arr)
    return arr

fps_r2 = np.stack(df["mol"].apply(lambda m: morgan_fp(m, 2)))
fps_r3 = np.stack(df["mol"].apply(lambda m: morgan_fp(m, 3)))
maccs  = np.stack(df["mol"].apply(lambda m: np.array(MACCSkeys.GenMACCSKeys(m), dtype=int)))

extra_desc = np.vstack([
    df["mol"].apply(Descriptors.BertzCT),
    df["mol"].apply(Descriptors.BalabanJ),
    df["mol"].apply(CalcHallKierAlpha)
]).T

physchem = np.vstack([
    df["mol"].apply(Descriptors.MolWt),
    df["mol"].apply(Descriptors.TPSA),
    df["mol"].apply(Descriptors.MolLogP)
]).T

# full feature matrix
X = np.hstack([fps_r2, fps_r3, maccs, extra_desc, physchem])
y = df["pIC50"].values

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 4. Base learners
svr_model = SVR(kernel="linear", C=0.015099038552218166,
                epsilon=0.20443241251689925, gamma="scale")
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=600,
                         max_depth=5, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.85,
                         seed=42, verbosity=0)
rf_model  = RandomForestRegressor(n_estimators=200, random_state=42)

# 5. Stacking ensemble
stack = StackingRegressor(
    estimators=[("svr", svr_model), ("xgb", xgb_model), ("rf", rf_model)],
    final_estimator=Ridge(alpha=1.0),
    passthrough=False,
    n_jobs=-1
)

# 6. Build pipeline
pipeline = Pipeline([
    ("var_thresh", VarianceThreshold(threshold=0.01)),
    ("select_k",   SelectKBest(f_regression, k=800)),  # will be tuned
    ("scale",      StandardScaler()),
    ("stack",      stack)
])

# 7. Hyperparameter search space
param_dist = {
    "select_k__k":                  [500, 800, 1000, 1500],
    "stack__final_estimator__alpha": [0.1, 1.0, 10.0]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# 8. Run search
search.fit(X_train, y_train)

# 9. Internal CV Q²
cv_q2 = search.best_score_

# 10. Evaluate on test set
best_pipe = search.best_estimator_
y_pred = best_pipe.predict(X_test)
test_r2   = r2_score(y_test, y_pred)
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 11. Save all metrics to CSV
metrics_df = pd.DataFrame({
    "metric": ["CV_Q2", "Test_R2", "Test_MAE", "Test_RMSE"],
    "value":  [cv_q2, test_r2, test_mae, test_rmse]
})
metrics_df.to_csv("stacked_optuna_full_metrics.csv", index=False)
print("Saved stacked_optuna_full_metrics.csv")

# 12. Plot Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "--", color="gray")
plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
plt.title(f"Stacked SVR+XGB+RF: CV_Q2={cv_q2:.3f}, Test_R2={test_r2:.3f}")
plt.tight_layout()
plt.savefig("stacked_optuna_full_r2.png", dpi=300)
print("Saved stacked_optuna_full_r2.png")
