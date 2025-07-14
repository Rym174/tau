# predict_tnd_ic50.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcHallKierAlpha
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

# 1) Define your best‐found pipeline hyperparameters:
SELECT_K = 800
RIDGE_ALPHA = 1.0

# 2) Pipeline construction (matches stacked_optuna_full_pipeline)
def build_pipeline():
    svr = SVR(kernel="linear",
              C=0.015099038552218166,
              epsilon=0.20443241251689925,
              gamma="scale")
    xgb = XGBRegressor(objective="reg:squarederror",
                       n_estimators=600,
                       max_depth=5,
                       learning_rate=0.02,
                       subsample=0.8,
                       colsample_bytree=0.85,
                       seed=42,
                       verbosity=0)
    rf  = RandomForestRegressor(n_estimators=200, random_state=42)
    kr  = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)

    stack = StackingRegressor(
        estimators=[("svr", svr), ("xgb", xgb), ("rf", rf), ("kr", kr)],
        final_estimator=Ridge(alpha=RIDGE_ALPHA),
        passthrough=False,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.01)),
        ("select_k",   SelectKBest(f_regression, k=SELECT_K)),
        ("scale",      StandardScaler()),
        ("stack",      stack)
    ])
    return pipe

# 3) Featurization helper (exactly as training)
def morgan_fp(mol, radius):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, 2048)
    arr = np.zeros((bv.GetNumBits(),), dtype=int)
    ConvertToNumpyArray(bv, arr)
    return arr

def featurize_df(df):
    # Morgan2 & 3
    fps2 = np.stack([morgan_fp(m, 2) for m in df["mol"]])
    fps3 = np.stack([morgan_fp(m, 3) for m in df["mol"]])
    # MACCS
    macc = np.stack([np.array(MACCSkeys.GenMACCSKeys(m), dtype=int)
                     for m in df["mol"]])
    # Extra 2D/topo
    extra = np.vstack([
        df["mol"].apply(Descriptors.BertzCT),
        df["mol"].apply(Descriptors.BalabanJ),
        df["mol"].apply(CalcHallKierAlpha)
    ]).T
    # Physchem
    phys = np.vstack([
        df["mol"].apply(Descriptors.MolWt),
        df["mol"].apply(Descriptors.TPSA),
        df["mol"].apply(Descriptors.MolLogP)
    ]).T

    X = np.hstack([fps2, fps3, macc, extra, phys])
    return X

# 4) Load and featurize your 16 TND compounds
#    Assumes a two-column SMI: SMILES [tab or space] ID (we only need SMILES)
new = pd.read_csv("tnd_smiles.smi",
                  sep=r"\s+",
                  header=None,
                  names=["SMILES", "ID"],
                  engine="python")
new["mol"] = new["SMILES"].apply(Chem.MolFromSmiles)
new = new[new["mol"].notnull()].reset_index(drop=True)

X_new = featurize_df(new)

# 5) Fit pipeline on full 289-compound set
#    (so we use all data to train final model)
full = pd.read_csv("tau_top289_with_MorganFP.csv")
full["mol"] = full["SMILES"].apply(Chem.MolFromSmiles)
full = full[full["mol"].notnull()].reset_index(drop=True)
X_full = featurize_df(full)
y_full = full["pIC50"].values

pipeline = build_pipeline()
pipeline.fit(X_full, y_full)

# 6) Predict pIC50 and convert to IC50 (nM)
pIC50_pred   = pipeline.predict(X_new)
IC50_nM_pred = 10 ** (9 - pIC50_pred)

new["pred_pIC50"]   = pIC50_pred
new["pred_IC50_nM"] = IC50_nM_pred

# 7) Save results
new[["ID", "SMILES", "pred_pIC50", "pred_IC50_nM"]] \
    .to_csv("tnd_predicted_IC50.csv", index=False)

print("Saved tnd_predicted_IC50.csv")
