import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

# 1. Load your CSV
df = pd.read_csv("tau_top289.csv")   # adjust path if needed

# 2. Parse SMILES to RDKit Mol objects; drop invalid SMILES
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

# 3. Function to convert a Mol into a Morgan fingerprint numpy array
def mol_to_morgan_fp_array(mol, radius=2, nBits=2048):
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=int)
    ConvertToNumpyArray(bitvect, arr)
    return arr

# 4. Generate fingerprints for every molecule
fps = df["mol"].apply(lambda m: mol_to_morgan_fp_array(m, radius=2, nBits=2048))

# 5. Stack into (n_samples × 2048) numpy matrix
X = np.stack(fps.values)

# 6. Turn to a DataFrame with meaningful column names
bit_cols = [f"FP_bit_{i}" for i in range(X.shape[1])]
fp_df   = pd.DataFrame(X, columns=bit_cols)

# 7. (Optional) concatenate back to your original data
out_df = pd.concat([df.drop(columns=["mol"]), fp_df], axis=1)

# 8. Save to a new CSV
out_df.to_csv("tau_top289_with_MorganFP.csv", index=False)

print("Done! Generated descriptors for", len(out_df), "molecules.")
