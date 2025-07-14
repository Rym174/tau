from chembl_webresource_client.new_client import new_client
import pandas as pd, numpy as np
from collections import OrderedDict

TARGET_IDS = ["CHEMBL1075117", "CHEMBL1293224"]      # human tau entries
LIMIT       = 1000                                    # how many unique ligands you want
CUTOFF_NM   = 30_000                                 # keep up to 30 µM
TYPES       = ("IC50", "Ki", "EC50")                 # activity fields we'll accept

def fetch_rows(tid):
    """Yield activity dicts (IC50/Ki/EC50) sorted by potency."""
    for t in TYPES:
        rows = (
            new_client.activity
            .filter(target_chembl_id=tid,
                    standard_type=t,
                    standard_relation="=",
                    standard_value__lt=CUTOFF_NM,
                    standard_units="nM")
            .only(["canonical_smiles", "standard_value"])
            .order_by("standard_value")[:1000]        # over-pull then deduplicate
        )
        for r in rows:
            yield r

unique = OrderedDict()
for tid in TARGET_IDS:
    for r in fetch_rows(tid):
        smi, val = r["canonical_smiles"], r["standard_value"]
        if smi and val and smi not in unique:
            unique[smi] = float(val)
        if len(unique) == LIMIT:
            break
    if len(unique) == LIMIT:
        break

# assemble DataFrame
df = pd.DataFrame({
    "SMILES": list(unique.keys()),
    "IC50_nM": list(unique.values())
})
df["pIC50"] = -np.log10(df["IC50_nM"] * 1e-9)

df.to_csv("tau_top100.csv", index=False)
print(f"Saved {len(df)} ligands → tau_top1000.csv")