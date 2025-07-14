import os

# File paths
receptor_file = "6hrf.pdb"
ligand_file = "TND-14_out.pdb"
output_file = "complex_TND14.pdb"

# Read receptor
with open(receptor_file, 'r') as rec:
    receptor_data = rec.readlines()

# Read ligand
with open(ligand_file, 'r') as lig:
    ligand_data = lig.readlines()

# Remove END lines if present
receptor_data = [line for line in receptor_data if not line.startswith("END")]
ligand_data = [line for line in ligand_data if not line.startswith("END")]

# Combine and write to new file
with open(output_file, 'w') as out:
    out.writelines(receptor_data)
    out.write("\n")  # separator
    out.writelines(ligand_data)
    out.write("END\n")

print(f"✅ Combined file saved as: {output_file}")
