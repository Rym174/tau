import os
import subprocess

# === Configuration ===
vina_path = r"C:\Users\poula\Documents\The Scripps Research Institute\Vina\vina.exe"
receptor = "6hrf.pdbqt"
ligands = ["TND-11.pdbqt", "TND-12.pdbqt", "TND-13.pdbqt", "TND-14.pdbqt", "TND-15.pdbqt"]

# === Grid box coordinates (from .gpf converted to Vina format) ===
center_x = 113.222
center_y = 165.477
center_z = 140.388
size_x = 27.0
size_y = 33.0
size_z = 16.5

# === Output directory (inside the current folder) ===
output_dir = "vina_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Dock each ligand ===
for ligand in ligands:
    name = ligand.split(".")[0]
    out_pdbqt = os.path.join(output_dir, f"{name}_out.pdbqt")
    log_file = os.path.join(output_dir, f"{name}_log.txt")

    command = [
        vina_path,
        "--receptor", receptor,
        "--ligand", ligand,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--out", out_pdbqt,
        "--log", log_file,
        "--exhaustiveness", "16"
    ]

    print(f"\n⏳ Docking {ligand}...")
    subprocess.run(command)

print("\n✅ Docking completed for all ligands.")
