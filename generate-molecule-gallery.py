import csv
import json
from pathlib import Path


SOURCE = Path("verde+PCs-02142026.csv")
OUTPUT = Path("molecule-gallery-data.js")


def clean_xyz(raw_xyz, inchi_key):
    lines = [line.rstrip() for line in raw_xyz.strip().splitlines() if line.strip()]
    if len(lines) < 4:
        return None, None
    try:
        atoms = int(lines[0].strip())
    except ValueError:
        return None, None
    return atoms, "\n".join([str(atoms), f"{inchi_key} | S0 geometry"] + lines[2:])


def build_gallery_file():
    molecules = []
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            inchi_key = row.get("inchi_key", "")
            atoms, xyz_s0 = clean_xyz(row.get("xyz_S0", ""), inchi_key)
            if not xyz_s0:
                continue

            molecules.append(
                {
                    "inchiKey": inchi_key,
                    "label": inchi_key.split("-")[0] or inchi_key or "Molecule",
                    "smiles": row.get("smiles", ""),
                    "atoms": atoms,
                    "gapEv": row.get("gap (eV)", "") or "n/a",
                    "absorptionNm": row.get("absorption_wavelength (nm)", "") or "n/a",
                    "xyzS0": xyz_s0,
                }
            )

    OUTPUT.write_text(
        "// All available S0 geometry samples extracted from verde+PCs-02142026.csv for the 3D gallery.\n"
        "// Run `python generate-molecule-gallery.py` after replacing the CSV.\n"
        "window.S0_GALLERY_MOLECULES = "
        + json.dumps(molecules, indent=2, ensure_ascii=True)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(molecules)} S0 geometries.")


if __name__ == "__main__":
    build_gallery_file()
