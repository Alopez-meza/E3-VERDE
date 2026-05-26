import csv
import json
import math
from pathlib import Path


SOURCE = Path("verde+PCs-02142026.csv")
OUTPUT = Path("property-distributions-data.js")
BIN_COUNT = 24

PROPERTIES = [
    ("homo (eV)", "HOMO", "eV", "Frontier Orbitals"),
    ("lumo (eV)", "LUMO", "eV", "Frontier Orbitals"),
    ("gap (eV)", "HOMO-LUMO Gap", "eV", "Frontier Orbitals"),
    ("ionization_potential (eV)", "Ionization Potential", "eV", "Frontier Orbitals"),
    ("oxidation_potential_S0 (eV)", "Oxidation Potential S0", "eV", "Redox Potentials"),
    ("reduction_potential_S0 (eV)", "Reduction Potential S0", "eV", "Redox Potentials"),
    ("oxidation_potential_S1 (eV)", "Oxidation Potential S1", "eV", "Redox Potentials"),
    ("reduction_potential_S1 (eV)", "Reduction Potential S1", "eV", "Redox Potentials"),
    ("oxidation_potential_T1 (eV)", "Oxidation Potential T1", "eV", "Redox Potentials"),
    ("reduction_potential_T1 (eV)", "Reduction Potential T1", "eV", "Redox Potentials"),
    ("dipole_magnitude_S0 (D)", "Dipole Magnitude S0", "D", "Dipoles"),
    ("dipole_magnitude_S1 (D)", "Dipole Magnitude S1", "D", "Dipoles"),
    ("dipole_magnitude_T1 (D)", "Dipole Magnitude T1", "D", "Dipoles"),
    ("0-0_S1 (eV)", "0-0 Energy S1", "eV", "Excited States"),
    ("0-0_T1 (eV)", "0-0 Energy T1", "eV", "Excited States"),
    ("vee_S1 (eV)", "Vertical Excitation S1", "eV", "Excited States"),
    ("vee_T1 (eV)", "Vertical Excitation T1", "eV", "Excited States"),
    ("vee_S1_Osc", "Oscillator Strength S1", "", "Excited States"),
    ("vee_T1_Osc", "Oscillator Strength T1", "", "Excited States"),
    ("absorption_wavelength (nm)", "Absorption Wavelength", "nm", "Optical Properties"),
]

DEFAULT_SELECTED = [
    "gap (eV)",
    "homo (eV)",
    "lumo (eV)",
    "oxidation_potential_S0 (eV)",
    "reduction_potential_S0 (eV)",
    "absorption_wavelength (nm)",
]


def as_float(value):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_bins(values):
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [{"start": round(minimum, 6), "end": round(maximum, 6), "count": len(values)}]

    width = (maximum - minimum) / BIN_COUNT
    counts = [0] * BIN_COUNT
    for value in values:
        index = int((value - minimum) / width)
        counts[min(index, BIN_COUNT - 1)] += 1

    return [
        {
            "start": round(minimum + index * width, 6),
            "end": round(minimum + (index + 1) * width, 6),
            "count": counts[index],
        }
        for index in range(BIN_COUNT)
    ]


def build_distribution_file():
    values = {key: [] for key, *_ in PROPERTIES}
    total_rows = 0
    subset_counts = {}

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            subset = row.get("dataset") or "Unlabeled"
            subset_counts[subset] = subset_counts.get(subset, 0) + 1

            for key, *_ in PROPERTIES:
                number = as_float(row.get(key))
                if number is not None:
                    values[key].append(number)

    summaries = []
    for key, label, unit, group in PROPERTIES:
        sorted_values = sorted(values[key])
        valid_count = len(sorted_values)
        missing_count = total_rows - valid_count

        if valid_count:
            mean = sum(sorted_values) / valid_count
            summary = {
                "key": key,
                "label": label,
                "unit": unit,
                "group": group,
                "validCount": valid_count,
                "missingCount": missing_count,
                "min": round(sorted_values[0], 4),
                "max": round(sorted_values[-1], 4),
                "mean": round(mean, 4),
                "p05": round(percentile(sorted_values, 0.05), 4),
                "p25": round(percentile(sorted_values, 0.25), 4),
                "median": round(percentile(sorted_values, 0.50), 4),
                "p75": round(percentile(sorted_values, 0.75), 4),
                "p95": round(percentile(sorted_values, 0.95), 4),
                "bins": build_bins(sorted_values),
            }
        else:
            summary = {
                "key": key,
                "label": label,
                "unit": unit,
                "group": group,
                "validCount": 0,
                "missingCount": missing_count,
                "min": None,
                "max": None,
                "mean": None,
                "p05": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p95": None,
                "bins": [],
            }
        summaries.append(summary)

    payload = {
        "source": SOURCE.name,
        "totalRows": total_rows,
        "datasetCounts": subset_counts,
        "binCount": BIN_COUNT,
        "properties": summaries,
        "defaultSelected": DEFAULT_SELECTED,
    }

    OUTPUT.write_text(
        "// Aggregated property distributions generated from verde+PCs-02142026.csv.\n"
        "// Run `python generate-property-distributions.py` after replacing the CSV.\n"
        "window.PROPERTY_DISTRIBUTIONS = "
        + json.dumps(payload, indent=2, ensure_ascii=True)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(summaries)} properties from {total_rows} rows.")


if __name__ == "__main__":
    build_distribution_file()
