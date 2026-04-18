import pandas as pd
import json
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_DIR    = Path("D:/Medirag/csv")
JPEG_DIR   = Path("D:/Medirag/jpeg")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build image lookup: folder_name -> list of jpg paths ──────────────────
print("🔍 Scanning image folders...")
image_lookup = {}
for folder in JPEG_DIR.iterdir():
    if folder.is_dir():
        jpgs = list(folder.glob("*.jpg"))
        if jpgs:
            # Use the largest jpg in the folder (that's the full mammogram)
            largest = max(jpgs, key=lambda f: f.stat().st_size)
            image_lookup[folder.name] = str(largest)

print(f"   Found {len(image_lookup)} image folders")

# ── Load CSVs ──────────────────────────────────────────────────────────────
print("\n📄 Loading CSVs...")
dfs = []
for csv_file in CSV_DIR.glob("*.csv"):
    if "dicom" in csv_file.name or "meta" in csv_file.name:
        continue  # skip dicom_info and meta
    df = pd.read_csv(csv_file)
    df["source_file"] = csv_file.name
    dfs.append(df)
    print(f"   {csv_file.name}: {len(df)} rows, columns: {list(df.columns)}")

combined = pd.concat(dfs, ignore_index=True)
print(f"\n   Total rows across all CSVs: {len(combined)}")

# ── Normalize column names ─────────────────────────────────────────────────
combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]
print(f"   Normalized columns: {list(combined.columns)}")

# ── Match images to metadata rows ─────────────────────────────────────────
print("\n🔗 Matching images to metadata...")

# The image_file_path column contains partial DICOM paths — 
# we match on the last folder segment
def find_image(row):
    for col in ["image_file_path", "cropped_image_file_path", "roi_mask_file_path"]:
        if col in row.index and pd.notna(row[col]):
            # Extract the DICOM folder ID from the path
            parts = str(row[col]).replace("\\", "/").split("/")
            for part in parts:
                if part in image_lookup:
                    return image_lookup[part]
    return None

combined["image_path"] = combined.apply(find_image, axis=1)
matched = combined[combined["image_path"].notna()]
print(f"   Matched {len(matched)} / {len(combined)} rows to images")

# ── Generate natural language descriptions ─────────────────────────────────
print("\n📝 Generating case descriptions...")

def generate_description(row):
    parts = []
    
    for col in ["side", "left_or_right_breast"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{str(row[col]).title()} breast")
            break
    
    for col in ["view", "image_view"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{str(row[col]).upper()} view")
            break

    for col in ["abnormality_type"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Abnormality: {str(row[col]).upper()}")

    for col in ["mass_shape"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Mass shape: {str(row[col]).upper()}")

    for col in ["mass_margins", "mass_margin"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Margins: {str(row[col]).upper()}")
            break

    for col in ["calc_type", "calc_morphology"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Calcification type: {str(row[col]).upper()}")
            break

    for col in ["calc_distribution"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Distribution: {str(row[col]).upper()}")

    for col in ["assessment", "birads_assessment"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"BI-RADS: {row[col]}")
            break

    for col in ["pathology"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Pathology: {str(row[col]).upper()}")

    for col in ["subtlety"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"Subtlety: {row[col]}/5")

    return ". ".join(parts) if parts else "Mammogram case."

cases = []
for _, row in matched.iterrows():
    desc = generate_description(row)
    cases.append({
        "case_id": f"cbis_{len(cases):05d}",
        "description": desc,
        "image_path": row["image_path"],
        "source_file": row.get("source_file", ""),
        "pathology": str(row.get("pathology", "")).upper(),
        "assessment": str(row.get("assessment", "")),
        "abnormality_type": str(row.get("abnormality_type", "")),
    })

# ── Save ───────────────────────────────────────────────────────────────────
output_path = OUTPUT_DIR / "cbis_cases.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"✅ Total cases saved: {len(cases)}")
print(f"💾 Saved to {output_path}")
print(f"\n📄 Sample case:")
if cases:
    print(json.dumps(cases[0], indent=2))