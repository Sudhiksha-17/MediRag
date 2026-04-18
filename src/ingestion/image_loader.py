# src/ingestion/image_loader.py

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

class CBISDDSMLoader:
    """
    Loads and processes the CBIS-DDSM dataset.
    Handles both Kaggle format and original TCIA format.
    """
    
    def __init__(self, data_dir="data/raw/cbis_ddsm"):
        self.data_dir = data_dir
        self.cases = []
    
    def load_metadata(self):
        """
        Load metadata CSVs. The exact filenames depend on which 
        version you downloaded.
        """
        metadata_files = []
        
        # Find all CSV files in the data directory
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".csv"):
                    metadata_files.append(os.path.join(root, f))
        
        if not metadata_files:
            print("❌ No CSV metadata files found!")
            print(f"   Looked in: {self.data_dir}")
            print("   Make sure you've downloaded the dataset.")
            return pd.DataFrame()
        
        print(f"Found {len(metadata_files)} metadata files:")
        for f in metadata_files:
            print(f"  - {os.path.basename(f)}")
        
        # Load and combine all CSVs
        dfs = []
        for f in metadata_files:
            try:
                df = pd.read_csv(f)
                # Standardize column names (different versions use different names)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                dfs.append(df)
                print(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
            except Exception as e:
                print(f"  ⚠️ Error loading {f}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"\nTotal rows: {len(combined)}")
            print(f"Columns: {list(combined.columns)}")
            return combined
        
        return pd.DataFrame()
    
    def find_images(self):
        """Find all image files (PNG, JPG, DICOM) in the data directory."""
        image_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
        images = {}
        
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in image_extensions:
                    full_path = os.path.join(root, f)
                    # Use filename without extension as key
                    key = os.path.splitext(f)[0]
                    images[key] = full_path
        
        print(f"Found {len(images)} images")
        return images
    
    def process_cases(self, metadata_df, images_dict):
        """
        Match metadata rows with image files and create case entries.
        """
        cases = []
        matched = 0
        unmatched = 0
        
        # Common column name mappings across different CBIS-DDSM versions
        col_maps = {
            "pathology": ["pathology", "diagnosis", "class"],
            "assessment": ["assessment", "bi-rads", "birads", "bi_rads"],
            "abnormality_type": ["abnormality_type", "abnormality", "type"],
            "image_path": ["image_file_path", "image_path", "file_path", "image"],
            "case_id": ["patient_id", "case_id", "subject_id", "id"],
            "laterality": ["left_or_right", "laterality", "side", "left or right"],
            "view": ["image_view", "view", "image view"],
            "calc_type": ["calc_type", "calcification_type", "calc type"],
            "calc_distribution": ["calc_distribution", "calcification_distribution",
                                  "calc distribution"],
            "mass_shape": ["mass_shape", "shape", "mass shape"],
            "mass_margins": ["mass_margins", "margins", "mass margins"],
            "subtlety": ["subtlety", "subtle"],
        }
        
        def find_col(df, possible_names):
            """Find the actual column name from possible variations."""
            for name in possible_names:
                if name in df.columns:
                    return name
            return None
        
        # Resolve column names
        resolved = {}
        for key, possibilities in col_maps.items():
            col = find_col(metadata_df, possibilities)
            if col:
                resolved[key] = col
        
        print(f"\nResolved columns: {resolved}")
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df),
                              desc="Processing cases"):
            case = {"row_index": idx}
            
            # Extract available fields
            for key, col_name in resolved.items():
                value = row.get(col_name, "")
                case[key] = str(value).strip() if pd.notna(value) else ""
            
            # Try to find matching image
            image_path = None
            if "image_path" in case and case["image_path"]:
                # Check if the path from CSV points to an actual file
                possible_path = os.path.join(self.data_dir, case["image_path"])
                if os.path.exists(possible_path):
                    image_path = possible_path
            
            if image_path is None and "case_id" in case:
                # Try to match by case_id in the images dictionary
                for img_key, img_path in images_dict.items():
                    if case["case_id"] in img_key:
                        image_path = img_path
                        break
            
            case["resolved_image_path"] = image_path or ""
            
            if image_path:
                matched += 1
            else:
                unmatched += 1
            
            cases.append(case)
        
        print(f"\n✅ Processed {len(cases)} cases")
        print(f"   Matched with images: {matched}")
        print(f"   No image found: {unmatched}")
        
        self.cases = cases
        return cases
    
    def generate_case_descriptions(self):
        """
        Generate natural language descriptions for each case.
        These descriptions will be stored alongside image embeddings
        in ChromaDB so the LLM can reason about similar cases.
        """
        descriptions = []
        
        for case in self.cases:
            parts = []
            
            # Basic info
            case_id = case.get("case_id", "Unknown")
            parts.append(f"Mammography case {case_id}.")
            
            if case.get("laterality"):
                parts.append(f"{case['laterality']} breast.")
            if case.get("view"):
                parts.append(f"{case['view']} view.")
            
            # Abnormality info
            abn_type = case.get("abnormality_type", "").lower()
            if abn_type:
                parts.append(f"Finding type: {abn_type}.")
            
            # Calcification details
            if "calc" in abn_type:
                if case.get("calc_type"):
                    parts.append(f"Calcification morphology: {case['calc_type']}.")
                if case.get("calc_distribution"):
                    parts.append(f"Calcification distribution: {case['calc_distribution']}.")
            
            # Mass details
            if "mass" in abn_type:
                if case.get("mass_shape"):
                    parts.append(f"Mass shape: {case['mass_shape']}.")
                if case.get("mass_margins"):
                    parts.append(f"Mass margins: {case['mass_margins']}.")
            
            # Assessment and pathology
            if case.get("assessment"):
                parts.append(f"BI-RADS assessment category: {case['assessment']}.")
            if case.get("pathology"):
                parts.append(f"Pathology result: {case['pathology']}.")
            if case.get("subtlety"):
                parts.append(f"Subtlety rating: {case['subtlety']}/5.")
            
            description = " ".join(parts)
            descriptions.append({
                "case_id": case_id,
                "description": description,
                "image_path": case.get("resolved_image_path", ""),
                "metadata": {
                    "pathology": case.get("pathology", ""),
                    "assessment": case.get("assessment", ""),
                    "abnormality_type": case.get("abnormality_type", ""),
                    "laterality": case.get("laterality", ""),
                    "view": case.get("view", ""),
                }
            })
        
        return descriptions
    
    def save_processed_data(self, output_path="data/processed/cbis_ddsm_cases.json"):
        """Save processed cases with descriptions."""
        descriptions = self.generate_case_descriptions()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2)
        
        print(f"\n💾 Saved {len(descriptions)} case descriptions to {output_path}")
        
        # Print summary statistics
        pathologies = {}
        for d in descriptions:
            p = d["metadata"].get("pathology", "Unknown")
            pathologies[p] = pathologies.get(p, 0) + 1
        
        print(f"\n📊 Pathology distribution:")
        for p, count in sorted(pathologies.items(), key=lambda x: -x[1]):
            print(f"   {p}: {count}")
        
        # Print sample
        if descriptions:
            print(f"\n📄 Sample case description:")
            print(f"   {descriptions[0]['description']}")
        
        return descriptions


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🚀 Processing CBIS-DDSM dataset...")
    
    loader = CBISDDSMLoader(data_dir="data/raw/cbis_ddsm")
    
    # Step 1: Load metadata
    print("\n=== Step 1: Loading metadata ===")
    metadata = loader.load_metadata()
    
    if metadata.empty:
        print("\n⚠️ No metadata found. Please download the dataset first.")
        print("See the instructions in STEP 3.1 or 3.2 above.")
        exit(1)
    
    # Step 2: Find images
    print("\n=== Step 2: Finding images ===")
    images = loader.find_images()
    
    # Step 3: Match and process
    print("\n=== Step 3: Processing cases ===")
    cases = loader.process_cases(metadata, images)
    
    # Step 4: Generate descriptions and save
    print("\n=== Step 4: Generating descriptions ===")
    descriptions = loader.save_processed_data()
    
    print("\n✅ CBIS-DDSM processing complete!")