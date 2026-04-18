import json
import chromadb
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
CASES_PATH = Path("data/processed/cbis_cases.json")
CHROMA_DIR = Path("data/vectorstore")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16

# ── Load BiomedCLIP ────────────────────────────────────────────────────────
print("Loading BiomedCLIP — first run will download ~900MB...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
model = model.to(device)
model.eval()
preprocess = preprocess_val
print("BiomedCLIP loaded.")

# ── Load cases ─────────────────────────────────────────────────────────────
print(f"\nLoading cases from {CASES_PATH}...")
with open(CASES_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)
print(f"Loaded {len(cases)} cases")

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection("image_cases")
    print("Deleted existing image_cases collection")
except:
    pass

collection = client.create_collection(
    name="image_cases",
    metadata={"hnsw:space": "cosine"}
)

# ── Embed images in batches ────────────────────────────────────────────────
print(f"\nEmbedding {len(cases)} images in batches of {BATCH_SIZE}...")
print("This will take 30-60 minutes on GPU...\n")

skipped = 0
stored  = 0

for i in tqdm(range(0, len(cases), BATCH_SIZE)):
    batch = cases[i:i + BATCH_SIZE]

    images     = []
    valid_cases = []

    for case in batch:
        img_path = Path(case["image_path"])
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = preprocess(Image.open(img_path).convert("RGB"))
            images.append(img)
            valid_cases.append(case)
        except Exception as e:
            skipped += 1
            continue

    if not images:
        continue

    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(image_tensor)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().tolist()

    collection.add(
        ids=[c["case_id"] for c in valid_cases],
        embeddings=embeddings,
        documents=[c["description"] for c in valid_cases],
        metadatas=[{
            "image_path":      c["image_path"],
            "pathology":       c["pathology"],
            "assessment":      c["assessment"],
            "abnormality_type":c["abnormality_type"],
            "source_file":     c["source_file"],
        } for c in valid_cases]
    )
    stored += len(valid_cases)

print(f"\n{'='*50}")
print(f"✅ Stored {stored} image embeddings in ChromaDB")
print(f"⚠️  Skipped {skipped} images (missing or unreadable)")
print(f"💾 Vector store saved to {CHROMA_DIR}")