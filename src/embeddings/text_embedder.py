import json
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
CHUNKS_PATH  = Path("data/processed/text_chunks.json")
CHROMA_DIR   = Path("data/vectorstore")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE   = 32
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ── Load model ─────────────────────────────────────────────────────────────
print(f"Loading BiomedBERT — first run will download ~400MB...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Model loaded.")

# ── Mean pooling ───────────────────────────────────────────────────────────
def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def embed_texts(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pool(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()

# ── Load chunks ────────────────────────────────────────────────────────────
print(f"\nLoading chunks from {CHUNKS_PATH}...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Delete existing collection if re-running
try:
    client.delete_collection("text_chunks")
    print("Deleted existing text_chunks collection")
except:
    pass

collection = client.create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ── Embed and store in batches ─────────────────────────────────────────────
print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
print("This will take 20-40 minutes on GPU...\n")

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]
    ids   = [c["chunk_id"] for c in batch]
    metas = [{
        "source":  c.get("source", ""),
        "pmid":    c.get("pmid", ""),
        "title":   c.get("title", "")[:200],  # ChromaDB metadata limit
        "journal": c.get("journal", "")[:200],
        "category":c.get("category", ""),
    } for c in batch]

    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas
    )

print(f"\n{'='*50}")
print(f"✅ Stored {collection.count()} chunks in ChromaDB")
print(f"💾 Vector store saved to {CHROMA_DIR}")