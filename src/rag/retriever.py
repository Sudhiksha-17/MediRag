import json
import chromadb
import torch
import open_clip
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR = Path("data/vectorstore")
TOP_K_TEXT  = 5
TOP_K_IMAGE = 3

# ── Load ChromaDB ──────────────────────────────────────────────────────────
client           = chromadb.PersistentClient(path=str(CHROMA_DIR))
text_collection  = client.get_collection("text_chunks")
image_collection = client.get_collection("image_cases")

# ── Load BiomedBERT for query embedding ────────────────────────────────────
print("Loading BiomedBERT for retrieval...")
device    = "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
bert_model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
).to(device)
bert_model.eval()

# ── Load BiomedCLIP for image query embedding ──────────────────────────────
print("Loading BiomedCLIP for image retrieval...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
clip_model = clip_model.to(device)
clip_model.eval()
tokenize_clip = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
print("Retriever ready.\n")

# ── Embed query text with BiomedBERT ───────────────────────────────────────
def embed_query_text(query: str):
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = bert_model(**encoded)
    token_emb = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
    pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1).cpu().tolist()[0]

# ── Embed query text with BiomedCLIP (for image search) ───────────────────
def embed_query_for_images(query: str):
    tokens = tokenize_clip([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().tolist()[0]

# ── Embed uploaded image with BiomedCLIP ──────────────────────────────────
def embed_image(image_path: str):
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().tolist()[0]

# ── Retrieve text chunks ───────────────────────────────────────────────────
def retrieve_text(query: str, top_k: int = TOP_K_TEXT):
    query_embedding = embed_query_text(query)
    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "title":    meta.get("title", ""),
            "pmid":     meta.get("pmid", ""),
            "score":    round(1 - dist, 4)
        })
    return chunks

# ── Retrieve similar images ────────────────────────────────────────────────
def retrieve_images(query: str = None, image_path: str = None, top_k: int = TOP_K_IMAGE):
    if image_path:
        query_embedding = embed_image(image_path)
    elif query:
        query_embedding = embed_query_for_images(query)
    else:
        return []

    results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    cases = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        cases.append({
            "description":      doc,
            "image_path":       meta.get("image_path", ""),
            "pathology":        meta.get("pathology", ""),
            "assessment":       meta.get("assessment", ""),
            "abnormality_type": meta.get("abnormality_type", ""),
            "score":            round(1 - dist, 4)
        })
    return cases

# ── Combined retrieval ─────────────────────────────────────────────────────
def retrieve(query: str, image_path: str = None):
    text_results  = retrieve_text(query)
    image_results = retrieve_images(query=query, image_path=image_path)
    return {
        "text_chunks":   text_results,
        "image_cases":   image_results
    }

# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "spiculated mass with irregular margins BI-RADS 5"
    print(f"Test query: '{test_query}'\n")
    results = retrieve(test_query)

    print(f"── Text Results ({len(results['text_chunks'])}) ──")
    for i, chunk in enumerate(results["text_chunks"]):
        print(f"{i+1}. [{chunk['score']}] {chunk['text'][:120]}...")
        print(f"   Source: {chunk['title'][:80]}\n")

    print(f"── Image Results ({len(results['image_cases'])}) ──")
    for i, case in enumerate(results["image_cases"]):
        print(f"{i+1}. [{case['score']}] {case['description']}")
        print(f"   Pathology: {case['pathology']}\n")