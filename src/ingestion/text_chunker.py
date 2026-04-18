import json
import os
from pathlib import Path

# LangChain v0.2+ import path
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────
PUBMED_PATH   = Path("data/pubmed_abstracts/abstracts.json")
BIRADS_PATH   = Path("data/birads_reference/birads_knowledge.json")
OUTPUT_PATH   = Path("data/processed/text_chunks.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Splitter ───────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []

# ── 1. PubMed Abstracts ────────────────────────────────────────────────────
print("\n--- PubMed Abstracts ---")
with open(PUBMED_PATH, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Loaded {len(abstracts)} abstracts")

for paper in abstracts:
    text = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"pubmed_{paper.get('pmid', 'unknown')}_{i}",
            "text": chunk,
            "source": "pubmed",
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "journal": paper.get("journal", ""),
        })

pubmed_chunk_count = len(all_chunks)
print(f"Created {pubmed_chunk_count} chunks from PubMed abstracts")

# ── 2. BI-RADS Reference ───────────────────────────────────────────────────
print("\n--- BI-RADS Reference ---")
with open(BIRADS_PATH, "r", encoding="utf-8") as f:
    birads_docs = json.load(f)

print(f"Loaded {len(birads_docs)} BI-RADS documents")

for doc in birads_docs:
    text = f"{doc.get('category', '')}. {doc.get('description', '')} {doc.get('content', '')}"
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"birads_{doc.get('category', 'unknown').replace(' ', '_')}_{i}",
            "text": chunk,
            "source": "birads",
            "category": doc.get("category", ""),
        })

birads_chunk_count = len(all_chunks) - pubmed_chunk_count
print(f"Created {birads_chunk_count} chunks from BI-RADS reference")

# ── 3. Save ────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"✅ Total chunks: {len(all_chunks)}")
print(f"   PubMed:  {pubmed_chunk_count}")
print(f"   BI-RADS: {birads_chunk_count}")
print(f"💾 Saved to {OUTPUT_PATH}")

# ── 4. Quick sanity check ──────────────────────────────────────────────────
print("\n📄 Sample chunk:")
print(json.dumps(all_chunks[0], indent=2))