# MediRAG — Multimodal Retrieval-Augmented Generation for Medical Imaging Q&A

A multimodal RAG system that enables natural language question-answering over breast mammography data. Users can submit text queries, upload mammogram images, or combine both to retrieve grounded, cited answers from a knowledge base of **9,296 medical literature chunks** and **3,568 annotated mammogram cases**.

The system uses **dual embedding models** (BiomedBERT for text retrieval, BiomedCLIP for cross-modal image-text retrieval), a **ChromaDB vector store** with separate text and image collections, and **Llama-3 via Groq API** for grounded answer generation with source citations.

Evaluated using the **RAGAS framework**: 82% context precision@5 and 0.78 faithfulness score across structured medical test cases.

---

## Demo

> **[Watch the demo video →](#)** *https://drive.google.com/file/d/1YygBWOIL1EeahSHBYHGAHSbWbkNtcZR3/view?usp=sharing*

**Example interactions:**

| Query Type | Example | System Behavior |
|---|---|---|
| Text only | *"What are the characteristics of a malignant mass on mammography?"* | BiomedBERT embeds query → retrieves top-5 from 9,296 text chunks + top-3 image cases via CLIP text encoder → LLM generates cited answer |
| Image only | *(upload a mammogram)* | BiomedCLIP encodes image → cosine similarity search over 3,568 CBIS-DDSM embeddings → returns similar cases with pathology labels |
| Multimodal | *"What does this calcification pattern suggest?"* + uploaded image | Parallel retrieval: BiomedBERT on text collection + BiomedCLIP on image collection → merged context → grounded generation |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (app.py)                  │
│          Text input + image upload + source visualization        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline (pipeline.py)                   │
│                                                                  │
│  1. retrieve(query, image_path)     → hybrid vector search       │
│  2. build_prompt(query, chunks, cases) → context assembly        │
│  3. Groq API (Llama-3.1 8B)        → grounded generation        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│   Text Retrieval     │  │   Image Retrieval         │
│                      │  │                            │
│  Query               │  │  Query (text or image)     │
│    ↓                 │  │    ↓                       │
│  BiomedBERT          │  │  BiomedCLIP                │
│  (768-dim, mean pool)│  │  (512-dim, L2 normalized)  │
│    ↓                 │  │    ↓                       │
│  ChromaDB            │  │  ChromaDB                  │
│  "text_chunks"       │  │  "image_cases"             │
│  9,296 chunks        │  │  3,568 cases               │
│  cosine similarity   │  │  cosine similarity         │
│    ↓                 │  │    ↓                       │
│  Top-5 chunks        │  │  Top-3 cases               │
│  + metadata          │  │  + pathology labels        │
└──────────────────────┘  └──────────────────────────┘
```

---

## Retrieval Pipeline — Technical Deep Dive

The retrieval system (`retriever.py`) is the core of MediRAG. It uses a **dual-encoder architecture** with modality-specific routing.

### Dual Embedding Strategy

The system deliberately uses **two separate embedding models** rather than a single unified encoder:

| Model | Used For | Embedding Dim | Rationale |
|---|---|---|---|
| **BiomedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) | Text-to-text retrieval | 768 | Pre-trained on 16.8M PubMed abstracts + 3.3M PMC full-text articles. Produces significantly better semantic representations for medical text than CLIP's text encoder, which is optimized for image-text alignment rather than text-text similarity. |
| **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | Image encoding, text-to-image search, image-to-image search | 512 | Contrastive vision-language model trained on 15M figure-caption pairs from PubMed Central. Enables cross-modal retrieval where a text query can find relevant mammogram images and vice versa. |

### Query Routing Logic

The retriever adapts its embedding and search strategy based on what the user provides:

```python
def retrieve(query: str, image_path: str = None):
    # ALWAYS run text retrieval with BiomedBERT
    text_results = retrieve_text(query)          # BiomedBERT → text_chunks collection

    # Image retrieval strategy depends on input modality
    if image_path:
        # User uploaded an image → encode with CLIP vision encoder
        image_results = retrieve_images(image_path=image_path)  # CLIP ViT → image_cases
    else:
        # Text-only → use CLIP text encoder for cross-modal search
        image_results = retrieve_images(query=query)  # CLIP text → image_cases

    return {"text_chunks": text_results, "image_cases": image_results}
```

**Why this matters:** When a user asks *"What does a spiculated mass look like?"*, the system:
1. Uses **BiomedBERT** to find the most semantically relevant medical literature about spiculated masses (text-to-text)
2. Uses **BiomedCLIP's text encoder** to find mammogram images that match the concept of "spiculated mass" (text-to-image cross-modal search)
3. Combines both into a unified context for the LLM

When a user **uploads an image**, step 2 switches to **BiomedCLIP's vision encoder** for direct image-to-image similarity search against the CBIS-DDSM collection.

### Embedding Details

**BiomedBERT text embedding** uses mean pooling with attention mask weighting:

```python
def embed_query_text(query: str):
    encoded = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    output = bert_model(**encoded)
    token_emb = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
    pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # masked mean
    return F.normalize(pooled, p=2, dim=1)  # L2 normalize for cosine similarity
```

This is more robust than using the `[CLS]` token alone, as it captures semantic information distributed across all tokens while ignoring padding.

**BiomedCLIP image embedding** uses the pre-trained vision transformer with L2 normalization:

```python
def embed_image(image_path: str):
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    image_features = clip_model.encode_image(img)
    return image_features / image_features.norm(dim=-1, keepdim=True)  # unit sphere
```

### Vector Store Design

ChromaDB with two persistent collections, both configured for cosine similarity:

**`text_chunks` collection (9,296 entries)**
- Source: 1,089 PubMed abstracts chunked via `RecursiveCharacterTextSplitter` (300-char, 50-overlap) + 14 BI-RADS reference documents (400-char, 75-overlap)
- Metadata per chunk: `source`, `title`, `pmid`, `journal`, `date`, `finding_type`, `doc_type`, `chunk_id`
- Indexed with BiomedBERT 768-dim embeddings

**`image_cases` collection (3,568 entries)**
- Source: CBIS-DDSM annotated mammogram cases
- Each entry stores: BiomedCLIP 512-dim image embedding + generated natural language case description + structured metadata
- Metadata per case: `pathology` (BENIGN/MALIGNANT), `assessment` (BI-RADS category), `abnormality_type`, `image_path`, `laterality`, `view`
- The case description enables the LLM to reason about image results in natural language

### Context Assembly and Generation

The pipeline (`pipeline.py`) assembles retrieved results into a structured prompt:

```
System instruction (expert radiologist, cite sources, don't hallucinate)
    ↓
=== RELEVANT MEDICAL LITERATURE ===
[1] {chunk text} — Source: {PubMed title}
[2] {chunk text} — Source: {BI-RADS reference}
...
    ↓
=== SIMILAR MAMMOGRAM CASES ===
[Case 1] {case description} — Pathology: MALIGNANT | BI-RADS: 5
[Case 2] {case description} — Pathology: BENIGN | BI-RADS: 2
...
    ↓
QUESTION: {user query}
```

Generation uses **Llama-3.1 8B** via Groq API with `temperature=0.2` for factual consistency and `max_tokens=1024`. The low temperature minimizes hallucination while still producing fluent, natural responses.

---

## Evaluation

### RAGAS Framework Evaluation

Evaluated using [RAGAS](https://docs.ragas.io/) with **Llama-3.3 70B as the judge model** (via Groq API) across 5 structured medical test cases with ground truth answers.

| Metric | Score | What It Measures |
|---|---|---|
| **Context Precision@5** | **0.82** | Are the top-5 retrieved chunks actually relevant to answering the question? Measures retrieval quality. |
| **Faithfulness** | **0.78** | Is the generated answer actually supported by the retrieved context? Measures hallucination resistance. |

**Judge setup:** Llama-3.3 70B Versatile (via Groq, temperature=0) evaluates whether each claim in the generated answer is supported by the retrieved context (faithfulness) and whether the retrieved documents contain the information needed to answer the question (context precision). Embeddings for RAGAS computed via `sentence-transformers/all-MiniLM-L6-v2`.

**Test cases covered:**
- Malignant mass characteristics on mammography
- BI-RADS category interpretation and recommended follow-up
- Significance of spiculated margins
- Calcification classification systems
- Role of deep learning in mammography CAD

### Keyword-Based Retrieval Evaluation

Separately evaluated retrieval quality across **20 diverse mammography queries** using keyword coverage analysis:

- Each query has 4 expected domain-specific keywords
- A query passes if ≥50% of keywords appear in the generated answer AND the answer is not a refusal AND sources were retrieved
- Tests cover: BI-RADS categories (0-5), mass descriptors, calcification types, screening guidelines, architectural distortion, breast density, tomosynthesis, and deep learning in CAD

### Evaluation Scripts

```bash
# Run RAGAS evaluation (requires GROQ_API_KEY, ~5-10 minutes)
python src/rag/evaluate_ragas.py

# Run keyword-based evaluation (requires GROQ_API_KEY, ~2-3 minutes)
python src/rag/evaluate.py
```

Results are saved to `data/processed/ragas_results.json` and `data/processed/eval_results.json`.

---

## Data Pipeline

### PubMed Abstract Collection
- **12 targeted queries** via Entrez API (Biopython): mammography classification, calcification detection, mass characterization, architectural distortion, screening guidelines, deep learning in CAD, and more
- 1,089 unique abstracts after deduplication by PMID
- Each abstract includes: title, abstract text, authors, journal, publication date, MeSH terms
- Chunked into 300-character passages with 50-character overlap using `RecursiveCharacterTextSplitter`

### BI-RADS Reference Knowledge Base
- 14 manually curated documents from publicly available radiology education resources
- Coverage: all 7 assessment categories (BI-RADS 0-6), calcification morphology (typically benign + suspicious types), calcification distribution patterns (diffuse, regional, grouped, linear, segmental), mass shape/margin/density descriptors, associated features (skin retraction, architectural distortion, lymphadenopathy)
- Chunked with larger window (400 characters, 75 overlap) to preserve definitional context

### CBIS-DDSM Image Dataset
- 3,568 annotated mammogram cases from the Curated Breast Imaging Subset of DDSM
- Per case: mammogram image, pathology label (BENIGN/MALIGNANT/BENIGN_WITHOUT_CALLBACK), BI-RADS assessment, abnormality type, calcification morphology and distribution, mass shape and margins
- Each case has a generated natural language description combining all metadata fields, enabling the LLM to reason about retrieved images

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| Text Embeddings | BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, 768-dim, mean pooling |
| Image Embeddings | BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, 512-dim, ViT-B/16 |
| Vector Database | ChromaDB | Persistent storage, cosine similarity, two collections |
| LLM (Generation) | Llama-3.1 8B | Via Groq API, temperature=0.2 |
| LLM (RAGAS Judge) | Llama-3.3 70B | Via Groq API, temperature=0 |
| Text Chunking | LangChain | `RecursiveCharacterTextSplitter` with domain-tuned parameters |
| Data Source (Text) | PubMed | Entrez API via Biopython, 12 targeted queries |
| Data Source (Images) | CBIS-DDSM | Cancer Imaging Archive, Kaggle mirror |
| Frontend | Streamlit | Dual-panel layout, image upload, expandable source panels |
| Language | Python 3.12 | PyTorch, Transformers, OpenCLIP |

---

## Performance

| Metric | Value |
|---|---|
| Text chunks indexed | 9,296 |
| Image cases indexed | 3,568 |
| Retrieval latency | ~2s per query |
| Generation latency | ~3s via Groq API |
| End-to-end (text-only) | ~5-6s |
| End-to-end (multimodal) | ~7-8s |
| Context Precision@5 (RAGAS) | 0.82 |
| Faithfulness (RAGAS) | 0.78 |

---

## Project Structure

```
medical-rag-assistant/
├── src/
│   ├── ingestion/                  # Data collection and preprocessing
│   │   ├── pubmed_fetch.py         # Fetch 1,089 PubMed abstracts via Entrez API
│   │   ├── birads_reference.py     # 14 curated BI-RADS knowledge documents
│   │   ├── text_chunker.py         # Chunk corpus → 9,296 passages
│   │   └── cbis_loader.py          # Load + process 3,568 CBIS-DDSM cases
│   ├── embeddings/                 # Embedding generation
│   │   ├── text_embedder.py        # BiomedBERT (768-dim, mean pooled)
│   │   └── image_embedder.py       # BiomedCLIP (512-dim, L2 normalized)
│   ├── vectorstore/                # Vector database
│   │   └── chroma_store.py         # ChromaDB: 2 collections, cosine similarity
│   └── rag/                        # Core RAG pipeline
│       ├── retriever.py            # Dual-encoder hybrid retrieval
│       ├── pipeline.py             # Context assembly + Llama-3 generation
│       ├── evaluate.py             # Keyword-based eval (20 queries)
│       └── evaluate_ragas.py       # RAGAS eval (faithfulness + precision)
├── app.py                          # Streamlit frontend
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup and Reproduction

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU recommended (CPU works but embedding generation is slower)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/Sudhiksha-17/MediRag.git
cd MediRag

python -m venv medicalrag
# Windows
medicalrag\Scripts\Activate.ps1
# Linux/Mac
source medicalrag/bin/activate

pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

### Data Ingestion (run once)

```bash
# 1. Fetch PubMed abstracts (~5-10 min)
python src/ingestion/pubmed_fetch.py

# 2. Generate BI-RADS reference knowledge base
python src/ingestion/birads_reference.py

# 3. Chunk all text data into passages
python src/ingestion/text_chunker.py

# 4. Download CBIS-DDSM from Kaggle:
#    https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
#    Unzip into data/raw/cbis_ddsm/

# 5. Process images, generate embeddings, build vector store
python src/ingestion/cbis_loader.py
```

### Run the Application

```bash
streamlit run app.py
```

### Run Evaluation

```bash
# RAGAS evaluation (faithfulness + context precision)
python src/rag/evaluate_ragas.py

# Keyword-based retrieval evaluation (20 queries)
python src/rag/evaluate.py
```

---

## Transferability to Search and Recommendation Systems

While built for medical imaging, the architecture maps directly to product search and marketplace recommendation:

| MediRAG Component | Marketplace / E-commerce Equivalent |
|---|---|
| BiomedCLIP image embeddings | Product image embeddings for visual search ("find similar items") |
| BiomedBERT text embeddings | Product description / search query embeddings |
| Dual-encoder hybrid retrieval | Multimodal product search (text query + photo input) |
| ChromaDB cosine similarity top-k | Candidate retrieval layer in a two-stage ranking pipeline |
| Metadata-filtered vector search | Faceted search (category, price range, brand, condition) |
| Context-aware LLM generation | Conversational search, recommendation explanations |
| RAGAS evaluation framework | Search relevance evaluation, ranking quality metrics |
| Cross-modal CLIP alignment | "Search by photo" or "find this look" features |

---

## Future Improvements

- **Re-ranking layer:** Add a cross-encoder (e.g., `ms-marco-MiniLM`) after initial vector retrieval for higher precision at the cost of latency
- **Query classification agent:** Route queries by modality and complexity to specialized retrieval strategies with different top-k and score thresholds
- **Fine-tune BiomedCLIP** on CBIS-DDSM for improved mammogram-specific embeddings beyond the general biomedical domain
- **Reciprocal Rank Fusion:** Merge text and image retrieval scores using RRF instead of simple concatenation for better multimodal ranking
- **Scale vector store:** Migrate from ChromaDB to Milvus or Qdrant with HNSW indexing for sub-100ms retrieval at million-scale

---

## Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for clinical diagnosis. Always consult a qualified radiologist for medical imaging interpretation.