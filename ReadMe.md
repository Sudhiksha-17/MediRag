# 🩻 MediRAG — Multimodal Mammography RAG Assistant

A retrieval-augmented generation (RAG) system for breast imaging Q&A, combining medical literature and mammogram image search with an LLM to generate grounded, cited answers.

## Architecture
```
User Query + Optional Image
        ↓
   BiomedBERT (text embedding)
   BiomedCLIP (image embedding)
        ↓
   ChromaDB Vector Search
   ├── Text collection (9,296 PubMed chunks)
   └── Image collection (3,568 CBIS-DDSM cases)
        ↓
   Context Builder
        ↓
   Llama3 via Groq API
        ↓
   Grounded Answer + Citations
```

## Dataset
- **PubMed**: 1,089 mammography abstracts → 9,296 text chunks
- **BI-RADS Reference**: 14 structured radiology knowledge documents
- **CBIS-DDSM**: 3,568 annotated mammogram cases with pathology labels

## Models
| Component | Model |
|-----------|-------|
| Text Embeddings | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |
| Image Embeddings | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 |
| Vector Store | ChromaDB (cosine similarity) |
| LLM | Llama3 8B via Groq API |
| Image Captioning | BiomedCLIP text-image alignment |

## Setup
```bash
# Clone and create environment
git clone <your-repo>
cd medical-rag-assistant
python -m venv medicalrag
medicalrag\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY=your_key_here  # Linux/Mac
$env:GROQ_API_KEY="your_key_here"  # Windows

# Run
streamlit run app.py
```

## Project Structure
```
medical-rag-assistant/
├── src/
│   ├── ingestion/          # Data collection scripts
│   │   ├── pubmed_fetch.py
│   │   ├── birads_reference.py
│   │   ├── text_chunker.py
│   │   └── cbis_loader.py
│   ├── embeddings/         # Embedding scripts
│   │   ├── text_embedder.py
│   │   └── image_embedder.py
│   └── rag/                # RAG pipeline
│       ├── retriever.py
│       └── pipeline.py
├── data/
│   ├── pubmed_abstracts/
│   ├── birads_reference/
│   └── processed/
├── app.py                  # Streamlit frontend
├── Dockerfile
└── README.md
```

## Results
- 9,296 text chunks indexed from PubMed mammography literature
- 3,568 CBIS-DDSM mammogram cases with pathology labels
- Retrieval latency: ~2s per query
- Answer generation: ~3s via Groq API

##  Disclaimer
This tool is for **educational and research purposes only**. It is not a substitute for clinical diagnosis by a qualified radiologist.