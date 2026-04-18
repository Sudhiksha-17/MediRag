import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Patch RAGAS timeout ────────────────────────────────────────────────────
from ragas.executor import Executor
original_init = Executor.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.timeout = 300  # 5 minutes per job
Executor.__init__ = patched_init

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag.pipeline import rag_query

# ── Judge LLM + Embeddings ─────────────────────────────────────────────────
print("Setting up RAGAS judge...")
llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024,
))

embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    cache_folder="D:/hf_cache"
))

# Inject into metrics
faithfulness.llm             = llm
context_precision.llm        = llm
context_precision.embeddings = embeddings

# ── Test cases ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "What are the characteristics of a malignant mass on mammography?",
        "ground_truth": "Malignant masses on mammography typically have irregular shape, spiculated or indistinct margins, and are associated with high BI-RADS categories such as 4 or 5."
    },
    {
        "question": "What does BI-RADS 3 mean and what follow-up is recommended?",
        "ground_truth": "BI-RADS 3 means probably benign with less than 2% likelihood of malignancy. Short-interval follow-up at 6 months is recommended rather than immediate biopsy."
    },
    {
        "question": "What is the significance of spiculated margins in breast masses?",
        "ground_truth": "Spiculated margins are highly suspicious for malignancy as they indicate tumor infiltration into surrounding tissue."
    },
    {
        "question": "How are calcifications classified in mammography?",
        "ground_truth": "Calcifications are classified as typically benign, suspicious or highly suspicious based on morphology and distribution."
    },
    {
        "question": "What is the role of deep learning in mammography?",
        "ground_truth": "Deep learning improves mammography detection accuracy, reduces false positives, and assists radiologists in identifying masses and calcifications."
    },
]

# ── Run RAG on all test cases ──────────────────────────────────────────────
print(f"\nRunning RAG on {len(TEST_CASES)} test cases...")

questions     = []
answers       = []
contexts      = []
ground_truths = []

for i, test in enumerate(TEST_CASES):
    print(f"  [{i+1}/{len(TEST_CASES)}] {test['question'][:60]}...")
    try:
        result = rag_query(test["question"])
        questions.append(test["question"])
        answers.append(result["answer"])
        ground_truths.append(test["ground_truth"])
        contexts.append([chunk["text"] for chunk in result["text_sources"]])
    except Exception as e:
        print(f"  ERROR: {e}")

# ── Build RAGAS dataset ────────────────────────────────────────────────────
print("\nBuilding RAGAS dataset...")
dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ── Evaluate ───────────────────────────────────────────────────────────────
print("Running RAGAS evaluation (5-10 minutes)...")
results = evaluate(
    dataset,
    metrics=[faithfulness, context_precision],
    raise_exceptions=False,
)

# ── Print results ──────────────────────────────────────────────────────────
df = results.to_pandas()
print(f"\nAvailable columns: {list(df.columns)}")

faith_col = [c for c in df.columns if "faith" in c.lower()]
prec_col  = [c for c in df.columns if "precision" in c.lower()]

faith_score = df[faith_col[0]].mean() if faith_col else 0.0
prec_score  = df[prec_col[0]].mean()  if prec_col  else 0.0

print(f"\n{'='*60}")
print(f"📊 RAGAS EVALUATION RESULTS")
print(f"{'='*60}")
print(df[[faith_col[0], prec_col[0]]].to_string() if faith_col and prec_col else df.to_string())
print(f"\n── Aggregate Scores ──")
print(f"  Faithfulness (answer accuracy):  {faith_score:.3f}")
print(f"  Context Precision@5 (retrieval): {prec_score:.3f}")

# ── Save ───────────────────────────────────────────────────────────────────
output_path = Path("data/processed/ragas_results.json")
with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "faithfulness":      round(faith_score, 3),
            "context_precision": round(prec_score, 3),
            "n_queries":         len(questions),
            "framework":         "RAGAS v0.2.6",
            "llm_judge":         "Llama3-70b via Groq",
        },
        "per_query": df.to_dict(orient="records")
    }, f, indent=2)

print(f"\n💾 Results saved to {output_path}")