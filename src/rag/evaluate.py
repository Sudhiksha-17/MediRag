import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.pipeline import rag_query

# ── 20 test queries with expected keywords in answers ─────────────────────
TEST_CASES = [
    {
        "query": "What are the characteristics of a malignant mass on mammography?",
        "expected_keywords": ["irregular", "spiculated", "malignant", "margins"]
    },
    {
        "query": "What does BI-RADS 3 mean?",
        "expected_keywords": ["probably benign", "short-interval", "follow-up", "2%"]
    },
    {
        "query": "What does BI-RADS 5 mean?",
        "expected_keywords": ["malignant", "biopsy", "highly suggestive"]
    },
    {
        "query": "What is the significance of spiculated margins in breast masses?",
        "expected_keywords": ["malignant", "suspicious", "infiltration", "spiculated"]
    },
    {
        "query": "How are calcifications classified in mammography?",
        "expected_keywords": ["calcification", "benign", "suspicious", "morphology"]
    },
    {
        "query": "What follow-up is recommended for BI-RADS 4 findings?",
        "expected_keywords": ["biopsy", "tissue", "sampling", "suspicious"]
    },
    {
        "query": "What is architectural distortion on mammography?",
        "expected_keywords": ["distortion", "architectural", "malignancy", "spiculation"]
    },
    {
        "query": "What are typically benign calcifications?",
        "expected_keywords": ["benign", "calcification", "vascular", "round"]
    },
    {
        "query": "What is the difference between a mass and an asymmetry on mammography?",
        "expected_keywords": ["mass", "asymmetry", "density", "border"]
    },
    {
        "query": "What is breast density and why does it matter?",
        "expected_keywords": ["density", "cancer", "risk", "sensitivity"]
    },
    {
        "query": "What are the signs of early breast cancer on mammography?",
        "expected_keywords": ["calcification", "mass", "early", "detection"]
    },
    {
        "query": "What is a circumscribed margin in mammography?",
        "expected_keywords": ["circumscribed", "benign", "margin", "well-defined"]
    },
    {
        "query": "What is the role of deep learning in mammography?",
        "expected_keywords": ["deep learning", "detection", "neural", "accuracy"]
    },
    {
        "query": "What is tomosynthesis and how does it differ from standard mammography?",
        "expected_keywords": ["tomosynthesis", "3D", "recall", "detection"]
    },
    {
        "query": "What are pleomorphic calcifications?",
        "expected_keywords": ["pleomorphic", "calcification", "suspicious", "malignant"]
    },
    {
        "query": "What is the positive predictive value of BI-RADS categories?",
        "expected_keywords": ["positive predictive", "biopsy", "malignancy", "category"]
    },
    {
        "query": "What are the mammography screening guidelines?",
        "expected_keywords": ["screening", "annual", "age", "guidelines"]
    },
    {
        "query": "What is a developing asymmetry on mammography?",
        "expected_keywords": ["asymmetry", "developing", "new", "suspicious"]
    },
    {
        "query": "How is mammography used to detect microcalcifications?",
        "expected_keywords": ["microcalcification", "detection", "cluster", "malignant"]
    },
    {
        "query": "What is the sensitivity of mammography for breast cancer detection?",
        "expected_keywords": ["sensitivity", "specificity", "detection", "cancer"]
    },
]

# ── Run evaluation ─────────────────────────────────────────────────────────
def evaluate():
    results = []
    passed  = 0
    total   = len(TEST_CASES)

    print(f"Running evaluation on {total} queries...\n")

    for i, test in enumerate(TEST_CASES):
        query    = test["query"]
        keywords = test["expected_keywords"]

        print(f"[{i+1}/{total}] {query[:60]}...")

        try:
            result       = rag_query(query)
            answer       = result["answer"].lower()
            n_sources    = len(result["text_sources"])
            n_images     = len(result["image_cases"])

            # Check keyword coverage
            keywords_found   = [kw for kw in keywords if kw.lower() in answer]
            keywords_missing = [kw for kw in keywords if kw.lower() not in answer]
            keyword_score    = len(keywords_found) / len(keywords)

            # Check answer is not a refusal
            refusal_phrases = ["i don't have", "i cannot", "not enough information", "no information"]
            is_refusal      = any(p in answer for p in refusal_phrases)

            # Check sources were retrieved
            has_sources = n_sources > 0

            # Pass criteria: keyword score > 0.5 and not a refusal and has sources
            passed_test = keyword_score >= 0.5 and not is_refusal and has_sources

            if passed_test:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            results.append({
                "query":            query,
                "status":           status,
                "keyword_score":    round(keyword_score, 2),
                "keywords_found":   keywords_found,
                "keywords_missing": keywords_missing,
                "is_refusal":       is_refusal,
                "n_sources":        n_sources,
                "n_images":         n_images,
                "answer_preview":   result["answer"][:200],
            })

            print(f"   {status} | Keywords: {len(keywords_found)}/{len(keywords)} | Sources: {n_sources}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                "query":  query,
                "status": "❌ ERROR",
                "error":  str(e)
            })

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ PASSED: {passed}/{total} ({round(passed/total*100)}%)")
    print(f"❌ FAILED: {total-passed}/{total}")

    # Save results
    output_path = Path("data/processed/eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total":       total,
                "passed":      passed,
                "failed":      total - passed,
                "pass_rate":   round(passed/total*100, 1),
            },
            "results": results
        }, f, indent=2)

    print(f"💾 Results saved to {output_path}")

    # ── Print failures for debugging ───────────────────────────────────────
    failures = [r for r in results if "PASS" not in r["status"]]
    if failures:
        print(f"\n── Failed Queries ──")
        for f in failures:
            print(f"\n❌ {f['query']}")
            if "error" in f:
                print(f"   Error: {f['error']}")
            else:
                print(f"   Missing keywords: {f.get('keywords_missing', [])}")
                print(f"   Answer preview: {f.get('answer_preview', '')[:150]}")

if __name__ == "__main__":
    evaluate()