import os
from groq import Groq
from src.rag.retriever import retrieve

# ── Groq client ────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

# ── Build prompt from retrieved context ────────────────────────────────────
def build_prompt(query: str, text_chunks: list, image_cases: list) -> str:
    context_parts = []

    # Text context
    if text_chunks:
        context_parts.append("=== RELEVANT MEDICAL LITERATURE ===")
        for i, chunk in enumerate(text_chunks):
            source = chunk.get("title", "Unknown source")[:80]
            context_parts.append(f"[{i+1}] {chunk['text']}\nSource: {source}")

    # Image context
    if image_cases:
        context_parts.append("\n=== SIMILAR MAMMOGRAM CASES ===")
        for i, case in enumerate(image_cases):
            context_parts.append(
                f"[Case {i+1}] {case['description']}\n"
                f"Pathology: {case['pathology']} | BI-RADS: {case['assessment']}"
            )

    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert radiologist assistant specializing in mammography and breast imaging.
Answer the question using ONLY the provided context. If the context doesn't contain enough information, say so clearly.
Always cite which sources or cases informed your answer.
Never make up clinical findings or diagnoses not supported by the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

# ── Main RAG function ──────────────────────────────────────────────────────
def rag_query(query: str, image_path: str = None) -> dict:
    # Step 1: Retrieve
    print(f"Retrieving context for: '{query}'")
    results = retrieve(query, image_path=image_path)
    text_chunks  = results["text_chunks"]
    image_cases  = results["image_cases"]

    # Step 2: Build prompt
    prompt = build_prompt(query, text_chunks, image_cases)

    # Step 3: Generate
    print("Generating answer with Llama3...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    # Step 4: Return structured result
    return {
        "query":        query,
        "answer":       answer,
        "text_sources": text_chunks,
        "image_cases":  image_cases,
    }

# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the characteristics of a malignant mass on mammography?",
        "What does BI-RADS 4 mean and what follow up is recommended?",
        "What is the significance of spiculated margins in breast masses?",
    ]

    for query in test_queries:
        print("\n" + "="*60)
        result = rag_query(query)
        print(f"Q: {result['query']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources used: {len(result['text_sources'])} text chunks, {len(result['image_cases'])} image cases")