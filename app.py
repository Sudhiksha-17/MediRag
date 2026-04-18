import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import tempfile
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent))

from src.rag.pipeline import rag_query

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRAG — Mammography Assistant",
    page_icon="🩻",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🩻 MediRAG — Multimodal Mammography Assistant")
st.caption("Powered by BiomedBERT + BiomedCLIP + Llama3 | RAG over PubMed & CBIS-DDSM")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📖 Example Queries")
    examples = [
        "What are the characteristics of a malignant mass?",
        "What does BI-RADS 4 mean?",
        "What is the significance of spiculated margins?",
        "How are calcifications classified in mammography?",
        "What follow-up is needed for BI-RADS 3 findings?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

    st.divider()
    st.caption("Created by Sudhiksha Kandavel Rajan. ⚠️ For educational purposes only. Not a substitute for clinical diagnosis.")

# ── Main layout ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Your Query")
    query = st.text_area(
        "Ask a mammography question",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. What are signs of malignancy on mammography?"
    )

    st.subheader("🖼️ Upload Mammogram (optional)")
    uploaded_file = st.file_uploader(
        "Upload a mammogram image for visual similarity search",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded mammogram", width=400)

    run = st.button("🔍 Search & Analyze", type="primary", use_container_width=True)

# ── Run RAG ────────────────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Please enter a question.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Make sure your .env file is set up correctly.")
    else:
        with col2:
            with st.spinner("Retrieving context and generating answer..."):
                image_path = None
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.read())
                        image_path = tmp.name

                result = rag_query(query, image_path=image_path)

            st.subheader("💡 Answer")
            st.markdown(result["answer"])
            st.divider()

            # ── Text sources ───────────────────────────────────────────
            with st.expander(f"📄 Text Sources ({len(result['text_sources'])} chunks)", expanded=False):
                for i, chunk in enumerate(result["text_sources"]):
                    score_color = "🟢" if chunk["score"] > 0.95 else "🟡"
                    st.markdown(f"**{score_color} [{i+1}] Score: {chunk['score']}**")
                    st.markdown(f"*{chunk['title'][:100]}*")
                    st.markdown(chunk["text"])
                    if chunk.get("pmid"):
                        st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{chunk['pmid']}/)")
                    st.divider()

            # ── Image cases ────────────────────────────────────────────
            with st.expander(f"🖼️ Similar Cases ({len(result['image_cases'])} cases)", expanded=False):
                for i, case in enumerate(result["image_cases"]):
                    path = Path(case["image_path"])
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        if path.exists():
                            st.image(str(path), width=200)
                        else:
                            st.caption("Image not available")
                    with col_info:
                        pathology_color = "🔴" if case["pathology"] == "MALIGNANT" else "🟢"
                        st.markdown(f"**Case {i+1}** | Score: {case['score']}")
                        st.markdown(f"{pathology_color} **{case['pathology']}** | BI-RADS: {case['assessment']}")
                        st.markdown(case["description"])
                    st.divider()