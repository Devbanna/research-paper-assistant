# ===============================
# SAFE SETTINGS (M1 Stability)
# ===============================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ===============================
# IMPORTS
# ===============================

import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
# 🤖 AI Research Paper Assistant
### Understand papers instantly • Ask questions • Extract insights
""")

# ===============================
# LOAD MODELS (CACHE)
# ===============================

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    qa = pipeline(
        "question-answering",
        model="deepset/minilm-uncased-squad2"
    )

    summarizer = pipeline(
        "text2text-generation",
        model="t5-small"
    )

    return embed_model, qa, summarizer


model, qa_pipeline, summarizer = load_models()

# ===============================
# SIDEBAR CONTROLS
# ===============================

with st.sidebar:
    st.header("⚙️ Controls")

    k_value = st.slider(
        "Context Depth",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.markdown("### ℹ️ About")
    st.write(
        "AI assistant for research papers using "
        "Retrieval-Augmented Generation (RAG)."
    )

# ===============================
# FUNCTIONS
# ===============================

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ===============================
# FILE UPLOAD
# ===============================

uploaded_files = st.file_uploader(
    "📂 Upload Research Papers",
    type="pdf",
    accept_multiple_files=True
)

# ===============================
# EMPTY STATE
# ===============================

if not uploaded_files:
    st.info("👆 Upload one or more research papers to begin")
    st.stop()

# ===============================
# PROCESS DOCUMENTS
# ===============================

st.success(f"{len(uploaded_files)} paper(s) uploaded successfully!")

all_text = ""
for file in uploaded_files:
    all_text += extract_text_from_pdf(file)

text = all_text

st.subheader("Extracted Text Preview")
st.write(text[:2000])

chunks = chunk_text(text)

# ===============================
# BUILD VECTOR INDEX
# ===============================

with st.spinner("Building knowledge base..."):
    embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

st.success("Knowledge base ready ✅")

# ===============================
# METRICS CARDS
# ===============================

col1, col2, col3 = st.columns(3)

col1.metric("📄 Papers", len(uploaded_files))
col2.metric("🧩 Chunks", len(chunks))
col3.metric("🧠 Vectors", index.ntotal)

# ===============================
# TABS
# ===============================

tab1, tab2, tab3 = st.tabs(
    ["💬 Chat", "📝 Summary", "📌 Insights"]
)

# ===============================
# TAB 1 — CHAT
# ===============================

with tab1:

    st.subheader("Chat with Your Papers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask something about the papers")

    if user_input:

        st.chat_message("user").write(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        query_embedding = model.encode([user_input])
        distances, indices = index.search(query_embedding, k_value)

        context = " ".join([chunks[i] for i in indices[0]])

        with st.spinner("AI is thinking..."):
            result = qa_pipeline(
                question=user_input,
                context=context
            )

        answer = result["answer"]

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.expander("🔍 View Retrieved Context"):
            st.write(context)

# ===============================
# TAB 2 — SUMMARY
# ===============================

with tab2:

    st.subheader("Paper Summary")

    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            prompt = "summarize: " + " ".join(chunks[:5])

            summary = summarizer(
                prompt,
                max_length=200,
                min_length=80,
                do_sample=False
            )

        summary_text = summary[0]["generated_text"]

        st.success(summary_text)

        st.download_button(
            label="📥 Download Summary",
            data=summary_text,
            file_name="paper_summary.txt",
            mime="text/plain"
        )

# ===============================
# TAB 3 — KEY INSIGHTS
# ===============================

with tab3:

    st.subheader("Key Contributions")

    if st.button("Extract Key Contributions"):

        with st.spinner("Analyzing paper..."):

            prompt = "extract key contributions: " + " ".join(chunks[:6])

            result = summarizer(
                prompt,
                max_length=150,
                min_length=50,
                do_sample=False
            )

        st.info(result[0]["generated_text"])