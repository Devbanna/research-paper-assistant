
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
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Dark Theme Consistency
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
# 🤖 AI Research Paper Assistant
### Understand papers instantly • Ask questions • Extract insights
""")

# ===============================
# LOAD MODELS (CACHE)
# ===============================
@st.cache_resource
def load_models():
    # Embedding model for Vector Search
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # QA Pipeline for Chat
    qa = pipeline(
    task="question-answering",
    model="deepset/minilm-uncased-squad2",
    tokenizer="deepset/minilm-uncased-squad2"
)

    # DIRECT LOADING of T5 to bypass Python 3.13 Task Registry errors
    model_name = "t5-small"
    summ_tokenizer = AutoTokenizer.from_pretrained(model_name)
    summ_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embed_model, qa, summ_model, summ_tokenizer


model, qa_pipeline, summ_model, summ_tokenizer = load_models()

# ===============================
# SIDEBAR CONTROLS
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")
    k_value = st.slider("Context Depth", 1, 10, 7)
    st.divider()
    st.markdown("### ℹ️ About")
    st.write("AI assistant for research papers using Retrieval-Augmented Generation (RAG).")

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

def generate_t5_text(prompt, prefix="summarize: ", max_tokens=200):
    # Helper to handle T5 generation without the pipeline shortcut
    inputs = summ_tokenizer.encode(prefix + prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = summ_model.generate(inputs, max_length=max_tokens, min_length=50, length_penalty=2.0, num_beams=4)
    return summ_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# FILE UPLOAD (Multi-File Support)
# ===============================
uploaded_files = st.file_uploader(
    "📂 Upload Research Papers",
    type="pdf",
    accept_multiple_files=True
)

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

# Show preview as requested
st.subheader("Extracted Text Preview")
st.text_area("Preview content", value=all_text[:2000], height=200)

chunks = chunk_text(all_text)

# ===============================
# BUILD VECTOR INDEX
# ===============================
with st.spinner("Building knowledge base..."):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

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
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summary", "📌 Insights"])

# TAB 1 — CHAT
with tab1:
    st.subheader("Chat with Your Papers")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask something about the papers")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Retrieval Logic
        q_emb = model.encode([user_input])
        _, indices = index.search(np.array(q_emb).astype('float32'), k_value)
        context = " ".join([chunks[i] for i in indices[0]])

        with st.spinner("AI is thinking..."):
            result = qa_pipeline(question=user_input, context=context)
            answer = result["answer"]

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("🔍 View Retrieved Context"):
            st.write(context)

# TAB 2 — SUMMARY
with tab2:
    st.subheader("Paper Summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            # Using first 5 chunks for summary synthesis
            summary_text = generate_t5_text(" ".join(chunks[:5]))
            st.success(summary_text)
            st.download_button("📥 Download Summary", summary_text, "paper_summary.txt")

# TAB 3 — KEY INSIGHTS
with tab3:
    st.subheader("Key Contributions")
    if st.button("Extract Key Contributions"):
        with st.spinner("Analyzing paper..."):
            insights = generate_t5_text(" ".join(chunks[:6]), prefix="extract key insights: ")
            st.info(insights)