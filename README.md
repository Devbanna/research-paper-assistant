# 🤖 Research-Genius: RAG-Powered Academic Assistant
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-paper-assistant-ikhawhapomdhypk5bxnmrx.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Research-Genius is a sophisticated **Retrieval-Augmented Generation (RAG)** system designed to automate the literature review process. By combining **Vector Embeddings** with **Transformer-based LLMs**, it allows researchers to converse with dense academic papers to extract grounded, non-hallucinated insights.

---

## 💡 The Motivation
As a B.Tech specialist in AIML with an **8.02 CGPA**, I recognized the "Information Overload" bottleneck in academic research. Processing hundreds of ArXiv papers manually is inefficient and prone to human oversight. I built this assistant to demonstrate how **Semantic Search** can outperform traditional keyword matching, providing a tool that doesn't just "search" but "understands" scientific context.

---

## 🏗️ System Architecture & Pipeline

The system follows a modular pipeline designed for high precision and low latency:

1.  **Ingestion Engine:** High-fidelity PDF text extraction using `PyMuPDF`.
2.  **Strategic Chunking:** Recursive character splitting with a 10% overlap to maintain semantic continuity between chunks.
3.  **Vectorization:** Transformation of text into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
4.  **Indexing:** High-speed similarity search powered by **FAISS (Facebook AI Similarity Search)**.
5.  **Inference Layer:** * **QA:** `MiniLM-L6-v2` fine-tuned on SQuAD2.0 for context-aware extraction.
    * **Summarization:** `T5-Small` for abstractive synthesis of key findings.



---

## 🚀 How to Run

Follow these steps to set up the Research Assistant on your local machine.

### 1️ Prerequisites:
- **Python 3.10 or 3.11** (Recommended for library stability)
- **Conda** or **venv** for environment management

### 2   Install Dependencies:
    
    pip install -r requirements.txt

### 3  Clone & Setup Environment:
    
    git clone https://github.com/Devbanna/research-paper-assistant.git
    cd research-paper-assistant

### 4   Run the Dashboard:
    
    streamlit run app/app.py



## 🧠 Engineering Challenges & Human-Centric Solutions

### 1. The "Lost in the Middle" Context Problem
**The Challenge:** Standard transformer models often lose context in long research papers (e.g., 20+ pages) due to token limits.
**The Solution:** I implemented an **Overlapping Window Chunking** strategy. By creating 500-token chunks with a 50-token overlap, the system ensures that "contextual bridges" aren't broken, preserving the meaning of sentences split across chunks.

### 2. Hallucination Control & Grounding
**The Challenge:** Generative AI can sometimes "make up" research findings if the answer is not present in the source material.
**The Solution:** I engineered a **Grounded Response** logic. The system is strictly constrained to only answer using the retrieved context from the FAISS index. If the answer isn't in the uploaded paper, the AI explicitly states its limitations, adhering to the high-stakes requirements of academic integrity.

### 3. Hardware Optimization (Apple Silicon)
**The Challenge:** Efficiently running heavy Transformers locally on Mac hardware without high latency.
**The Solution:** I optimized the pipeline to utilize **MPS (Metal Performance Shaders)** where available, ensuring stable, real-time performance on local machines while maintaining a low memory footprint.

---

## ✨ Key Features
* **Multi-Paper Support:** Upload and compare findings across multiple PDFs simultaneously.
* **Semantic Insight Extraction:** Automatically identifies core contributions, methodologies, and results.
* **Metrics Dashboard:** Real-time display of retrieval confidence scores.
* **Transparent Context:** View the specific document snippets the AI used to generate its response for easy verification.

---

## 📂 Project Structure

```text
research-paper-assistant/
├── app/
│   └── app.py              # Main Streamlit UI & RAG Logic
├── images/                 # Architecture Diagrams & UI Screenshots
├── requirements.txt        # Dependency mapping for reproducibility
└── README.md               # Engineering Journey