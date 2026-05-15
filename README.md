# 🔬 Research Synthesis & Comparison Engine

A **RAG-powered** (Retrieval-Augmented Generation) application that enables cross-document analysis of 3–5 research papers. Upload PDFs on a specific niche and get an AI assistant that identifies conflicting or supporting evidence between them — powered by **Google Gemini**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.0-4285F4?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Paper-wise Retrieval** | Every answer cites which paper it comes from: *"Paper A claims 20%, while Paper B says 25%"* |
| 📊 **Cross-Document Summary** | Unified synthesis covering themes, methods, and findings of all papers |
| 🔬 **Methodology Synthesis** | Deep-dive technical analysis pulling details from multiple paper sections |
| ⚖️ **Findings Comparison** | Side-by-side comparison with agreement/conflict detection |
| 💬 **Conversational Q&A** | Chat with your papers using context-aware follow-up questions |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
copy .env.example .env
# Edit .env → add your GOOGLE_API_KEY from https://aistudio.google.com/apikey

# Run
streamlit run app.py
```

## 🏗️ Architecture

```
PDF Upload → Text Extraction (PyPDF2)
           → Chunking + Paper-ID Tagging (LangChain)
           → Embedding (Gemini embedding-001)
           → Vector Storage (ChromaDB)

User Query → Query Embedding → Multi-Paper Retrieval
           → Grouped Context → Gemini 2.0 Flash → Cited Response
```

## 📁 Project Structure

```
├── app.py                 # Streamlit UI & orchestration
├── pdf_processor.py       # PDF extraction & chunking
├── embeddings.py          # Gemini embedding generation
├── vector_store.py        # ChromaDB management
├── synthesis_engine.py    # LLM synthesis prompts
├── requirements.txt       # Dependencies
└── .env.example           # API key template
```

## 📋 Requirements

- Python 3.10+
- Google Gemini API key (free tier available)
- 3–5 research papers in PDF format

## 🎯 Use Cases

- **Literature Reviews**: Quickly synthesize findings across related papers
- **Research Gap Analysis**: Identify what's been studied and what's missing
- **Methodology Comparison**: Understand different technical approaches
- **Evidence Assessment**: Find agreements and conflicts in the literature

## 📜 License

MIT License
