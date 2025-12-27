# FinSight RAG

**FinSight RAG** is a Retrieval-Augmented Generation (RAG) application combined with **financial sentiment analysis**, designed to help users analyze and query **financial reports of companies or entities**.
The system leverages **Large Language Models (LLMs)**, **AI agents**, and **domain-specific NLP models** to deliver context-aware, sentiment-driven insights grounded directly in source documents.

---

## 🚀 Project Overview

FinSight RAG enables users to:

* Upload and store financial reports (e.g., annual reports, earnings statements)
* Ask natural language questions about the content
* Retrieve relevant document chunks using vector search
* Generate grounded answers using LLMs
* Perform **financial sentiment analysis** on extracted statements
* Interact with the system via a **Gradio-based UI**

The project combines **RAG pipelines**, **agent-based reasoning**, and **fine-tuned financial NLP models** for robust financial analysis.

---

## 🔨 Main Tools

* LangGraph – Agent orchestration  
* LangChain – Retrieval-Augmented Generation (RAG)  
* Hugging Face Transformers – Financial sentiment classification  
* Gradio – Web-based user interface

## 🧠 Core Features

* **Retrieval-Augmented Generation (RAG)**

  * Semantic document retrieval using embeddings
  * Context-aware answer generation
* **Financial Sentiment Analysis**

  * Fine-tuned DistilBERT model on financial text
* **Agent-Based Reasoning**

  * Multi-hop RAG using LangGraph agents

* **Interactive UI**

  * Gradio-powered web interface

---

## 🏗️ Project Structure

```text
root/
├── .venv/                # Virtual environment
├── .vscode/              # VS Code configuration
├── data/                 # Documents and datasets
├── models/               # Saved and fine-tuned models
├── finsight_rag/         # Core application package
│   ├── agent/            # LangGraph agent logic
│   ├── config/           # Configuration files
│   ├── datasets/         # Dataset download scripts
│   ├── ingest/           # Document ingestion
│   ├── llms/             # Functions returning LLM instances
│   ├── rag/              # RAG pipelines and retrievers
│   ├── sentiment/        # Sentiment analysis module
│   ├── train/            # Model training scripts
│   ├── app.py            # Application entry point
│   ├── utils.py          # Shared utilities
│   └── tests/            # Unit and integration tests
├── pyproject.toml        # Project dependencies
├── poetry.lock           # Locked dependency versions
└── .gitignore
```

---

## 🧪 Models & Technologies

### 🔍 Embeddings

* **sentence-transformers/all-MiniLM-L6-v2**

Used for:

* Semantic document chunking
* Vector-based retrieval

---

### 📈 Sentiment Analysis Model

* **distilbert-base-uncased**
* Fine-tuned on **Financial PhraseBank**

Purpose:

* Classify sentiment in financial statements
* Capture domain-specific financial tone (positive, neutral, negative)

---

### 💬 Language Models (Chat Models)

* **meta-llama/Llama-3.1-8B-Instruct**
* **gemini-2.5-flash**

Used for:

* Answer generation
* Multi-step reasoning
* Agent decision-making

---

### 🧠 AI Agents

* **LangGraph**

  * Orchestrates reasoning steps
  * Coordinates RAG retrieval + sentiment analysis
  * Enables structured, explainable workflows

---

### 🔗 RAG Framework

* **LangChain**

  * Document loaders
  * Chunking & indexing
  * Retrieval pipelines

---

### 🖥️ User Interface

* **Gradio**

  * Simple and interactive UI
  * Chat-style financial analysis interface

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/finsight-rag.git
cd finsight-rag
poetry config virtualenvs.in-project true
poetry install
poetry env activate
```

---

## ▶️ Running the Application

```bash
python finsight_rag/app.py
```

Access the UI at `http://localhost:7860`.

---

## 🧪 Training the Sentiment Model

```bash
python finsight_rag/train/train_sentiment.py
```

---

## 📚 References

* Financial PhraseBank Dataset
* Hugging Face Transformers
* LangChain
* LangGraph
* Sentence Transformers

---

## 📄 License

This project is licensed under the **MIT License**.

---
