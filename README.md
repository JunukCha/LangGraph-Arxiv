# 🌐 LangGraph-Arxiv

**LangGraph-Arxiv** is a base project for experimenting with [LangGraph](https://github.com/langchain-ai/langgraph) and [Streamlit](https://streamlit.io). It provides a simple interface to summarize and evaluate Arxiv papers using a modular, graph-based approach. While it supports interactive evaluation, it is **not intended for high-accuracy or production-level summarization**.

## 🧠 Reasoning Flow with LangGraph

This project demonstrates a basic implementation of **Chain of Thought (CoT) reasoning** using the `LangGraph` framework. Each step in the reasoning pipeline is modeled as a node, and transitions are managed via **conditional edges**. These edges allow dynamic control over the flow—such as verifying output quality and deciding whether to proceed, retry, or terminate—creating a flexible reasoning structure ideal for experimentation.

## 📆 Requirements

- Python 3.8+
- LangGraph
- Streamlit

## ⚙️ Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

Run the app with:

```bash
streamlit run main.py
```

## 📌 Features

- 🧠 Graph-based reasoning with condition edges (CoT-style flow)
- 📄 Arxiv paper summarization with evaluation checkpoints
- 🔁 Dynamic routing based on output quality (pass/fail logic)
- 🎨 Simple and interactive UI with Streamlit

---
