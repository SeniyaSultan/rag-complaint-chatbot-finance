RAG Complaint Analysis Chatbot for Financial Services

An end-to-end Retrieval-Augmented Generation (RAG) system that transforms unstructured customer complaints into actionable insights for financial service teams.

This project was developed as part of the “Intelligent Complaint Analysis for Financial Services” challenge and demonstrates how vector search and large language models can be combined to support product, support, and compliance decision-making.

🧠 Project Overview

CrediTrust Financial receives thousands of customer complaints across multiple financial products. Manually analyzing these complaints is slow, error-prone, and reactive.

This system enables internal stakeholders to:

Ask plain-English questions about customer complaints

Retrieve relevant complaint evidence using semantic search

Generate concise, grounded answers using an LLM

View source complaints to build trust and transparency

✨ Key Features

Exploratory Data Analysis (EDA) on real CFPB complaint data

Text cleaning and preprocessing for NLP tasks

Stratified sampling to avoid product bias

Text chunking and embedding for semantic retrieval

Vector database indexing (FAISS / ChromaDB)

RAG pipeline with prompt-controlled LLM generation

Qualitative evaluation of system responses

Interactive chat interface built with Gradio

📁 Project Structure
rag-complaint-chatbot/
├── data/
│   ├── raw/                          # Raw CFPB complaint data
│   └── processed/                   # Cleaned & filtered data
├── vector_store/                    # Persisted FAISS / ChromaDB index
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb     # Task 1: EDA & preprocessing
│   ├── 02_chunk_embed_index.ipynb    # Task 2: Chunking & embedding
│   └── README.md
├── src/
│   ├── data_preprocessing.py         # Data cleaning utilities
│   ├── vector_store_builder.py       # Embedding & indexing logic
│   └── rag_pipeline.py               # Core RAG retrieval & generation
├── app.py                            # Gradio chat interface
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Containerized deployment
├── README.md                         # Project documentation
└── .gitignore

🧪 Task 1: Exploratory Data Analysis & Preprocessing
Objectives

Understand complaint structure and quality

Identify product imbalance and narrative variability

Prepare clean text for semantic embedding

Key Analyses

Complaint distribution across product categories

Narrative length (word count) distribution

Missing and empty narrative detection

Preprocessing Steps

Filter to core products:

Credit Cards

Personal Loans

Savings Accounts

Money Transfers

Remove empty narratives

Normalize text:

Lowercasing

Boilerplate removal

Special character removal

Whitespace normalization

Output

Cleaned dataset saved to:

data/processed/filtered_complaints.csv

🔗 Task 2: Text Chunking, Embedding & Vector Indexing
Stratified Sampling

Sample size: 10,000–15,000 complaints

Ensures proportional representation across product categories

Prevents retrieval bias during development

Chunking Strategy

Chunk size: 500 characters

Overlap: 50 characters

Rationale: balances semantic coherence and retrieval accuracy

Embedding Model

sentence-transformers/all-MiniLM-L6-v2

384-dimensional embeddings

Lightweight, fast, and effective for semantic similarity

Vector Stores

FAISS for fast similarity search

ChromaDB for persistence and metadata-rich querying

Metadata stored per chunk:

complaint ID

product category

issue / sub-issue

company

date received

chunk index

🧠 Task 3: RAG Pipeline & Evaluation
Retrieval

User question → embedded using same embedding model

Top-k (k=5) most relevant chunks retrieved from vector store

Prompt Engineering

The LLM is instructed to:

Act as a financial complaint analyst

Use only retrieved complaint context

Avoid hallucination when evidence is insufficient

Generation

Retrieved chunks + question → passed to LLM

Generates concise, evidence-backed answers

Evaluation

5–10 representative business questions tested

Results analyzed using a qualitative evaluation table:

Question

Generated Answer

Retrieved Sources

Quality Score (1–5)

Analysis

💬 Task 4: Interactive Chat Interface
UI Features

Built with Gradio

Text input for user questions

AI-generated answer display

Source complaint excerpts shown below each response

Clear/reset functionality

Goal

Enable non-technical users to explore complaint data confidently and transparently.

🚀 Setup Instructions
git clone https://github.com/SeniyaSultan/rag-complaint-chatbot-finance
cd rag-complaint-chatbot
pip install -r requirements.txt
python app.py

📌 Key Learnings

Long narratives require chunking for effective semantic retrieval

Stratified sampling prevents product-level bias

RAG significantly reduces hallucination compared to vanilla LLMs

Showing sources is critical for user trust

Lightweight models are often better for local and constrained environments

🔮 Future Improvements

Add product-level filters in the UI

Implement response streaming

Improve ranking with hybrid (BM25 + vector) search

Deploy to Hugging Face Spaces or cloud infrastructure

Add monitoring for retrieval quality over time
