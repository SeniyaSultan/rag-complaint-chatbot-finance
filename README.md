# RAG Complaint Analysis Chatbot - Interim Submission

This repository contains the code for Tasks 1 and 2 of the "Intelligent Complaint Analysis for Financial Services" project.

## Project Structure

rag-complaint-chatbot/
├── data/
│ ├── raw/ # Raw CFPB complaint data
│ └── processed/ # Processed and filtered data
├── vector_store/ # Vector store files (FAISS/ChromaDB)
├── notebooks/
│ ├── 01_eda_and_cleaning.ipynb # Task 1: EDA and preprocessing
│ └── 02_chunk_embed_index.ipynb # Task 2: Chunking, embedding, indexing
├── src/
│ ├── data_preprocessing.py # Task 1 module
│ └── vector_store_builder.py # Task 2 module
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
└── README.md # This file

## Task 1: Exploratory Data Analysis and Data Preprocessing

### Objectives:

1. Understand the structure and quality of CFPB complaint data
2. Filter data for 4 product categories
3. Clean text narratives for embedding
4. Save processed data for Task 2

### Key Steps:

1. **Data Loading**: Load CFPB complaint dataset
2. **EDA Analysis**:
   - Product distribution analysis
   - Narrative length analysis
   - Missing data analysis
3. **Data Filtering**:
   - Keep only: Credit Card, Personal Loan, Savings Account, Money Transfer
   - Remove empty narratives
4. **Text Cleaning**:
   - Lowercasing
   - Remove boilerplate text
   - Remove special characters
   - Normalize whitespace
5. **Data Saving**: Save to `data/processed/filtered_complaints.csv`

### Files:

- `notebooks/01_eda_and_cleaning.ipynb`: Interactive notebook
- `src/data_preprocessing.py`: Reusable Python module

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objectives:

1. Create stratified sample of 10K-15K complaints
2. Implement text chunking strategy
3. Generate embeddings using all-MiniLM-L6-v2
4. Build vector stores (FAISS and ChromaDB)
5. Store metadata with vectors

### Key Steps:

1. **Stratified Sampling**: Create proportional sample across products
2. **Text Chunking**:
   - Chunk size: 500 characters
   - Overlap: 50 characters
   - Using LangChain's RecursiveCharacterTextSplitter
3. **Embedding Generation**:
   - Model: `all-MiniLM-L6-v2` (384 dimensions)
   - Lightweight but effective for semantic search
4. **Vector Store Creation**:
   - FAISS: Lightweight, fast similarity search
   - ChromaDB: Feature-rich, persistent store
   - Both stores include full metadata
5. **Persistence**: Save vector stores to `vector_store/`

### Files:

- `notebooks/02_chunk_embed_index.ipynb`: Interactive notebook
- `src/vector_store_builder.py`: Reusable Python module

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-complaint-chatbot
   ```
