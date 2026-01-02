# Intelligent Complaint Analysis for Financial Services

## RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

### Project Overview

This project develops an intelligent AI tool for **CrediTrust Financial**, a fast-growing digital finance company serving East African markets. The tool transforms unstructured customer complaint data into strategic, actionable insights using Retrieval-Augmented Generation (RAG) technology.

### Business Objective

CrediTrust Financial serves over 500,000 users across three countries through a mobile-first platform offering:

- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

With thousands of customer complaints received monthly through in-app channels, email, and regulatory reporting portals, internal teams need a tool that can quickly analyze and synthesize complaint data to identify trends and issues.

### Key Performance Indicators (KPIs)

1. **Speed**: Decrease the time for Product Managers to identify major complaint trends from **days to minutes**
2. **Accessibility**: Empower non-technical teams (Support and Compliance) to get answers without needing a data analyst
3. **Proactivity**: Shift from reactive problem-solving to proactive identification and resolution based on real-time customer feedback

### Project Goals

Build a RAG-powered chatbot that:

- Allows internal users to ask plain-English questions about customer complaints
- Uses semantic search (via vector databases like FAISS or ChromaDB) to retrieve relevant complaint narratives
- Generates concise, insightful answers using language models (LLMs)
- Supports multi-product querying across financial services categories

### Data Sources

The project uses complaint data from the **Consumer Financial Protection Bureau (CFPB)**:

1. **Full CFPB Dataset** (`data/raw/complaints.csv`)

   - Complete complaint dataset
   - Used for: Task 1 (EDA and preprocessing)

2. **Pre-built Embeddings** (`data/raw/complaint_embeddings.parquet`)
   - Pre-built embeddings, text chunks, and metadata for all complaints
   - Used for: Tasks 3-4 (RAG pipeline and UI)
   - Specifications:
     - Total chunks: ~1.37 million (from 464K complaints)
     - Embedding model: all-MiniLM-L6-v2 (384 dimensions, ~80MB)
     - Vector database: ChromaDB (raw embeddings also provided for FAISS users)
     - Chunk size: 500 characters
     - Chunk overlap: 50 characters

### Data Schema

Each complaint record includes:

- `complaint_id`: Original complaint ID
- `product_category`: Product category (Credit Card, Personal Loan, etc.)
- `product`: Specific product name
- `issue`: Main issue category
- `sub_issue`: Detailed sub-issue
- `company`: Company name
- `state`: US state code
- `date_received`: Date complaint was received
- `consumer_complaint_narrative`: Free-text narrative written by the consumer

### Project Structure

```
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                       # Original data files
│   │   ├── complaints.csv
│   │   └── complaint_embeddings.parquet
│   └── processed/                 # Processed and cleaned data
├── vector_store/                  # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── app.py                         # Gradio/Streamlit interface
├── requirements.txt
├── README.md
└── .gitignore
```

### Branch Structure

- `main`: Initial project setup and documentation
- `task-1`: Exploratory Data Analysis and preprocessing
- `task-2`: Chunking and embedding pipeline (sample data)
- `task-3`: RAG pipeline development and evaluation
- `task-4`: User interface development

### Learning Outcomes

By completing this challenge, you will:

- Learn how to combine vector similarity search with language models to answer user questions based on unstructured data
- Gain experience handling noisy, unstructured consumer complaint narratives and extracting meaningful insights
- Learn how to create and query a vector database (FAISS or ChromaDB) using embedding models for semantic search
- Develop a chatbot that uses real retrieved documents as context for generating intelligent, grounded answers using LLMs
- Create a system that can analyze and respond across multiple financial product categories
- Build and test a simple user interface that allows natural-language querying of large-scale complaint data

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd week7
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data files**
   - Ensure `data/raw/complaints.csv` exists
   - Ensure `data/raw/complaint_embeddings.parquet` exists

### Usage

This README will be updated as tasks are completed. Each task branch will contain specific implementation details and instructions.

### Technologies

- **Vector Databases**: FAISS, ChromaDB
- **Embedding Models**: all-MiniLM-L6-v2
- **Language Models**: (To be specified in task branches)
- **Python**: Core development language

### Contributors

Data & AI Engineering Team - CrediTrust Financial

---

**Note**: This project is part of the Kifiya AI Mastery Training program, Week 7.
