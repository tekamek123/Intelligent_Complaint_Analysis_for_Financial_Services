# Task 2: Text Chunking, Embedding, and Vector Store Indexing

## Overview

This task converts cleaned text narratives into a format suitable for efficient semantic search by:

1. Creating a stratified sample of complaints
2. Chunking the narratives
3. Generating embeddings
4. Creating a vector store with metadata

## Prerequisites

- Task 1 must be completed first (cleaned dataset must exist)
- Required packages: pandas, numpy, langchain, sentence-transformers, chromadb

## Running the Script

### Option 1: Run as Python Script

```bash
python notebooks/task2_chunking_embedding.py
```

### Option 2: Run in Jupyter Notebook

1. Start Jupyter: `jupyter notebook`
2. Create a new notebook or open existing one
3. Import and run the functions

## What the Script Does

1. **Loads Cleaned Data**

   - Reads `data/processed/filtered_complaints.csv` from Task 1
   - Validates that the file exists

2. **Creates Stratified Sample**

   - Samples 10,000-15,000 complaints (default: 12,000)
   - Ensures proportional representation across all product categories
   - Uses random seed (42) for reproducibility

3. **Chunks Text Narratives**

   - Uses LangChain's `RecursiveCharacterTextSplitter`
   - Chunk size: 500 characters
   - Chunk overlap: 50 characters
   - Preserves semantic structure (paragraphs, sentences, words)

4. **Generates Embeddings**

   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - 384-dimensional embeddings
   - Batch processing for efficiency

5. **Creates Vector Store**

   - Uses ChromaDB for persistent storage
   - Stores embeddings with rich metadata:
     - Complaint ID
     - Product category
     - Chunk index and total chunks
     - Issue, Sub-issue, Company, State, Date received
   - Saves to `vector_store/chroma_db/`

6. **Generates Reports**
   - Sampling strategy report
   - Chunking approach report
   - Embedding model choice report

## Expected Output Files

- `vector_store/chroma_db/` - ChromaDB vector store directory
- `data/processed/task2_sampling_report.txt` - Sampling strategy documentation
- `data/processed/task2_chunking_report.txt` - Chunking approach documentation
- `data/processed/task2_embedding_report.txt` - Embedding model documentation

## Configuration

You can modify these parameters in the script:

- **Sample size**: Change `target_sample_size` in `main()` (default: 12000)
- **Chunk size**: Change `chunk_size` in `chunk_texts()` (default: 500)
- **Chunk overlap**: Change `chunk_overlap` in `chunk_texts()` (default: 50)
- **Embedding model**: Change `model_name` in `load_embedding_model()` (default: all-MiniLM-L6-v2)
- **Batch size**: Change `batch_size` in `generate_embeddings()` (default: 32)

## Notes

- **First run**: The embedding model will be downloaded automatically (may take a few minutes)
- **Processing time**: Expect 10-30 minutes depending on sample size and hardware
- **Memory usage**: The script processes data in batches to manage memory efficiently
- **Vector store**: ChromaDB creates a persistent database that can be reused in Task 3

## Troubleshooting

### Error: "Cleaned dataset not found"

- Solution: Run Task 1 first to generate `data/processed/filtered_complaints.csv`

### Out of Memory Error

- Solution: Reduce `target_sample_size` or `batch_size` in the script

### ChromaDB Permission Error

- Solution: Ensure write permissions for `vector_store/` directory

### Model Download Issues

- Solution: Check internet connection. The model downloads automatically on first use.

## Next Steps

After completing Task 2, the vector store is ready for use in:

- **Task 3**: RAG pipeline development
- **Task 4**: User interface development

The vector store can be loaded in Task 3 using:

```python
import chromadb
client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = client.get_collection("complaint_chunks")
```
