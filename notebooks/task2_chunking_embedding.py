"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
Objective: Convert the cleaned text narratives into a format suitable for efficient semantic search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime

# LangChain imports - handle different versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "Could not import RecursiveCharacterTextSplitter. "
            "Please install langchain-text-splitters: pip install langchain-text-splitters"
        )

# Embedding model
from sentence_transformers import SentenceTransformer

# Vector store - using ChromaDB for better metadata support
import chromadb
from chromadb.config import Settings

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Create directories if they don't exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned and filtered dataset from Task 1.
    
    Returns:
        pd.DataFrame: Loaded cleaned dataset
        
    Raises:
        FileNotFoundError: If the cleaned dataset file is not found
        ValueError: If the file cannot be read or is empty
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    print("="*80)
    print("LOADING CLEANED DATASET")
    print("="*80)
    
    data_path = DATA_PROCESSED / "filtered_complaints.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. "
            "Please run Task 1 first to generate the cleaned dataset."
        )
    
    print(f"\nLoading cleaned dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Cleaned dataset file is empty: {e}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse cleaned dataset CSV: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading cleaned dataset: {e}") from e
    
    if df.empty:
        raise ValueError("Loaded dataset is empty. Please check the data file.")
    
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Identify key columns in the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (product_col, narrative_col, complaint_id_col), each may be None if not found
    """
    # Find product column
    product_col = None
    for col in ['Product', 'product', 'product_category', 'Product category']:
        if col in df.columns:
            product_col = col
            break
    
    # Find narrative column
    narrative_col = None
    for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative']:
        if col in df.columns:
            narrative_col = col
            break
    
    # Find complaint ID column
    complaint_id_col = None
    for col in ['Complaint ID', 'complaint_id', 'Complaint ID', 'id']:
        if col in df.columns:
            complaint_id_col = col
            break
    
    print(f"\nIdentified columns:")
    print(f"  Product column: {product_col}")
    print(f"  Narrative column: {narrative_col}")
    print(f"  Complaint ID column: {complaint_id_col}")
    
    return product_col, narrative_col, complaint_id_col


def create_stratified_sample(
    df: pd.DataFrame, 
    product_col: Optional[str], 
    narrative_col: Optional[str], 
    target_size: int = 12000
) -> pd.DataFrame:
    """
    Create a stratified sample of complaints ensuring proportional representation
    across all product categories.
    
    Args:
        df: DataFrame with cleaned complaints
        product_col: Name of the product column, or None if not found
        narrative_col: Name of the narrative column, or None if not found
        target_size: Target sample size (default 12000, between 10K-15K)
    
    Returns:
        pd.DataFrame: Stratified sample DataFrame
        
    Raises:
        ValueError: If narrative_col is None or DataFrame is empty
    """
    if narrative_col is None:
        raise ValueError("Narrative column is required for sampling but was not found.")
    
    if df.empty:
        raise ValueError("Cannot create sample from empty DataFrame.")
    print("\n" + "="*80)
    print("CREATING STRATIFIED SAMPLE")
    print("="*80)
    
    # Filter out rows with missing narratives
    df_valid = df[df[narrative_col].notna() & (df[narrative_col].astype(str).str.strip() != '')].copy()
    print(f"\nValid complaints with narratives: {len(df_valid):,}")
    
    if product_col is None:
        print("Warning: Product column not found. Creating simple random sample.")
        sample_size = min(target_size, len(df_valid))
        df_sample = df_valid.sample(n=sample_size, random_state=42)
        print(f"Random sample created: {len(df_sample):,} complaints")
        return df_sample
    
    # Get product distribution
    product_counts = df_valid[product_col].value_counts()
    print(f"\nProduct distribution in full dataset:")
    for product, count in product_counts.items():
        pct = (count / len(df_valid)) * 100
        print(f"  {product}: {count:,} ({pct:.2f}%)")
    
    # Calculate proportional sample sizes
    total_complaints = len(df_valid)
    sample_sizes = {}
    
    for product in product_counts.index:
        proportion = product_counts[product] / total_complaints
        sample_size = max(1, int(target_size * proportion))  # At least 1 per category
        sample_sizes[product] = sample_size
    
    # Adjust if total exceeds target (rounding may cause this)
    total_sample_size = sum(sample_sizes.values())
    if total_sample_size > target_size:
        # Proportionally reduce
        scale_factor = target_size / total_sample_size
        sample_sizes = {k: max(1, int(v * scale_factor)) for k, v in sample_sizes.items()}
        total_sample_size = sum(sample_sizes.values())
    
    print(f"\nTarget sample size: {target_size:,}")
    print(f"Calculated sample sizes by product:")
    for product, size in sample_sizes.items():
        print(f"  {product}: {size:,}")
    print(f"Total sample size: {sum(sample_sizes.values()):,}")
    
    # Create stratified sample
    samples = []
    for product, sample_size in sample_sizes.items():
        product_df = df_valid[df_valid[product_col] == product]
        if len(product_df) >= sample_size:
            sample = product_df.sample(n=sample_size, random_state=42)
        else:
            # If not enough samples, take all available
            sample = product_df
            print(f"  Warning: Only {len(product_df):,} available for {product}, taking all")
        samples.append(sample)
    
    df_sample = pd.concat(samples, ignore_index=True)
    
    # Shuffle the final sample
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal stratified sample: {len(df_sample):,} complaints")
    print("\nSample distribution:")
    sample_dist = df_sample[product_col].value_counts()
    for product, count in sample_dist.items():
        pct = (count / len(df_sample)) * 100
        print(f"  {product}: {count:,} ({pct:.2f}%)")
    
    return df_sample


def chunk_texts(
    df: pd.DataFrame, 
    narrative_col: Optional[str], 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk the complaint narratives using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        df: DataFrame with complaint narratives
        narrative_col: Name of the narrative column, or None if not found
        chunk_size: Target size of each chunk in characters (default: 500)
        chunk_overlap: Number of characters to overlap between chunks (default: 50)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing chunk text and metadata
        
    Raises:
        ValueError: If narrative_col is None or DataFrame is empty
        RuntimeError: If text splitting fails
    """
    if narrative_col is None:
        raise ValueError("Narrative column is required for chunking but was not found.")
    
    if df.empty:
        raise ValueError("Cannot chunk empty DataFrame.")
    print("\n" + "="*80)
    print("CHUNKING TEXT NARRATIVES")
    print("="*80)
    
    print(f"\nChunking parameters:")
    print(f"  Chunk size: {chunk_size} characters")
    print(f"  Chunk overlap: {chunk_overlap} characters")
    
    # Initialize text splitter
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs, sentences, words
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize text splitter: {e}") from e
    
    all_chunks = []
    total_chunks = 0
    
    print(f"\nProcessing {len(df):,} complaints...")
    
    for idx, row in df.iterrows():
        narrative = str(row[narrative_col])
        
        # Split the narrative into chunks
        chunks = text_splitter.split_text(narrative)
        
        # Create metadata for each chunk
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_text': chunk_text,
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                'row_index': idx
            }
            all_chunks.append(chunk_metadata)
            total_chunks += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} complaints, created {total_chunks:,} chunks so far...")
    
    print(f"\nChunking complete!")
    print(f"  Total complaints processed: {len(df):,}")
    print(f"  Total chunks created: {total_chunks:,}")
    print(f"  Average chunks per complaint: {total_chunks / len(df):.2f}")
    
    # Calculate chunk statistics
    chunk_lengths = [len(chunk['chunk_text']) for chunk in all_chunks]
    print(f"\nChunk statistics:")
    print(f"  Average length: {np.mean(chunk_lengths):.0f} characters")
    print(f"  Median length: {np.median(chunk_lengths):.0f} characters")
    print(f"  Min length: {min(chunk_lengths)} characters")
    print(f"  Max length: {max(chunk_lengths)} characters")
    
    return all_chunks


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the embedding model.
    
    Args:
        model_name: Name of the sentence transformer model (default: all-MiniLM-L6-v2)
    
    Returns:
        SentenceTransformer: Loaded embedding model
        
    Raises:
        OSError: If model cannot be downloaded or loaded
        RuntimeError: If model initialization fails
    """
    print("\n" + "="*80)
    print("LOADING EMBEDDING MODEL")
    print("="*80)
    
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes on first run (downloading model)...")
    
    try:
        model = SentenceTransformer(model_name)
    except OSError as e:
        raise OSError(
            f"Failed to download or load model {model_name}. "
            "Please check your internet connection and try again."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading embedding model: {e}") from e
    
    print(f"Model loaded successfully!")
    print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
    
    return model


def generate_embeddings(
    chunks: List[Dict[str, Any]], 
    model: SentenceTransformer, 
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for all text chunks.
    
    Args:
        chunks: List of chunk dictionaries containing 'chunk_text' key
        model: SentenceTransformer model for generating embeddings
        batch_size: Batch size for embedding generation (default: 32)
    
    Returns:
        np.ndarray: Array of embeddings with shape (n_chunks, embedding_dim)
        
    Raises:
        ValueError: If chunks list is empty or missing required keys
        RuntimeError: If embedding generation fails
    """
    if not chunks:
        raise ValueError("Cannot generate embeddings for empty chunks list.")
    
    if not all('chunk_text' in chunk for chunk in chunks):
        raise ValueError("All chunks must contain 'chunk_text' key.")
    print("\n" + "="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)
    
    # Extract chunk texts
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    
    print(f"\nGenerating embeddings for {len(chunk_texts):,} chunks...")
    print(f"Using batch size: {batch_size}")
    
    # Generate embeddings in batches
    try:
        embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e
    
    print(f"\nEmbeddings generated successfully!")
    print(f"  Total embeddings: {len(embeddings):,}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings


def create_vector_store(
    chunks: List[Dict[str, Any]], 
    embeddings: np.ndarray, 
    df: pd.DataFrame, 
    product_col: Optional[str], 
    narrative_col: Optional[str], 
    complaint_id_col: Optional[str]
) -> Tuple[Any, Path]:
    """
    Create a ChromaDB vector store with embeddings and metadata.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Numpy array of embeddings
        df: Original DataFrame
        product_col: Name of product column, or None if not found
        narrative_col: Name of narrative column, or None if not found
        complaint_id_col: Name of complaint ID column, or None if not found
    
    Returns:
        Tuple containing:
            - ChromaDB collection object
            - Path to the vector store directory
            
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If vector store creation fails
    """
    if not chunks:
        raise ValueError("Cannot create vector store with empty chunks list.")
    
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
        )
    print("\n" + "="*80)
    print("CREATING VECTOR STORE")
    print("="*80)
    
    # Initialize ChromaDB client
    chroma_path = VECTOR_STORE_DIR / "chroma_db"
    print(f"\nInitializing ChromaDB at: {chroma_path}")
    
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB client: {e}") from e
    
    # Create or get collection
    collection_name = "complaint_chunks"
    print(f"Creating collection: {collection_name}")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print("  Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "CFPB complaint narrative chunks with embeddings"}
    )
    
    print(f"\nPreparing data for vector store...")
    
    # Prepare documents, embeddings, and metadata
    documents = []
    metadatas = []
    ids = []
    
    for idx, chunk in enumerate(chunks):
        row_idx = chunk['row_index']
        row = df.iloc[row_idx]
        
        # Create document
        documents.append(chunk['chunk_text'])
        
        # Create metadata
        metadata = {
            'chunk_index': chunk['chunk_index'],
            'total_chunks': chunk['total_chunks'],
        }
        
        # Add product information
        if product_col:
            metadata['product'] = str(row[product_col]) if pd.notna(row[product_col]) else 'Unknown'
        
        # Add complaint ID
        if complaint_id_col:
            complaint_id = str(row[complaint_id_col]) if pd.notna(row[complaint_id_col]) else f"row_{row_idx}"
            metadata['complaint_id'] = complaint_id
        else:
            metadata['complaint_id'] = f"row_{row_idx}"
        
        # Add other useful metadata
        for col in ['Issue', 'Sub-issue', 'Company', 'State', 'Date received']:
            if col in df.columns and pd.notna(row[col]):
                # Clean column name for metadata (ChromaDB has restrictions)
                clean_col = col.lower().replace(' ', '_').replace('-', '_')
                metadata[clean_col] = str(row[col])[:100]  # Limit length
        
        metadatas.append(metadata)
        ids.append(f"chunk_{idx}")
        
        if (idx + 1) % 5000 == 0:
            print(f"  Prepared {idx + 1:,} chunks...")
    
    print(f"\nAdding {len(documents):,} chunks to vector store...")
    
    # Convert embeddings to list format for ChromaDB
    embeddings_list = embeddings.tolist()
    
    # Add to collection in batches (ChromaDB handles batching internally, but we can do it manually)
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings_list[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        if (i + batch_size) % 5000 == 0:
            print(f"  Added {batch_end:,} chunks...")
    
    print(f"\nVector store created successfully!")
    print(f"  Total chunks stored: {len(documents):,}")
    print(f"  Collection name: {collection_name}")
    print(f"  Storage path: {chroma_path}")
    
    # Test the vector store
    print(f"\nTesting vector store...")
    test_result = collection.get(ids=[ids[0]])
    print(f"  Successfully retrieved test chunk")
    print(f"  Test chunk metadata keys: {list(test_result['metadatas'][0].keys())}")
    
    return collection, chroma_path


def save_sampling_report(df_sample, product_col, target_size):
    """Save a report documenting the sampling strategy."""
    print("\n" + "="*80)
    print("SAVING SAMPLING REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("TASK 2: SAMPLING STRATEGY REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. SAMPLING METHODOLOGY")
    report.append("-" * 80)
    report.append("   Method: Stratified Random Sampling")
    report.append("   Target Sample Size: {:,}".format(target_size))
    report.append("   Actual Sample Size: {:,}".format(len(df_sample)))
    report.append("")
    report.append("   Rationale:")
    report.append("   - Stratified sampling ensures proportional representation across all")
    report.append("     product categories, maintaining the distribution of the full dataset")
    report.append("   - This approach prevents over-representation of dominant product categories")
    report.append("   - Each product category is sampled proportionally to its occurrence in")
    report.append("     the full dataset")
    report.append("")
    
    if product_col:
        report.append("2. SAMPLE DISTRIBUTION BY PRODUCT")
        report.append("-" * 80)
        product_dist = df_sample[product_col].value_counts()
        total = len(df_sample)
        
        for product, count in product_dist.items():
            pct = (count / total) * 100
            report.append(f"   {product}: {count:,} ({pct:.2f}%)")
        report.append("")
    
    report.append("3. SAMPLING PARAMETERS")
    report.append("-" * 80)
    report.append("   Random Seed: 42 (for reproducibility)")
    report.append("   Minimum samples per category: 1")
    report.append("   Sampling method: Random sampling within each stratum")
    report.append("")
    
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = DATA_PROCESSED / "task2_sampling_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nSampling report saved to: {report_path}")
    
    return report_text


def save_chunking_report(chunks, chunk_size, chunk_overlap):
    """Save a report documenting the chunking strategy."""
    print("\n" + "="*80)
    print("SAVING CHUNKING REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("TASK 2: CHUNKING STRATEGY REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. CHUNKING METHODOLOGY")
    report.append("-" * 80)
    report.append("   Library: LangChain RecursiveCharacterTextSplitter")
    report.append("   Chunk Size: {} characters".format(chunk_size))
    report.append("   Chunk Overlap: {} characters".format(chunk_overlap))
    report.append("")
    
    report.append("   Rationale for chunk size (500 characters):")
    report.append("   - Balances context preservation with embedding quality")
    report.append("   - Large enough to capture meaningful context and complete thoughts")
    report.append("   - Small enough to ensure embeddings capture focused semantic meaning")
    report.append("   - Aligns with the pre-built embeddings specification (500 chars)")
    report.append("")
    
    report.append("   Rationale for chunk overlap (50 characters):")
    report.append("   - Prevents loss of context at chunk boundaries")
    report.append("   - Ensures important information spanning chunk boundaries is preserved")
    report.append("   - 10% overlap provides good balance between redundancy and coverage")
    report.append("")
    
    report.append("2. CHUNKING STATISTICS")
    report.append("-" * 80)
    chunk_lengths = [len(chunk['chunk_text']) for chunk in chunks]
    report.append("   Total chunks created: {:,}".format(len(chunks)))
    report.append("   Average chunk length: {:.0f} characters".format(np.mean(chunk_lengths)))
    report.append("   Median chunk length: {:.0f} characters".format(np.median(chunk_lengths)))
    report.append("   Min chunk length: {} characters".format(min(chunk_lengths)))
    report.append("   Max chunk length: {} characters".format(max(chunk_lengths)))
    report.append("")
    
    # Calculate chunks per complaint
    chunks_per_complaint = {}
    for chunk in chunks:
        row_idx = chunk['row_index']
        if row_idx not in chunks_per_complaint:
            chunks_per_complaint[row_idx] = 0
        chunks_per_complaint[row_idx] += 1
    
    avg_chunks = np.mean(list(chunks_per_complaint.values()))
    report.append("   Average chunks per complaint: {:.2f}".format(avg_chunks))
    report.append("")
    
    report.append("3. SPLITTING STRATEGY")
    report.append("-" * 80)
    report.append("   The RecursiveCharacterTextSplitter uses the following separators in order:")
    report.append("   1. Double newlines (\\n\\n) - Split on paragraph boundaries")
    report.append("   2. Single newlines (\\n) - Split on line boundaries")
    report.append("   3. Periods followed by spaces (. ) - Split on sentence boundaries")
    report.append("   4. Spaces - Split on word boundaries")
    report.append("   5. Empty string - Split on character boundaries (last resort)")
    report.append("")
    report.append("   This hierarchical approach preserves semantic structure when possible.")
    report.append("")
    
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = DATA_PROCESSED / "task2_chunking_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nChunking report saved to: {report_path}")
    
    return report_text


def save_embedding_report(model_name, embedding_dim, num_embeddings):
    """Save a report documenting the embedding model choice."""
    print("\n" + "="*80)
    print("SAVING EMBEDDING MODEL REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("TASK 2: EMBEDDING MODEL REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. EMBEDDING MODEL")
    report.append("-" * 80)
    report.append("   Model: {}".format(model_name))
    report.append("   Dimension: {}".format(embedding_dim))
    report.append("   Total embeddings generated: {:,}".format(num_embeddings))
    report.append("")
    
    report.append("2. MODEL SELECTION RATIONALE")
    report.append("-" * 80)
    report.append("   The sentence-transformers/all-MiniLM-L6-v2 model was chosen for the following reasons:")
    report.append("")
    report.append("   a) Performance vs. Size Balance:")
    report.append("      - Provides good semantic understanding while maintaining efficiency")
    report.append("      - 384-dimensional embeddings offer sufficient expressiveness")
    report.append("      - Model size (~80MB) allows for fast inference")
    report.append("")
    report.append("   b) Compatibility:")
    report.append("      - Matches the pre-built embeddings specification for Tasks 3-4")
    report.append("      - Ensures consistency across the project pipeline")
    report.append("")
    report.append("   c) Domain Suitability:")
    report.append("      - Trained on diverse text, suitable for consumer complaint narratives")
    report.append("      - Handles informal language and domain-specific terminology well")
    report.append("")
    report.append("   d) Industry Standard:")
    report.append("      - Widely used in production RAG systems")
    report.append("      - Good balance of speed and quality for semantic search")
    report.append("")
    
    report.append("3. EMBEDDING GENERATION")
    report.append("-" * 80)
    report.append("   Batch processing: Yes (batch size: 32)")
    report.append("   Progress tracking: Enabled")
    report.append("   Format: NumPy arrays (converted to lists for ChromaDB)")
    report.append("")
    
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = DATA_PROCESSED / "task2_embedding_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nEmbedding model report saved to: {report_path}")
    
    return report_text


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 2: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING")
    print("="*80)
    
    # Step 1: Load cleaned data
    df = load_cleaned_data()
    
    # Step 2: Identify columns
    product_col, narrative_col, complaint_id_col = identify_columns(df)
    
    if narrative_col is None:
        raise ValueError("Narrative column not found. Cannot proceed with chunking.")
    
    # Step 3: Create stratified sample
    target_sample_size = 12000  # Between 10K-15K
    df_sample = create_stratified_sample(df, product_col, narrative_col, target_sample_size)
    
    # Step 4: Chunk texts
    chunk_size = 500
    chunk_overlap = 50
    chunks = chunk_texts(df_sample, narrative_col, chunk_size, chunk_overlap)
    
    # Step 5: Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = load_embedding_model(model_name)
    
    # Step 6: Generate embeddings
    embeddings = generate_embeddings(chunks, model)
    
    # Step 7: Create vector store
    collection, chroma_path = create_vector_store(
        chunks, embeddings, df_sample, product_col, narrative_col, complaint_id_col
    )
    
    # Step 8: Save reports
    save_sampling_report(df_sample, product_col, target_sample_size)
    save_chunking_report(chunks, chunk_size, chunk_overlap)
    save_embedding_report(model_name, embeddings.shape[1], len(embeddings))
    
    print("\n" + "="*80)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nDeliverables:")
    print(f"  1. Vector store: {chroma_path}")
    print(f"  2. Sampling report: {DATA_PROCESSED / 'task2_sampling_report.txt'}")
    print(f"  3. Chunking report: {DATA_PROCESSED / 'task2_chunking_report.txt'}")
    print(f"  4. Embedding report: {DATA_PROCESSED / 'task2_embedding_report.txt'}")
    print("\nVector store is ready for use in Task 3 (RAG pipeline)!")
    print("="*80)


if __name__ == "__main__":
    main()

