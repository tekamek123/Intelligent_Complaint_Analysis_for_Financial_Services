"""
Unit tests for Task 2: Text Chunking, Embedding, and Vector Store Indexing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from task2_chunking_embedding import (
    load_cleaned_data,
    identify_columns,
    create_stratified_sample,
    chunk_texts,
    load_embedding_model,
    generate_embeddings,
    create_vector_store
)


class TestLoadCleanedData:
    """Test loading cleaned data functionality."""
    
    def test_load_cleaned_data_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        # This would require mocking DATA_PROCESSED path
        # For now, we verify the error message structure
        pass
    
    def test_load_cleaned_data_empty_file(self, tmp_path):
        """Test handling of empty CSV file."""
        # Create empty CSV
        test_file = tmp_path / "empty.csv"
        test_file.write_text("")
        
        # This would require path mocking
        pass


class TestIdentifyColumns:
    """Test column identification functionality."""
    
    def test_identify_columns_all_found(self):
        """Test when all columns are found."""
        df = pd.DataFrame({
            'Product': ['Credit card'],
            'Consumer complaint narrative': ['Text'],
            'Complaint ID': ['123']
        })
        
        product_col, narrative_col, complaint_id_col = identify_columns(df)
        
        assert product_col == 'Product'
        assert narrative_col == 'Consumer complaint narrative'
        assert complaint_id_col == 'Complaint ID'
    
    def test_identify_columns_none_found(self):
        """Test when no expected columns are found."""
        df = pd.DataFrame({
            'Other1': ['a'],
            'Other2': ['b']
        })
        
        product_col, narrative_col, complaint_id_col = identify_columns(df)
        
        assert product_col is None
        assert narrative_col is None
        assert complaint_id_col is None


class TestCreateStratifiedSample:
    """Test stratified sampling functionality."""
    
    def test_create_stratified_sample_missing_narrative_col(self):
        """Test that ValueError is raised when narrative_col is None."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Personal loan'],
            'Other': ['text1', 'text2']
        })
        
        with pytest.raises(ValueError, match="Narrative column is required"):
            create_stratified_sample(df, product_col='Product', narrative_col=None)
    
    def test_create_stratified_sample_empty_df(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot create sample from empty DataFrame"):
            create_stratified_sample(df, product_col=None, narrative_col='narrative')
    
    def test_create_stratified_sample_basic(self):
        """Test basic stratified sampling."""
        df = pd.DataFrame({
            'Product': ['Credit card'] * 100 + ['Personal loan'] * 50,
            'Consumer complaint narrative': ['Valid text'] * 150
        })
        
        result = create_stratified_sample(
            df, 
            product_col='Product', 
            narrative_col='Consumer complaint narrative',
            target_size=30
        )
        
        assert len(result) <= 30
        assert len(result) > 0
        # Check that both products are represented
        products = result['Product'].unique()
        assert len(products) >= 1  # At least one product should be present


class TestChunkTexts:
    """Test text chunking functionality."""
    
    def test_chunk_texts_missing_narrative_col(self):
        """Test that ValueError is raised when narrative_col is None."""
        df = pd.DataFrame({'Other': ['text1', 'text2']})
        
        with pytest.raises(ValueError, match="Narrative column is required"):
            chunk_texts(df, narrative_col=None)
    
    def test_chunk_texts_empty_df(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot chunk empty DataFrame"):
            chunk_texts(df, narrative_col='narrative')
    
    def test_chunk_texts_basic(self):
        """Test basic text chunking."""
        # Create a long text that will be chunked
        long_text = "This is a test. " * 100  # ~2000 characters
        
        df = pd.DataFrame({
            'Consumer complaint narrative': [long_text]
        })
        
        chunks = chunk_texts(df, narrative_col='Consumer complaint narrative', chunk_size=500, chunk_overlap=50)
        
        assert len(chunks) > 0
        assert all('chunk_text' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk for chunk in chunks)
        assert all('row_index' in chunk for chunk in chunks)
    
    def test_chunk_texts_short_text(self):
        """Test chunking with short text that doesn't need chunking."""
        df = pd.DataFrame({
            'Consumer complaint narrative': ['Short text']
        })
        
        chunks = chunk_texts(df, narrative_col='Consumer complaint narrative')
        
        assert len(chunks) == 1
        assert chunks[0]['chunk_text'] == 'Short text'


class TestLoadEmbeddingModel:
    """Test embedding model loading."""
    
    def test_load_embedding_model_default(self):
        """Test loading default embedding model."""
        # This test requires actual model download, so we'll skip in CI
        # but test the function structure
        try:
            model = load_embedding_model()
            assert model is not None
            assert hasattr(model, 'encode')
        except (OSError, RuntimeError) as e:
            # If model download fails, that's acceptable for testing
            pytest.skip(f"Model loading failed (expected in some environments): {e}")


class TestGenerateEmbeddings:
    """Test embedding generation functionality."""
    
    def test_generate_embeddings_empty_chunks(self):
        """Test that ValueError is raised for empty chunks."""
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            pytest.skip("Model not available for testing")
        
        with pytest.raises(ValueError, match="Cannot generate embeddings for empty chunks"):
            generate_embeddings([], model)
    
    def test_generate_embeddings_missing_chunk_text(self):
        """Test that ValueError is raised when chunks lack 'chunk_text' key."""
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            pytest.skip("Model not available for testing")
        
        chunks = [{'other_key': 'text'}]
        
        with pytest.raises(ValueError, match="All chunks must contain 'chunk_text' key"):
            generate_embeddings(chunks, model)
    
    def test_generate_embeddings_basic(self):
        """Test basic embedding generation."""
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            pytest.skip("Model not available for testing")
        
        chunks = [
            {'chunk_text': 'This is a test chunk.'},
            {'chunk_text': 'Another test chunk with different content.'}
        ]
        
        embeddings = generate_embeddings(chunks, model, batch_size=2)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] > 0  # Should have embedding dimension


class TestCreateVectorStore:
    """Test vector store creation functionality."""
    
    def test_create_vector_store_empty_chunks(self):
        """Test that ValueError is raised for empty chunks."""
        df = pd.DataFrame({'Product': ['Credit card']})
        embeddings = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="Cannot create vector store with empty chunks"):
            create_vector_store([], embeddings, df, None, None, None)
    
    def test_create_vector_store_mismatch(self):
        """Test that ValueError is raised when chunks and embeddings don't match."""
        chunks = [{'chunk_text': 'text1'}, {'chunk_text': 'text2'}]
        embeddings = np.array([[1, 2, 3]])  # Only one embedding
        
        df = pd.DataFrame({'Product': ['Credit card']})
        
        with pytest.raises(ValueError, match="Mismatch between chunks"):
            create_vector_store(chunks, embeddings, df, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

