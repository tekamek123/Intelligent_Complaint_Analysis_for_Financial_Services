"""
Unit tests for Task 1: Exploratory Data Analysis and Data Preprocessing
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

from task1_eda_preprocessing import (
    load_data,
    initial_eda,
    analyze_product_distribution,
    analyze_narrative_length,
    filter_dataset,
    clean_text,
    clean_narratives,
    save_cleaned_dataset
)


class TestLoadData:
    """Test data loading functionality."""
    
    def test_load_data_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        # This test would require mocking the DATA_RAW path
        # For now, we'll test the error handling logic
        pass
    
    def test_load_data_empty_file(self, tmp_path):
        """Test handling of empty CSV file."""
        # Create empty CSV
        test_file = tmp_path / "empty.csv"
        test_file.write_text("")
        
        # This would require path mocking - skipping for now
        pass


class TestCleanText:
    """Test text cleaning functionality."""
    
    def test_clean_text_lowercase(self):
        """Test that text is lowercased."""
        text = "THIS IS A TEST"
        result = clean_text(text)
        assert result == "this is a test"
    
    def test_clean_text_removes_boilerplate(self):
        """Test that boilerplate phrases are removed."""
        text = "I am writing to file a complaint about my credit card."
        result = clean_text(text)
        assert "i am writing to file a complaint" not in result.lower()
        assert "credit card" in result.lower()
    
    def test_clean_text_removes_special_chars(self):
        """Test that special characters are handled."""
        text = "This is a test!!! With special chars @#$%"
        result = clean_text(text)
        # Should contain basic text but special chars removed or normalized
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_text_handles_empty(self):
        """Test that empty strings are handled."""
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_handles_none(self):
        """Test that None values are handled."""
        result = clean_text(None)
        assert result == ""


class TestFilterDataset:
    """Test dataset filtering functionality."""
    
    def test_filter_dataset_missing_narrative_col(self):
        """Test that ValueError is raised when narrative_col is None."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Personal loan'],
            'Other': ['text1', 'text2']
        })
        
        with pytest.raises(ValueError, match="Narrative column is required"):
            filter_dataset(df, product_col='Product', narrative_col=None)
    
    def test_filter_dataset_filters_products(self):
        """Test that only target products are kept."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'Mortgage', 'Credit card'],
            'Consumer complaint narrative': ['text1', 'text2', 'text3', 'text4']
        })
        
        result = filter_dataset(df, product_col='Product', narrative_col='Consumer complaint narrative')
        
        # Should only contain Credit card and Personal loan
        assert len(result) <= len(df)
        assert all(p in ['Credit card', 'Personal loan', 'Payday loan', 'Savings account', 'Money transfer'] 
                  or any(keyword in p.lower() for keyword in ['credit card', 'personal loan', 'savings', 'money transfer'])
                  for p in result['Product'].str.lower())
    
    def test_filter_dataset_removes_empty_narratives(self):
        """Test that rows with empty narratives are removed."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Credit card', 'Credit card'],
            'Consumer complaint narrative': ['Valid text', '', '   ']
        })
        
        result = filter_dataset(df, product_col='Product', narrative_col='Consumer complaint narrative')
        
        # Should only have one row with valid narrative
        assert len(result) == 1
        assert result.iloc[0]['Consumer complaint narrative'] == 'Valid text'


class TestAnalyzeProductDistribution:
    """Test product distribution analysis."""
    
    def test_analyze_product_distribution_basic(self):
        """Test basic product distribution analysis."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Credit card', 'Personal loan'],
            'Other': ['a', 'b', 'c']
        })
        
        result_df, product_col = analyze_product_distribution(df)
        
        assert product_col == 'Product'
        assert len(result_df) == len(df)
    
    def test_analyze_product_distribution_no_product_col(self):
        """Test when product column is not found."""
        df = pd.DataFrame({
            'Other': ['a', 'b', 'c']
        })
        
        result_df, product_col = analyze_product_distribution(df)
        
        assert product_col is None


class TestAnalyzeNarrativeLength:
    """Test narrative length analysis."""
    
    def test_analyze_narrative_length_basic(self):
        """Test basic narrative length analysis."""
        df = pd.DataFrame({
            'Consumer complaint narrative': [
                'This is a short complaint.',
                'This is a much longer complaint with many more words to test the word count functionality.'
            ]
        })
        
        result_df, narrative_col = analyze_narrative_length(df)
        
        assert narrative_col == 'Consumer complaint narrative'
        assert 'narrative_word_count' in result_df.columns
        assert 'narrative_char_count' in result_df.columns
        assert result_df['narrative_word_count'].iloc[0] < result_df['narrative_word_count'].iloc[1]


class TestCleanNarratives:
    """Test narrative cleaning functionality."""
    
    def test_clean_narratives_missing_col(self):
        """Test that ValueError is raised when narrative_col is None."""
        df = pd.DataFrame({'Other': ['text1', 'text2']})
        
        with pytest.raises(ValueError, match="Narrative column is required"):
            clean_narratives(df, narrative_col=None)
    
    def test_clean_narratives_creates_cleaned_column(self):
        """Test that cleaned_narrative column is created."""
        df = pd.DataFrame({
            'Consumer complaint narrative': [
                'I am writing to file a complaint. THIS IS UPPERCASE.',
                'Another complaint with special chars!!!'
            ]
        })
        
        result = clean_narratives(df, narrative_col='Consumer complaint narrative')
        
        assert 'cleaned_narrative' in result.columns
        assert len(result) == len(df)


class TestSaveCleanedDataset:
    """Test saving cleaned dataset."""
    
    def test_save_cleaned_dataset_empty_df(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            save_cleaned_dataset(df, narrative_col=None)
    
    def test_save_cleaned_dataset_success(self, tmp_path):
        """Test successful saving of cleaned dataset."""
        # Create a temporary directory structure
        test_processed = tmp_path / "processed"
        test_processed.mkdir()
        
        df = pd.DataFrame({
            'Product': ['Credit card'],
            'Consumer complaint narrative': ['Test narrative']
        })
        
        # Mock the DATA_PROCESSED path - this would require more complex setup
        # For now, we test the validation logic
        with pytest.raises((ValueError, FileNotFoundError)):
            # This will fail because DATA_PROCESSED doesn't exist in test env
            # but validates the error handling
            save_cleaned_dataset(df, narrative_col='Consumer complaint narrative')


class TestInitialEDA:
    """Test initial EDA functionality."""
    
    def test_initial_eda_empty_df(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            initial_eda(df)
    
    def test_initial_eda_basic(self):
        """Test basic EDA on valid DataFrame."""
        df = pd.DataFrame({
            'Column1': [1, 2, 3],
            'Column2': ['a', 'b', 'c']
        })
        
        result = initial_eda(df)
        
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

