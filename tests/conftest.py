"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_complaints_df():
    """Create a sample complaints DataFrame for testing."""
    return pd.DataFrame({
        'Date received': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Product': ['Credit card', 'Personal loan', 'Credit card'],
        'Sub-product': ['General purpose credit card', 'Personal line of credit', 'General purpose credit card'],
        'Issue': ['Billing dispute', 'Loan servicing', 'Billing dispute'],
        'Sub-issue': ['Billing statement', 'Problems when you are unable to pay', 'Billing statement'],
        'Consumer complaint narrative': [
            'I am writing to file a complaint about my credit card billing. This is a valid complaint with details.',
            'I have issues with my personal loan payment processing. The system is not working correctly.',
            'Another credit card complaint with billing issues that need to be resolved.'
        ],
        'Company': ['Bank A', 'Bank B', 'Bank A'],
        'State': ['CA', 'NY', 'TX'],
        'Complaint ID': ['123', '456', '789']
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    return {
        'base': tmp_path,
        'data': data_dir,
        'raw': raw_dir,
        'processed': processed_dir
    }


@pytest.fixture
def sample_cleaned_df():
    """Create a sample cleaned DataFrame for Task 2 testing."""
    return pd.DataFrame({
        'Product': ['Credit card', 'Personal loan', 'Savings account'] * 10,
        'Consumer complaint narrative': [
            'Valid complaint text about credit card issues. ' * 5
        ] * 30,
        'Complaint ID': [f'ID_{i}' for i in range(30)],
        'Issue': ['Billing'] * 30,
        'Company': ['Bank A'] * 30,
        'State': ['CA'] * 30
    })

