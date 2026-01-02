# Task 1: Exploratory Data Analysis and Data Preprocessing

## Overview

This task performs comprehensive EDA on the CFPB complaint dataset and prepares it for the RAG pipeline.

## Running the Script

### Option 1: Run as Python Script

```bash
python notebooks/task1_eda_preprocessing.py
```

### Option 2: Run in Jupyter Notebook

1. Start Jupyter: `jupyter notebook`
2. Open `notebooks/task1_eda_preprocessing.ipynb` (if available)
3. Run all cells

## What the Script Does

1. **Loads the dataset** from `data/raw/complaints.csv`
2. **Performs Initial EDA**:
   - Dataset shape and structure
   - Column information
   - Missing value analysis
3. **Analyzes Product Distribution**:
   - Counts complaints by product category
   - Creates visualization
4. **Analyzes Narrative Length**:
   - Calculates word and character counts
   - Identifies very short/long narratives
   - Creates visualizations
5. **Analyzes Missing Narratives**:
   - Counts complaints with/without narratives
   - Creates visualization
6. **Filters Dataset**:
   - Includes only target products (Credit Card, Personal Loan, Savings Account, Money Transfer)
   - Removes records with empty narratives
7. **Cleans Text Narratives**:
   - Lowercases text
   - Removes boilerplate phrases
   - Removes special characters
   - Normalizes whitespace and punctuation
8. **Saves Outputs**:
   - Cleaned dataset: `data/processed/filtered_complaints.csv`
   - Summary report: `data/processed/eda_summary_report.txt`
   - Visualizations: `data/processed/*.png`

## Expected Output Files

- `data/processed/filtered_complaints.csv` - Cleaned and filtered dataset
- `data/processed/eda_summary_report.txt` - Summary of findings
- `data/processed/product_distribution.png` - Product distribution chart
- `data/processed/narrative_length_distribution.png` - Narrative length analysis
- `data/processed/missing_narratives.png` - Missing narratives pie chart

## Notes

- The script handles large datasets efficiently using pandas
- Processing may take several minutes for the full dataset
- All visualizations are saved automatically
- The summary report provides key findings and statistics
