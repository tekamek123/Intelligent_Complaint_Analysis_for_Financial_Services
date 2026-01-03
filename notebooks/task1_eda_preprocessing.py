"""
Task 1: Exploratory Data Analysis and Data Preprocessing
Objective: Understand the structure, content, and quality of the complaint data 
and prepare it for the RAG pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings
from typing import Tuple, Optional

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Create processed directory if it doesn't exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """
    Load the CFPB complaint dataset using chunking for large files.
    
    Returns:
        pd.DataFrame: Loaded complaint dataset
        
    Raises:
        FileNotFoundError: If the complaints.csv file is not found
        ValueError: If the file cannot be read or is empty
        MemoryError: If system runs out of memory during processing
    """
    print("Loading CFPB complaint dataset...")
    print("This may take a few minutes for large datasets...")
    
    file_path = DATA_RAW / "complaints.csv"
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Complaints file not found at {file_path}. "
            "Please ensure data/raw/complaints.csv exists."
        )
    
    if not file_path.is_file():
        raise ValueError(f"Path {file_path} exists but is not a file.")
    
    # Read in chunks to handle large files efficiently
    chunk_size = 50000  # Read 50k rows at a time
    chunks = []
    
    print(f"Reading file in chunks of {chunk_size:,} rows...")
    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding='utf-8', on_bad_lines='skip')):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * chunk_size:,} rows...")
        
        # Combine all chunks
        print("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except TypeError:
        # For older pandas versions that don't have on_bad_lines parameter
        print("Using compatibility mode for older pandas version...")
        chunks = []
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * chunk_size:,} rows...")
        
        print("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative method with Python engine...")
        try:
            # Fallback: use Python engine (slower but more memory efficient)
            df = pd.read_csv(file_path, engine='python', encoding='utf-8', on_bad_lines='skip')
            print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
        except TypeError:
            # For older pandas versions
            df = pd.read_csv(file_path, engine='python', encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)
            print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e2:
            raise ValueError(f"Failed to parse CSV file: {e2}") from e2
        except MemoryError as e2:
            raise MemoryError(
                "Out of memory while loading dataset. "
                "Try reducing chunk_size or using a machine with more RAM."
            ) from e2
        except Exception as e2:
            raise RuntimeError(f"Unexpected error loading data with Python engine: {e2}") from e2


def initial_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial exploratory data analysis on the complaint dataset.
    
    Args:
        df: Input DataFrame containing complaint data
        
    Returns:
        pd.DataFrame: The same DataFrame (for method chaining)
        
    Raises:
        ValueError: If DataFrame is empty or invalid
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None. Cannot perform EDA.")
    print("\n" + "="*80)
    print("INITIAL EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Basic information
    print("\n1. Dataset Shape:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    print("\n2. Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print("\n3. Data Types:")
    print(df.dtypes)
    
    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    print(missing_df)
    
    print("\n5. First Few Rows:")
    print(df.head())
    
    return df


def analyze_product_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Analyze the distribution of complaints across different products.
    
    Args:
        df: DataFrame containing complaint data
        
    Returns:
        Tuple containing:
            - pd.DataFrame: Original DataFrame
            - Optional[str]: Name of the product column found, or None if not found
    """
    print("\n" + "="*80)
    print("PRODUCT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Identify the product column (could be 'Product' or 'product')
    product_col = None
    for col in ['Product', 'product', 'product_category', 'Product category']:
        if col in df.columns:
            product_col = col
            break
    
    if product_col is None:
        print("Warning: Product column not found. Available columns:", df.columns.tolist())
        return df
    
    print(f"\nUsing column: '{product_col}'")
    
    # Product distribution
    product_counts = df[product_col].value_counts()
    product_pct = (product_counts / len(df)) * 100
    
    product_dist = pd.DataFrame({
        'Count': product_counts,
        'Percentage': product_pct
    })
    
    print("\nComplaint Distribution by Product:")
    print(product_dist)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    product_counts.plot(kind='barh')
    plt.title('Complaint Distribution by Product', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Complaints', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'product_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {DATA_PROCESSED / 'product_distribution.png'}")
    plt.close()
    
    return df, product_col


def analyze_narrative_length(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Calculate and visualize the length (word count) of Consumer complaint narratives.
    
    Args:
        df: DataFrame containing complaint data
        
    Returns:
        Tuple containing:
            - pd.DataFrame: DataFrame with added narrative length columns
            - Optional[str]: Name of the narrative column found, or None if not found
    """
    print("\n" + "="*80)
    print("NARRATIVE LENGTH ANALYSIS")
    print("="*80)
    
    # Identify the narrative column
    narrative_col = None
    for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative', 'Consumer complaint narrative']:
        if col in df.columns:
            narrative_col = col
            break
    
    if narrative_col is None:
        print("Warning: Narrative column not found. Available columns:", df.columns.tolist())
        return df, None
    
    print(f"\nUsing column: '{narrative_col}'")
    
    # Calculate word counts (handle NaN values)
    df['narrative_word_count'] = df[narrative_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    # Calculate character counts
    df['narrative_char_count'] = df[narrative_col].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    
    # Statistics
    print("\nNarrative Length Statistics (Word Count):")
    print(df['narrative_word_count'].describe())
    
    print("\nNarrative Length Statistics (Character Count):")
    print(df['narrative_char_count'].describe())
    
    # Identify very short and very long narratives
    short_threshold = 10  # words
    long_threshold = 1000  # words
    
    short_narratives = df[df['narrative_word_count'] < short_threshold]
    long_narratives = df[df['narrative_word_count'] > long_threshold]
    
    print(f"\nVery Short Narratives (< {short_threshold} words): {len(short_narratives):,} ({len(short_narratives)/len(df)*100:.2f}%)")
    print(f"Very Long Narratives (> {long_threshold} words): {len(long_narratives):,} ({len(long_narratives)/len(df)*100:.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Word count distribution
    axes[0].hist(df[df['narrative_word_count'] > 0]['narrative_word_count'], bins=50, edgecolor='black')
    axes[0].axvline(short_threshold, color='r', linestyle='--', label=f'Short threshold ({short_threshold} words)')
    axes[0].axvline(long_threshold, color='orange', linestyle='--', label=f'Long threshold ({long_threshold} words)')
    axes[0].set_xlabel('Word Count', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Narrative Word Counts', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, min(2000, df['narrative_word_count'].quantile(0.99)))
    
    # Box plot
    axes[1].boxplot(df[df['narrative_word_count'] > 0]['narrative_word_count'], vert=True)
    axes[1].set_ylabel('Word Count', fontsize=12)
    axes[1].set_title('Box Plot of Narrative Word Counts', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'narrative_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {DATA_PROCESSED / 'narrative_length_distribution.png'}")
    plt.close()
    
    return df, narrative_col


def analyze_missing_narratives(df, narrative_col):
    """Identify the number of complaints with and without narratives."""
    print("\n" + "="*80)
    print("MISSING NARRATIVES ANALYSIS")
    print("="*80)
    
    if narrative_col is None:
        print("Warning: Narrative column not identified.")
        return df
    
    # Count missing narratives
    missing_narratives = df[narrative_col].isna() | (df[narrative_col].astype(str).str.strip() == '')
    has_narratives = ~missing_narratives
    
    missing_count = missing_narratives.sum()
    has_count = has_narratives.sum()
    total = len(df)
    
    print(f"\nComplaints with Narratives: {has_count:,} ({has_count/total*100:.2f}%)")
    print(f"Complaints without Narratives: {missing_count:,} ({missing_count/total*100:.2f}%)")
    print(f"Total Complaints: {total:,}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    labels = ['With Narratives', 'Without Narratives']
    sizes = [has_count, missing_count]
    colors = ['#2ecc71', '#e74c3c']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Complaints with vs without Narratives', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'missing_narratives.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {DATA_PROCESSED / 'missing_narratives.png'}")
    plt.close()
    
    return df


def filter_dataset(
    df: pd.DataFrame, 
    product_col: Optional[str], 
    narrative_col: Optional[str]
) -> pd.DataFrame:
    """
    Filter the dataset to meet project requirements.
    
    Args:
        df: Input DataFrame
        product_col: Name of the product column, or None if not found
        narrative_col: Name of the narrative column, or None if not found
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only target products with valid narratives
        
    Raises:
        ValueError: If narrative_col is None (required for filtering)
    """
    if narrative_col is None:
        raise ValueError(
            "Narrative column is required for filtering but was not found. "
            "Cannot proceed with dataset filtering."
        )
    print("\n" + "="*80)
    print("FILTERING DATASET")
    print("="*80)
    
    initial_count = len(df)
    print(f"\nInitial dataset size: {initial_count:,} rows")
    
    # Filter 1: Include only specified products
    # CFPB dataset product names are more descriptive, so we'll match by keywords
    target_keywords = {
        'credit card': ['credit card'],
        'personal loan': ['personal loan', 'payday loan', 'title loan', 'advance loan'],
        'savings account': ['savings account', 'checking or savings account'],
        'money transfer': ['money transfer', 'money service']
    }
    
    if product_col:
        # Create a filter that matches products containing target keywords
        product_lower = df[product_col].astype(str).str.strip().str.lower()
        
        # Build a boolean mask for matching products
        mask = pd.Series([False] * len(df), index=df.index)
        
        for category, keywords in target_keywords.items():
            for keyword in keywords:
                mask |= product_lower.str.contains(keyword, case=False, na=False)
        
        df_filtered = df[mask].copy()
        
        print(f"\nAfter filtering for target products: {len(df_filtered):,} rows")
        print(f"Removed: {initial_count - len(df_filtered):,} rows")
        
        # Show product distribution after filtering
        print("\nProduct distribution after filtering:")
        print(df_filtered[product_col].value_counts())
    else:
        print("Warning: Product column not found. Skipping product filter.")
        df_filtered = df.copy()
    
    # Filter 2: Remove records with empty narratives
    if narrative_col:
        before_narrative_filter = len(df_filtered)
        df_filtered = df_filtered[
            df_filtered[narrative_col].notna() & 
            (df_filtered[narrative_col].astype(str).str.strip() != '')
        ].copy()
        
        print(f"\nAfter removing empty narratives: {len(df_filtered):,} rows")
        print(f"Removed: {before_narrative_filter - len(df_filtered):,} rows")
    else:
        print("Warning: Narrative column not found. Skipping narrative filter.")
    
    print(f"\nFinal filtered dataset size: {len(df_filtered):,} rows")
    print(f"Total removed: {initial_count - len(df_filtered):,} rows ({((initial_count - len(df_filtered))/initial_count)*100:.2f}%)")
    
    return df_filtered


def clean_text(text: str) -> str:
    """
    Clean text narratives to improve embedding quality.
    
    Args:
        text: Raw text string to clean
        
    Returns:
        str: Cleaned text string
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove common boilerplate phrases
    boilerplate_phrases = [
        r'i am writing to file a complaint',
        r'i am writing to complain',
        r'this is a complaint',
        r'i would like to file a complaint',
        r'please be advised that',
    ]
    
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    # Keep: letters, numbers, spaces, basic punctuation (. , ! ? : ; - ' " ( ) [ ])
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'\"\(\)\[\]]', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[\.]{2,}', '.', text)
    text = re.sub(r'[\!]{2,}', '!', text)
    text = re.sub(r'[\?]{2,}', '?', text)
    
    # Clean up whitespace again
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_narratives(df: pd.DataFrame, narrative_col: Optional[str]) -> pd.DataFrame:
    """
    Apply text cleaning to all narratives in the DataFrame.
    
    Args:
        df: DataFrame containing narratives
        narrative_col: Name of the narrative column, or None if not found
        
    Returns:
        pd.DataFrame: DataFrame with cleaned narratives added as 'cleaned_narrative' column
        
    Raises:
        ValueError: If narrative_col is None
    """
    if narrative_col is None:
        raise ValueError("Narrative column is required but was not found.")
    print("\n" + "="*80)
    print("CLEANING TEXT NARRATIVES")
    print("="*80)
    
    if narrative_col is None:
        print("Warning: Narrative column not found. Skipping text cleaning.")
        return df
    
    print(f"\nCleaning narratives in column: '{narrative_col}'")
    print("This may take a few minutes for large datasets...")
    
    # Create a cleaned version
    df['cleaned_narrative'] = df[narrative_col].apply(clean_text)
    
    # Statistics on cleaning
    original_lengths = df[narrative_col].astype(str).str.len()
    cleaned_lengths = df['cleaned_narrative'].str.len()
    
    print("\nText Cleaning Statistics:")
    print(f"Average original length: {original_lengths.mean():.0f} characters")
    print(f"Average cleaned length: {cleaned_lengths.mean():.0f} characters")
    print(f"Average reduction: {((original_lengths - cleaned_lengths) / original_lengths * 100).mean():.2f}%")
    
    # Show examples
    print("\nExample of cleaned text (first 3):")
    for idx in range(min(3, len(df))):
        if pd.notna(df.iloc[idx][narrative_col]):
            print(f"\nOriginal ({idx+1}):")
            print(df.iloc[idx][narrative_col][:200] + "...")
            print(f"\nCleaned ({idx+1}):")
            print(df.iloc[idx]['cleaned_narrative'][:200] + "...")
            print("-" * 80)
    
    return df


def save_cleaned_dataset(df: pd.DataFrame, narrative_col: Optional[str]) -> pd.DataFrame:
    """
    Save the cleaned and filtered dataset to disk.
    
    Args:
        df: DataFrame to save
        narrative_col: Name of the narrative column, or None if not found
        
    Returns:
        pd.DataFrame: Final cleaned DataFrame
        
    Raises:
        IOError: If file cannot be written to disk
        ValueError: If DataFrame is empty
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame.")
    print("\n" + "="*80)
    print("SAVING CLEANED DATASET")
    print("="*80)
    
    # Use cleaned narrative if available, otherwise use original
    if 'cleaned_narrative' in df.columns and narrative_col:
        # Replace original narrative with cleaned version
        df_final = df.copy()
        df_final[narrative_col] = df_final['cleaned_narrative']
        df_final = df_final.drop(columns=['cleaned_narrative'], errors='ignore')
    else:
        df_final = df.copy()
    
    # Remove temporary columns
    columns_to_drop = ['narrative_word_count', 'narrative_char_count']
    df_final = df_final.drop(columns=[col for col in columns_to_drop if col in df_final.columns])
    
    # Save to CSV
    output_path = DATA_PROCESSED / "filtered_complaints.csv"
    try:
        df_final.to_csv(output_path, index=False)
    except (IOError, PermissionError) as e:
        raise IOError(f"Failed to save cleaned dataset to {output_path}: {e}") from e
    
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    
    return df_final


def generate_summary_report(df_original, df_filtered, product_col, narrative_col):
    """Generate a summary report of the EDA findings."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("EDA AND PREPROCESSING SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"   Original dataset size: {len(df_original):,} rows, {len(df_original.columns)} columns")
    report.append(f"   Filtered dataset size: {len(df_filtered):,} rows, {len(df_filtered.columns)} columns")
    report.append(f"   Reduction: {len(df_original) - len(df_filtered):,} rows ({(len(df_original) - len(df_filtered))/len(df_original)*100:.2f}%)")
    report.append("")
    
    if product_col:
        report.append("2. PRODUCT DISTRIBUTION")
        report.append("-" * 80)
        product_dist = df_filtered[product_col].value_counts()
        for product, count in product_dist.items():
            pct = (count / len(df_filtered)) * 100
            report.append(f"   {product}: {count:,} ({pct:.2f}%)")
        report.append("")
    
    if narrative_col:
        report.append("3. NARRATIVE ANALYSIS")
        report.append("-" * 80)
        if 'narrative_word_count' in df_filtered.columns:
            word_stats = df_filtered['narrative_word_count'].describe()
            report.append(f"   Average word count: {word_stats['mean']:.1f}")
            report.append(f"   Median word count: {word_stats['50%']:.1f}")
            report.append(f"   Min word count: {word_stats['min']:.0f}")
            report.append(f"   Max word count: {word_stats['max']:.0f}")
        report.append("")
    
    report.append("4. DATA QUALITY IMPROVEMENTS")
    report.append("-" * 80)
    report.append("   - Filtered to include only target products (Credit Card, Personal Loan, Savings Account, Money Transfer)")
    report.append("   - Removed all records with empty or missing narratives")
    report.append("   - Applied text cleaning: lowercasing, boilerplate removal, special character handling")
    report.append("   - Normalized whitespace and punctuation")
    report.append("")
    
    report.append("5. KEY FINDINGS")
    report.append("-" * 80)
    if product_col:
        most_common_product = df_filtered[product_col].value_counts().index[0]
        most_common_count = df_filtered[product_col].value_counts().iloc[0]
        report.append(f"   - Most common product category: {most_common_product} ({most_common_count:,} complaints)")
    
    if narrative_col:
        has_narratives_original = (df_original[narrative_col].notna() & 
                                   (df_original[narrative_col].astype(str).str.strip() != '')).sum()
        report.append(f"   - Original dataset had {has_narratives_original:,} complaints with narratives")
        report.append(f"   - Final dataset contains {len(df_filtered):,} complaints with valid narratives")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = DATA_PROCESSED / "eda_summary_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\nSummary report saved to:", report_path)
    print("\n" + report_text)
    
    return report_text


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 1: EXPLORATORY DATA ANALYSIS AND PREPROCESSING")
    print("="*80)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Initial EDA
    df = initial_eda(df)
    
    # Step 3: Analyze product distribution
    df, product_col = analyze_product_distribution(df)
    
    # Step 4: Analyze narrative length
    df, narrative_col = analyze_narrative_length(df)
    
    # Step 5: Analyze missing narratives
    df = analyze_missing_narratives(df, narrative_col)
    
    # Step 6: Filter dataset
    df_filtered = filter_dataset(df, product_col, narrative_col)
    
    # Step 7: Clean narratives
    df_filtered = clean_narratives(df_filtered, narrative_col)
    
    # Step 8: Save cleaned dataset
    df_final = save_cleaned_dataset(df_filtered, narrative_col)
    
    # Step 9: Generate summary report
    generate_summary_report(df, df_final, product_col, narrative_col)
    
    print("\n" + "="*80)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nDeliverables:")
    print(f"  1. Processed dataset: {DATA_PROCESSED / 'filtered_complaints.csv'}")
    print(f"  2. Summary report: {DATA_PROCESSED / 'eda_summary_report.txt'}")
    print(f"  3. Visualizations: {DATA_PROCESSED / '*.png'}")
    print("="*80)


if __name__ == "__main__":
    main()

