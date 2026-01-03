"""
Regenerate visualizations with improved formatting and clear labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Load the filtered dataset
df = pd.read_csv(DATA_PROCESSED / "filtered_complaints.csv")

# Identify product column
product_col = None
for col in ['Product', 'product', 'product_category', 'Product category']:
    if col in df.columns:
        product_col = col
        break

# Identify narrative column
narrative_col = None
for col in ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative']:
    if col in df.columns:
        narrative_col = col
        break

print("Regenerating visualizations with improved formatting...")

# ===== 1. Product Distribution - Improved =====
if product_col:
    print("\n1. Generating Product Distribution chart...")
    product_counts = df[product_col].value_counts()
    
    # Create figure with better formatting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(product_counts)), product_counts.values, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set y-axis labels
    ax.set_yticks(range(len(product_counts)))
    ax.set_yticklabels(product_counts.index, fontsize=10)
    
    # Set x-axis label
    ax.set_xlabel('Number of Complaints', fontsize=12, fontweight='bold')
    ax.set_ylabel('Product Category', fontsize=12, fontweight='bold')
    ax.set_title('Complaint Distribution by Product Category', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(product_counts.items()):
        ax.text(val + val*0.01, i, f'{val:,}', va='center', fontsize=9, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format x-axis with comma separators
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'product_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {DATA_PROCESSED / 'product_distribution.png'}")
    plt.close()

# ===== 2. Narrative Length Distribution - Improved =====
if narrative_col:
    print("\n2. Generating Narrative Length Distribution chart...")
    
    # Calculate word counts
    df['narrative_word_count'] = df[narrative_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    # Filter out zero-length narratives for visualization
    word_counts = df[df['narrative_word_count'] > 0]['narrative_word_count']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Histogram
    axes[0].hist(word_counts, bins=50, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    axes[0].axvline(word_counts.quantile(0.25), color='red', linestyle='--', linewidth=2, label=f'25th percentile: {int(word_counts.quantile(0.25))} words')
    axes[0].axvline(word_counts.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {int(word_counts.median())} words')
    axes[0].axvline(word_counts.quantile(0.75), color='green', linestyle='--', linewidth=2, label=f'75th percentile: {int(word_counts.quantile(0.75))} words')
    
    axes[0].set_xlabel('Word Count per Narrative', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency (Number of Complaints)', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Narrative Word Counts', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    
    # Limit x-axis to 99th percentile for better visibility
    x_max = min(2000, word_counts.quantile(0.99))
    axes[0].set_xlim(0, x_max)
    
    # Format x-axis
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Right plot: Box plot
    bp = axes[1].boxplot([word_counts], vert=True, patch_artist=True, 
                         boxprops=dict(facecolor='steelblue', alpha=0.7),
                         medianprops=dict(color='orange', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
    
    axes[1].set_ylabel('Word Count per Narrative', fontsize=12, fontweight='bold')
    axes[1].set_title('Box Plot of Narrative Word Counts', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_xticklabels(['All Complaints'], fontsize=11)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    
    # Add statistics text
    stats_text = f'Mean: {word_counts.mean():.0f}\nMedian: {word_counts.median():.0f}\nMin: {word_counts.min()}\nMax: {word_counts.max():,}'
    axes[1].text(1.15, word_counts.median(), stats_text, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, verticalalignment='center')
    
    # Format y-axis
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'narrative_length_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {DATA_PROCESSED / 'narrative_length_distribution.png'}")
    plt.close()

# ===== 3. Missing Narratives - Improved =====
if narrative_col:
    print("\n3. Generating Missing Narratives chart...")
    
    # Count missing narratives
    missing_narratives = df[narrative_col].isna() | (df[narrative_col].astype(str).str.strip() == '')
    has_narratives = ~missing_narratives
    
    missing_count = missing_narratives.sum()
    has_count = has_narratives.sum()
    total = len(df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    labels = ['With Narratives', 'Without Narratives']
    sizes = [has_count, missing_count]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)  # Slight separation for emphasis
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                       startangle=90, explode=explode, shadow=True,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Enhance autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Add count information
    info_text = f'Total Complaints: {total:,}\nWith Narratives: {has_count:,}\nWithout Narratives: {missing_count:,}'
    ax.text(0, -1.3, info_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Complaints with vs without Narratives', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'missing_narratives.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {DATA_PROCESSED / 'missing_narratives.png'}")
    plt.close()

print("\n" + "="*80)
print("All visualizations regenerated with improved formatting!")
print("="*80)

