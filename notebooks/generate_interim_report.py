"""
Generate Interim Report PDF - Covering Task 1 and Task 2
This script creates a comprehensive PDF report summarizing the work completed.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from pathlib import Path
from datetime import datetime
import pandas as pd

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Output file
OUTPUT_FILE = OUTPUT_DIR / "interim_report.pdf"


def load_report_data():
    """Load data from generated reports."""
    data = {}
    
    # Load EDA summary
    eda_path = DATA_PROCESSED / "eda_summary_report.txt"
    if eda_path.exists():
        with open(eda_path, 'r', encoding='utf-8') as f:
            data['eda_report'] = f.read()
    
    # Load Task 2 reports
    sampling_path = DATA_PROCESSED / "task2_sampling_report.txt"
    if sampling_path.exists():
        with open(sampling_path, 'r', encoding='utf-8') as f:
            data['sampling_report'] = f.read()
    
    chunking_path = DATA_PROCESSED / "task2_chunking_report.txt"
    if chunking_path.exists():
        with open(chunking_path, 'r', encoding='utf-8') as f:
            data['chunking_report'] = f.read()
    
    embedding_path = DATA_PROCESSED / "task2_embedding_report.txt"
    if embedding_path.exists():
        with open(embedding_path, 'r', encoding='utf-8') as f:
            data['embedding_report'] = f.read()
    
    return data


def create_pdf_report():
    """Create the interim report PDF."""
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=15,
        textColor=colors.HexColor('#283593'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#3949ab'),
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        leftIndent=20,
        spaceAfter=4
    )
    
    # Load report data
    report_data = load_report_data()
    
    # ===== TITLE PAGE =====
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Interim Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Intelligent Complaint Analysis for Financial Services", 
                          ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                       fontSize=16, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#5c6bc0'))))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("RAG-Powered Chatbot Development", 
                          ParagraphStyle('Subtitle2', parent=styles['Normal'], 
                                       fontSize=12, alignment=TA_CENTER, 
                                       textColor=colors.grey)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('Date', parent=styles['Normal'], 
                                       fontSize=11, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Covering Task 1 and Task 2", 
                          ParagraphStyle('Coverage', parent=styles['Normal'], 
                                       fontSize=11, alignment=TA_CENTER, 
                                       textColor=colors.grey)))
    story.append(PageBreak())
    
    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("Table of Contents", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Understanding and Defining the Business Objective",
        "3. Discussion of Completed Work and Initial Analysis",
        "4. Next Steps and Key Areas of Focus",
        "5. Conclusion"
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 0.05*inch))
    
    story.append(Spacer(1, 0.2*inch))
    
    # ===== EXECUTIVE SUMMARY =====
    story.append(Paragraph("1. Executive Summary", heading1_style))
    
    exec_summary = """
    This interim report documents the progress made on developing an intelligent RAG-powered 
    chatbot for analyzing customer complaints in the financial services sector. The project 
    aims to transform unstructured complaint data into actionable insights for CrediTrust 
    Financial, a digital finance company serving East African markets.
    
    <b>Key Accomplishments:</b>
    <br/>• Completed comprehensive exploratory data analysis (EDA) on 9.6 million CFPB complaint records
    <br/>• Processed and cleaned 456,218 complaint narratives across four product categories
    <br/>• Created a stratified sample of 12,000 complaints for embedding generation
    <br/>• Generated 37,158 text chunks with semantic embeddings
    <br/>• Established a ChromaDB vector store ready for semantic search
    
    The foundation for the RAG pipeline has been successfully established, with data 
    preprocessing and vectorization completed. The system is now ready for RAG pipeline 
    development and user interface implementation.
    """
    story.append(Paragraph(exec_summary, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # ===== BUSINESS OBJECTIVE =====
    story.append(Paragraph("2. Understanding and Defining the Business Objective", heading1_style))
    
    business_obj = """
    <b>2.1 Business Context</b>
    <br/><br/>
    CrediTrust Financial is a fast-growing digital finance company serving over 500,000 users 
    across three countries through a mobile-first platform. The company offers four primary 
    financial products:
    <br/>• Credit Cards
    <br/>• Personal Loans
    <br/>• Savings Accounts
    <br/>• Money Transfers
    
    <br/><br/><b>2.2 Business Challenge</b>
    <br/><br/>
    With thousands of customer complaints received monthly through multiple channels (in-app, 
    email, regulatory portals), internal teams face significant challenges in:
    <br/>• <b>Speed:</b> Product Managers currently take days to identify major complaint trends
    <br/>• <b>Accessibility:</b> Non-technical teams (Support and Compliance) depend on data analysts 
    for insights
    <br/>• <b>Proactivity:</b> The organization operates reactively rather than proactively identifying 
    issues from customer feedback
    
    <br/><br/><b>2.3 Project Objective</b>
    <br/><br/>
    Develop a RAG-powered chatbot that enables internal users to:
    <br/>• Ask plain-English questions about customer complaints
    <br/>• Receive instant, insightful answers using semantic search
    <br/>• Query across multiple product categories simultaneously
    <br/>• Access actionable insights without technical expertise
    
    <br/><br/><b>2.4 Key Performance Indicators (KPIs)</b>
    <br/><br/>
    The solution targets three critical KPIs:
    <br/>1. <b>Speed:</b> Reduce complaint trend identification time from days to minutes
    <br/>2. <b>Accessibility:</b> Empower non-technical teams to get answers independently
    <br/>3. <b>Proactivity:</b> Shift from reactive to proactive problem-solving based on real-time 
    customer feedback
    
    <br/><br/><b>2.5 Technical Approach</b>
    <br/><br/>
    The solution leverages Retrieval-Augmented Generation (RAG) technology:
    <br/>• <b>Semantic Search:</b> Vector databases (ChromaDB) enable similarity-based retrieval
    <br/>• <b>Language Models:</b> LLMs generate concise, contextual answers from retrieved complaints
    <br/>• <b>Multi-Product Support:</b> Unified interface for querying across all product categories
    """
    story.append(Paragraph(business_obj, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # ===== COMPLETED WORK =====
    story.append(Paragraph("3. Discussion of Completed Work and Initial Analysis", heading1_style))
    
    # Task 1 Section
    story.append(Paragraph("3.1 Task 1: Exploratory Data Analysis and Preprocessing", heading2_style))
    
    task1_content = """
    <b>3.1.1 Dataset Overview</b>
    <br/><br/>
    The project utilizes the Consumer Financial Protection Bureau (CFPB) complaint dataset, 
    which contains comprehensive consumer complaint data across multiple financial product 
    categories. The initial dataset comprised:
    <br/>• <b>Total Records:</b> 9,609,797 complaints
    <br/>• <b>Columns:</b> 20 attributes including product category, issue type, company, 
    state, date, and consumer narratives
    <br/>• <b>Time Period:</b> Historical complaint data spanning multiple years
    
    <br/><br/><b>3.1.2 Data Quality Analysis</b>
    <br/><br/>
    Comprehensive EDA revealed several key insights (see visualizations below):
    <br/>• <b>Narrative Coverage:</b> 2,980,756 complaints (31%) contained consumer narratives 
    (visualized in Figure 3)
    <br/>• <b>Product Distribution:</b> Complaints spanned multiple product categories, with 
    significant variation in volume (see Figure 1)
    <br/>• <b>Narrative Length:</b> Wide variation in narrative length, requiring careful 
    handling for chunking (see Figure 2 for distribution analysis)
    
    <br/><br/><b>3.1.3 Data Filtering and Cleaning</b>
    <br/><br/>
    The dataset was filtered to focus on four target product categories:
    <br/>• Credit Card
    <br/>• Personal Loan
    <br/>• Savings Account
    <br/>• Money Transfer
    
    <b>Filtering Results:</b>
    <br/>• <b>Original Dataset:</b> 9,609,797 rows
    <br/>• <b>After Product Filtering:</b> Reduced to target product categories
    <br/>• <b>After Narrative Filtering:</b> 456,218 rows with valid narratives
    <br/>• <b>Final Reduction:</b> 95.25% reduction, focusing on high-quality, relevant data
    
    <br/><br/><b>3.1.4 Text Preprocessing</b>
    <br/><br/>
    Comprehensive text cleaning was applied to improve embedding quality:
    <br/>• <b>Normalization:</b> Lowercasing all text for consistency
    <br/>• <b>Boilerplate Removal:</b> Removed common complaint phrases that add noise
    <br/>• <b>Special Character Handling:</b> Cleaned and normalized punctuation
    <br/>• <b>Whitespace Normalization:</b> Standardized spacing and formatting
    
    <b>Product Distribution in Final Dataset:</b>
    <br/>• Checking or savings account: 140,319 (30.76%)
    <br/>• Credit card or prepaid card: 108,667 (23.82%)
    <br/>• Money transfer, virtual currency, or money service: 97,188 (21.30%)
    <br/>• Credit card: 80,667 (17.68%)
    <br/>• Personal loan categories: 27,377 (6.00%)
    
    <br/><br/><b>3.1.5 Exploratory Data Analysis Visualizations</b>
    <br/><br/>
    The following visualizations provide visual evidence of the key findings from the EDA process:
    """
    story.append(Paragraph(task1_content, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Add visualizations with proper error handling
    viz_paths = {
        'product_distribution.png': {
            'caption': 'Figure 1: Product Distribution - Shows the distribution of complaints across different product categories in the filtered dataset. The horizontal bar chart displays the number of complaints for each product category with clear axis labels.',
            'width': 5.5*inch,
            'height': 4*inch
        },
        'narrative_length_distribution.png': {
            'caption': 'Figure 2: Narrative Length Distribution - Displays the distribution of word counts in complaint narratives. The left panel shows a histogram with percentile markers, and the right panel shows a box plot with summary statistics, demonstrating the variation in narrative length.',
            'width': 6*inch,
            'height': 3*inch
        },
        'missing_narratives.png': {
            'caption': 'Figure 3: Missing Narratives Analysis - Pie chart showing the proportion of complaints with and without narratives. The chart includes percentage labels and total counts for each category.',
            'width': 4.5*inch,
            'height': 4.5*inch
        }
    }
    
    for viz_file, viz_info in viz_paths.items():
        viz_path = DATA_PROCESSED / viz_file
        if viz_path.exists():
            try:
                # Add caption before image
                story.append(Paragraph(f"<b>{viz_info['caption']}</b>", 
                    ParagraphStyle('Caption', parent=styles['Normal'], 
                                 fontSize=9, alignment=TA_LEFT, 
                                 textColor=colors.HexColor('#333333'), 
                                 spaceAfter=6, spaceBefore=4)))
                
                # Load and add image with proper sizing
                img = Image(str(viz_path), width=viz_info['width'], height=viz_info['height'])
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.15*inch))
                print(f"Successfully added {viz_file} to PDF")
            except Exception as e:
                print(f"ERROR: Could not add image {viz_file}: {e}")
                import traceback
                traceback.print_exc()
                # Add error message
                story.append(Paragraph(f"<i>[ERROR: Visualization {viz_file} could not be loaded: {str(e)}]</i>", 
                    ParagraphStyle('Caption', parent=styles['Normal'], 
                                 fontSize=9, alignment=TA_LEFT, 
                                 textColor=colors.red)))
        else:
            print(f"WARNING: Visualization file not found: {viz_path}")
            story.append(Paragraph(f"<i>[WARNING: Visualization {viz_file} not found at {viz_path}]</i>", 
                ParagraphStyle('Caption', parent=styles['Normal'], 
                             fontSize=9, alignment=TA_LEFT, 
                             textColor=colors.red)))
    
    story.append(Spacer(1, 0.1*inch))
    
    # Task 2 Section
    story.append(Paragraph("3.2 Task 2: Text Chunking, Embedding, and Vector Store Indexing", heading2_style))
    
    task2_content = """
    <b>3.2.1 Stratified Sampling Strategy</b>
    <br/><br/>
    To balance computational efficiency with representativeness, a stratified random sample 
    was created:
    <br/>• <b>Sample Size:</b> 12,000 complaints (within the 10,000-15,000 target range)
    <br/>• <b>Method:</b> Stratified random sampling ensuring proportional representation 
    across all product categories
    <br/>• <b>Rationale:</b> Maintains the distribution of the full dataset while enabling 
    efficient processing
    <br/>• <b>Reproducibility:</b> Random seed (42) ensures consistent results
    
    <b>Sample Distribution:</b>
    <br/>• Checking or savings account: 3,690 (30.76%)
    <br/>• Credit card or prepaid card: 2,858 (23.83%)
    <br/>• Money transfer, virtual currency, or money service: 2,556 (21.31%)
    <br/>• Credit card: 2,121 (17.68%)
    <br/>• Personal loan categories: 770 (6.42%)
    
    <br/><br/><b>3.2.2 Text Chunking Methodology</b>
    <br/><br/>
    Text narratives were chunked using LangChain's RecursiveCharacterTextSplitter:
    <br/>• <b>Chunk Size:</b> 500 characters (optimal balance between context and embedding quality)
    <br/>• <b>Chunk Overlap:</b> 50 characters (10% overlap prevents context loss at boundaries)
    <br/>• <b>Splitting Strategy:</b> Hierarchical approach prioritizing paragraph, sentence, 
    word, and character boundaries
    
    <b>Chunking Results:</b>
    <br/>• <b>Total Chunks Created:</b> 37,158 chunks from 11,995 complaints
    <br/>• <b>Average Chunks per Complaint:</b> 3.10 chunks
    <br/>• <b>Average Chunk Length:</b> 371 characters
    <br/>• <b>Median Chunk Length:</b> 407 characters
    
    <br/><br/><b>3.2.3 Embedding Generation</b>
    <br/><br/>
    Semantic embeddings were generated using the sentence-transformers model:
    <br/>• <b>Model:</b> sentence-transformers/all-MiniLM-L6-v2
    <br/>• <b>Embedding Dimension:</b> 384 dimensions
    <br/>• <b>Model Size:</b> ~80MB (efficient for production use)
    <br/>• <b>Total Embeddings:</b> 37,158 embeddings generated
    
    <b>Model Selection Rationale:</b>
    <br/>• <b>Performance vs. Size:</b> Excellent balance of semantic understanding and efficiency
    <br/>• <b>Compatibility:</b> Matches pre-built embeddings specification for consistency
    <br/>• <b>Domain Suitability:</b> Handles informal language and financial terminology well
    <br/>• <b>Industry Standard:</b> Widely used in production RAG systems
    
    <br/><br/><b>3.2.4 Vector Store Creation</b>
    <br/><br/>
    A persistent ChromaDB vector store was created with rich metadata:
    <br/>• <b>Storage System:</b> ChromaDB (persistent, production-ready)
    <br/>• <b>Collection Name:</b> complaint_chunks
    <br/>• <b>Metadata Fields:</b> Complaint ID, product category, chunk index, issue, 
    sub-issue, company, state, date received
    <br/>• <b>Total Documents:</b> 37,158 chunks with embeddings and metadata
    
    The vector store is now ready for semantic search operations in the RAG pipeline.
    """
    story.append(Paragraph(task2_content, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Initial Analysis Section
    story.append(Paragraph("3.3 Initial Analysis and Insights", heading2_style))
    
    analysis_content = """
    <b>3.3.1 Data Quality Insights</b>
    <br/><br/>
    The EDA process revealed important patterns:
    <br/>• <b>Narrative Availability:</b> Only 31% of complaints contained narratives, 
    highlighting the importance of filtering
    <br/>• <b>Product Concentration:</b> Savings accounts and credit cards dominate complaint 
    volume, representing over 50% of filtered data
    <br/>• <b>Text Quality:</b> Significant variation in narrative quality and length, 
    necessitating robust preprocessing
    
    <br/><br/><b>3.3.2 Technical Decisions</b>
    <br/><br/>
    Several key technical decisions were made based on analysis:
    <br/>• <b>Chunk Size (500 chars):</b> Chosen to balance context preservation with 
    embedding quality, aligning with pre-built embeddings specification
    <br/>• <b>Stratified Sampling:</b> Ensures all product categories are represented 
    proportionally, critical for multi-product querying
    <br/>• <b>Embedding Model:</b> all-MiniLM-L6-v2 selected for optimal balance of 
    performance, size, and compatibility
    
    <br/><br/><b>3.3.3 Pipeline Readiness</b>
    <br/><br/>
    The completed work establishes a solid foundation:
    <br/>• <b>Clean Data:</b> 456,218 high-quality complaint narratives ready for processing
    <br/>• <b>Vector Store:</b> 37,158 chunks with embeddings and metadata indexed in ChromaDB
    <br/>• <b>Reproducibility:</b> All processes use fixed random seeds and documented parameters
    <br/>• <b>Scalability:</b> Pipeline designed to handle larger datasets if needed
    """
    story.append(Paragraph(analysis_content, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # ===== NEXT STEPS =====
    story.append(Paragraph("4. Next Steps and Key Areas of Focus", heading1_style))
    
    next_steps_content = """
    <b>4.1 Immediate Next Steps</b>
    <br/><br/>
    <b>Task 3: RAG Pipeline Development</b>
    <br/>• Integrate the vector store with a language model (LLM) for answer generation
    <br/>• Implement semantic search functionality using ChromaDB queries
    <br/>• Develop prompt engineering strategies for complaint-specific queries
    <br/>• Create retrieval and generation pipeline with error handling
    <br/>• Test and evaluate retrieval quality and answer relevance
    <br/>• <b>Qualitative Evaluation:</b> Conduct comprehensive evaluation using 5-10 carefully 
    designed test questions covering different query types (product-specific, cross-product, 
    trend analysis, specific issue identification) to assess answer quality, relevance, 
    and accuracy
    
    <br/><br/><b>Task 4: User Interface Development</b>
    <br/>• Build interactive chatbot interface (Gradio or Streamlit)
    <br/>• Implement query input and response display
    <br/>• Add metadata display (product, company, date) for retrieved complaints
    <br/>• Create user-friendly error messages
    <br/>• Design intuitive UI for non-technical users
    
    <br/><br/><b>4.2 Key Areas of Focus</b>
    <br/><br/>
    <b>4.2.1 Retrieval Quality</b>
    <br/>• Optimize similarity search parameters (top-k retrieval)
    <br/>• Evaluate retrieval precision and recall
    <br/>• Test query variations and edge cases
    <br/>• Implement query expansion if needed
    
    <br/><br/><b>4.2.2 Answer Generation</b>
    <br/>• Develop effective prompt templates for complaint analysis
    <br/>• Ensure answers are concise, actionable, and cite sources
    <br/>• Handle multi-product queries effectively
    <br/>• Implement answer quality evaluation metrics
    <br/>• <b>Qualitative Evaluation Plan:</b> Design and execute a systematic evaluation 
    using 5-10 test questions that cover:
    <br/>  - Product-specific queries (e.g., "What are common issues with credit cards?")
    <br/>  - Cross-product comparisons (e.g., "Compare complaint patterns between credit 
    cards and personal loans")
    <br/>  - Trend identification (e.g., "What complaint trends have emerged recently?")
    <br/>  - Specific issue queries (e.g., "Find complaints about unauthorized charges")
    <br/>  - Complex multi-faceted questions (e.g., "What are the main concerns about 
    money transfer services in California?")
    <br/>  Each test question will be evaluated on answer relevance, accuracy, completeness, 
    and citation quality
    
    <br/><br/><b>4.2.3 User Experience</b>
    <br/>• Design intuitive interface for non-technical users
    <br/>• Provide clear examples of effective queries
    <br/>• Display retrieved complaint details for transparency
    <br/>• Implement feedback mechanisms for continuous improvement
    
    <br/><br/><b>4.2.4 Performance Optimization</b>
    <br/>• Optimize query response time (target: < 5 seconds)
    <br/>• Implement caching for common queries
    <br/>• Monitor resource usage and scalability
    <br/>• Test with larger sample sizes if needed
    
    <br/><br/><b>4.3 Potential Challenges and Mitigation</b>
    <br/><br/>
    <b>Challenge 1: Query Understanding</b>
    <br/>• <b>Risk:</b> Users may phrase queries in ways that don't match complaint language
    <br/>• <b>Mitigation:</b> Implement query preprocessing and provide example queries
    
    <b>Challenge 2: Multi-Product Queries</b>
    <br/>• <b>Risk:</b> Queries spanning multiple products may return mixed results
    <br/>• <b>Mitigation:</b> Implement product filtering and result categorization
    
    <b>Challenge 3: Answer Quality</b>
    <br/>• <b>Risk:</b> Generated answers may lack specificity or accuracy
    <br/>• <b>Mitigation:</b> Iterative prompt engineering and answer evaluation
    
    <br/><br/><b>4.4 Success Metrics and Evaluation</b>
    <br/><br/>
    The following metrics will be used to evaluate success:
    <br/>• <b>Retrieval Accuracy:</b> Percentage of relevant complaints retrieved
    <br/>• <b>Answer Quality:</b> User satisfaction with generated answers
    <br/>• <b>Response Time:</b> Average time to generate answers
    <br/>• <b>User Adoption:</b> Usage by non-technical teams
    <br/>• <b>Business Impact:</b> Reduction in time to identify complaint trends
    
    <br/><br/><b>Qualitative Evaluation Framework:</b>
    <br/>A structured qualitative evaluation will be conducted using 5-10 comprehensive test 
    questions designed to assess the RAG pipeline across multiple dimensions:
    <br/>• <b>Query Diversity:</b> Questions will span different complexity levels and query types
    <br/>• <b>Evaluation Criteria:</b> Each answer will be assessed on relevance, accuracy, 
    completeness, source citation, and actionability
    <br/>• <b>Iterative Refinement:</b> Results will inform prompt engineering improvements 
    and retrieval parameter optimization
    <br/>• <b>Documentation:</b> All test questions, answers, and evaluation scores will be 
    documented for reproducibility and continuous improvement
    """
    story.append(Paragraph(next_steps_content, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # ===== CONCLUSION =====
    story.append(Paragraph("5. Conclusion", heading1_style))
    
    conclusion_content = """
    This interim report documents significant progress on the RAG-powered complaint analysis 
    system for CrediTrust Financial. The foundation has been successfully established through 
    comprehensive data analysis, preprocessing, and vector store creation.
    
    <br/><br/><b>Key Achievements:</b>
    <br/>• Processed 9.6 million complaint records to identify 456,218 high-quality narratives
    <br/>• Created a representative sample of 12,000 complaints with proportional product 
    distribution
    <br/>• Generated 37,158 semantic embeddings and established a production-ready vector store
    <br/>• Documented all methodologies and decisions for reproducibility
    
    <br/><br/><b>Project Status:</b>
    <br/>The data pipeline is complete and ready for RAG development. The vector store 
    contains rich metadata enabling sophisticated semantic search across product categories. 
    All technical decisions have been made with scalability and production-readiness in mind.
    
    <br/><br/><b>Path Forward:</b>
    <br/>With Tasks 1 and 2 complete, the project is well-positioned to proceed with RAG 
    pipeline development (Task 3) and user interface creation (Task 4). The established 
    foundation ensures that the final system will meet the business objectives of speed, 
    accessibility, and proactivity in complaint analysis.
    
    <br/><br/>The next phase will focus on integrating the vector store with language models 
    and creating an intuitive interface that empowers non-technical teams to extract 
    actionable insights from customer complaints.
    """
    story.append(Paragraph(conclusion_content, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Appendices
    story.append(Paragraph("Appendix A: Technical Specifications", heading2_style))
    
    appendix_content = """
    <b>Data Processing Pipeline:</b>
    <br/>• Input: CFPB complaints.csv (9.6M records)
    <br/>• Output: filtered_complaints.csv (456K records)
    <br/>• Tools: pandas, numpy, matplotlib, seaborn
    
    <br/><br/><b>Embedding Pipeline:</b>
    <br/>• Sample Size: 12,000 complaints
    <br/>• Chunking: LangChain RecursiveCharacterTextSplitter
    <br/>• Embedding Model: sentence-transformers/all-MiniLM-L6-v2
    <br/>• Vector Store: ChromaDB
    <br/>• Total Chunks: 37,158
    
    <br/><br/><b>System Requirements:</b>
    <br/>• Python 3.8+
    <br/>• Key Libraries: pandas, numpy, langchain, sentence-transformers, chromadb
    <br/>• Storage: ~500MB for vector store
    <br/>• Memory: 8GB+ recommended for processing
    """
    story.append(Paragraph(appendix_content, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"\n{'='*80}")
    print("INTERIM REPORT GENERATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nReport saved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    create_pdf_report()

