"""
Task 3: RAG Pipeline Evaluation
Creates evaluation framework with 5-10 test questions and generates evaluation report.
"""

from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

import sys
from pathlib import Path

# Add notebooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from task3_rag_pipeline import RAGPipeline, TOP_K

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# Evaluation test questions covering different query types
EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What are the most common issues with credit cards?",
        "type": "product-specific",
        "expected_focus": "Credit card related complaints, common issues like unauthorized charges, billing errors"
    },
    {
        "id": 2,
        "question": "What problems do customers face with personal loans?",
        "type": "product-specific",
        "expected_focus": "Personal loan complaints, issues with repayment, interest rates, loan terms"
    },
    {
        "id": 3,
        "question": "Compare complaint patterns between credit cards and savings accounts.",
        "type": "cross-product",
        "expected_focus": "Differences in complaint types between two product categories"
    },
    {
        "id": 4,
        "question": "What are the main concerns about money transfer services?",
        "type": "product-specific",
        "expected_focus": "Money transfer issues, transaction problems, fees, delays"
    },
    {
        "id": 5,
        "question": "Find complaints about unauthorized charges or fraudulent transactions.",
        "type": "specific-issue",
        "expected_focus": "Security-related complaints, unauthorized transactions, fraud"
    },
    {
        "id": 6,
        "question": "What complaint trends have emerged recently regarding customer service?",
        "type": "trend-analysis",
        "expected_focus": "Customer service issues, response times, support quality"
    },
    {
        "id": 7,
        "question": "What are the main issues with billing and payment processing?",
        "type": "specific-issue",
        "expected_focus": "Billing errors, payment processing problems, transaction issues"
    },
    {
        "id": 8,
        "question": "What are customers saying about interest rates and fees?",
        "type": "specific-issue",
        "expected_focus": "Interest rate complaints, fee-related issues, pricing concerns"
    },
    {
        "id": 9,
        "question": "What are the primary concerns about account management and access?",
        "type": "specific-issue",
        "expected_focus": "Account access issues, account management problems, login difficulties"
    },
    {
        "id": 10,
        "question": "What are the main complaints across all financial products we offer?",
        "type": "cross-product",
        "expected_focus": "Overall complaint patterns, common themes across all products"
    }
]


def evaluate_rag_pipeline(pipeline: RAGPipeline, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the RAG pipeline on a set of test questions.
    
    Args:
        pipeline: Initialized RAGPipeline instance
        questions: List of question dictionaries
        
    Returns:
        List of evaluation results
    """
    results = []
    
    print("="*80)
    print("RAG PIPELINE EVALUATION")
    print("="*80)
    print(f"Evaluating {len(questions)} test questions...\n")
    
    for i, q_dict in enumerate(questions, 1):
        question = q_dict['question']
        print(f"[{i}/{len(questions)}] Processing: {question}")
        print("-" * 80)
        
        try:
            # Query the pipeline
            result = pipeline.query(question, top_k=TOP_K)
            
            # Extract retrieved sources (top 2 for report)
            retrieved_sources = []
            for j, chunk in enumerate(result['retrieved_chunks'][:2], 1):
                metadata = chunk.get('metadata', {})
                source_info = {
                    'excerpt_num': j,
                    'text_preview': chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text'],
                    'product': metadata.get('product', 'N/A'),
                    'complaint_id': metadata.get('complaint_id', 'N/A'),
                    'issue': metadata.get('issue', 'N/A')
                }
                retrieved_sources.append(source_info)
            
            # Manual quality scoring (1-5 scale)
            # In a real scenario, this would be done by human evaluators
            # For now, we'll create a placeholder that can be filled manually
            quality_score = None  # To be filled manually or by evaluator
            
            evaluation_result = {
                'question_id': q_dict['id'],
                'question': question,
                'question_type': q_dict['type'],
                'expected_focus': q_dict['expected_focus'],
                'generated_answer': result['answer'],
                'retrieved_sources': retrieved_sources,
                'num_chunks_retrieved': result['num_chunks'],
                'quality_score': quality_score,
                'comments': ""
            }
            
            results.append(evaluation_result)
            print(f"[OK] Completed. Retrieved {result['num_chunks']} chunks.\n")
            
        except Exception as e:
            print(f"[ERROR] Error: {e}\n")
            evaluation_result = {
                'question_id': q_dict['id'],
                'question': question,
                'question_type': q_dict['type'],
                'expected_focus': q_dict['expected_focus'],
                'generated_answer': f"ERROR: {str(e)}",
                'retrieved_sources': [],
                'num_chunks_retrieved': 0,
                'quality_score': 0,
                'comments': f"Error occurred: {str(e)}"
            }
            results.append(evaluation_result)
    
    return results


def generate_evaluation_report(results: List[Dict[str, Any]], output_path: Path):
    """
    Generate evaluation report in Markdown format.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the report
    """
    report_lines = []
    
    # Header
    report_lines.append("# RAG Pipeline Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report presents a qualitative evaluation of the RAG (Retrieval-Augmented Generation) pipeline ")
    report_lines.append("for analyzing customer complaints. The evaluation uses 10 representative test questions covering ")
    report_lines.append("different query types: product-specific, cross-product comparisons, specific issue identification, ")
    report_lines.append("and trend analysis.")
    report_lines.append("")
    
    # Evaluation Table
    report_lines.append("## Evaluation Results")
    report_lines.append("")
    report_lines.append("| Question ID | Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |")
    report_lines.append("|-------------|----------|------------------|------------------|---------------|-------------------|")
    
    for result in results:
        q_id = result['question_id']
        question = result['question'].replace('|', '\\|')  # Escape pipes
        answer = result['generated_answer'][:200].replace('|', '\\|').replace('\n', ' ')  # Truncate and clean
        if len(result['generated_answer']) > 200:
            answer += "..."
        
        # Format retrieved sources
        sources_text = ""
        for src in result['retrieved_sources']:
            sources_text += f"**Excerpt {src['excerpt_num']}:** {src['text_preview']} "
            sources_text += f"(Product: {src['product']}, Issue: {src['issue']})\\n"
        sources_text = sources_text.replace('|', '\\|').replace('\n', '<br>')
        
        quality_score = result['quality_score'] if result['quality_score'] is not None else "TBD"
        comments = result['comments'].replace('|', '\\|').replace('\n', ' ') if result['comments'] else "Pending evaluation"
        
        report_lines.append(f"| {q_id} | {question} | {answer} | {sources_text} | {quality_score} | {comments} |")
    
    report_lines.append("")
    
    # Detailed Analysis Section
    report_lines.append("## Detailed Analysis")
    report_lines.append("")
    
    for result in results:
        report_lines.append(f"### Question {result['question_id']}: {result['question']}")
        report_lines.append("")
        report_lines.append(f"**Question Type:** {result['question_type']}")
        report_lines.append("")
        report_lines.append(f"**Expected Focus:** {result['expected_focus']}")
        report_lines.append("")
        report_lines.append(f"**Generated Answer:**")
        report_lines.append("")
        report_lines.append(f"{result['generated_answer']}")
        report_lines.append("")
        report_lines.append(f"**Retrieved Chunks:** {result['num_chunks_retrieved']}")
        report_lines.append("")
        
        for src in result['retrieved_sources']:
            report_lines.append(f"- **Excerpt {src['excerpt_num']}:**")
            report_lines.append(f"  - Product: {src['product']}")
            report_lines.append(f"  - Issue: {src['issue']}")
            report_lines.append(f"  - Complaint ID: {src['complaint_id']}")
            report_lines.append(f"  - Preview: {src['text_preview']}")
            report_lines.append("")
        
        if result['comments']:
            report_lines.append(f"**Analysis:** {result['comments']}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Summary and Recommendations
    report_lines.append("## Summary and Recommendations")
    report_lines.append("")
    report_lines.append("### What Worked Well")
    report_lines.append("")
    report_lines.append("- [To be filled based on evaluation results]")
    report_lines.append("")
    report_lines.append("### Areas for Improvement")
    report_lines.append("")
    report_lines.append("- [To be filled based on evaluation results]")
    report_lines.append("")
    report_lines.append("### Recommendations")
    report_lines.append("")
    report_lines.append("1. **Prompt Engineering:** Refine prompt templates based on answer quality")
    report_lines.append("2. **Retrieval Optimization:** Adjust top-k parameter or similarity thresholds")
    report_lines.append("3. **Model Selection:** Consider using larger or instruction-tuned models for better generation")
    report_lines.append("4. **Context Formatting:** Improve how retrieved chunks are formatted in the context")
    report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nEvaluation report saved to: {output_path}")


def save_evaluation_json(results: List[Dict[str, Any]], output_path: Path):
    """Save evaluation results as JSON for programmatic access."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation JSON saved to: {output_path}")


def main():
    """Main evaluation function."""
    print("="*80)
    print("TASK 3: RAG PIPELINE EVALUATION")
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nNote: If you encounter LLM model loading issues, you may need to:")
        print("1. Install transformers and torch: pip install transformers torch")
        print("2. Use a different model or API-based solution")
        print("3. Check available system resources")
        return
    
    # Run evaluation
    results = evaluate_rag_pipeline(pipeline, EVALUATION_QUESTIONS)
    
    # Generate reports
    report_path = REPORTS_DIR / "task3_evaluation_report.md"
    json_path = REPORTS_DIR / "task3_evaluation_results.json"
    
    generate_evaluation_report(results, report_path)
    save_evaluation_json(results, json_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTotal questions evaluated: {len(results)}")
    print(f"Successful queries: {sum(1 for r in results if r['num_chunks_retrieved'] > 0)}")
    print(f"Failed queries: {sum(1 for r in results if r['num_chunks_retrieved'] == 0)}")
    print(f"\nReports generated:")
    print(f"  1. Markdown report: {report_path}")
    print(f"  2. JSON results: {json_path}")
    print("\n" + "="*80)
    print("\nNote: Quality scores and detailed comments should be filled manually")
    print("by reviewing the generated answers and retrieved sources.")
    print("="*80)


if __name__ == "__main__":
    main()

