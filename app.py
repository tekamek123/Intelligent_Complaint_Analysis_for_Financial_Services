"""
Task 4: Interactive Chat Interface for RAG System
A user-friendly Gradio interface for interacting with the RAG pipeline.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import gradio as gr

# Add notebooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "notebooks"))

from task3_rag_pipeline import RAGPipeline, TOP_K

# Initialize RAG pipeline (will be loaded once at startup)
pipeline: Optional[RAGPipeline] = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        print("Initializing RAG pipeline...")
        try:
            pipeline = RAGPipeline()
            print("Pipeline initialized successfully!")
            return "✓ RAG Pipeline ready!"
        except Exception as e:
            error_msg = f"Error initializing pipeline: {str(e)}"
            print(error_msg)
            return f"✗ {error_msg}"
    return "✓ RAG Pipeline ready!"


def format_source_chunks(chunks):
    """Format retrieved chunks for display with metadata."""
    if not chunks:
        return "No sources retrieved."
    
    formatted_sources = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        text = chunk.get('text', '')
        
        # Extract metadata
        product = metadata.get('product', 'N/A')
        complaint_id = metadata.get('complaint_id', 'N/A')
        issue = metadata.get('issue', 'N/A')
        distance = chunk.get('distance')
        
        # Format distance if available
        similarity = f" (Similarity: {1 - distance:.3f})" if distance is not None else ""
        
        # Truncate text if too long
        text_preview = text[:300] + "..." if len(text) > 300 else text
        
        source_text = f"""
**Source {i}**{similarity}
- **Product:** {product}
- **Issue:** {issue}
- **Complaint ID:** {complaint_id}
- **Text:** {text_preview}
"""
        formatted_sources.append(source_text)
    
    return "\n---\n".join(formatted_sources)


def query_rag(question: str) -> Tuple[str, str]:
    """
    Process a question through the RAG pipeline.
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (answer, formatted_sources)
    """
    if not question or not question.strip():
        return "Please enter a question.", ""
    
    if pipeline is None:
        return "Error: Pipeline not initialized. Please restart the application.", ""
    
    try:
        # Query the pipeline
        result = pipeline.query(question, top_k=TOP_K)
        
        # Extract answer
        answer = result.get('answer', 'No answer generated.')
        
        # Format source chunks
        retrieved_chunks = result.get('retrieved_chunks', [])
        sources = format_source_chunks(retrieved_chunks)
        
        # Add metadata about retrieval
        num_chunks = result.get('num_chunks', 0)
        answer_with_meta = f"{answer}\n\n*Retrieved {num_chunks} relevant source(s) from the complaint database.*"
        
        return answer_with_meta, sources
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, ""


def clear_conversation():
    """Clear the conversation interface."""
    return "", "", ""


def create_interface():
    """Create and launch the Gradio interface."""
    
    # Initialize pipeline
    init_status = initialize_pipeline()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .answer-box {
        min-height: 200px;
    }
    .source-box {
        font-size: 0.9em;
    }
    """
    
    with gr.Blocks(css=css, title="RAG Complaint Analysis System") as demo:
        # Header
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Complaint Analysis System
            **Ask questions about customer complaints and get AI-powered insights**
            
            This system uses Retrieval-Augmented Generation (RAG) to answer questions based on 
            customer complaint data. All answers are backed by source documents for verification.
            """,
            elem_classes=["main-header"]
        )
        
        # Status indicator
        status_box = gr.Textbox(
            value=init_status,
            label="System Status",
            interactive=False,
            visible=True
        )
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., What are the most common issues with credit cards?",
                    lines=3,
                    elem_id="question_input"
                )
                
                # Buttons
                with gr.Row():
                    submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                
                # Answer display
                answer_output = gr.Textbox(
                    label="AI-Generated Answer",
                    lines=8,
                    interactive=False,
                    elem_classes=["answer-box"]
                )
            
            with gr.Column(scale=1):
                # Source chunks display
                sources_output = gr.Markdown(
                    label="Source Documents",
                    elem_classes=["source-box"]
                )
        
        # Example questions
        gr.Markdown("### 💡 Example Questions")
        examples = gr.Examples(
            examples=[
                ["What are the most common issues with credit cards?"],
                ["What problems do customers face with personal loans?"],
                ["Find complaints about unauthorized charges or fraudulent transactions."],
                ["What are the main concerns about money transfer services?"],
                ["What are customers saying about interest rates and fees?"],
            ],
            inputs=question_input
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This system retrieves relevant complaint narratives from the database 
            and generates answers based on the retrieved context. Source documents are displayed 
            below each answer for verification and transparency.
            """
        )
        
        # Event handlers
        submit_btn.click(
            fn=query_rag,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        question_input.submit(
            fn=query_rag,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[question_input, answer_output, sources_output]
        )
    
    return demo


def main():
    """Main function to launch the application."""
    print("="*80)
    print("TASK 4: INTERACTIVE CHAT INTERFACE")
    print("="*80)
    print("\nStarting Gradio application...")
    print("The interface will open in your default web browser.")
    print("You can also access it at the URL shown in the terminal.")
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        show_error=True
    )


if __name__ == "__main__":
    main()

