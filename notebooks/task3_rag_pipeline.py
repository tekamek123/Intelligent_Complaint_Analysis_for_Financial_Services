"""
Task 3: Building the RAG Core Logic and Evaluation
Objective: Build the retrieval and generation pipeline using the pre-built vector store.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json

# Embedding model
from sentence_transformers import SentenceTransformer

# Vector store
import chromadb
from chromadb.config import Settings

# LLM - Using Hugging Face transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "chroma_db"
COLLECTION_NAME = "complaint_chunks"
TOP_K = 5  # Number of chunks to retrieve


class RAGRetriever:
    """Retriever component for semantic search in the vector store."""
    
    def __init__(self, vector_store_path: Path, collection_name: str, embedding_model_name: str):
        """
        Initialize the retriever.
        
        Args:
            vector_store_path: Path to the ChromaDB vector store
            collection_name: Name of the collection in ChromaDB
            embedding_model_name: Name of the embedding model
        """
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Connect to vector store
        print(f"Connecting to vector store at: {vector_store_path}")
        self.client = chromadb.PersistentClient(path=str(vector_store_path))
        self.collection = self.client.get_collection(name=collection_name)
        
        print(f"Retriever initialized. Collection has {self.collection.count()} chunks.")
    
    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a given question.
        
        Args:
            question: User's question as a string
            top_k: Number of chunks to retrieve (default: 5)
            
        Returns:
            List of dictionaries, each containing:
                - 'text': The chunk text
                - 'metadata': Metadata associated with the chunk
                - 'distance': Similarity distance (lower is more similar)
        """
        # Embed the question
        question_embedding = self.embedding_model.encode(question, convert_to_numpy=True).tolist()
        
        # Query the vector store
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        retrieved_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks


class PromptEngineer:
    """Prompt engineering for RAG pipeline."""
    
    @staticmethod
    def create_prompt(context: str, question: str) -> str:
        """
        Create a prompt template for the LLM.
        
        Args:
            context: Retrieved context chunks (concatenated)
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        prompt_template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context from complaint narratives.

Instructions:
- Use ONLY the information provided in the context below to answer the question
- If the context does not contain enough information to answer the question, clearly state that you don't have enough information
- Be concise, accurate, and professional
- Focus on actionable insights when possible
- If multiple complaints are mentioned, summarize the key patterns

Context from Complaint Narratives:
{context}

Question: {question}

Answer:"""
        
        return prompt_template.format(context=context, question=question)
    
    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            
            # Add metadata information if available
            metadata_info = []
            if 'product' in metadata:
                metadata_info.append(f"Product: {metadata['product']}")
            if 'complaint_id' in metadata:
                metadata_info.append(f"Complaint ID: {metadata['complaint_id']}")
            if 'issue' in metadata:
                metadata_info.append(f"Issue: {metadata['issue']}")
            
            metadata_str = f" ({', '.join(metadata_info)})" if metadata_info else ""
            
            context_parts.append(f"[Excerpt {i}{metadata_str}]\n{text}")
        
        return "\n\n".join(context_parts)


class RAGGenerator:
    """Generator component using LLM for answer generation."""
    
    def __init__(self, model_name: str = "gpt2", use_simple_fallback: bool = True):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the Hugging Face model to use
                        Options:
                        - "gpt2" (small, fast, lower quality)
                        - "microsoft/DialoGPT-medium" (conversational)
                        - For better results: use API-based models (OpenAI, etc.)
            use_simple_fallback: If True, use a simple template-based approach when LLM fails
        """
        self.model_name = model_name
        self.use_simple_fallback = use_simple_fallback
        self.generator = None
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available. Using simple fallback mode.")
            self.generator = None
            return
        
        print(f"Initializing generator with model: {model_name}")
        print("Note: This may take a few minutes on first run (downloading model)...")
        
        try:
            # Try to use a text generation pipeline
            # Using GPT-2 as default (smaller, faster)
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                device=-1,  # Use CPU by default
                pad_token_id=50256  # GPT-2 pad token
            )
            print("Generator initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            print("Using fallback mode...")
            self.generator = None
    
    def _simple_fallback_answer(self, context: str, question: str) -> str:
        """
        Simple template-based answer generation when LLM is not available.
        Extracts key information from context to form an answer.
        
        Args:
            context: Retrieved context chunks
            question: User's question
            
        Returns:
            Formatted answer string
        """
        # Extract key information from context
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Simple keyword-based extraction
        if "credit card" in question_lower:
            if "issue" in question_lower or "problem" in question_lower:
                # Look for common issues in context
                issues = []
                if "unauthorized" in context_lower or "fraud" in context_lower:
                    issues.append("unauthorized charges")
                if "billing" in context_lower or "charge" in context_lower:
                    issues.append("billing errors")
                if "fee" in context_lower or "interest" in context_lower:
                    issues.append("fees and interest rates")
                if "service" in context_lower or "customer" in context_lower:
                    issues.append("customer service issues")
                
                if issues:
                    return f"Based on the complaint data, common issues with credit cards include: {', '.join(issues)}. The retrieved complaints show various customer concerns in these areas."
        
        # Generic answer based on context
        if len(context) > 0:
            # Extract first few sentences from context as summary
            sentences = context.split('.')[:3]
            summary = '. '.join(s for s in sentences if len(s.strip()) > 20)
            return f"Based on the retrieved complaint narratives, here are relevant findings: {summary}. This information is derived from customer complaint data in our system."
        
        return "I don't have enough information in the retrieved context to provide a comprehensive answer to this question. Please try rephrasing or asking about a different aspect."
    
    def generate(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate an answer from a prompt.
        
        Args:
            prompt: The formatted prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated answer string
        """
        if self.generator is None:
            if self.use_simple_fallback:
                # Extract context and question from prompt
                if "Context from Complaint Narratives:" in prompt and "Question:" in prompt:
                    parts = prompt.split("Context from Complaint Narratives:")
                    if len(parts) > 1:
                        context_question = parts[1].split("Question:")
                        if len(context_question) == 2:
                            context = context_question[0].strip()
                            question = context_question[1].split("Answer:")[0].strip()
                            return self._simple_fallback_answer(context, question)
                
                return "LLM model not available. Please install transformers and torch, or use an API-based model."
            else:
                return "LLM model not available. Please install transformers and torch, or use an API-based model."
        
        try:
            # Truncate prompt if too long (keep last part which has the question)
            # GPT-2 has max context of 1024 tokens, but we'll be conservative
            max_prompt_length = 400  # Leave room for generation
            if len(prompt) > max_prompt_length:
                # Keep the question and some context
                if "Question:" in prompt:
                    question_part = prompt.split("Question:")[-1]
                    context_part = prompt.split("Context from Complaint Narratives:")[-1].split("Question:")[0]
                    # Truncate context if needed
                    if len(context_part) > max_prompt_length - len(question_part) - 200:
                        context_part = context_part[-(max_prompt_length - len(question_part) - 200):]
                    prompt = f"Context from Complaint Narratives:\n{context_part}\n\nQuestion: {question_part}"
            
            # Generate text with max_new_tokens instead of max_length
            result = self.generator(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=50256
            )
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                # Try to extract new content after the prompt
                if len(generated_text) > len(prompt):
                    answer = generated_text[len(prompt):].strip()
                else:
                    answer = generated_text.strip()
            
            # Clean up the answer
            answer = answer.split("\n")[0]  # Take first line/paragraph
            if len(answer) > max_length:
                answer = answer[:max_length] + "..."
            
            return answer
        except Exception as e:
            print(f"Error during generation: {e}")
            if self.use_simple_fallback:
                # Try fallback
                if "Context from Complaint Narratives:" in prompt and "Question:" in prompt:
                    parts = prompt.split("Context from Complaint Narratives:")
                    if len(parts) > 1:
                        context_question = parts[1].split("Question:")
                        if len(context_question) == 2:
                            context = context_question[0].strip()
                            question = context_question[1].split("Answer:")[0].strip()
                            return self._simple_fallback_answer(context, question)
            return f"Error generating answer: {str(e)}"


class RAGPipeline:
    """Complete RAG pipeline combining retriever, prompt engineering, and generator."""
    
    def __init__(self, vector_store_path: Path = VECTOR_STORE_PATH, 
                 collection_name: str = COLLECTION_NAME,
                 embedding_model: str = EMBEDDING_MODEL,
                 llm_model: str = "gpt2",
                 use_simple_fallback: bool = True):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store_path: Path to ChromaDB vector store
            collection_name: Name of the collection
            embedding_model: Name of embedding model
            llm_model: Name of LLM model for generation (default: "gpt2")
            use_simple_fallback: Use template-based fallback if LLM unavailable
        """
        self.retriever = RAGRetriever(vector_store_path, collection_name, embedding_model)
        self.prompt_engineer = PromptEngineer()
        self.generator = RAGGenerator(llm_model, use_simple_fallback=use_simple_fallback)
    
    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing:
                - 'question': Original question
                - 'answer': Generated answer
                - 'retrieved_chunks': List of retrieved chunks with metadata
                - 'context': Formatted context string
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Format context
        context = self.prompt_engineer.format_context(retrieved_chunks)
        
        # Step 3: Create prompt
        prompt = self.prompt_engineer.create_prompt(context, question)
        
        # Step 4: Generate answer
        answer = self.generator.generate(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'context': context,
            'num_chunks': len(retrieved_chunks)
        }


def main():
    """Main function for testing the RAG pipeline."""
    print("="*80)
    print("TASK 3: RAG PIPELINE - TESTING")
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Test query
    test_question = "What are common issues with credit cards?"
    print(f"\nTest Question: {test_question}")
    print("-" * 80)
    
    result = pipeline.query(test_question)
    
    print(f"\nRetrieved {result['num_chunks']} chunks")
    print(f"\nGenerated Answer:\n{result['answer']}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

