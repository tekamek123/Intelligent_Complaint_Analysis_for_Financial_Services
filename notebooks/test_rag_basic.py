"""Quick test of RAG pipeline before full evaluation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from task3_rag_pipeline import RAGPipeline

def test_basic():
    print("Testing RAG Pipeline...")
    print("="*80)
    
    try:
        pipeline = RAGPipeline()
        print("\n[OK] Pipeline initialized successfully")
        
        # Test query
        test_question = "What are common issues with credit cards?"
        print(f"\nTest Question: {test_question}")
        print("-" * 80)
        
        result = pipeline.query(test_question, top_k=3)
        
        print(f"\n✓ Retrieved {result['num_chunks']} chunks")
        print(f"\nGenerated Answer:")
        print(result['answer'])
        print("\n" + "="*80)
        print("✓ Basic test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic()

