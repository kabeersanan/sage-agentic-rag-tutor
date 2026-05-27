import os
import sys
import time

# Add the project root to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.database.vector_store import load_vector_db

def evaluate_retrieval(test_queries):
    """
    Runs a set of queries against the Vector Database and measures:
    1. Retrieval Speed (Latency)
    2. Average Confidence Score (Normalized for L2 Distance)
    """
    print("Starting Retrieval Evaluation System...\n")
    
    try:
        db = load_vector_db()
    except Exception as e:
        print(f"Error: Could not load database. Run main.py first. Details: {e}")
        return

    total_score = 0
    total_time = 0
    
    print(f"{'QUERY':<50} | {'CONFIDENCE':<12} | {'SOURCE (Page)':<20}")

    for query in test_queries:
        start_time = time.time()
        
        # Get top 3 results with scores
        # Note: ChromaDB returns L2 DISTANCE by default (Lower is Better)
        results = db.similarity_search_with_score(query, k=3)
        
        end_time = time.time()
        latency = end_time - start_time
        total_time += latency

        if not results:
            print(f"{query[:47]:<50} ... | 0.0%         | NO RESULTS")
            continue

        best_doc, best_score = results[0]
        
        # Normalize L2 distance to a 0-100 percentage score.
        confidence = round((1 / (1 + best_score)) * 100, 2)
        
        # Get metadata
        source = best_doc.metadata.get('source', 'Unknown').split('/')[-1]
        page = best_doc.metadata.get('page', '??')

        print(f"{query[:47]:<50} ... | {confidence}%      | {source} (p.{page})")
        
        total_score += confidence

    # Summary Metrics
    avg_score = round(total_score / len(test_queries), 2)
    avg_latency = round(total_time / len(test_queries), 4)

    print("\nFINAL PERFORMANCE METRICS")
    print(f"Average Confidence Score: {avg_score}/100")
    print(f"Average Retrieval Time:   {avg_latency} seconds")

    # Interpretation
    if avg_score > 50:
        print("RATING: EXCELLENT (High relevance)")
    elif avg_score > 30:
        print("RATING: GOOD (Acceptable relevance)")
    else:
        print("RATING: POOR (Check chunk size or embeddings)")

if __name__ == "__main__":
    # Test Queries tailored to your specific content
    sample_queries = [
        "When was the pH scale introduced?", 
        "What is the meaning of water of crystallization?", 
        "Who proposed the Arrhenius theory of acids and bases?", 
        "Why does distilled water not conduct electricity?", 
        "Describe the Chlor-alkali process.", 
        "What did the reaction of metal carbonates with acids produce?"
    ]
    
    evaluate_retrieval(sample_queries)