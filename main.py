import os
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pdf_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db
from src.database.retriever import get_retriever
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.config import DB_DIR

def ensure_knowledge_base():
    """
    Checks if the Vector DB exists. If not, builds it from scratch.
    """
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logging.info(f"Knowledge Base found in {DB_DIR}. Skipping ingestion.")
        return

    logging.info("No Knowledge Base found. Starting Ingestion Pipeline...")
    
    documents = load_documents()
    if not documents:
        logging.error("No PDFs found in data/raw/. Please add a file.")
        sys.exit(1)
    
    chunks = chunk_documents(documents)
    create_vector_db(chunks)
    logging.info("Ingestion Complete. Database built.")

def format_docs_for_agent(docs):
    """
    Prepares retrieved documents for the LLM.
    """
    return "\n\n".join([f"Content: {d.page_content}\nSource: Page {d.metadata.get('page', 'Unknown')}" for d in docs])

def main():
    ensure_knowledge_base()
    retriever = get_retriever(k=4) 
    chat_history = [] 
    
    print("AI Tutor CLI - Hybrid RAG Active")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nStudent: ")
        
        if query.lower() in ["exit", "quit", "bye", "stop"]:
            print("Goodbye.")
            break

        if not query.strip():
            continue

        try:
            retrieved_docs = retriever.invoke(query)
            
            logging.info("Retrieved Sources (Hybrid Search):")
            for i, doc in enumerate(retrieved_docs[:3]):
                page = doc.metadata.get("page", "Unknown")
                topic = doc.metadata.get("topic", "General")
                logging.info(f"  {i+1}. [Page {page}] Topic: {topic}...")

            context_text = format_docs_for_agent(retrieved_docs)
            intent = route_query(query).strip().upper()
            
            if "QUIZ" in intent:
                logging.info(f"Generating Quiz on: {query}")
                response = generate_quiz(query, context_text)
            
            elif "CHAT" in intent:
                response = "Hello. I am your AI Tutor. Ask me anything about your Class 10 chapter."
            
            else:
                logging.info("Generating Explanation...")
                response = generate_explanation(query, context_text, chat_history)

            chat_history.append((query, response))
            if len(chat_history) > 3:
                chat_history.pop(0)

            print(f"\nAI Tutor ({intent}):")
            
            if isinstance(response, list):
                for i, q in enumerate(response):
                    print(f"\nQ{i+1}: {q['question']}")
                    for option in q['options']:
                        print(f"   {option}")
                    
                    print(f"   [Answer: {q['answer']} | Reason: {q['explanation']}]")
            else:
                print(f"\n{response}\n")
                
        except Exception as e:
            logging.error("Failed to execute query", exc_info=True)

if __name__ == "__main__":
    main()