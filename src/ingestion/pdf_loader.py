import os
from langchain_community.document_loaders import PyPDFLoader
from src.config import DATA_DIR

def load_single_document(file_path):
    """
    Loads a single PDF file. Used when a user uploads one new chapter so we
    ingest only that file instead of re-processing the whole data directory
    (which would create duplicate vectors for already-ingested PDFs).

    Returns:
        List[Document]: One Document per page, or [] on failure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        print(f" - Loading single file: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} pages.")
        return docs
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_documents():
    """
    Scans the configured data directory and loads all PDF documents.
    
    Returns:
        List[Document]: A list of LangChain Document objects, where each object 
                        represents a page from a PDF with metadata (page number, source).
    """
    documents = []
    
    # 1. Validation: Ensure the directory exists to avoid cryptic errors later
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"The directory {DATA_DIR} does not exist. Please create it and add your PDFs.")

    # 2. Iteration: robustly find only PDF files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    if not files:
        print(f"Warning: No PDF files found in {DATA_DIR}. Please add the NCERT chapter.")
        return []

    print(f"Found {len(files)} PDF(s) in {DATA_DIR}...")

    # 3. Loading: Process files one by one
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            print(f" - Loading: {filename}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"Successfully loaded {len(documents)} pages in total.")
    return documents