import os
import re
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove page markers
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)  # 3/25 style page markers
    
    # Remove common PDF artifacts
    text = re.sub(r'\f', ' ', text)  # form feed characters
    text = re.sub(r'\.{3,}', '...', text)  # multiple dots
    
    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_documents(docs):
    """Preprocess document contents"""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        # Ensure metadata has source information
        if 'source' not in doc.metadata:
            doc.metadata['source'] = 'unknown'
    return docs

def load_all_pdfs(data_dir: str):
    """Load all PDF files from the data directory"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pdf_names = [n for n in os.listdir(data_dir) if n.lower().endswith(".pdf")]
    if not pdf_names:
        raise FileNotFoundError(f"No PDFs found in: {data_dir}")

    print(f"Found {len(pdf_names)} PDF files:")
    for name in pdf_names:
        print(f"  - {name}")

    docs = []
    for name in pdf_names:
        path = os.path.join(data_dir, name)
        try:
            print(f"Loading {name}...")
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            print(f"  Loaded {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Warning: failed to load {name}: {e}")
    
    return docs

def load_and_preprocess_pdfs(data_dir: str):
    """Main function to load and preprocess all PDFs"""
    print("Loading PDF documents...")
    docs = load_all_pdfs(data_dir)
    print(f"Total pages loaded: {len(docs)}")
    
    print("Preprocessing documents...")
    docs = preprocess_documents(docs)
    print("Preprocessing completed")
    
    return docs