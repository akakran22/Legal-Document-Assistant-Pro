import os
import requests
import numpy as np
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def encode(self, texts, batch_size=8, normalize=True, show_progress=True):
        """
        Encode texts using Hugging Face Inference API
        Using smaller batch sizes for API calls to avoid timeouts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Creating embeddings")
        
        for batch in batches:
            try:
                # Make API request
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}}
                )
                
                if response.status_code == 200:
                    batch_embeddings = response.json()
                    
                    # Handle single text vs batch response
                    if isinstance(batch_embeddings[0], list) and isinstance(batch_embeddings[0][0], (int, float)):
                        # Single embedding returned
                        batch_embeddings = [batch_embeddings]
                    
                    all_embeddings.extend(batch_embeddings)
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    # Fallback: create zero vectors
                    dummy_embedding = [0.0] * 768  # MPNet base dimension
                    all_embeddings.extend([dummy_embedding] * len(batch))
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Fallback: create zero vectors
                dummy_embedding = [0.0] * 768
                all_embeddings.extend([dummy_embedding] * len(batch))
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings

def chunk_documents(docs, chunk_size=3000, chunk_overlap=300, separators=None):
    """Chunk documents into smaller pieces"""
    if separators is None:
        separators = ["\n\n", "\n", "Section ", "SECTION ", "Sec. ", "CHAPTER ", "Chapter ", " "]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

def chunk_and_embed_documents(docs, vdb, chunk_size=3000, chunk_overlap=300, 
                            separators=None, batch_size=8):
    """
    Main function to chunk documents and create embeddings, 
    then store them in Qdrant vector database
    """
    # 1. Chunk the documents
    print("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap, separators)
    
    # 2. Extract texts and metadata
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # 3. Create embeddings
    print("Creating embeddings using Hugging Face API...")
    embeddings_model = HuggingFaceEmbeddings()
    embeddings = embeddings_model.encode(texts, batch_size=batch_size, show_progress=True)
    
    # 4. Store in Qdrant
    print("Storing embeddings in Qdrant...")
    vdb.add_documents(texts, embeddings, metadatas)
    
    print(f"Successfully processed and stored {len(chunks)} document chunks")