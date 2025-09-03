import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np

class QdrantDB:
    def __init__(self, url: str, api_key: str, collection_name: str):
        """
        Initialize Qdrant client for cloud-based vector storage
        
        Args:
            url: Qdrant cloud cluster URL
            api_key: Qdrant API key
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = 768  # MPNet base model dimension
        
        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created successfully")
            else:
                print(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise

    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return self.collection_name in [col.name for col in collections.collections]
        except Exception:
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"points_count": 0, "vectors_count": 0, "status": "error"}

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        """
        Add documents to the Qdrant collection
        
        Args:
            texts: List of text chunks
            embeddings: Numpy array of embeddings
            metadatas: List of metadata dictionaries
        """
        points = []
        
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            point_id = str(uuid.uuid4())
            
            # Prepare payload
            payload = {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                **metadata  # Include all metadata
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error uploading batch: {e}")
                raise

        print(f"Successfully added {len(points)} points to collection")

    def search(self, query_embedding: np.ndarray, top_k: int = 6) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector as numpy array
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Ensure query_embedding is the right format
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim == 2:
                    query_vector = query_embedding[0].tolist()
                else:
                    query_vector = query_embedding.tolist()
            else:
                query_vector = query_embedding

            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            hits = []
            for rank, hit in enumerate(search_result, 1):
                # Convert cosine similarity to a percentage (0-100%)
                # Qdrant cosine similarity ranges from -1 to 1, normalize to 0-100%
                relevance_score = max(0, (float(hit.score) + 1) / 2 * 100)
                
                hits.append({
                    "rank": rank,
                    "score": relevance_score,
                    "text": hit.payload.get("text", ""),
                    "payload": hit.payload
                })

            return hits

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def clear_collection(self):
        """Clear all points from the collection"""
        try:
            # Delete collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            print(f"Collection '{self.collection_name}' cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {e}")