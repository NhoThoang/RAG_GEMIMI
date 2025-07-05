from qdrant_client import QdrantClient as PyQdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from config.settings import settings

class QdrantClient:
    def __init__(self):
        self.client = PyQdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()
        print("▶ Qdrant connected")
    
    def _ensure_collection_exists(self):
        """Ensure collection exists, create if not"""
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print(f"▶ Created collection: {self.collection_name}")
    
    def search(self, query_vector, limit=5):
        """Search for similar vectors"""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
    
    def upsert(self, points):
        """Upsert points to collection"""
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def delete(self, points_selector):
        """Delete points from collection"""
        return self.client.delete(
            collection_name=self.collection_name,
            points_selector=points_selector
        ) 