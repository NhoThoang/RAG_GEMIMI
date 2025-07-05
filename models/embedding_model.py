import torch
from sentence_transformers import SentenceTransformer
from config.settings import settings

class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=self.device)
        print(f"▶ Embedding model loaded on device: {self.device}")
    
    def encode(self, text):
        """Encode text to vector"""
        return self.model.encode(text)
    
    def encode_batch(self, texts):
        """Encode multiple texts to vectors"""
        return self.model.encode(texts) 