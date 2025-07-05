from models.embedding_model import EmbeddingModel
from models.gemini_model import GeminiModel
from database.mongo_client import MongoClient
from database.qdrant_client import QdrantClient
from database.redis_client import RedisClient
from utils.text_utils import TextUtils
from utils.prompt_utils import PromptUtils

class QuestionService:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.gemini_model = GeminiModel()
        self.mongo_client = MongoClient()
        self.qdrant_client = QdrantClient()
        self.redis_client = RedisClient()
    
    async def ask_question(self, question: str, question_type: str, history: list = None):
        """Process question and generate answer"""
        q = question.strip()
        norm_q = TextUtils.normalize_question(q)
        
        # Check cache for followup questions
        if question_type != "new":
            cached = self.redis_client.get_json(f"q:{norm_q}")
            if cached:
                print("✅ Cache hit")
                return {
                    "question": q,
                    "generated_answer": cached["answer"],
                    "answers": cached["sources"]
                }
        
        # Search in vector database
        question_embedding = self.embedding_model.encode(q).tolist()
        search_result = self.qdrant_client.search(question_embedding, limit=5)
        
        # Get documents from MongoDB
        ids = [point.id for point in search_result]
        docs = {doc["_id"]: doc for doc in self.mongo_client.find({"_id": {"$in": ids}})}
        chunks_text = [docs.get(hit.id, {}).get("chunk_text", "") for hit in search_result]
        
        # Generate answer
        if question_type == "followup" and history:
            prompt = PromptUtils.build_followup_prompt(q, "\n\n".join(chunks_text), history)
            final_answer = self.gemini_model.generate_content(prompt)
        else:
            context = "\n\n".join(chunks_text)
            prompt = PromptUtils.build_prompt(q, context)
            final_answer = self.gemini_model.generate_content(prompt)
        
        # Prepare source information
        source_info = [
            {
                "score": float(hit.score),
                "summary": docs.get(hit.id, {}).get("summary", ""),
                "text": docs.get(hit.id, {}).get("chunk_text", ""),
                "source_file": docs.get(hit.id, {}).get("source_file", ""),
                "chunk_index": docs.get(hit.id, {}).get("chunk_index", -1),
                "page": docs.get(hit.id, {}).get("page", -1)
            }
            for hit in search_result
        ]
        
        # Cache for followup questions
        if question_type == "followup":
            self.redis_client.setex(
                f"q:{norm_q}",
                60 * 60 * 6,  # 6 hours
                {
                    "answer": final_answer,
                    "sources": source_info
                }
            )
        
        return {
            "question": q,
            "generated_answer": final_answer,
            "answers": source_info
        } 