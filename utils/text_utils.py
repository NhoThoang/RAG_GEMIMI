import re
from sklearn.metrics.pairwise import cosine_similarity

class TextUtils:
    @staticmethod
    def split_sentences(text):
        """Split text into sentences"""
        return re.findall(r'[^.!?\n]+[.!?\n]', text)
    
    @staticmethod
    def normalize_question(text):
        """Normalize question text"""
        return re.sub(r"\s+", " ", text.strip().lower())
    
    @staticmethod
    def chunk_text(text, model, max_chunk_words=300, threshold=0.75):
        """Chunk text into smaller pieces based on semantic similarity"""
        sentences = TextUtils.split_sentences(text)
        chunks = []
        current_chunk = []
        current_len = 0
        prev_embedding = None

        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            current_chunk.append(sent)
            current_len += len(words)
            joined_chunk = " ".join(current_chunk)
            embedding = model.encode(joined_chunk)

            if prev_embedding is not None:
                sim = cosine_similarity([embedding], [prev_embedding])[0][0]
                if sim < threshold or current_len >= max_chunk_words:
                    chunks.append(joined_chunk)
                    current_chunk = []
                    current_len = 0
                    prev_embedding = None
                    continue

            prev_embedding = embedding

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks 