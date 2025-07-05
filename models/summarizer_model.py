import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from config.settings import settings

class SummarizerModel:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(settings.SUMMARIZER_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.SUMMARIZER_MODEL)
        self.pipeline = pipeline(
            "text2text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=self.device
        )
        print(f"▶ Summarizer model loaded on device: {self.device}")
    
    def summarize(self, text, max_new_tokens=100):
        """Summarize text"""
        try:
            result = self.pipeline(text[:1024], max_new_tokens=max_new_tokens, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:300] 