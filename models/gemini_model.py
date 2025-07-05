import google.generativeai as genai
from config.settings import settings

class GeminiModel:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        print("▶ Gemini model configured")
    
    def generate_content(self, prompt):
        """Generate content using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            return "Không thể trả lời câu hỏi này." 