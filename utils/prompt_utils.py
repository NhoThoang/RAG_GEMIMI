class PromptUtils:
    PROMPT_TEMPLATES = {
        "main_topic": "Tóm tắt nội dung chính của tài liệu sau:\n{context}",
        "year_of_event": "Dựa vào đoạn văn sau, cho biết năm diễn ra sự kiện liên quan:\n{context}\n\nCâu hỏi: {question}",
        "number_question": "Đếm và cho biết số lượng theo yêu cầu sau:\n{context}\n\nCâu hỏi: {question}",
        "who_is": "Cho biết thông tin về nhân vật hoặc sự kiện được hỏi:\n{context}\n\nCâu hỏi: {question}",
        "compare": "So sánh nội dung theo yêu cầu sau:\n{context}\n\nCâu hỏi: {question}",
        "why_reason": "Giải thích lý do theo câu hỏi sau:\n{context}\n\nCâu hỏi: {question}",
        "how_process": "Mô tả quá trình, cách làm hoặc diễn biến:\n{context}\n\nCâu hỏi: {question}",
        "default": "Trả lời câu hỏi sau dựa trên đoạn văn:\n{context}\n\nCâu hỏi: {question}"
    }
    
    @staticmethod
    def classify_question(question: str) -> str:
        """Classify question type"""
        q = question.lower()
        if "nội dung chính" in q or "tóm tắt" in q:
            return "main_topic"
        if "năm" in q:
            return "year_of_event"
        if "bao nhiêu" in q or "số lượng" in q:
            return "number_question"
        if "ai là" in q or q.startswith("ai "):
            return "who_is"
        if "khác" in q or "so sánh" in q:
            return "compare"
        if "tại sao" in q or "vì sao" in q:
            return "why_reason"
        if "như thế nào" in q or "làm sao" in q:
            return "how_process"
        return "default"
    
    @staticmethod
    def build_prompt(question: str, context: str, intent: str = None) -> str:
        """Build prompt for question answering"""
        if intent is None:
            intent = PromptUtils.classify_question(question)
        
        template = PromptUtils.PROMPT_TEMPLATES.get(intent, PromptUtils.PROMPT_TEMPLATES["default"])
        return template.format(context=context, question=question)
    
    @staticmethod
    def build_followup_prompt(question: str, context: str, history: list) -> str:
        """Build prompt for followup questions"""
        history_text = "\n\n".join(
            f"Hỏi: {h['question']}\nĐáp: {h['answer']}" for h in history
        )
        
        return f"""Lịch sử hội thoại:
{history_text}

Văn bản tham khảo:
{context}

Câu hỏi hiện tại: {question}
""" 