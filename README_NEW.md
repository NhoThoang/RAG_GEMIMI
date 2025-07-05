# Book Reading Assistant API - Refactored Version

Hệ thống AI hỗ trợ đọc sách với khả năng upload PDF và trả lời câu hỏi thông minh.

## 🏗️ Cấu trúc Project

```
reading_book/
├── config/
│   └── settings.py          # Cấu hình từ biến môi trường
├── schemas/
│   ├── __init__.py
│   ├── input_schemas.py     # Schema cho input
│   └── output_schemas.py    # Schema cho output
├── models/
│   ├── __init__.py
│   ├── embedding_model.py   # Model embedding
│   ├── summarizer_model.py  # Model tóm tắt
│   └── gemini_model.py      # Model Gemini
├── database/
│   ├── __init__.py
│   ├── mongo_client.py      # MongoDB client
│   ├── qdrant_client.py     # Qdrant vector DB
│   └── redis_client.py      # Redis cache
├── utils/
│   ├── __init__.py
│   ├── text_utils.py        # Xử lý text
│   └── prompt_utils.py      # Xử lý prompt
├── services/
│   ├── __init__.py
│   ├── book_service.py      # Service xử lý sách
│   └── question_service.py  # Service xử lý câu hỏi
├── api/
│   ├── __init__.py
│   └── routes.py            # API routes
├── main_new.py              # Entry point mới
├── env.example              # Mẫu file .env
└── requirement.txt          # Dependencies
```

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirement.txt
```

### 2. Tạo file .env
Copy file `env.example` thành `.env` và cập nhật các giá trị:
```bash
cp env.example .env
```

Cập nhật file `.env`:
```env
# Database Configuration
MONGODB_URI=mongodb://admin:admin123@localhost:27017
MONGODB_DATABASE=books
MONGODB_COLLECTION=chunks

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=book_chunks

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-base
SUMMARIZER_MODEL=pengold/t5-vietnamese-summarization

# Application Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

### 3. Khởi động các service
- MongoDB
- Qdrant
- Redis

### 4. Chạy ứng dụng
```bash
python main_new.py
```

## 📚 API Endpoints

### Upload Book
```http
POST /api/v1/upload-book
Content-Type: multipart/form-data

file: [PDF file]
```

### Ask Question
```http
POST /api/v1/ask
Content-Type: application/json

{
  "question": "Nội dung chính của sách là gì?",
  "type": "new",
  "history": []
}
```

### Follow-up Question
```http
POST /api/v1/ask
Content-Type: application/json

{
  "question": "Chi tiết hơn về phần đó?",
  "type": "followup",
  "history": [
    {
      "question": "Nội dung chính của sách là gì?",
      "answer": "Sách nói về..."
    }
  ]
}
```

## 🔧 Tính năng

### ✅ Đã cải thiện
- **Cấu trúc modular**: Code được tổ chức thành các module riêng biệt
- **Bảo mật**: API keys và cấu hình DB được lưu trong file .env
- **Dễ mở rộng**: Cấu trúc cho phép dễ dàng thêm tính năng mới
- **Tách biệt concerns**: Models, services, database được tách riêng
- **Type safety**: Sử dụng Pydantic schemas cho input/output
- **Error handling**: Xử lý lỗi tốt hơn

### 🎯 Các module chính

#### Config
- Quản lý cấu hình từ biến môi trường
- Tự động load file .env

#### Models
- **EmbeddingModel**: Xử lý vector embedding
- **SummarizerModel**: Tóm tắt văn bản
- **GeminiModel**: Tạo câu trả lời

#### Database
- **MongoClient**: Lưu trữ chunks và metadata
- **QdrantClient**: Vector database cho semantic search
- **RedisClient**: Cache cho câu hỏi follow-up

#### Services
- **BookService**: Xử lý upload và process PDF
- **QuestionService**: Xử lý câu hỏi và trả lời

#### Utils
- **TextUtils**: Xử lý text, chunking, normalization
- **PromptUtils**: Tạo prompt và classify questions

## 🔄 Migration từ version cũ

1. Backup file `app.py` hiện tại
2. Sử dụng `main_new.py` thay vì `app.py`
3. Tạo file `.env` từ `env.example`
4. Cập nhật API key và cấu hình DB

## 📝 Ghi chú

- API endpoints đã được prefix với `/api/v1`
- Tất cả cấu hình được lưu trong file `.env`
- Code được tổ chức theo nguyên tắc SOLID
- Dễ dàng thêm middleware, authentication, logging 