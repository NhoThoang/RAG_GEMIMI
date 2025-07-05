from fastapi import APIRouter, File, UploadFile, Body
from fastapi.responses import JSONResponse

from schemas.input_schemas import AskRequest
from schemas.output_schemas import AskResponse, UploadResponse
from services.book_service import BookService
from services.question_service import QuestionService

router = APIRouter()

# Initialize services
book_service = BookService()
question_service = QuestionService()

@router.post("/upload-book", response_model=UploadResponse)
async def upload_book(file: UploadFile = File(...)):
    """Upload and process PDF book"""
    if not file.filename.endswith(".pdf"):
        return JSONResponse(
            content={"error": "Only PDF files allowed"}, 
            status_code=400
        )
    
    try:
        file_content = await file.read()
        result = await book_service.upload_book(file_content, file.filename)
        return UploadResponse(**result)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Upload failed: {str(e)}"}, 
            status_code=500
        )

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest = Body(...)):
    """Ask question about uploaded books"""
    try:
        result = await question_service.ask_question(
            question=request.question,
            question_type=request.type,
            history=request.history
        )
        return AskResponse(**result)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Question processing failed: {str(e)}"}, 
            status_code=500
        ) 