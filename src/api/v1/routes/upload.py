from fastapi import APIRouter, UploadFile, File,  HTTPException
import shutil
import os
from src.ingestion.ingestion import run_ingestion

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/admin/upload")
async def upload_file(
    file: UploadFile = File(...),
):
    
   
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are allowed"
        )

   
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   
    try:
        result = run_ingestion(file_path=file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {
        "status": result.get("status", "Failed"),
        "message": "File uploaded and ingested successfully",
        "file_name": file.filename,
        "doc_id": result.get("doc_id", "Null"),
        "chunks_created": result.get("chunks_ingested", 0)
    }