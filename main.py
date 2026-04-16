from fastapi import FastAPI
from src.api.v1.routes.query import router as query_router
from src.api.v1.routes.upload import router as upload_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])

@app.get("/")
def root():
    return {
        "status":"MULTI MODEL RAG SYSTEM"
    }

@app.get("/health")
def health_check():
      return {
        "status":"working fine"
    }
   
