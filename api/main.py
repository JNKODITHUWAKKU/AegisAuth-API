import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import enroll, verify

app = FastAPI(
    title="AegisAuth API",
    description="Industry-level API for Deep Learning based Liveness Detection and Identity Verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(enroll.router, tags=["Enrollment"])
app.include_router(verify.router, tags=["Verification"])

if __name__ == "__main__":
    import uvicorn
    print("Starting AegisAuth API server...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
