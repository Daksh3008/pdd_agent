import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import query, video
from config import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings()
settings.configure_env()

app = FastAPI(
    title="VectorDB Frame Service",
    description="Upload videos, extract & index frames, and query them by text.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted frame images as static files
data_dir = os.path.abspath(settings.data_dir)
os.makedirs(data_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=data_dir), name="static")

# Register routers
app.include_router(video.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
