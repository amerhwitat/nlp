"""Artifact-enabled web entry point.

Run with:
    uvicorn ThamudicScan.server.artifact_app:app --reload

It reuses the existing scanner API and adds the unified artifact evidence API.
"""
from .main import create_app
from .artifact_api import router as artifact_router

app = create_app()
app.include_router(artifact_router)
