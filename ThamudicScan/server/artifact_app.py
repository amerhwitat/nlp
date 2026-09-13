"""Artifact-enabled web entry point.

Run with:
    uvicorn ThamudicScan.server.artifact_app:app --reload

It reuses the existing scanner API and adds the unified artifact evidence APIs.
"""
from .main import create_app
from .artifact_api import router as artifact_router
from .artifact_features_api import router as artifact_features_router

app = create_app()
app.include_router(artifact_router)
app.include_router(artifact_features_router)
