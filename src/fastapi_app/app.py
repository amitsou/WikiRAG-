""" FastAPI application for handling user queries. """

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.fastapi_app.core.huggingface_api_client import HuggingFaceAPIClient
from src.fastapi_app.core.models import QueryRequest, QueryResponse
from src.fastapi_app.core.qdrant_manager import QdrantManager
from src.utils.general_utils import load_config

# Load configuration
ROOT_DIR = Path(__file__).resolve().parents[2]
config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
config = load_config(config_path)

llm_provider = config.get("llm_provider", "huggingface").lower()

if llm_provider == "huggingface":
    client = HuggingFaceAPIClient()
else:
    raise ValueError(f"Invalid LLM provider in config: {llm_provider}")

qdrant_manager = QdrantManager()

app = FastAPI(title="RAG API", version="1.0")


@app.post("/query", response_model=QueryResponse)
async def query_api(request: QueryRequest):
    """
    Handles user queries by retrieving context from Qdrant and calling the selected LLM.
    """
    try:
        context_enriched_query = qdrant_manager.search_with_context(request.query)
        llm_response = client.generate_response(context_enriched_query)
        return QueryResponse(response=llm_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


app.mount("/static", StaticFiles(directory="src/fastapi_app/static"), name="static")


@app.get("/")
async def serve_ui():
    """Serves the chat interface."""
    return FileResponse("src/fastapi_app/static/user_interface.html")
