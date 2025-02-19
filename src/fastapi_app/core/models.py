""" Core models for the FastAPI app. """

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """
    Request model for the /query endpoint.
    """

    query: str


class QueryResponse(BaseModel):
    """
    Response model for the /query endpoint.
    """

    response: str
