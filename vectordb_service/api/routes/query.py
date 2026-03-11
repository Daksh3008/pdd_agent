from fastapi import APIRouter, Depends

from api.dependencies import get_query_service
from domain.models import QueryRequest, QueryResponse, FrameMatch
from services.query_service import QueryService

router = APIRouter(prefix="/api/frames", tags=["frames"])


@router.post("/query", response_model=QueryResponse)
def query_frames(
    body: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """Query indexed frames by natural-language text. Returns matching
    frames ranked by cosine similarity."""

    result = query_service.query_frames(body.query, body.top_k, body.video_id)

    return QueryResponse(
        query=result["query"],
        results=[FrameMatch(**m) for m in result["results"]],
    )
