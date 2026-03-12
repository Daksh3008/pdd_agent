from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_query_service
from domain.models import (
    FrameMatch,
    LocateRequest,
    LocateResponse,
    QueryRequest,
    QueryResponse,
)
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


@router.post("/query/locate", response_model=LocateResponse)
def query_and_locate(
    body: LocateRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """Full pipeline: vector search → Gemini frame selection → OmniParser → bbox annotation."""

    result = query_service.query_and_locate(body.query, body.top_k, body.video_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return LocateResponse(
        query=result["query"],
        selected_frame=FrameMatch(**result["selected_frame"]),
        selection_reason=result["selection_reason"],
        omniparser_elements_count=result["omniparser_elements_count"],
        located_element=result["located_element"],
        bbox=result["bbox"],
        bbox_reason=result["bbox_reason"],
        annotated_image_url=result["annotated_image_url"],
    )
