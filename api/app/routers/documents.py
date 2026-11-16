from fastapi import APIRouter, UploadFile

from app.errors import DocumentNotFoundError
from app.routers.deps import RuntimeDep
from app.schemas import UploadResponse, error_responses
from app.services import ingestion

router = APIRouter(tags=["documents"])


@router.post(
    "/api/documents",
    summary="Upload a document",
    responses=error_responses(400, 413, 415, 422, 429, 500),
)
async def upload_document(file: UploadFile, runtime: RuntimeDep) -> UploadResponse:
    """Indexes a `.md`, `.txt` or `.pdf` file (up to 1 MB) into a private collection that expires
    after 24 hours, and describes it (summary, language, suggested questions)."""
    data = await file.read(ingestion.MAX_BYTES + 1)  # never read more than the limit
    upload = await ingestion.ingest(runtime, file.filename or "document.txt", data)
    return UploadResponse.model_validate(upload)


@router.get(
    "/api/documents/{collection}",
    summary="Get an uploaded document",
    responses=error_responses(404, 500),
)
async def get_document(collection: str, runtime: RuntimeDep) -> UploadResponse:
    """The metadata of an upload, until it expires."""
    upload = (
        await ingestion.get_upload(runtime, collection)
        if ingestion.is_upload_collection(collection)
        else None
    )
    if upload is None:
        raise DocumentNotFoundError()
    return UploadResponse.model_validate(upload)
