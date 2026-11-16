from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError

from app.domain import MONDRIA
from app.errors import DocumentNotFoundError
from app.routers.deps import RuntimeDep
from app.schemas import AskRequest, AskResponse, error_responses
from app.services import ingestion, rag

router = APIRouter(tags=["ask"])


@router.post(
    "/api/ask",
    summary="Ask a question",
    responses=error_responses(404, 422, 429, 500),
)
async def ask(request: AskRequest, runtime: RuntimeDep) -> AskResponse:
    """Retrieves the most relevant sections with the chosen strategy and answers **only** from
    them, citing their IDs, in the language of the question. Works on the Mondria knowledge base
    or on a document uploaded with `POST /api/documents`."""
    question = request.question.strip()
    if not question:
        raise _empty_question()
    if request.collection != MONDRIA and await ingestion.get_upload(runtime, request.collection) is None:
        raise DocumentNotFoundError()
    answer = await rag.answer_question(runtime, question, request.strategy, request.collection)
    return AskResponse.model_validate(answer)


def _empty_question() -> RequestValidationError:
    """Whitespace-only questions pass the length check; reject them like any validation error."""
    return RequestValidationError(
        [{"type": "string_too_short", "loc": ("body", "question"), "msg": "Empty question", "input": ""}]
    )
