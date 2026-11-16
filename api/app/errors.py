"""Domain errors and the single error format of the REST API.

Every REST error body looks like:
    {"detail": "<message>", "error": {"code": "...", "message": "...", "request_id": "..."}}
`detail` stays for compatibility with clients written against FastAPI's default format.
The MCP endpoint (JSON-RPC) and the Vercel Firewall (429) keep their own formats.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    default_message = "Internal server error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"
    default_message = "Document not found or expired"


class InvalidDocumentError(AppError):
    status_code = 400
    code = "invalid_document"
    default_message = "The document could not be processed."


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"
    default_message = "Le fichier dépasse 1 Mo."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"
    default_message = "Formats acceptés : .md, .txt et .pdf."


class StrategyNotAvailableError(AppError):
    status_code = 422
    code = "strategy_not_available"
    default_message = "This strategy is not available in the LangChain pipeline."


def error_body(code: str, message: str, request_id: str | None, **extra: Any) -> dict[str, Any]:
    return {"detail": message, "error": {"code": code, "message": message, "request_id": request_id, **extra}}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _app_error_handler(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, AppError)  # noqa: S101 - registered only for AppError
    return JSONResponse(
        error_body(error.code, error.message, _request_id(request)), status_code=error.status_code
    )


async def _validation_error_handler(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)  # noqa: S101 - registered only for this type
    fields = [
        {"location": ".".join(map(str, item["loc"])), "message": item["msg"]} for item in error.errors()
    ]
    return JSONResponse(
        error_body("validation_error", "Invalid request", _request_id(request), fields=fields),
        status_code=422,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Unexpected exceptions are turned into a 500 by the request middleware (app/logging.py),
    which also logs them with the request ID."""
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
