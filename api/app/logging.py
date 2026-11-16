"""Request IDs and one structured log line per request.

A pure ASGI middleware (no buffering, so uploads and the MCP transport stream through):
- reuses a valid incoming X-Request-ID, or generates one, returns it in the response header and
  keeps it in a ContextVar so every log line of the request carries it;
- logs method, route template (never the real path: upload collection IDs are private),
  status and duration as JSON;
- turns an unexpected exception into the generic 500 error body, with the same request ID in the
  body, the header and the logged stack trace. Questions, documents and credentials are never logged.
"""

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_body

REQUEST_ID_HEADER = "X-Request-ID"
VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("app.request")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Logs go to stderr (stdout is the MCP stdio channel). Idempotent."""
    root = logging.getLogger()
    if any(isinstance(f, RequestIdFilter) for handler in root.handlers for f in handler.filters):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return "/api/mcp" if str(scope.get("path", "")).startswith("/api/mcp") else "unmatched"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers", [])).get(REQUEST_ID_HEADER.lower().encode(), b"").decode()
        request_id = incoming if VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        started_at = time.perf_counter()
        status = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("Unhandled error")
            if response_started:
                raise
            response = JSONResponse(
                error_body("internal_error", "Internal server error", request_id),
                status_code=500,
                headers={REQUEST_ID_HEADER: request_id},
            )
            await response(scope, receive, send)
            status = 500
        finally:
            logger.info(
                json.dumps(
                    {
                        "method": scope.get("method"),
                        "route": _route_template(scope),
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started_at) * 1000),
                        "request_id": request_id,
                    }
                )
            )
            request_id_var.reset(token)
