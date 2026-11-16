"""FastAPI dependencies."""

from typing import Annotated, cast

from fastapi import Depends, Request

from app.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    """The runtime opened in the app's lifespan (app.main.create_app)."""
    return cast(Runtime, request.app.state.runtime)


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
