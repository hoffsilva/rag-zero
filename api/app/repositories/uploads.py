"""Metadata of visitors' uploaded documents, and their 24-hour expiry."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from app.domain import UPLOAD_TTL_HOURS, Upload, UploadKind
from app.repositories.database import Connection

TTL = f"interval '{UPLOAD_TTL_HOURS} hours'"


async def save_upload(connection: Connection, upload: Upload) -> None:
    await connection.execute(
        """
        INSERT INTO uploads (collection, title, file, kind, size_bytes, sections, language, summary,
                             suggested_questions)
        VALUES (%(collection)s, %(title)s, %(file)s, %(kind)s, %(size_bytes)s, %(sections)s, %(language)s,
                %(summary)s, %(suggested_questions)s)
        """,
        dict(upload),
    )


async def get_upload(connection: Connection, collection: str) -> Upload | None:
    """The upload's metadata, or None when it doesn't exist or has expired."""
    cursor = await connection.execute(
        f"""
        SELECT collection, title, file, kind, size_bytes, sections, language, summary, suggested_questions,
               created_at + {TTL}
        FROM uploads
        WHERE collection = %s AND created_at > now() - {TTL}
        """,  # noqa: S608 - only the constant TTL is interpolated; the collection is a bound parameter
        (collection,),
    )
    row: Sequence[Any] | None = await cursor.fetchone()
    if row is None:
        return None
    expires_at = cast(datetime, row[9])
    return Upload(
        collection=str(row[0]),
        title=str(row[1]),
        file=str(row[2]),
        kind=cast(UploadKind, row[3]),
        size_bytes=int(row[4]),
        sections=[str(section) for section in row[5]],
        language=str(row[6]),
        summary=str(row[7]),
        suggested_questions=[str(question) for question in row[8]],
        expires_at=expires_at.isoformat(),
    )


async def delete_expired_uploads(connection: Connection) -> None:
    """Uploads live 24 h. Called on every new upload, so no scheduled job is needed."""
    async with connection.transaction():
        await connection.execute(
            f"DELETE FROM chunks WHERE collection LIKE 'upload-%%' AND created_at < now() - {TTL}"  # noqa: S608
        )
        await connection.execute(f"DELETE FROM uploads WHERE created_at < now() - {TTL}")  # noqa: S608
