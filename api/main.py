"""Vercel entrypoint (vercel.json: "entrypoint": "main:app"). The app lives in app/main.py."""

from app.main import app

__all__ = ["app"]
