import re
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

SECTION_PATTERN = re.compile(r"^## (?:(?P<id>[A-Z]+-\d+-\d+) — )?(?P<title>.+)$", re.MULTILINE)


def split_sections(text: str) -> list[dict[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    sections = []
    for index, match in enumerate(matches):
        if not match["id"]:
            continue  # "Documents associés", "Sources": boundary only, not a chunk
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append({
            "id": match["id"],
            "title": match["title"].strip(),
            "text": text[match.start():end].strip(),
        })
    return sections


# Load documents and split them into sections
folder = Path("./mondria-rag")
chunks = [
    section
    for file in sorted(folder.glob("*.md"))
    for section in split_sections(file.read_text(encoding="utf-8"))
]

# Local embedding model (requires "query: " / "passage: " prefixes)
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Indexing
embeddings = model.encode(
    [chunk["text"] for chunk in chunks],
    prompt="passage: ",
    normalize_embeddings=True,
    convert_to_tensor=True,
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


app = FastAPI()


@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = model.encode(
        request.query,
        prompt="query: ",
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    scores = model.similarity(query_embedding, embeddings)[0]
    top = scores.topk(k=min(request.top_k, len(chunks)))

    return {
        "query": request.query,
        "results": [
            {**chunks[index], "score": round(score, 4)}
            for score, index in zip(top.values.tolist(), top.indices.tolist())
        ],
    }