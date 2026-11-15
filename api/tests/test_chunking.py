from pathlib import Path

import rag

SAMPLE = """---
id: MON-09
title: "Document de test"
source_urls:
  - https://example.com/a
  - https://example.com/b
---

# Document de test

> Intro that belongs to no section.

## MON-09-01 — Première section

Texte un.

## MON-09-02 — Deuxième section

Texte deux,
sur deux lignes.

## Documents associés

- [MON-01](01.md)

## Sources

- https://example.com/a
"""


def test_parse_document_reads_frontmatter():
    document = rag.parse_document(Path("09-test.md"), SAMPLE)

    assert document == {
        "id": "MON-09",
        "title": "Document de test",
        "file": "09-test.md",
        "source_urls": ["https://example.com/a", "https://example.com/b"],
    }


def test_parse_document_without_frontmatter_falls_back_to_file_name():
    document = rag.parse_document(Path("notes.md"), "## MON-01-01 — X\n\ntext")

    assert document["id"] == "notes"
    assert document["title"] == "notes"
    assert document["source_urls"] == []


def test_split_sections_keeps_only_sections_with_an_id():
    document = rag.parse_document(Path("09-test.md"), SAMPLE)

    chunks = rag.split_sections(document, SAMPLE)

    assert [chunk["id"] for chunk in chunks] == ["MON-09-01", "MON-09-02"]
    assert [chunk["title"] for chunk in chunks] == ["Première section", "Deuxième section"]


def test_split_sections_stops_at_the_next_heading():
    document = rag.parse_document(Path("09-test.md"), SAMPLE)

    first, second = rag.split_sections(document, SAMPLE)

    assert first["text"] == "## MON-09-01 — Première section\n\nTexte un."
    # The last section ends at "## Documents associés", so the links don't leak into it.
    assert second["text"].endswith("sur deux lignes.")
    assert "Documents associés" not in second["text"]
    assert second["document"]["id"] == "MON-09"


def test_load_chunks_on_the_real_knowledge_base():
    chunks = rag.load_chunks()

    ids = [chunk["id"] for chunk in chunks]
    assert len(chunks) == 21
    assert len(set(ids)) == len(ids)
    for chunk in chunks:
        assert chunk["id"].startswith(chunk["document"]["id"] + "-")
        assert chunk["text"].startswith(f"## {chunk['id']} — ")
        assert chunk["document"]["source_urls"], chunk["id"]
