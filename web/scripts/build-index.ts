// Builds data/index.json from the Markdown docs. Run with: bun run index
// Bun loads OPENAI_API_KEY from .env.local automatically.
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { EMBEDDING_MODEL, embed } from "../lib/openai";
import type { Chunk, IndexFile } from "../lib/retrieval";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.join(SCRIPT_DIR, "../../mondria-rag");
const OUTPUT_FILE = path.join(SCRIPT_DIR, "../data/index.json");

// Every "## " heading is a boundary; only headings with an ID ("## MON-01-01 — Title") become chunks.
// Group 1 = ID, group 2 = title.
const SECTION_PATTERN = /^## (?:([A-Z]+-\d+-\d+) — )?(.+)$/gm;

function splitSections(text: string): Chunk[] {
  const matches = [...text.matchAll(SECTION_PATTERN)];
  const sections: Chunk[] = [];

  matches.forEach((match, i) => {
    const [, id, title] = match;
    if (!id || !title) {
      return;
    }
    const end = matches[i + 1]?.index ?? text.length;
    sections.push({ id, title: title.trim(), text: text.slice(match.index, end).trim() });
  });

  return sections;
}

const files = (await readdir(DOCS_DIR)).filter((name) => name.endsWith(".md")).toSorted();
const chunks: Chunk[] = [];
for (const file of files) {
  chunks.push(...splitSections(await readFile(path.join(DOCS_DIR, file), "utf-8")));
}

const embeddings = await embed(chunks.map((chunk) => chunk.text));

const index: IndexFile = {
  model: EMBEDDING_MODEL,
  chunks: chunks.map((chunk, i) => ({ ...chunk, embedding: embeddings[i] ?? [] })),
};

await writeFile(OUTPUT_FILE, `${JSON.stringify(index)}\n`);
console.log(`Indexed ${chunks.length} chunks from ${files.length} files into data/index.json`);
