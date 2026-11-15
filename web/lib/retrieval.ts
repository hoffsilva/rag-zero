import indexData from "@/data/index.json";

export interface Chunk {
  id: string;
  title: string;
  text: string;
}

interface IndexedChunk extends Chunk {
  embedding: number[];
}

export interface IndexFile {
  model: string;
  chunks: IndexedChunk[];
}

export interface ScoredChunk extends Chunk {
  score: number;
}

const index: IndexFile = indexData;

// OpenAI embeddings are normalized to length 1, so the dot product is the cosine similarity.
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

export function search(queryEmbedding: number[], topK: number): ScoredChunk[] {
  return index.chunks
    .map(({ embedding, ...chunk }) => ({ ...chunk, score: dot(queryEmbedding, embedding) }))
    .toSorted((a, b) => b.score - a.score)
    .slice(0, topK);
}
