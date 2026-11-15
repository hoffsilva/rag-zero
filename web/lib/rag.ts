import { chat, embed } from "@/lib/openai";
import { search, type ScoredChunk } from "@/lib/retrieval";

const TOP_K = 4;

const SYSTEM_PROMPT = `Tu es l'assistant d'une base de connaissances sur l'entreprise Mondria.
Réponds uniquement à partir du contexte fourni.
Cite les identifiants des sections utilisées entre crochets, par exemple [MON-01-04].
Si l'information n'est pas dans le contexte, dis clairement qu'elle n'est pas disponible ; n'invente rien.
Réponds dans la langue de la question, de façon concise.`;

export interface RagAnswer {
  answer: string;
  sources: ScoredChunk[];
}

// Augmented step: the retrieved chunks become the context of the prompt.
function buildUserPrompt(question: string, sources: ScoredChunk[]): string {
  const context = sources.map((chunk) => `[${chunk.id}] ${chunk.text}`).join("\n\n");
  return `Contexte :\n${context}\n\nQuestion : ${question}`;
}

export async function answerQuestion(question: string): Promise<RagAnswer> {
  // Retrieval
  const [queryEmbedding] = await embed([question]);
  if (!queryEmbedding) {
    throw new Error("Embedding request returned no vector");
  }
  const sources = search(queryEmbedding, TOP_K);

  // Generation
  const answer = await chat([
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: buildUserPrompt(question, sources) },
  ]);

  return { answer, sources };
}
