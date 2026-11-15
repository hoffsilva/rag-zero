const OPENAI_BASE_URL = "https://api.openai.com/v1";

export const EMBEDDING_MODEL = "text-embedding-3-small";

function apiKey(): string {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  return key;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${OPENAI_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`OpenAI ${path} failed (${response.status}): ${await response.text()}`);
  }
  return (await response.json()) as T;
}

interface EmbeddingResponse {
  data: { index: number; embedding: number[] }[];
}

export async function embed(inputs: string[]): Promise<number[][]> {
  const result = await post<EmbeddingResponse>("/embeddings", {
    model: EMBEDDING_MODEL,
    input: inputs,
  });
  return result.data
    .toSorted((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}

export interface ChatMessage {
  role: "system" | "user";
  content: string;
}

interface ChatCompletionResponse {
  choices: { message: { content: string | null } }[];
}

export async function chat(messages: ChatMessage[]): Promise<string> {
  const model = process.env.OPENAI_CHAT_MODEL;
  if (!model) {
    throw new Error("OPENAI_CHAT_MODEL is not set");
  }

  const result = await post<ChatCompletionResponse>("/chat/completions", {
    model,
    messages,
  });
  return result.choices[0]?.message.content?.trim() ?? "";
}
