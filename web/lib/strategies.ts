// Retrieval strategies of api/retrieval.py, with the labels shown in the UI.
export type Strategy = "vector" | "keyword" | "hybrid" | "mmr" | "multi_query" | "rerank";

export const STRATEGIES: { value: Strategy; label: string; extraLlmCall: boolean }[] = [
  { value: "vector", label: "Vectorielle (cosinus)", extraLlmCall: false },
  { value: "keyword", label: "Mots-clés (full-text)", extraLlmCall: false },
  { value: "hybrid", label: "Hybride (RRF)", extraLlmCall: false },
  { value: "mmr", label: "MMR (diversité)", extraLlmCall: false },
  { value: "multi_query", label: "Multi-requêtes (LLM)", extraLlmCall: true },
  { value: "rerank", label: "Reranking (LLM)", extraLlmCall: true },
];

export const strategyLabel = (strategy: string) =>
  STRATEGIES.find((option) => option.value === strategy)?.label ?? strategy;

export const strategyOrder = (strategy: string) => STRATEGIES.findIndex((option) => option.value === strategy);
