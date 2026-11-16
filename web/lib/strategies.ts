// Retrieval strategies of api/app/services/retrieval.py, with the labels shown in the UI.
export type Strategy = "vector" | "keyword" | "hybrid" | "mmr" | "multi_query" | "rerank";

// inLangChain: offered by the LangChain pipeline too (api/app/services/langchain_rag.py).
export const STRATEGIES: { value: Strategy; label: string; extraLlmCall: boolean; inLangChain: boolean }[] = [
  { value: "vector", label: "Vectorielle (cosinus)", extraLlmCall: false, inLangChain: true },
  { value: "keyword", label: "Mots-clés (full-text)", extraLlmCall: false, inLangChain: false },
  { value: "hybrid", label: "Hybride (RRF)", extraLlmCall: false, inLangChain: true },
  { value: "mmr", label: "MMR (diversité)", extraLlmCall: false, inLangChain: false },
  { value: "multi_query", label: "Multi-requêtes (LLM)", extraLlmCall: true, inLangChain: true },
  { value: "rerank", label: "Reranking (LLM)", extraLlmCall: true, inLangChain: true },
];

export type Pipeline = "native" | "langchain";

export const PIPELINES: { value: Pipeline; label: string }[] = [
  { value: "native", label: "Code maison" },
  { value: "langchain", label: "LangChain" },
];

export const pipelineLabel = (pipeline: string) =>
  PIPELINES.find((option) => option.value === pipeline)?.label ?? pipeline;

export const strategyLabel = (strategy: string) =>
  STRATEGIES.find((option) => option.value === strategy)?.label ?? strategy;

export const strategyOrder = (strategy: string) => STRATEGIES.findIndex((option) => option.value === strategy);
