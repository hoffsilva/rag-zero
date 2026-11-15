// Shape of web/data/evaluation.json, written by api/eval/run_eval.py --save.
export interface Retrieved {
  id: string;
  score: number;
}

export interface QuestionResult {
  question: string;
  expected: string[];
  retrieved: Retrieved[];
  rank: number | null;
  latency_ms: number;
  answer?: string;
  answer_sources?: string[];
  cited?: string[];
  citation_ok?: boolean;
}

export interface ThresholdRow {
  threshold: number;
  false_positives: number;
  false_negatives: number;
  error_rate: number;
}

export interface Run {
  embedding_model: string;
  strategy: string;
  date: string;
  k: number;
  retrieval: {
    answerable: number;
    hit_rate_at_1: number;
    hit_rate_at_k: number;
    mrr: number;
    document_hit_rate_at_k: number;
    latency_ms_median: number;
  };
  // Only the vector strategy has a similarity score a threshold can be put on.
  threshold: {
    answerable_top1: { min: number; max: number };
    unanswerable_top1: { min: number; max: number };
    rows: ThresholdRow[];
    best: ThresholdRow;
  } | null;
  answers: {
    chat_model: string;
    answer_k: number;
    citation_error_rate: number;
    answerable_wrong: number;
    unanswerable_answered: number;
  } | null;
  questions: QuestionResult[];
}

export interface EvaluationFile {
  runs: Record<string, Run>; // key: "<embedding model>|<strategy>"
  default_strategy: string;
}

export const percent = (value: number) => `${Math.round(value * 100)} %`;
