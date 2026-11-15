"use client";

import { useState } from "react";

import type { QuestionResult, Run } from "@/lib/evaluation";
import { strategyLabel } from "@/lib/strategies";

function QuestionRow({ q, k }: { q: QuestionResult; k: number }) {
  const answerable = q.expected.length > 0;
  const retrievalLabel = !answerable ? "—" : q.rank ? `rang ${q.rank}` : `hors top ${k}`;
  return (
    <details className="group rounded-2xl border border-ink/10 bg-navy-raised/95 p-5">
      <summary className="flex cursor-pointer list-none flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <span className="font-medium">{q.question}</span>
        <span className="flex shrink-0 gap-3 text-sm">
          <span className={q.rank === 1 || !answerable ? "text-ink/60" : "text-accent"}>{retrievalLabel}</span>
          {q.citation_ok !== undefined && (
            <span className={q.citation_ok ? "text-ink/60" : "text-accent"}>
              {q.citation_ok ? "citation ✓" : "citation ✗"}
            </span>
          )}
        </span>
      </summary>
      <div className="mt-4 flex flex-col gap-3 text-sm text-ink/75">
        <p>
          <span className="text-ink/50">Attendu : </span>
          {answerable ? q.expected.join(", ") : "aucune section (la réponse n’est pas dans la base)"}
        </p>
        <p>
          <span className="text-ink/50">Récupéré : </span>
          {q.retrieved.map((r) => `${r.id} (${r.score.toFixed(3)})`).join(" · ")}
          <span className="text-ink/50"> · {q.latency_ms} ms</span>
        </p>
        {q.answer_sources && (
          <p>
            <span className="text-ink/50">Envoyé au LLM : </span>
            {q.answer_sources.join(", ")}
          </p>
        )}
        {q.answer && (
          <p className="whitespace-pre-line">
            <span className="text-ink/50">Réponse : </span>
            {q.answer}
          </p>
        )}
      </div>
    </details>
  );
}

// Per-question detail, with one tab per retrieval strategy.
export function EvaluationDetails({ runs, initialStrategy }: { runs: Run[]; initialStrategy: string }) {
  const [selected, setSelected] = useState(initialStrategy);
  const run = runs.find((r) => r.strategy === selected) ?? runs[0];
  if (!run) {
    return null;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Stratégie de recherche">
        {runs.map((r) => (
          <button
            key={r.strategy}
            type="button"
            role="tab"
            aria-selected={r.strategy === run.strategy}
            onClick={() => setSelected(r.strategy)}
            className={
              r.strategy === run.strategy
                ? "rounded-full bg-accent px-4 py-1.5 text-sm font-semibold text-white"
                : "rounded-full border border-ink/15 px-4 py-1.5 text-sm text-ink/75 transition-colors hover:border-ink/40 hover:text-ink"
            }
          >
            {strategyLabel(r.strategy)}
          </button>
        ))}
      </div>
      <div className="flex flex-col gap-3">
        {run.questions.map((q) => (
          <QuestionRow key={q.question} q={q} k={run.k} />
        ))}
      </div>
    </div>
  );
}
