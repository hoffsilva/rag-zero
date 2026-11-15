"use client";

import { useState, type FormEvent } from "react";

import type { RagAnswer } from "@/lib/rag";

const MAX_QUESTION_LENGTH = 300;

const EXAMPLES = [
  "Où sont les bureaux de Mondria ?",
  "Quelle est la différence entre services gérés et support à la demande ?",
  "Combien coûtent les services TI ?",
];

type Status = "idle" | "loading" | "error";

export function AskForm() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<RagAnswer | null>(null);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || status === "loading") {
      return;
    }
    setQuestion(trimmed);
    setStatus("loading");
    setResult(null);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }
      setResult((await response.json()) as RagAnswer);
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void ask(question);
  }

  return (
    <div className="flex flex-col gap-6">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          maxLength={MAX_QUESTION_LENGTH}
          placeholder="Posez une question sur Mondria…"
          className="flex-1 rounded-lg border border-zinc-300 bg-white px-4 py-3 text-base outline-none focus:border-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:focus:border-zinc-300"
        />
        <button
          type="submit"
          disabled={status === "loading" || !question.trim()}
          className="rounded-lg bg-zinc-900 px-5 py-3 font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {status === "loading" ? "Recherche…" : "Demander"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => void ask(example)}
            className="rounded-full border border-zinc-300 px-3 py-1 text-sm text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-900"
          >
            {example}
          </button>
        ))}
      </div>

      {status === "error" && (
        <p className="text-red-600 dark:text-red-400">Une erreur est survenue. Réessayez.</p>
      )}

      {result && (
        <section className="flex flex-col gap-4">
          <p className="whitespace-pre-line rounded-lg bg-zinc-100 p-4 leading-relaxed dark:bg-zinc-900">
            {result.answer}
          </p>
          <div className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
              Sections récupérées
            </h2>
            {result.sources.map((source) => (
              <details key={source.id} className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <summary className="cursor-pointer text-sm">
                  <span className="font-mono">{source.id}</span> — {source.title}
                  <span className="ml-2 text-zinc-500">score {source.score.toFixed(3)}</span>
                </summary>
                <p className="mt-2 whitespace-pre-line text-sm text-zinc-600 dark:text-zinc-400">{source.text}</p>
              </details>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
