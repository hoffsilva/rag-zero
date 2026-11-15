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

function Eyebrow({ children }: { children: string }) {
  return <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">{children}</p>;
}

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
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-3 rounded-2xl border border-ink/10 bg-navy-raised/95 p-3 shadow-[0_18px_50px_-28px_rgba(3,29,42,0.55)] sm:flex-row"
        >
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={MAX_QUESTION_LENGTH}
            placeholder="Posez une question sur Mondria…"
            className="flex-1 bg-transparent px-3 py-3 text-base text-ink outline-none placeholder:text-ink/40"
          />
          <button
            type="submit"
            disabled={status === "loading" || !question.trim()}
            className="rounded-full bg-accent px-6 py-3 font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
          >
            {status === "loading" ? "Recherche…" : "Demander →"}
          </button>
        </form>

        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => void ask(example)}
              className="rounded-full border border-ink/15 px-4 py-1.5 text-sm text-ink/75 transition-colors hover:border-ink/40 hover:text-ink"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {status === "error" && <p className="text-accent">Une erreur est survenue. Réessayez.</p>}

      {result && (
        <section className="flex flex-col gap-8">
          <div className="rounded-2xl border border-ink/10 bg-navy-raised/95 p-7">
            <span className="block size-2 bg-accent" aria-hidden />
            <div className="mt-4">
              <Eyebrow>Réponse</Eyebrow>
            </div>
            <p className="mt-3 whitespace-pre-line text-lg leading-relaxed">{result.answer}</p>
          </div>

          <div className="flex flex-col gap-4">
            <div>
              <Eyebrow>Sources</Eyebrow>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Sections récupérées</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {result.sources.map((source) => (
                <details
                  key={source.id}
                  className="group rounded-2xl border border-ink/10 bg-navy-raised/95 p-6 transition-colors hover:border-ink/25"
                >
                  <summary className="flex cursor-pointer list-none flex-col gap-2">
                    <span className="text-xs font-medium uppercase tracking-[0.18em] text-ink/60">
                      {source.id} · score {source.score.toFixed(3)}
                    </span>
                    <span className="font-display text-xl font-bold">{source.title}</span>
                    <span className="text-sm font-semibold group-open:hidden">Voir le texte →</span>
                  </summary>
                  <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-ink/75">{source.text}</p>
                </details>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
