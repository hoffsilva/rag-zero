"use client";

import { useState, type FormEvent } from "react";

import { STRATEGIES, strategyLabel, type Strategy } from "@/lib/strategies";

// Response of POST /api/ask, served by the Python API in api/main.py.
interface SourceDocument {
  id: string;
  title: string;
  file: string;
  source_urls: string[];
}

interface Source {
  id: string;
  title: string;
  text: string;
  document: SourceDocument;
  score: number;
  cited: boolean;
}

interface RagAnswer {
  answer: string;
  sources: Source[];
  strategy: Strategy;
}

const DOCS_BASE_URL = "https://github.com/hoffsilva/rag-zero/blob/main/api/mondria-rag-docs";

// The section text starts with its own "## ID — Title" heading, already shown as the card title.
function sectionBody(text: string): string {
  return text.replace(/^## .*\n+/, "");
}

function CitedSource({ source, showLinks }: { source: Source; showLinks: boolean }) {
  const pageUrl = source.document.source_urls[0];
  return (
    <article className="rounded-2xl border border-accent/40 bg-navy-raised/95 p-6">
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-ink/60">
        {source.document.id} · {source.document.title}
      </p>
      <h3 className="mt-2 font-display text-xl font-bold">
        <span className="font-mono text-base text-accent">[{source.id}]</span> {source.title}
      </h3>
      <p className="mt-3 whitespace-pre-line leading-relaxed text-ink/80">{sectionBody(source.text)}</p>
      {showLinks && (
        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm font-semibold">
          <a href={`${DOCS_BASE_URL}/${source.document.file}`} target="_blank" rel="noreferrer" className="hover:text-accent">
            Document complet ↗
          </a>
          {pageUrl && (
            <a href={pageUrl} target="_blank" rel="noreferrer" className="hover:text-accent">
              Page d’origine ↗
            </a>
          )}
        </div>
      )}
    </article>
  );
}

function RetrievedSource({ source }: { source: Source }) {
  return (
    <details className="group rounded-2xl border border-ink/10 bg-navy-raised/95 p-6 transition-colors hover:border-ink/25">
      <summary className="flex cursor-pointer list-none flex-col gap-2">
        <span className="text-xs font-medium uppercase tracking-[0.18em] text-ink/60">
          {source.id} · score {source.score.toFixed(3)}
        </span>
        <span className="font-display text-lg font-bold">{source.title}</span>
        <span className="text-sm text-ink/60">{source.document.title}</span>
        <span className="text-sm font-semibold group-open:hidden">Voir le texte →</span>
      </summary>
      <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-ink/75">{sectionBody(source.text)}</p>
    </details>
  );
}

const MAX_QUESTION_LENGTH = 300;

const MONDRIA_EXAMPLES = [
  "Où sont les bureaux de Mondria ?",
  "Quelle est la différence entre services gérés et support à la demande ?",
  "Combien coûtent les services TI ?",
];

type Status = "idle" | "loading" | "error" | "expired";

interface AskFormProps {
  // "mondria" (the demo knowledge base) or the collection of an uploaded document.
  collection?: string;
  examples?: string[];
  placeholder?: string;
  // Links to the full Mondria document and its original page; uploads have neither.
  showSourceLinks?: boolean;
}

function Eyebrow({ children }: { children: string }) {
  return <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">{children}</p>;
}

export function AskForm({
  collection = "mondria",
  examples = MONDRIA_EXAMPLES,
  placeholder = "Posez une question sur Mondria…",
  showSourceLinks = true,
}: AskFormProps) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<RagAnswer | null>(null);
  const [strategy, setStrategy] = useState<Strategy | "">("");

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
        body: JSON.stringify({ question: trimmed, collection, ...(strategy && { strategy }) }),
      });
      if (response.status === 404) {
        setStatus("expired");
        return;
      }
      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }
      setResult((await response.json()) as RagAnswer);
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  const cited = result?.sources.filter((source) => source.cited) ?? [];
  const others = result?.sources.filter((source) => !source.cited) ?? [];

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
            placeholder={placeholder}
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

        <label className="flex items-center gap-3 text-sm text-ink/70">
          Stratégie de recherche
          <select
            value={strategy}
            onChange={(event) => setStrategy(event.target.value as Strategy | "")}
            className="rounded-full border border-ink/15 bg-navy-raised px-3 py-1.5 text-ink outline-none focus:border-ink/40"
          >
            <option value="">Par défaut</option>
            {STRATEGIES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <div className="flex flex-wrap gap-2">
          {examples.map((example) => (
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
      {status === "expired" && <p className="text-accent">Ce document a expiré. Envoyez-le de nouveau.</p>}

      {result && (
        <section className="flex flex-col gap-8">
          <div className="rounded-2xl border border-ink/10 bg-navy-raised/95 p-7">
            <span className="block size-2 bg-accent" aria-hidden />
            <div className="mt-4 flex flex-wrap items-baseline justify-between gap-2">
              <Eyebrow>Réponse</Eyebrow>
              <span className="text-xs text-ink/50">Recherche : {strategyLabel(result.strategy)}</span>
            </div>
            <p className="mt-3 whitespace-pre-line text-lg leading-relaxed">{result.answer}</p>
          </div>

          {cited.length > 0 && (
            <div className="flex flex-col gap-4">
              <div>
                <Eyebrow>Sources</Eyebrow>
                <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Documents utilisés</h2>
              </div>
              {cited.map((source) => (
                <CitedSource key={source.id} source={source} showLinks={showSourceLinks} />
              ))}
            </div>
          )}

          {others.length > 0 && (
            <div className="flex flex-col gap-4">
              <h2 className="font-display text-xl font-bold tracking-tight">
                {cited.length > 0 ? "Autres sections récupérées" : "Sections récupérées"}
              </h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {others.map((source) => (
                  <RetrievedSource key={source.id} source={source} />
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
