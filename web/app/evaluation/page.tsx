import type { Metadata } from "next";

import evaluationData from "@/data/evaluation.json";

export const metadata: Metadata = {
  title: "Évaluation — Mondria RAG",
  description: "Retrieval and answer evaluation of the Mondria RAG demo",
};

// Shape of web/data/evaluation.json, written by api/eval/run_eval.py --save.
interface Retrieved {
  id: string;
  score: number;
}

interface QuestionResult {
  question: string;
  expected: string[];
  retrieved: Retrieved[];
  rank: number | null;
  answer?: string;
  cited?: string[];
  answer_correct?: boolean;
}

interface ThresholdRow {
  threshold: number;
  false_positives: number;
  false_negatives: number;
  error_rate: number;
}

interface Run {
  embedding_model: string;
  date: string;
  k: number;
  retrieval: {
    answerable: number;
    hit_rate_at_1: number;
    hit_rate_at_k: number;
    mrr: number;
    document_hit_rate_at_k: number;
  };
  threshold: {
    answerable_top1: { min: number; max: number };
    unanswerable_top1: { min: number; max: number };
    rows: ThresholdRow[];
    best: ThresholdRow;
  };
  answers: {
    chat_model: string;
    error_rate: number;
    answerable_wrong: number;
    unanswerable_answered: number;
  } | null;
  questions: QuestionResult[];
}

interface EvaluationFile {
  runs: Record<string, Run>;
}

const evaluation: EvaluationFile = evaluationData;

// The model the live demo uses; the other runs are shown for comparison.
const PRODUCTION_MODEL = "text-embedding-3-small";

const percent = (value: number) => `${Math.round(value * 100)} %`;

function Eyebrow({ children }: { children: string }) {
  return <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">{children}</p>;
}

function Card({ children }: { children: React.ReactNode }) {
  return <div className="rounded-2xl border border-ink/10 bg-navy-raised/95 p-6">{children}</div>;
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card>
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-ink/60">{label}</p>
      <p className="mt-2 font-display text-4xl font-bold">{value}</p>
      <p className="mt-2 text-sm text-ink/60">{hint}</p>
    </Card>
  );
}

// Top-1 similarities sit between 0.1 and 0.75, so the axis stops at 0.8 to spread the dots.
const AXIS_MAX = 0.8;
const axisPosition = (score: number) => `${(Math.min(score, AXIS_MAX) / AXIS_MAX) * 100}%`;

// Top-1 similarity of every question, with the best threshold as a line.
function ScoreStrip({ run }: { run: Run }) {
  const rows = [
    { label: "Avec réponse", questions: run.questions.filter((q) => q.expected.length > 0), color: "bg-ink" },
    { label: "Sans réponse", questions: run.questions.filter((q) => q.expected.length === 0), color: "bg-accent" },
  ];
  return (
    <div className="flex flex-col gap-4">
      {rows.map((row) => (
        <div key={row.label} className="flex items-center gap-4">
          <span className="w-28 shrink-0 text-sm text-ink/70">{row.label}</span>
          <div className="relative h-8 flex-1 rounded-full bg-ink/5">
            {row.questions.map((q) => (
              <span
                key={q.question}
                title={`${q.question} — ${(q.retrieved[0]?.score ?? 0).toFixed(3)}`}
                className={`absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${row.color} opacity-80`}
                style={{ left: axisPosition(q.retrieved[0]?.score ?? 0) }}
              />
            ))}
            <span
              className="absolute inset-y-0 w-px bg-accent"
              style={{ left: axisPosition(run.threshold.best.threshold) }}
              aria-hidden
            />
          </div>
        </div>
      ))}
      <div className="flex justify-between pl-32 text-xs text-ink/50">
        <span>0</span>
        <span>{AXIS_MAX}</span>
      </div>
      <p className="text-center text-xs text-ink/50">
        Similarité du premier résultat · ligne orange : seuil {run.threshold.best.threshold.toFixed(2)}
      </p>
    </div>
  );
}

function QuestionRow({ q, k }: { q: QuestionResult; k: number }) {
  const answerable = q.expected.length > 0;
  const retrievalLabel = !answerable ? "—" : q.rank ? `rang ${q.rank}` : `hors top ${k}`;
  return (
    <details className="group rounded-2xl border border-ink/10 bg-navy-raised/95 p-5">
      <summary className="flex cursor-pointer list-none flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <span className="font-medium">{q.question}</span>
        <span className="flex shrink-0 gap-3 text-sm">
          <span className={q.rank === 1 || !answerable ? "text-ink/60" : "text-accent"}>{retrievalLabel}</span>
          {q.answer_correct !== undefined && (
            <span className={q.answer_correct ? "text-ink/60" : "text-accent"}>
              {q.answer_correct ? "réponse ✓" : "réponse ✗"}
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
        </p>
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

export default function EvaluationPage() {
  const runs = Object.values(evaluation.runs);
  const run = evaluation.runs[PRODUCTION_MODEL] ?? runs[0];
  if (!run) {
    return <main className="flex-1 px-4 py-24 text-center">Aucune évaluation enregistrée.</main>;
  }
  const answerable = run.questions.filter((q) => q.expected.length > 0).length;
  const unanswerable = run.questions.length - answerable;
  const answers = run.answers;

  return (
    <main className="flex flex-1 flex-col">
      <section className="hero-grid px-4 pb-14 pt-20 text-center sm:pt-24">
        <span className="inline-flex items-center gap-2 rounded-full border border-ink/15 bg-ink/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-ink/75">
          <span className="size-2 bg-accent" aria-hidden />
          Évaluation
        </span>
        <h1 className="mx-auto mt-8 max-w-4xl font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
          Le RAG trouve-t-il la bonne section ?
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-ink/75">
          {run.questions.length} questions en français, anglais et portugais : {answerable} dont la réponse est
          dans la base, {unanswerable} dont elle ne l’est pas. La recherche est mesurée séparément de la réponse
          du LLM. Instantané du {run.date}, modèle {run.embedding_model}.
        </p>
      </section>

      <div className="mx-auto flex w-full max-w-5xl flex-col gap-16 px-4 pb-24">
        <section className="flex flex-col gap-6">
          <div>
            <Eyebrow>Recherche</Eyebrow>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Résultats</h2>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              label={`Hit rate@${run.k}`}
              value={percent(run.retrieval.hit_rate_at_k)}
              hint={`Section attendue dans les ${run.k} premiers résultats`}
            />
            <Metric
              label="Hit rate@1"
              value={percent(run.retrieval.hit_rate_at_1)}
              hint="Section attendue en première position"
            />
            <Metric label="MRR" value={run.retrieval.mrr.toFixed(3)} hint="Moyenne de 1 / rang de la bonne section" />
            {answers && (
              <Metric
                label="Erreur des réponses"
                value={percent(answers.error_rate)}
                hint={`Citation attendue ou refus, ${answers.chat_model}`}
              />
            )}
          </div>
        </section>

        {runs.length > 1 && (
          <section className="flex flex-col gap-6">
            <div>
              <Eyebrow>Comparaison</Eyebrow>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Modèles d’embedding</h2>
            </div>
            <div className="overflow-x-auto rounded-2xl border border-ink/10">
              <table className="w-full text-left text-sm">
                <thead className="bg-navy-raised text-ink/60">
                  <tr>
                    <th className="px-4 py-3 font-medium">Modèle</th>
                    <th className="px-4 py-3 font-medium">Hit rate@1</th>
                    <th className="px-4 py-3 font-medium">Hit rate@k</th>
                    <th className="px-4 py-3 font-medium">MRR</th>
                    <th className="px-4 py-3 font-medium">Erreur des réponses</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.embedding_model} className="border-t border-ink/10">
                      <td className="whitespace-nowrap px-4 py-3 font-mono">
                        {r.embedding_model}
                        {r.embedding_model === PRODUCTION_MODEL && <span className="ml-2 text-accent">démo</span>}
                      </td>
                      <td className="px-4 py-3">{percent(r.retrieval.hit_rate_at_1)}</td>
                      <td className="px-4 py-3">{percent(r.retrieval.hit_rate_at_k)}</td>
                      <td className="px-4 py-3">{r.retrieval.mrr.toFixed(3)}</td>
                      <td className="px-4 py-3">{r.answers ? percent(r.answers.error_rate) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="flex flex-col gap-6">
          <div>
            <Eyebrow>Seuil</Eyebrow>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Faux positifs et faux négatifs</h2>
            <p className="mt-3 max-w-3xl text-ink/75">
              Si la similarité du premier résultat est sous le seuil, on considère que la réponse n’est pas dans la
              base. Faux positif : question sans réponse au-dessus du seuil. Faux négatif : question avec réponse
              en dessous.
            </p>
          </div>
          <Card>
            <ScoreStrip run={run} />
          </Card>
          <div className="overflow-x-auto rounded-2xl border border-ink/10">
            <table className="w-full text-left text-sm">
              <thead className="bg-navy-raised text-ink/60">
                <tr>
                  <th className="px-4 py-3 font-medium">Seuil</th>
                  <th className="px-4 py-3 font-medium">Faux positifs</th>
                  <th className="px-4 py-3 font-medium">Faux négatifs</th>
                  <th className="px-4 py-3 font-medium">Taux d’erreur</th>
                </tr>
              </thead>
              <tbody>
                {run.threshold.rows.map((row) => (
                  <tr key={row.threshold} className="border-t border-ink/10">
                    <td className="px-4 py-3 font-mono">{row.threshold.toFixed(2)}</td>
                    <td className="px-4 py-3">{row.false_positives}</td>
                    <td className="px-4 py-3">{row.false_negatives}</td>
                    <td className="px-4 py-3">{percent(row.error_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="max-w-3xl text-ink/75">
            Un seuil n’écarte que les questions hors sujet. Les questions sur Mondria dont la réponse n’est pas dans
            la base obtiennent des scores proches de celles qui ont une réponse : c’est la consigne du prompt
            (répondre uniquement à partir du contexte) qui les refuse.
          </p>
        </section>

        <section className="flex flex-col gap-6">
          <div>
            <Eyebrow>Détail</Eyebrow>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Question par question</h2>
          </div>
          <div className="flex flex-col gap-3">
            {run.questions.map((q) => (
              <QuestionRow key={q.question} q={q} k={run.k} />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
