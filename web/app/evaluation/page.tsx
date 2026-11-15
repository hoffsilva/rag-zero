import type { Metadata } from "next";

import { EvaluationDetails } from "@/components/evaluation-details";
import evaluationData from "@/data/evaluation.json";
import { percent, type EvaluationFile, type Run } from "@/lib/evaluation";
import { STRATEGIES, strategyLabel, strategyOrder } from "@/lib/strategies";

export const metadata: Metadata = {
  title: "Évaluation — Mondria RAG",
  description: "Retrieval and answer evaluation of the Mondria RAG demo",
};

const evaluation: EvaluationFile = evaluationData;

// The embedding model the live demo uses; the other model is shown for comparison.
const PRODUCTION_MODEL = "text-embedding-3-small";

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

function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-ink/10">
      <table className="w-full text-left text-sm">
        <thead className="bg-navy-raised text-ink/60">
          <tr>
            {head.map((label) => (
              <th key={label} className="whitespace-nowrap px-4 py-3 font-medium">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

const CELL = "whitespace-nowrap px-4 py-3";

// Top-1 similarities sit between 0.1 and 0.75, so the axis stops at 0.8 to spread the dots.
const AXIS_MAX = 0.8;
const axisPosition = (score: number) => `${(Math.min(score, AXIS_MAX) / AXIS_MAX) * 100}%`;

// Top-1 similarity of every question, with the best threshold as a line.
function ScoreStrip({ run, threshold }: { run: Run; threshold: number }) {
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
            <span className="absolute inset-y-0 w-px bg-accent" style={{ left: axisPosition(threshold) }} aria-hidden />
          </div>
        </div>
      ))}
      <div className="flex justify-between pl-32 text-xs text-ink/50">
        <span>0</span>
        <span>{AXIS_MAX}</span>
      </div>
      <p className="text-center text-xs text-ink/50">
        Similarité du premier résultat · ligne orange : seuil {threshold.toFixed(2)}
      </p>
    </div>
  );
}

export default function EvaluationPage() {
  const runs = Object.values(evaluation.runs);
  const defaultStrategy = evaluation.default_strategy;
  const strategyRuns = runs
    .filter((r) => r.embedding_model === PRODUCTION_MODEL)
    .sort((a, b) => strategyOrder(a.strategy) - strategyOrder(b.strategy));
  const modelRuns = runs.filter((r) => r.strategy === "vector");
  const run = strategyRuns.find((r) => r.strategy === defaultStrategy) ?? strategyRuns[0];
  const vectorRun = strategyRuns.find((r) => r.strategy === "vector");
  if (!run) {
    return <main className="flex-1 px-4 py-24 text-center">Aucune évaluation enregistrée.</main>;
  }
  const answerable = run.questions.filter((q) => q.expected.length > 0).length;
  const unanswerable = run.questions.length - answerable;
  const answers = run.answers;
  const threshold = vectorRun?.threshold;

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
          dans la base, {unanswerable} dont elle ne l’est pas. {STRATEGIES.length} stratégies de recherche,
          mesurées séparément de la réponse du LLM. Instantané du {run.date}, modèle {run.embedding_model}.
        </p>
      </section>

      <div className="mx-auto flex w-full max-w-5xl flex-col gap-16 px-4 pb-24">
        <section className="flex flex-col gap-6">
          <div>
            <Eyebrow>Démo</Eyebrow>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">
              Résultats — {strategyLabel(run.strategy)}
            </h2>
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
                label="Erreur de citation"
                value={percent(answers.citation_error_rate)}
                hint={`Cite une section attendue, ou rien si la réponse n’est pas dans la base (${answers.chat_model}, ${answers.answer_k} sections). Ne vérifie pas l’exactitude des faits.`}
              />
            )}
          </div>
        </section>

        {strategyRuns.length > 1 && (
          <section className="flex flex-col gap-6">
            <div>
              <Eyebrow>Comparaison</Eyebrow>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Stratégies de recherche</h2>
              <p className="mt-3 max-w-3xl text-ink/75">
                Même modèle d’embedding et mêmes questions ; seule la façon de choisir les sections change. La
                latence couvre l’embedding, la base et, pour deux stratégies, un appel LLM en plus.
              </p>
            </div>
            <Table head={["Stratégie", "Hit rate@1", `Hit rate@${run.k}`, "MRR", "Latence médiane", "Appel LLM en plus", "Erreur de citation"]}>
              {strategyRuns.map((r) => (
                <tr key={r.strategy} className="border-t border-ink/10">
                  <td className={CELL}>
                    {strategyLabel(r.strategy)}
                    {r.strategy === defaultStrategy && <span className="ml-2 text-accent">démo</span>}
                  </td>
                  <td className={CELL}>{percent(r.retrieval.hit_rate_at_1)}</td>
                  <td className={CELL}>{percent(r.retrieval.hit_rate_at_k)}</td>
                  <td className={CELL}>{r.retrieval.mrr.toFixed(3)}</td>
                  <td className={CELL}>{r.retrieval.latency_ms_median} ms</td>
                  <td className={CELL}>
                    {STRATEGIES.find((s) => s.value === r.strategy)?.extraLlmCall ? "oui" : "non"}
                  </td>
                  <td className={CELL}>{r.answers ? percent(r.answers.citation_error_rate) : "—"}</td>
                </tr>
              ))}
            </Table>
          </section>
        )}

        {modelRuns.length > 1 && (
          <section className="flex flex-col gap-6">
            <div>
              <Eyebrow>Comparaison</Eyebrow>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Modèles d’embedding</h2>
              <p className="mt-3 max-w-3xl text-ink/75">Recherche vectorielle, seul le modèle change.</p>
            </div>
            <Table head={["Modèle", "Hit rate@1", "Hit rate@k", "MRR", "Erreur de citation"]}>
              {modelRuns.map((r) => (
                <tr key={r.embedding_model} className="border-t border-ink/10">
                  <td className={`${CELL} font-mono`}>
                    {r.embedding_model}
                    {r.embedding_model === PRODUCTION_MODEL && <span className="ml-2 text-accent">démo</span>}
                  </td>
                  <td className={CELL}>{percent(r.retrieval.hit_rate_at_1)}</td>
                  <td className={CELL}>{percent(r.retrieval.hit_rate_at_k)}</td>
                  <td className={CELL}>{r.retrieval.mrr.toFixed(3)}</td>
                  <td className={CELL}>{r.answers ? percent(r.answers.citation_error_rate) : "—"}</td>
                </tr>
              ))}
            </Table>
          </section>
        )}

        {vectorRun && threshold && (
          <section className="flex flex-col gap-6">
            <div>
              <Eyebrow>Seuil</Eyebrow>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Faux positifs et faux négatifs</h2>
              <p className="mt-3 max-w-3xl text-ink/75">
                Recherche vectorielle : si la similarité du premier résultat est sous le seuil, on considère que la
                réponse n’est pas dans la base. Faux positif : question sans réponse au-dessus du seuil. Faux
                négatif : question avec réponse en dessous.
              </p>
            </div>
            <Card>
              <ScoreStrip run={vectorRun} threshold={threshold.best.threshold} />
            </Card>
            <Table head={["Seuil", "Faux positifs", "Faux négatifs", "Taux d’erreur"]}>
              {threshold.rows.map((row) => (
                <tr key={row.threshold} className="border-t border-ink/10">
                  <td className={`${CELL} font-mono`}>{row.threshold.toFixed(2)}</td>
                  <td className={CELL}>{row.false_positives}</td>
                  <td className={CELL}>{row.false_negatives}</td>
                  <td className={CELL}>{percent(row.error_rate)}</td>
                </tr>
              ))}
            </Table>
            <p className="max-w-3xl text-ink/75">
              Un seuil n’écarte que les questions hors sujet. Les questions sur Mondria dont la réponse n’est pas dans
              la base obtiennent des scores proches de celles qui ont une réponse : c’est la consigne du prompt
              (répondre uniquement à partir du contexte) qui les refuse.
            </p>
          </section>
        )}

        <section className="flex flex-col gap-6">
          <div>
            <Eyebrow>Détail</Eyebrow>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">Question par question</h2>
          </div>
          <EvaluationDetails runs={strategyRuns} initialStrategy={run.strategy} />
        </section>
      </div>
    </main>
  );
}
