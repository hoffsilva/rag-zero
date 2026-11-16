"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { AskForm } from "@/components/ask-form";

// Response of POST /api/documents and GET /api/documents/{collection} (api/db.py Upload).
interface Upload {
  collection: string;
  title: string;
  file: string;
  kind: "markdown" | "text" | "pdf";
  size_bytes: number;
  sections: string[];
  language: string;
  summary: string;
  suggested_questions: string[];
  expires_at: string;
}

type Status = "idle" | "uploading" | "ready" | "error";

const STORAGE_KEY = "rag-zero:upload";
const MAX_BYTES = 1_000_000;
const ACCEPT = ".md,.markdown,.txt,.pdf";
const KIND_LABELS: Record<Upload["kind"], string> = { markdown: "Markdown", text: "Texte", pdf: "PDF" };

// localStorage can be unavailable (private mode, blocked storage): the page must work without it.
function readStoredCollection(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeCollection(collection: string | null) {
  try {
    if (collection) {
      window.localStorage.setItem(STORAGE_KEY, collection);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // ignore: the upload still works for this visit
  }
}

function formatSize(bytes: number): string {
  return bytes < 1024 ? `${bytes} o` : `${(bytes / 1024).toFixed(1)} Ko`;
}

function timeLeft(expiresAt: string, now: number): string {
  const minutes = Math.max(0, Math.round((new Date(expiresAt).getTime() - now) / 60_000));
  return minutes >= 60 ? `${Math.floor(minutes / 60)} h ${minutes % 60} min` : `${minutes} min`;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: { message?: unknown }; detail?: unknown };
    const message = body.error?.message ?? body.detail;
    if (typeof message === "string") {
      return message;
    }
  } catch {
    // fall through
  }
  return response.status === 429 ? "Trop de requêtes. Réessayez dans une minute." : "L’envoi a échoué. Réessayez.";
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-[0.18em] text-ink/50">{label}</dt>
      <dd className="mt-1 font-medium">{value}</dd>
    </div>
  );
}

export function DocumentWorkspace() {
  const [status, setStatus] = useState<Status>("idle");
  const [upload, setUpload] = useState<Upload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  // Current time for the expiry countdown: set when a document loads and every minute after,
  // never during the (pre)render.
  const [now, setNow] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Restore the visitor's document after a reload, unless it expired.
  useEffect(() => {
    const collection = readStoredCollection();
    if (!collection) {
      return;
    }
    fetch(`/api/documents/${collection}`)
      .then(async (response) => {
        if (!response.ok) {
          storeCollection(null);
          return;
        }
        setUpload((await response.json()) as Upload);
        setNow(Date.now());
        setStatus("ready");
      })
      .catch(() => storeCollection(null));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  async function send(file: File) {
    if (file.size > MAX_BYTES) {
      setError("Le fichier dépasse 1 Mo.");
      setStatus("error");
      return;
    }
    setStatus("uploading");
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/documents", { method: "POST", body: form });
      if (!response.ok) {
        setError(await errorMessage(response));
        setStatus("error");
        return;
      }
      const created = (await response.json()) as Upload;
      storeCollection(created.collection);
      setUpload(created);
      setNow(Date.now());
      setStatus("ready");
    } catch {
      setError("L’envoi a échoué. Réessayez.");
      setStatus("error");
    }
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      void send(file);
    }
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) {
      void send(file);
    }
  }

  function reset() {
    storeCollection(null);
    setUpload(null);
    setError(null);
    setStatus("idle");
  }

  if (status === "ready" && upload) {
    return (
      <div className="flex flex-col gap-10">
        <section className="rounded-2xl border border-ink/10 bg-navy-raised/95 p-7">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">Votre document</p>
              <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">{upload.title}</h2>
              <p className="mt-1 text-sm text-ink/60">{upload.file}</p>
            </div>
            <button
              type="button"
              onClick={reset}
              className="self-start rounded-full border border-ink/15 px-4 py-1.5 text-sm text-ink/75 transition-colors hover:border-ink/40 hover:text-ink"
            >
              Nouveau document
            </button>
          </div>

          <dl className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Fact label="Type" value={`${KIND_LABELS[upload.kind]} · ${formatSize(upload.size_bytes)}`} />
            <Fact label="Sections" value={String(upload.sections.length)} />
            <Fact label="Langue" value={upload.language} />
            <Fact label="Expire dans" value={timeLeft(upload.expires_at, now)} />
          </dl>

          <p className="mt-6 leading-relaxed text-ink/80">{upload.summary}</p>

          <details className="mt-5">
            <summary className="cursor-pointer text-sm font-semibold">Voir les sections indexées →</summary>
            <ol className="mt-3 list-decimal space-y-1 pl-6 text-sm text-ink/75">
              {upload.sections.map((section, index) => (
                <li key={`${index}-${section}`}>
                  <span className="font-mono text-ink/50">DOC-01-{String(index + 1).padStart(2, "0")}</span>{" "}
                  {section}
                </li>
              ))}
            </ol>
          </details>
        </section>

        <AskForm
          key={upload.collection}
          collection={upload.collection}
          examples={upload.suggested_questions}
          placeholder="Posez une question sur votre document…"
          showSourceLinks={false}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center gap-4 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
          dragging ? "border-accent bg-accent/5" : "border-ink/20 bg-navy-raised/60"
        }`}
      >
        <span className="block size-2.5 bg-accent" aria-hidden />
        <p className="font-display text-2xl font-bold">
          {status === "uploading" ? "Analyse en cours…" : "Déposez un fichier ici"}
        </p>
        <p className="max-w-md text-sm text-ink/60">
          {status === "uploading"
            ? "Extraction du texte, découpage en sections, embeddings et résumé."
            : ".md, .txt ou .pdf, jusqu’à 1 Mo."}
        </p>
        <button
          type="button"
          disabled={status === "uploading"}
          onClick={() => inputRef.current?.click()}
          className="rounded-full bg-accent px-6 py-3 font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
        >
          Choisir un fichier
        </button>
        <input ref={inputRef} type="file" accept={ACCEPT} onChange={handleFile} className="hidden" />
      </div>
      {status === "error" && error && <p className="text-accent">{error}</p>}
      <p className="text-sm text-ink/50">
        Votre document est isolé : seul ce navigateur peut l’interroger, et il est supprimé après 24 heures. Évitez
        les documents confidentiels.
      </p>
    </div>
  );
}
