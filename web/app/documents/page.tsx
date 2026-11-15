import type { Metadata } from "next";

import { DocumentWorkspace } from "@/components/document-workspace";

export const metadata: Metadata = {
  title: "Vos documents — Mondria RAG",
  description: "Upload a document and ask questions about it with the same RAG pipeline",
};

export default function DocumentsPage() {
  return (
    <main className="flex flex-1 flex-col">
      <section className="hero-grid px-4 pb-14 pt-20 text-center sm:pt-24">
        <span className="inline-flex items-center gap-2 rounded-full border border-ink/15 bg-ink/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-ink/75">
          <span className="size-2 bg-accent" aria-hidden />
          Vos documents
        </span>
        <h1 className="mx-auto mt-8 max-w-4xl font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
          Interrogez votre propre document.
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-ink/75">
          Le même pipeline que la démo Mondria : le texte est découpé en sections, chaque section devient un
          embedding dans pgvector, et le LLM répond uniquement à partir des sections retrouvées, en les citant.
        </p>
      </section>

      <section className="mx-auto w-full max-w-4xl px-4 pb-24">
        <DocumentWorkspace />
      </section>
    </main>
  );
}
