import { AskForm } from "@/components/ask-form";

export default function Home() {
  return (
    <>
      <header className="border-b border-ink/10">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-5">
          <span className="flex items-center gap-2 font-display text-lg font-bold">
            <span className="size-2.5 bg-accent" aria-hidden />
            Mondria RAG
          </span>
          <a
            href="https://github.com/hoffsilva/rag-zero"
            className="text-sm text-ink/70 transition-colors hover:text-ink"
          >
            Code source
          </a>
        </div>
      </header>

      <main className="flex flex-1 flex-col">
        <section className="hero-grid px-4 pb-16 pt-20 text-center sm:pt-28">
          <span className="inline-flex items-center gap-2 rounded-full border border-ink/15 bg-ink/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-ink/75">
            <span className="size-2 bg-accent" aria-hidden />
            Démo non officielle
          </span>
          <h1 className="mx-auto mt-8 max-w-4xl font-display text-5xl font-bold leading-[1.05] tracking-tight sm:text-7xl">
            Posez vos questions sur Mondria.
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-ink/75">
            La question devient un embedding, les sections les plus proches de la base de connaissances sont
            récupérées, puis un LLM répond uniquement à partir de ces sections, en citant ses sources.
          </p>
        </section>

        <section className="mx-auto w-full max-w-4xl px-4 pb-24">
          <AskForm />
        </section>
      </main>

      <footer className="border-t border-ink/10">
        <p className="mx-auto max-w-6xl px-4 py-6 text-sm text-ink/60">
          Projet personnel de Hoff Silva, non affilié à Mondria. Contenu résumé à partir des pages publiques de
          mondria.com.
        </p>
      </footer>
    </>
  );
}
