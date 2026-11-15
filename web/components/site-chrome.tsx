import Link from "next/link";

const NAV_LINK = "text-sm text-ink/70 transition-colors hover:text-ink";

export function SiteHeader() {
  return (
    <header className="border-b border-ink/10">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-5">
        <Link href="/" className="flex items-center gap-2 font-display text-lg font-bold">
          <span className="size-2.5 bg-accent" aria-hidden />
          Mondria RAG
        </Link>
        <nav className="flex items-center gap-5">
          <Link href="/" className={NAV_LINK}>
            Démo
          </Link>
          <Link href="/evaluation" className={NAV_LINK}>
            Évaluation
          </Link>
          <a href="https://github.com/hoffsilva/rag-zero" className={NAV_LINK}>
            Code source
          </a>
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-t border-ink/10">
      <p className="mx-auto max-w-6xl px-4 py-6 text-sm text-ink/60">
        Projet personnel de Hoff Silva, non affilié à Mondria. Contenu résumé à partir des pages publiques de
        mondria.com.
      </p>
    </footer>
  );
}
