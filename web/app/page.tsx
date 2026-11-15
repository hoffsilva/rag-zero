import { AskForm } from "@/components/ask-form";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-4 py-16">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Mondria RAG</h1>
        <p className="text-zinc-600 dark:text-zinc-400">
          Démo de Retrieval-Augmented Generation : la question est convertie en embedding, les sections les plus
          proches de la base de connaissances sont récupérées, puis un LLM répond uniquement à partir de ces
          sections, en citant ses sources.
        </p>
      </header>
      <AskForm />
    </main>
  );
}
