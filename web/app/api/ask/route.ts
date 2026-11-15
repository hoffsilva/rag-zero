import { answerQuestion } from "@/lib/rag";

const MAX_QUESTION_LENGTH = 300;

export async function POST(request: Request) {
  const body: unknown = await request.json().catch(() => null);
  const question =
    typeof body === "object" && body !== null && "question" in body && typeof body.question === "string"
      ? body.question.trim()
      : "";

  if (!question) {
    return Response.json({ error: "Missing question" }, { status: 400 });
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return Response.json({ error: `Question longer than ${MAX_QUESTION_LENGTH} characters` }, { status: 400 });
  }

  try {
    return Response.json(await answerQuestion(question));
  } catch (error) {
    console.error(error);
    return Response.json({ error: "Could not answer the question" }, { status: 500 });
  }
}
