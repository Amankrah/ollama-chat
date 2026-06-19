import {
  CHAT_DEFAULTS,
  ollamaFetch,
  ollamaUnreachableMessage,
} from "@/lib/ollama";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

interface SummarizeBody {
  model: string;
  /** Existing rolling summary (may be empty). */
  previousSummary: string;
  /** New conversation text to fold into the summary. */
  newText: string;
}

/**
 * POST /api/summarize
 * Maintains a compact rolling summary of a conversation so older turns that no
 * longer fit the context window are still "remembered". Non-streaming.
 */
export async function POST(req: Request) {
  let body: SummarizeBody;
  try {
    body = (await req.json()) as SummarizeBody;
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const { model, previousSummary, newText } = body;
  if (!model || !newText?.trim()) {
    return Response.json(
      { error: "`model` and `newText` are required." },
      { status: 400 },
    );
  }

  const system =
    "You maintain a running summary of an ongoing conversation. Produce an " +
    "updated summary that preserves key facts, user preferences, decisions, " +
    "names, and unresolved questions. Be concise — at most ~180 words. " +
    "Output ONLY the summary text, no preamble.";

  const user =
    (previousSummary?.trim()
      ? `Previous summary:\n${previousSummary.trim()}\n\n`
      : "") +
    `New conversation since then:\n${newText.trim()}\n\n` +
    "Updated summary:";

  let upstream: Response;
  try {
    upstream = await ollamaFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        stream: false,
        options: { temperature: 0.2, num_ctx: CHAT_DEFAULTS.maxNumCtx },
      }),
      signal: req.signal,
    });
  } catch {
    return Response.json({ error: ollamaUnreachableMessage() }, { status: 502 });
  }

  if (!upstream.ok) {
    const detail = await upstream.text().catch(() => "");
    return Response.json(
      { error: detail || `Ollama returned ${upstream.status}.` },
      { status: 502 },
    );
  }

  const data = (await upstream.json()) as {
    message?: { content?: string };
  };
  let summary = (data.message?.content ?? "").trim();
  if (summary.length > CHAT_DEFAULTS.summaryMaxChars) {
    summary = summary.slice(0, CHAT_DEFAULTS.summaryMaxChars).trimEnd() + "…";
  }

  return Response.json({ summary });
}
