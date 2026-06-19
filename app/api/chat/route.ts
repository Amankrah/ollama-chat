import {
  ollamaFetch,
  ollamaUnreachableMessage,
  type OllamaMessage,
} from "@/lib/ollama";

// Streaming responses must always run at request time.
export const dynamic = "force-dynamic";
// Allow long generations on large local models.
export const maxDuration = 300;

interface ChatRequestBody {
  model: string;
  messages: OllamaMessage[];
  /** Ollama generation options, e.g. { num_ctx, temperature }. */
  options?: Record<string, number | string | boolean>;
}

/**
 * POST /api/chat
 * Proxies a chat request to the local Ollama server and streams the NDJSON
 * response straight back to the browser, which parses it chunk-by-chunk.
 */
export async function POST(req: Request) {
  let body: ChatRequestBody;
  try {
    body = (await req.json()) as ChatRequestBody;
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const { model, messages, options } = body;
  if (!model || !Array.isArray(messages) || messages.length === 0) {
    return Response.json(
      { error: "`model` and a non-empty `messages` array are required." },
      { status: 400 },
    );
  }

  let upstream: Response;
  try {
    upstream = await ollamaFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        ...(options ? { options } : {}),
      }),
      // Abort the upstream request if the client disconnects.
      signal: req.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return new Response(null, { status: 499 });
    }
    return Response.json({ error: ollamaUnreachableMessage() }, { status: 502 });
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return Response.json(
      {
        error:
          detail ||
          `Ollama returned ${upstream.status}. Is the model "${model}" installed?`,
      },
      { status: 502 },
    );
  }

  // Pass the raw NDJSON stream through unchanged — the client parses it.
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
