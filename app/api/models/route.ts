import {
  ollamaFetch,
  ollamaUnreachableMessage,
  type OllamaModel,
} from "@/lib/ollama";

// Always run at request time — the installed model list can change.
export const dynamic = "force-dynamic";

/**
 * GET /api/models
 * Returns the locally installed models, each enriched with its capabilities
 * (notably "vision") so the UI knows which models accept images.
 */
export async function GET() {
  let tagsRes: Response;
  try {
    tagsRes = await ollamaFetch("/api/tags");
  } catch {
    return Response.json({ error: ollamaUnreachableMessage() }, { status: 502 });
  }

  if (!tagsRes.ok) {
    return Response.json(
      { error: `Ollama returned ${tagsRes.status} listing models.` },
      { status: 502 },
    );
  }

  const { models = [] } = (await tagsRes.json()) as { models: OllamaModel[] };

  // /api/tags doesn't include capabilities, so ask /api/show per model.
  // These are fast local calls; run them in parallel.
  const enriched = await Promise.all(
    models.map(async (m) => {
      try {
        const showRes = await ollamaFetch("/api/show", {
          method: "POST",
          body: JSON.stringify({ name: m.name }),
        });
        if (showRes.ok) {
          const info = (await showRes.json()) as {
            capabilities?: string[];
            model_info?: Record<string, unknown>;
          };
          // The context length key is namespaced by architecture, e.g.
          // "llama.context_length" or "llama4.context_length".
          const ctxEntry = Object.entries(info.model_info ?? {}).find(([k]) =>
            k.endsWith(".context_length"),
          );
          const contextLength =
            typeof ctxEntry?.[1] === "number" ? ctxEntry[1] : undefined;
          return {
            ...m,
            capabilities: info.capabilities ?? [],
            contextLength,
          };
        }
      } catch {
        // Fall through to no-capabilities below.
      }
      return { ...m, capabilities: [] as string[] };
    }),
  );

  enriched.sort((a, b) => a.name.localeCompare(b.name));

  return Response.json({ models: enriched });
}
