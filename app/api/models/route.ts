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
          const mi = info.model_info ?? {};
          // Keys are namespaced by architecture (e.g. "llama.context_length",
          // "gemma3.block_count"), so match by suffix.
          const num = (suffix: string): number | undefined => {
            for (const [k, v] of Object.entries(mi)) {
              if (k.endsWith(suffix) && typeof v === "number") return v;
            }
            return undefined;
          };

          const contextLength = num(".context_length");
          const blockCount = num(".block_count");
          const headCount = num(".attention.head_count");
          const headCountKv = num(".attention.head_count_kv") ?? headCount;
          const embedding = num(".embedding_length");
          const headDim =
            num(".attention.key_length") ??
            (embedding && headCount ? embedding / headCount : undefined);
          // KV cache per token (f16): layers × kv_heads × headDim × 2 (K and V)
          // × 2 bytes.
          const kvBytesPerToken =
            blockCount && headCountKv && headDim
              ? blockCount * headCountKv * headDim * 2 * 2
              : undefined;

          return {
            ...m,
            capabilities: info.capabilities ?? [],
            contextLength,
            kvBytesPerToken,
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
