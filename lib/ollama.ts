// Thin helpers around the local Ollama HTTP API.
// Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
//
// We talk to Ollama directly instead of going through an LLM SDK: the native
// API is stable, streams NDJSON, and supports multimodal input (base64 images
// per message) with no extra dependencies.

export const OLLAMA_HOST =
  process.env.OLLAMA_HOST?.replace(/\/$/, "") || "http://127.0.0.1:11434";

/** A single chat message in the shape Ollama expects. */
export interface OllamaMessage {
  role: "system" | "user" | "assistant";
  content: string;
  /** Base64-encoded images (no `data:` prefix). Only used by vision models. */
  images?: string[];
}

export interface OllamaModel {
  name: string;
  model: string;
  size: number;
  /** Capabilities such as "completion", "vision", "tools". Filled in by /api/show. */
  capabilities?: string[];
  /** Native context window in tokens (from model_info). Filled in by /api/show. */
  contextLength?: number;
  details?: {
    family?: string;
    parameter_size?: string;
    quantization_level?: string;
  };
}

/** Tunables for conversation / context-window management. */
export const CHAT_DEFAULTS = {
  /** Default behaviour for the assistant. Editable per-session in the UI. */
  systemPrompt:
    "You are a helpful, knowledgeable assistant running locally via Ollama. " +
    "Be clear, accurate, and concise. Use Markdown, and fenced code blocks for code. " +
    "If you are unsure or lack information, say so rather than inventing an answer. " +
    "When the user attaches an image, describe and reason about what you actually see in it.",
  /** Context window we ask Ollama to use. Capped to keep memory/latency sane even
   *  when a model technically supports far more (e.g. 128K). 16K is a comfortable
   *  default for ~12B models on a 24 GB GPU; the rolling summary covers anything
   *  beyond it. Raise toward a model's native limit if you have VRAM to spare. */
  maxNumCtx: 16384,
  /** Tokens held back within num_ctx for the model's reply. */
  responseReserveTokens: 1536,
  /** Rough bytes-per-token for estimating prompt size without a real tokenizer. */
  charsPerToken: 4,
  /** Approx token cost of a single attached image for a llava-class vision model. */
  imageTokens: 2048,
  /** Cap on the rolling-summary length (chars) so "memory" can't itself overflow. */
  summaryMaxChars: 2000,
  /** Sampling temperature. */
  temperature: 0.7,
} as const;

/** One NDJSON chunk streamed back from /api/chat. */
export interface OllamaChatChunk {
  message?: { role: string; content: string };
  done: boolean;
  error?: string;
}

export async function ollamaFetch(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  return fetch(`${OLLAMA_HOST}${path}`, {
    ...init,
    // Never cache anything from the local model server.
    cache: "no-store",
  });
}

/** Friendly message when the local server isn't reachable. */
export function ollamaUnreachableMessage(): string {
  return (
    `Couldn't reach Ollama at ${OLLAMA_HOST}. ` +
    `Make sure Ollama is installed and running (\`ollama serve\`), ` +
    `then reload this page.`
  );
}
