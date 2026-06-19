// Conversation context-window management.
//
// Ollama (like any LLM) has a fixed context window. If we naively send the whole
// growing history, we eventually overflow it — the model silently forgets the
// oldest turns and prompt processing slows down. This module budgets the prompt:
// it keeps the system prompt, a rolling "memory" summary, and as many of the most
// recent turns as fit, reporting which older turns dropped out (so they can be
// folded into the summary).

import { CHAT_DEFAULTS, type OllamaMessage, type OllamaModel } from "@/lib/ollama";
import { estimateFootprint, totalVram, type SystemInfo } from "@/lib/recommend";

/** Minimal message shape this module needs (compatible with the UI's ChatMessage). */
export interface BudgetMessage {
  role: "user" | "assistant";
  content: string;
  /** Data URLs or raw base64; only the count matters for token estimation. */
  images?: string[];
}

/** Strip a "data:...;base64," prefix Ollama doesn't want. */
function toRawBase64(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma === -1 ? dataUrl : dataUrl.slice(comma + 1);
}

/** Estimate the token cost of a message: text length plus a flat cost per image. */
export function estimateTokens(msg: { content: string; images?: string[] }): number {
  const textTokens = Math.ceil(msg.content.length / CHAT_DEFAULTS.charsPerToken);
  const imageTokens = (msg.images?.length ?? 0) * CHAT_DEFAULTS.imageTokens;
  return textTokens + imageTokens + 4; // +4 for role/formatting overhead
}

/**
 * Choose the context window (num_ctx) adaptively for a model on this machine.
 *
 * It's the largest window that fits within the model's native limit, the VRAM
 * left after the model's weights (so the KV cache won't spill to CPU), and a
 * latency ceiling — whichever is smallest. Falls back to a fixed modest window
 * when VRAM or architecture details aren't available (e.g. CPU-only).
 */
export function adaptiveNumCtx(
  model: OllamaModel | undefined,
  system: SystemInfo | null,
): number {
  const native = model?.contextLength;
  const kvPerToken = model?.kvBytesPerToken;
  const fallback = Math.min(
    native || CHAT_DEFAULTS.fallbackNumCtx,
    CHAT_DEFAULTS.fallbackNumCtx,
  );

  if (!model || !system || !native || !kvPerToken) return fallback;

  const vram = totalVram(system);
  if (vram <= 0) return fallback; // CPU-only — keep it modest.

  const headroom =
    vram - estimateFootprint(model.size) - CHAT_DEFAULTS.vramSafetyMarginBytes;
  if (headroom <= 0) return CHAT_DEFAULTS.minNumCtx;

  const maxFromVram = Math.floor(headroom / kvPerToken);
  let ctx = Math.min(native, maxFromVram, CHAT_DEFAULTS.ctxCeiling);
  ctx = Math.min(Math.max(ctx, CHAT_DEFAULTS.minNumCtx), native);
  // Round down to a tidy multiple of 1024.
  return Math.floor(ctx / 1024) * 1024 || CHAT_DEFAULTS.minNumCtx;
}

export interface BuiltContext {
  /** Ready-to-send messages: [system?, memory?, ...recent turns]. */
  messages: OllamaMessage[];
  /** Oldest turns that didn't fit and should be summarized into memory. */
  dropped: BudgetMessage[];
  /** Estimated prompt tokens used. */
  usedTokens: number;
}

/**
 * Assemble the message list to send, fitting within the model's context budget.
 *
 * Order of priority (highest first): system prompt → memory summary → newest
 * turns. We walk turns newest-to-oldest, keeping each while it fits; everything
 * older than the first one that doesn't fit is returned in `dropped`. The single
 * latest turn is always kept even if it alone is large.
 */
export function buildContext(opts: {
  history: BudgetMessage[];
  systemPrompt: string;
  summary: string;
  /** The context window (num_ctx) the request will use. */
  numCtx: number;
}): BuiltContext {
  const { history, systemPrompt, summary, numCtx } = opts;
  let budget = numCtx - CHAT_DEFAULTS.responseReserveTokens;

  const head: OllamaMessage[] = [];

  if (systemPrompt.trim()) {
    const m: OllamaMessage = { role: "system", content: systemPrompt.trim() };
    head.push(m);
    budget -= estimateTokens(m);
  }

  if (summary.trim()) {
    const m: OllamaMessage = {
      role: "system",
      content: `Summary of earlier conversation (for your memory):\n${summary.trim()}`,
    };
    head.push(m);
    budget -= estimateTokens(m);
  }

  // Walk newest → oldest, keeping turns that fit.
  const kept: BudgetMessage[] = [];
  let firstDroppedIdx = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    const cost = estimateTokens(history[i]);
    if (budget - cost < 0 && kept.length > 0) {
      firstDroppedIdx = i;
      break;
    }
    kept.unshift(history[i]);
    budget -= cost;
  }

  const dropped = firstDroppedIdx >= 0 ? history.slice(0, firstDroppedIdx + 1) : [];

  const recent: OllamaMessage[] = kept.map((m) => ({
    role: m.role,
    content: m.content,
    ...(m.images?.length ? { images: m.images.map(toRawBase64) } : {}),
  }));

  const messages = [...head, ...recent];
  const usedTokens = numCtx - CHAT_DEFAULTS.responseReserveTokens - budget;
  return { messages, dropped, usedTokens };
}

/** Render dropped turns as plain text for the summarizer (images become a note). */
export function serializeForSummary(messages: BudgetMessage[]): string {
  return messages
    .map((m) => {
      const who = m.role === "user" ? "User" : "Assistant";
      const imgNote = m.images?.length ? ` [sent ${m.images.length} image(s)]` : "";
      return `${who}${imgNote}: ${m.content}`;
    })
    .join("\n");
}
