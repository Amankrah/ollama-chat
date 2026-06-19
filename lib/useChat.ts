"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CHAT_DEFAULTS, type OllamaChatChunk } from "@/lib/ollama";
import { buildContext, serializeForSummary } from "@/lib/context";

/** A message as the UI holds it. Images are kept as data URLs for rendering. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Data URLs (e.g. "data:image/png;base64,...") for attached images. */
  images?: string[];
}

const HISTORY_KEY = "ollama-chat:history";
const SUMMARY_KEY = "ollama-chat:summary";

let idCounter = 0;
function nextId(): string {
  idCounter += 1;
  return `${Date.now()}-${idCounter}`;
}

export interface UseChatOptions {
  model: string | null;
  /** Context window for requests, chosen adaptively for the active model. */
  numCtx: number;
  systemPrompt: string;
}

export function useChat({ model, numCtx, systemPrompt }: UseChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [summary, setSummary] = useState(""); // rolling long-term memory
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<"idle" | "summarizing" | "generating">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Restore history + memory from the previous session. Must run in an effect:
  // localStorage is unavailable during SSR and reading it at render time would
  // cause a hydration mismatch.
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem(HISTORY_KEY);
      const savedSummary = localStorage.getItem(SUMMARY_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional one-time hydration
      if (savedHistory) setMessages(JSON.parse(savedHistory) as ChatMessage[]);
      if (savedSummary) setSummary(savedSummary);
    } catch {
      // Ignore corrupt storage.
    }
  }, []);

  // Persist when idle (avoid thrashing localStorage during streaming).
  useEffect(() => {
    if (isStreaming) return;
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
      localStorage.setItem(SUMMARY_KEY, summary);
    } catch {
      // Storage full / unavailable — non-fatal.
    }
  }, [messages, summary, isStreaming]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
    setStatus("idle");
  }, []);

  const clear = useCallback(() => {
    stop();
    setMessages([]);
    setSummary("");
    setError(null);
  }, [stop]);

  const send = useCallback(
    async (text: string, images: string[]) => {
      const trimmed = text.trim();
      if ((!trimmed && images.length === 0) || !model || isStreaming) return;

      setError(null);

      const userMessage: ChatMessage = {
        id: nextId(),
        role: "user",
        content: trimmed,
        images: images.length ? images : undefined,
      };
      const assistantId = nextId();
      const history = [...messages, userMessage];

      setMessages((prev) => [
        ...prev,
        userMessage,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      setIsStreaming(true);
      setStatus("generating");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        // 1) Fit the conversation into the context budget. If older turns drop
        //    out, fold them into the rolling summary so they aren't forgotten.
        let built = buildContext({
          history,
          systemPrompt,
          summary,
          numCtx,
        });

        if (built.dropped.length > 0) {
          setStatus("summarizing");
          try {
            const sumRes = await fetch("/api/summarize", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model,
                previousSummary: summary,
                newText: serializeForSummary(built.dropped),
              }),
              signal: controller.signal,
            });
            if (sumRes.ok) {
              const { summary: updated } = (await sumRes.json()) as {
                summary: string;
              };
              if (updated) {
                setSummary(updated);
                // Rebuild so the fresh memory is included and dropped turns excluded.
                built = buildContext({
                  history,
                  systemPrompt,
                  summary: updated,
                  numCtx,
                });
              }
            }
          } catch (sumErr) {
            if (sumErr instanceof DOMException && sumErr.name === "AbortError") {
              throw sumErr; // user stopped — bubble up to the outer handler
            }
            // Summarization failed: proceed with the sliding window alone.
          }
          setStatus("generating");
        }

        // 2) Stream the actual reply.
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            messages: built.messages,
            options: {
              num_ctx: numCtx,
              temperature: CHAT_DEFAULTS.temperature,
            },
          }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          const data = await res.json().catch(() => null);
          throw new Error(data?.error || `Request failed (${res.status}).`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // Ollama streams newline-delimited JSON objects.
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let newline: number;
          while ((newline = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            if (!line) continue;

            const chunk = JSON.parse(line) as OllamaChatChunk;
            if (chunk.error) throw new Error(chunk.error);
            const piece = chunk.message?.content;
            if (piece) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + piece }
                    : m,
                ),
              );
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // User stopped — keep whatever streamed so far.
        } else {
          const message =
            err instanceof Error ? err.message : "Something went wrong.";
          setError(message);
          // Drop the empty assistant placeholder on hard failure.
          setMessages((prev) =>
            prev.filter((m) => !(m.id === assistantId && m.content === "")),
          );
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        setStatus("idle");
      }
    },
    [messages, model, isStreaming, systemPrompt, summary, numCtx],
  );

  return {
    messages,
    isStreaming,
    status,
    error,
    summary,
    hasMemory: summary.trim().length > 0,
    send,
    stop,
    clear,
  };
}
