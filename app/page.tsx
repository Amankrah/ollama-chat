"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { CHAT_DEFAULTS, type OllamaModel } from "@/lib/ollama";
import { useChat, type ChatMessage } from "@/lib/useChat";
import { adaptiveNumCtx } from "@/lib/context";
import {
  fitForModel,
  recommendModel,
  totalVram,
  type FitTier,
  type SystemInfo,
} from "@/lib/recommend";

const MODEL_STORAGE_KEY = "ollama-chat:model";
const SYSTEM_PROMPT_KEY = "ollama-chat:system-prompt";

function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

const TIER_DOT: Record<FitTier, string> = {
  fast: "🟢",
  moderate: "🟡",
  slow: "🔴",
  unknown: "",
};

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

// Tailwind-styled element mapping so Markdown from the model renders cleanly.
const markdownComponents: Components = {
  p: ({ children }) => (
    <p className="my-2 leading-7 break-words first:mt-0 last:mb-0">{children}</p>
  ),
  h1: ({ children }) => (
    <h1 className="mt-4 mb-2 text-xl font-semibold first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-4 mb-2 text-lg font-semibold first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-3 mb-1.5 text-base font-semibold first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-3 mb-1.5 font-semibold first:mt-0">{children}</h4>
  ),
  ul: ({ children }) => (
    <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="leading-7 marker:text-zinc-400">{children}</li>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-600 underline underline-offset-2 break-words dark:text-blue-400"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-4 border-zinc-300 pl-3 text-zinc-600 italic dark:border-zinc-600 dark:text-zinc-300">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-4 border-zinc-200 dark:border-zinc-700" />,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-zinc-900 p-3 text-sm text-zinc-100 dark:bg-black">
      {children}
    </pre>
  ),
  code: ({ className, children }) => {
    const text = String(children);
    const isBlock = /language-/.test(className ?? "") || text.includes("\n");
    if (isBlock) return <code className="font-mono">{children}</code>;
    return (
      <code className="rounded bg-black/10 px-1.5 py-0.5 font-mono text-[0.85em] dark:bg-white/15">
        {children}
      </code>
    );
  },
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-zinc-300 bg-zinc-100 px-2 py-1 text-left font-semibold dark:border-zinc-700 dark:bg-zinc-800">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-zinc-300 px-2 py-1 dark:border-zinc-700">
      {children}
    </td>
  ),
};

/** Render assistant Markdown (GFM) with clean, themed styling. */
function MessageContent({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {content}
    </ReactMarkdown>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-[15px] ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
        }`}
      >
        {message.images && message.images.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {message.images.map((src, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={i}
                src={src}
                alt="attachment"
                className="max-h-48 rounded-lg object-cover"
              />
            ))}
          </div>
        )}
        {message.content ? (
          isUser ? (
            <p className="leading-7 break-words whitespace-pre-wrap">
              {message.content}
            </p>
          ) : (
            <MessageContent content={message.content} />
          )
        ) : (
          <span className="inline-block h-4 w-4 animate-pulse rounded-full bg-current opacity-40" />
        )}
      </div>
    </div>
  );
}

export default function Chat() {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [recommended, setRecommended] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [systemPrompt, setSystemPrompt] = useState(CHAT_DEFAULTS.systemPrompt);
  const [showSettings, setShowSettings] = useState(false);

  const activeModel = useMemo(
    () => models.find((m) => m.name === model),
    [models, model],
  );

  // Context window sized adaptively to the active model + this machine's VRAM.
  const numCtx = useMemo(
    () => adaptiveNumCtx(activeModel, system),
    [activeModel, system],
  );

  const { messages, isStreaming, status, error, hasMemory, send, stop, clear } =
    useChat({ model, numCtx, systemPrompt });
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportsVision = activeModel?.capabilities?.includes("vision") ?? false;
  const activeFit = activeModel ? fitForModel(activeModel, system) : null;

  const gpuLabel = useMemo(() => {
    if (!system) return null;
    if (system.gpus.length === 0) return "No GPU detected — CPU only";
    const vram = totalVram(system);
    const names = system.gpus.map((g) => g.name).join(", ");
    return `${names} · ${formatSize(vram)} VRAM`;
  }, [system]);

  // Detect hardware, then load models and pick the best default for this machine.
  useEffect(() => {
    (async () => {
      // Hardware detection is best-effort; never block the model list on it.
      let sys: SystemInfo | null = null;
      try {
        const sysRes = await fetch("/api/system");
        if (sysRes.ok) {
          sys = (await sysRes.json()) as SystemInfo;
          setSystem(sys);
        }
      } catch {
        // Ignore — fall back to size-only recommendation below.
      }

      try {
        const res = await fetch("/api/models");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load models.");
        const list: OllamaModel[] = data.models ?? [];
        setModels(list);

        const best = recommendModel(list, sys);
        setRecommended(best);

        const saved = localStorage.getItem(MODEL_STORAGE_KEY);
        const initial =
          (saved && list.some((m) => m.name === saved) && saved) ||
          best ||
          list[0]?.name ||
          null;
        setModel(initial);
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : "Failed to load models.");
      }
    })();
  }, []);

  // Persist model choice.
  useEffect(() => {
    if (model) localStorage.setItem(MODEL_STORAGE_KEY, model);
  }, [model]);

  // Restore a customised system prompt on mount.
  useEffect(() => {
    const saved = localStorage.getItem(SYSTEM_PROMPT_KEY);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time hydration from storage
    if (saved !== null) setSystemPrompt(saved);
  }, []);

  // Persist the system prompt.
  useEffect(() => {
    localStorage.setItem(SYSTEM_PROMPT_KEY, systemPrompt);
  }, [systemPrompt]);

  const handleModelChange = useCallback(
    (next: string) => {
      setModel(next);
      const nextSupportsVision =
        models.find((m) => m.name === next)?.capabilities?.includes("vision") ??
        false;
      if (!nextSupportsVision) setAttachments([]);
    },
    [models],
  );

  // Auto-scroll to newest content.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  // Tick an elapsed-seconds counter while a response is being generated, so a
  // slow first token (large model loading) never looks like a frozen UI.
  useEffect(() => {
    if (!isStreaming) return;
    const started = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => {
      clearInterval(id);
      setElapsed(0);
    };
  }, [isStreaming]);

  // True while streaming but no assistant tokens have arrived yet.
  const waitingForFirstToken =
    isStreaming && messages[messages.length - 1]?.content === "";

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files) return;
    const imgs = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const urls = await Promise.all(imgs.map(fileToDataUrl));
    setAttachments((prev) => [...prev, ...urls]);
  }, []);

  const submit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      if (isStreaming) return;
      send(input, attachments);
      setInput("");
      setAttachments([]);
    },
    [send, input, attachments, isStreaming],
  );

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex h-full flex-col bg-white dark:bg-zinc-950">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <h1 className="text-lg font-semibold tracking-tight">Ollama Chat</h1>
        <div className="ml-auto flex items-center gap-2">
          <select
            aria-label="Select model"
            value={model ?? ""}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={models.length === 0}
            className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-900"
          >
            {models.length === 0 && <option value="">No models</option>}
            {models.map((m) => {
              const fit = fitForModel(m, system);
              return (
                <option key={m.name} value={m.name}>
                  {TIER_DOT[fit.tier] && `${TIER_DOT[fit.tier]} `}
                  {m.name}
                  {m.capabilities?.includes("vision") ? " 👁" : ""}
                  {m.name === recommended ? " ★" : ""} · {formatSize(m.size)}
                </option>
              );
            })}
          </select>
          <button
            type="button"
            onClick={() => setShowSettings((s) => !s)}
            title="Settings"
            className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            ⚙
          </button>
          <button
            type="button"
            onClick={clear}
            disabled={messages.length === 0}
            className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 transition-colors hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Clear
          </button>
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div className="border-b border-zinc-200 bg-zinc-50 px-4 py-4 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mx-auto w-full max-w-3xl space-y-2">
            <div className="flex items-center justify-between">
              <label
                htmlFor="system-prompt"
                className="text-sm font-medium text-zinc-700 dark:text-zinc-200"
              >
                System prompt — sets the assistant&apos;s behaviour
              </label>
              <button
                type="button"
                onClick={() => setSystemPrompt(CHAT_DEFAULTS.systemPrompt)}
                className="text-xs text-blue-600 hover:underline dark:text-blue-400"
              >
                Reset to default
              </button>
            </div>
            <textarea
              id="system-prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
              className="w-full resize-y rounded-lg border border-zinc-300 bg-white p-2 text-sm outline-none focus:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-950"
            />
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Context window: {numCtx.toLocaleString()} tokens — sized
              automatically to {activeModel?.name ?? "this model"}
              {activeModel?.contextLength
                ? ` (native ${activeModel.contextLength.toLocaleString()})`
                : ""}{" "}
              and your available VRAM. Older turns are auto-summarized into
              long-term memory when they no longer fit. Changes apply to your
              next message.
            </p>
          </div>
        </div>
      )}

      {/* Hardware detection strip */}
      {system && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-zinc-200 px-4 py-1.5 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <span title={system.cpuModel}>🖥 {gpuLabel}</span>
          <span>🧠 {formatSize(system.totalRamBytes)} RAM</span>
          {activeFit && activeModel && (
            <span>
              {TIER_DOT[activeFit.tier]} {activeModel.name}: {activeFit.label}
            </span>
          )}
          <span title="Context window used for each request">
            🪟 {numCtx.toLocaleString()} ctx
          </span>
          {hasMemory && (
            <span
              className="text-emerald-600 dark:text-emerald-400"
              title="Earlier turns have been summarized into long-term memory"
            >
              💾 memory active
            </span>
          )}
          {recommended && recommended !== model && (
            <button
              type="button"
              onClick={() => handleModelChange(recommended)}
              className="ml-auto rounded-md px-2 py-0.5 text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-950"
            >
              ★ Best for this machine: {recommended} — switch
            </button>
          )}
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 py-6">
          {loadError ? (
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
              {loadError}
            </div>
          ) : messages.length === 0 ? (
            <div className="mt-24 text-center text-zinc-400">
              <p className="text-lg">Start a conversation with your local model.</p>
              {supportsVision && (
                <p className="mt-2 text-sm">
                  This model supports images — attach one with the 📎 button.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
            </div>
          )}

          {waitingForFirstToken && (
            <div className="mt-3 flex items-center gap-2 px-1 text-sm text-zinc-400">
              <span className="h-2 w-2 animate-ping rounded-full bg-zinc-400" />
              <span>
                {status === "summarizing"
                  ? `Updating memory (summarizing earlier turns)… ${elapsed}s`
                  : elapsed < 8
                    ? `Generating… ${elapsed}s`
                    : `Loading model & generating… ${elapsed}s — large models can take a while on first run`}
              </span>
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
        <form onSubmit={submit} className="mx-auto w-full max-w-3xl">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((src, i) => (
                <div key={i} className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={src}
                    alt="pending attachment"
                    className="h-16 w-16 rounded-lg object-cover"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setAttachments((prev) => prev.filter((_, j) => j !== i))
                    }
                    className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 text-xs text-white"
                    aria-label="Remove image"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex items-end gap-2 rounded-2xl border border-zinc-300 bg-white p-2 focus-within:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={!supportsVision}
              title={
                supportsVision
                  ? "Attach image"
                  : "Selected model doesn't support images"
              }
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-lg transition-colors hover:bg-zinc-100 disabled:opacity-30 dark:hover:bg-zinc-800"
            >
              📎
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => {
                handleFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={
                model ? `Message ${model}…` : "No model available"
              }
              disabled={!model}
              className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-[15px] outline-none disabled:opacity-50"
            />
            {isStreaming ? (
              <button
                type="button"
                onClick={stop}
                className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg bg-zinc-800 px-4 text-sm font-medium text-white dark:bg-zinc-200 dark:text-zinc-900"
              >
                <span className="h-2.5 w-2.5 rounded-[2px] bg-current" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!model || (!input.trim() && attachments.length === 0)}
                className="flex h-9 shrink-0 items-center rounded-lg bg-blue-600 px-4 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-40"
              >
                Send
              </button>
            )}
          </div>
          <p className="mt-1.5 px-1 text-xs text-zinc-400">
            Enter to send · Shift+Enter for a new line · runs fully locally
          </p>
        </form>
      </div>
    </div>
  );
}
