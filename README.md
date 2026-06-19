# Local Ollama Chat

A local, multimodal chat interface for [Ollama](https://ollama.com). Install Ollama,
pull a model, run this app — your conversations and images never leave your machine.

- **Streaming** responses, token by token
- **Multimodal** — attach images to vision-capable models (e.g. `llava`)
- **Hardware-aware model picker** — detects your CPU/RAM/GPU and recommends the
  best installed model, tagging each as 🟢 fast / 🟡 ok / 🔴 slow for your machine
- **Smart context management** — editable system prompt, a token budget that never
  overflows the model's context window, and a rolling summary that gives long
  conversations effectively unlimited memory
- **Local history + memory** persisted in the browser (`localStorage`)
- **No AI SDK / no API keys** — talks straight to Ollama's HTTP API

## Prerequisites

- **Node.js 18+** (Next.js 16 will not run on older Node). If you use `nvm`:

  ```bash
  nvm use 20   # or any 18+
  ```

  > Some machines have an old system `node` (e.g. v12) on the default `PATH`,
  > which is too old. Make sure `node --version` reports 18+ before continuing.

- **Ollama** installed and running. Install it from
  [ollama.com/download](https://ollama.com/download), then verify:

  ```bash
  ollama --version
  ollama serve        # usually already running as a background service
  ```

## Installing models

This app does **not** download models for you — it lists whatever you've pulled
with Ollama. Pull at least one model first:

```bash
# Browse the catalog at https://ollama.com/library
ollama pull llava            # vision (multimodal), ~4.7 GB — recommended starting point
ollama pull llama3.1:8b      # fast text-only chat, ~4.9 GB

ollama list                  # see what you have installed
ollama rm <model>            # remove one you no longer want
```

### Which model should I pull?

Pick a model whose **loaded size fits your GPU VRAM** (or system RAM if you have
no GPU). Loaded size is roughly `disk size × 1.2 + ~2 GB` for the context window.

| Your hardware              | Good text models             | Good vision (multimodal) models |
| -------------------------- | ---------------------------- | ------------------------------- |
| 8 GB VRAM / 16 GB RAM      | `llama3.2:3b`, `qwen2.5:7b`  | `llava:7b`, `moondream`         |
| 12–16 GB VRAM              | `llama3.1:8b`, `qwen2.5:14b` | `llava:13b`, `bakllava`         |
| 24 GB VRAM (e.g. RTX 4090) | `qwen2.5:14b`, `gemma2:27b`  | `llava:13b`, `llava:34b`        |
| No GPU (CPU only)          | `llama3.2:3b`                | `moondream`, `llava:7b`         |

Bigger models are more capable but slower; if a model doesn't fit your VRAM,
Ollama offloads the rest to CPU/RAM and generation slows down a lot. Once memory
is exhausted it swaps to disk and becomes unusable for interactive chat.

> **Compatibility note:** a few vision models (notably `llama3.2-vision`, which
> uses the `mllama` architecture) require a recent Ollama build and will fail to
> load on older ones with `unknown model architecture: 'mllama'`. If that
> happens, use `llava` (or `bakllava`) instead, or upgrade Ollama.

## Run

```bash
npm install      # first time only
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The app detects your hardware,
auto-selects the best-fitting installed model, and you can start chatting. Models
marked with 👁 accept image attachments (📎 button); ★ marks the recommended model.

## How the hardware detection works

On load, the app calls `/api/system` to read your machine's specs and ranks every
installed model against them:

- **CPU / RAM** via Node's `os` module.
- **GPU VRAM** via `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), or unified memory on
  Apple Silicon. If none is found, it falls back to a CPU/RAM-only estimate.

Each model is then badged in the picker:

- 🟢 **Fast** — estimated loaded size fits in GPU VRAM → fully GPU-accelerated
- 🟡 **OK** — fits in system RAM → partial GPU + CPU, noticeably slower
- 🔴 **Slow** — exceeds total memory → swaps to disk, not practical for chat

The header strip shows your detected GPU/RAM and offers a one-click switch to the
recommended model. The recommendation prefers the **most capable model that fits
the GPU with headroom to spare** — a model that maxes out VRAM (shown as "tight
VRAM") leaves no room to grow the context window, so a slightly smaller, roomier
model is preferred as the default.

## Conversation, context window & memory

LLMs have a fixed **context window** (max tokens per request). `llava` (v1.6 /
Mistral-7B) supports 32K tokens, but images are costly — each can use up to
~2,880 tokens. Naively resending an ever-growing history eventually overflows
the window, at which point the model silently forgets the oldest turns. This app
manages that for you ([`lib/context.ts`](lib/context.ts)):

- **System prompt** — sets the assistant's behaviour. Edit it via the ⚙ button;
  it's saved per-browser and applied to your next message.
- **Token budget** — each request is capped to `num_ctx` (default 16,384 tokens),
  always reserving room for the reply. The most recent turns that fit are sent.
- **Rolling memory** — when older turns fall outside the budget, they're
  automatically summarized (via [`/api/summarize`](app/api/summarize/route.ts))
  into a running "memory" that's prepended to every request. The header shows
  **💾 memory active** once this kicks in, so long chats keep their context.

To use a larger raw context instead of (or alongside) summarization, raise
`maxNumCtx` in [`lib/ollama.ts`](lib/ollama.ts) — bigger windows use more VRAM and
slow down prompt processing, so 16K is a balanced default for ~12B models on a
24 GB GPU (gemma3:12b uses only ~8.4 GB even at 16K, leaving plenty of headroom).

## Configuration

By default the app talks to Ollama at `http://127.0.0.1:11434`. To point at a
different host/port, set `OLLAMA_HOST` before starting:

```bash
OLLAMA_HOST=http://192.168.1.50:11434 npm run dev
```

## How it works

```text
Browser ──▶ /api/system     GET  → os + nvidia-smi/rocm-smi (hardware detection)
Browser ──▶ /api/models     GET  → Ollama /api/tags + /api/show (caps + context)
Browser ──▶ /api/summarize  POST → Ollama /api/chat (non-stream) → rolling memory
Browser ──▶ /api/chat       POST → Ollama /api/chat (stream: true), proxied as NDJSON
```

- [`app/api/chat/route.ts`](app/api/chat/route.ts) — streams the Ollama NDJSON
  response straight back to the browser; aborts upstream if the client disconnects.
- [`app/api/models/route.ts`](app/api/models/route.ts) — lists installed models and
  enriches each with its capabilities so the UI knows which support `vision`.
- [`app/api/system/route.ts`](app/api/system/route.ts) — best-effort CPU/RAM/GPU
  detection used for the model recommendations.
- [`app/api/summarize/route.ts`](app/api/summarize/route.ts) — maintains the
  rolling conversation summary (long-term memory).
- [`lib/recommend.ts`](lib/recommend.ts) — pure functions that estimate each
  model's memory footprint and rank them for the detected hardware.
- [`lib/context.ts`](lib/context.ts) — pure functions that estimate tokens and
  fit the conversation (system prompt + memory + recent turns) into the budget.
- [`lib/useChat.ts`](lib/useChat.ts) — client hook: manages messages, builds the
  budgeted context, triggers summarization, parses the NDJSON stream, and persists
  history + memory.
- [`app/page.tsx`](app/page.tsx) — the chat UI (model picker, settings, hardware
  strip, image attachments).

Images are sent as base64 in each message's `images` array, exactly as Ollama's
multimodal API expects.
