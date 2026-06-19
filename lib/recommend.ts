// Hardware detection + model-fit ranking.
//
// Given the machine's memory/GPU and the installed models' sizes, estimate how
// well each model will run and pick the best default. Pure functions so they
// can be used on the client and unit-tested without any I/O.

import type { OllamaModel } from "@/lib/ollama";

const GB = 1024 ** 3;

export interface GpuInfo {
  name: string;
  vramTotalBytes: number;
  vramFreeBytes: number;
}

export interface SystemInfo {
  platform: string;
  cpuModel: string;
  cpuCount: number;
  totalRamBytes: number;
  freeRamBytes: number;
  gpus: GpuInfo[];
}

export type FitTier = "fast" | "moderate" | "slow" | "unknown";

export interface ModelFit {
  tier: FitTier;
  /** Short human label, e.g. "Fast · fits GPU". */
  label: string;
  /** Estimated loaded memory footprint in bytes. */
  estBytes: number;
  /** True if the model fits with VRAM headroom to spare for a large context. */
  comfortable: boolean;
}

/**
 * Fraction of VRAM a model's weights should stay under to be considered a
 * *comfortable* fit — the remaining headroom covers the KV cache for a large
 * context window and other overhead. A model that fits VRAM but exceeds this is
 * still GPU-resident, just with little room to grow its context.
 */
export const COMFORT_FACTOR = 0.72;

/** Total VRAM across all detected GPUs (Ollama can split a model across them). */
export function totalVram(system: SystemInfo): number {
  return system.gpus.reduce((sum, g) => sum + g.vramTotalBytes, 0);
}

/**
 * Estimate a model's loaded memory footprint from its on-disk size.
 * Loaded size ≈ weights (~1.2× disk for runtime overhead) plus a fixed buffer
 * for the KV cache / context window. Empirically this lands close: a 4.7 GB
 * llava loads to ~8.6 GB, a 67 GB llama4 to ~85 GB.
 */
export function estimateFootprint(sizeBytes: number): number {
  return sizeBytes * 1.2 + 2 * GB;
}

/** Classify how well a single model will run on the given machine. */
export function fitForModel(
  model: OllamaModel,
  system: SystemInfo | null,
): ModelFit {
  const estBytes = estimateFootprint(model.size);
  if (!system) {
    return { tier: "unknown", label: "", estBytes, comfortable: false };
  }

  const vram = totalVram(system);
  if (vram > 0 && estBytes <= vram) {
    const comfortable = estBytes <= vram * COMFORT_FACTOR;
    return {
      tier: "fast",
      label: comfortable ? "Fast · fits GPU" : "Fast · tight VRAM",
      estBytes,
      comfortable,
    };
  }
  if (estBytes <= system.totalRamBytes) {
    return {
      tier: "moderate",
      label: vram > 0 ? "OK · partial GPU + CPU" : "OK · runs on CPU",
      estBytes,
      comfortable: false,
    };
  }
  return { tier: "slow", label: "Slow · exceeds memory", estBytes, comfortable: false };
}

const TIER_RANK: Record<FitTier, number> = {
  fast: 3,
  moderate: 2,
  slow: 1,
  unknown: 0,
};

/**
 * Pick the best default model. Priority:
 *  1. Highest fit tier (fast > moderate > slow).
 *  2. Within the "fast" tier, prefer a *comfortable* fit (VRAM headroom for a
 *     big context) over a tight one — a model maxing out VRAM leaves no room to
 *     grow the context window and runs slower.
 *  3. Then the *largest* model (more parameters → more capable), except in the
 *     "slow" tier where the *smallest* (least painful) model wins.
 *
 * On a 24 GB GPU this picks e.g. gemma3:12b (capable + roomy) over a 24B model
 * that technically fits but pegs VRAM.
 */
export function recommendModel(
  models: OllamaModel[],
  system: SystemInfo | null,
): string | null {
  if (models.length === 0) return null;

  const ranked = [...models].sort((a, b) => {
    const fa = fitForModel(a, system);
    const fb = fitForModel(b, system);
    if (fa.tier !== fb.tier) return TIER_RANK[fb.tier] - TIER_RANK[fa.tier];
    if (fa.tier === "fast" && fa.comfortable !== fb.comfortable) {
      return Number(fb.comfortable) - Number(fa.comfortable);
    }
    // Same tier (and comfort): largest first, except "slow" where smallest wins.
    return fa.tier === "slow" ? a.size - b.size : b.size - a.size;
  });

  return ranked[0]?.name ?? null;
}
