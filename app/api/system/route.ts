import os from "node:os";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { GpuInfo, SystemInfo } from "@/lib/recommend";

const execAsync = promisify(exec);

// Needs Node APIs (os, child_process) and reflects live machine state.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MIB = 1024 * 1024;

/** Probe NVIDIA GPUs via nvidia-smi. Returns [] if unavailable. */
async function detectNvidia(): Promise<GpuInfo[]> {
  try {
    const { stdout } = await execAsync(
      "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
      { timeout: 4000 },
    );
    return stdout
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [name, total, free] = line.split(",").map((s) => s.trim());
        return {
          name,
          vramTotalBytes: Number(total) * MIB,
          vramFreeBytes: Number(free) * MIB,
        };
      });
  } catch {
    return [];
  }
}

/** Probe AMD GPUs via rocm-smi. Returns [] if unavailable. */
async function detectAmd(): Promise<GpuInfo[]> {
  try {
    const { stdout } = await execAsync(
      "rocm-smi --showmeminfo vram --json",
      { timeout: 4000 },
    );
    const data = JSON.parse(stdout) as Record<
      string,
      Record<string, string>
    >;
    const gpus: GpuInfo[] = [];
    for (const [card, info] of Object.entries(data)) {
      const total = Number(
        info["VRAM Total Memory (B)"] ?? info["vram total memory (b)"] ?? 0,
      );
      const used = Number(
        info["VRAM Total Used Memory (B)"] ??
          info["vram total used memory (b)"] ??
          0,
      );
      if (total > 0) {
        gpus.push({
          name: `AMD GPU ${card}`,
          vramTotalBytes: total,
          vramFreeBytes: Math.max(0, total - used),
        });
      }
    }
    return gpus;
  } catch {
    return [];
  }
}

/**
 * On Apple Silicon the GPU shares system RAM (unified memory). Ollama can use
 * the bulk of it, so model the "VRAM" as ~70% of total RAM.
 */
function detectAppleSilicon(totalRam: number): GpuInfo[] {
  if (process.platform !== "darwin" || os.arch() !== "arm64") return [];
  const usable = Math.floor(totalRam * 0.7);
  return [
    {
      name: "Apple Silicon (unified memory)",
      vramTotalBytes: usable,
      vramFreeBytes: usable,
    },
  ];
}

/**
 * GET /api/system
 * Best-effort hardware detection used to recommend a model. Always returns
 * 200 with whatever it could determine (GPU detection may be empty).
 */
export async function GET() {
  const totalRamBytes = os.totalmem();
  const cpus = os.cpus();

  // Probe discrete GPUs in parallel; fall back to Apple unified memory.
  const [nvidia, amd] = await Promise.all([detectNvidia(), detectAmd()]);
  let gpus = [...nvidia, ...amd];
  if (gpus.length === 0) gpus = detectAppleSilicon(totalRamBytes);

  const info: SystemInfo = {
    platform: `${process.platform} ${os.arch()}`,
    cpuModel: cpus[0]?.model?.trim() ?? "Unknown CPU",
    cpuCount: cpus.length,
    totalRamBytes,
    freeRamBytes: os.freemem(),
    gpus,
  };

  return Response.json(info);
}
