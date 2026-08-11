#!/usr/bin/env node

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";

const repoRoot = path.resolve(import.meta.dirname, "..");
const port = Number(process.env.WATER_WEBGPU_BENCH_PORT || 3142);
const baseUrl = process.env.WATER_WEBGPU_BENCH_URL || `http://127.0.0.1:${port}`;
const outputDir = path.join(repoRoot, "benchmarks");
const stamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
const sampleCount = Number(process.env.WATER_WEBGPU_BENCH_SAMPLES || 240);

const cases = [
  { id: "surface-optimized-a", mode: "optimized", view: "surface", mesh: 240, simulation: 256, scale: 1 },
  { id: "surface-optimized-b", mode: "optimized", view: "surface", mesh: 240, simulation: 256, scale: 1 },
  { id: "surface-reference", mode: "reference", view: "surface", mesh: 240, simulation: 256, scale: 1 },
  { id: "underwater-optimized-a", mode: "optimized", view: "underwater", mesh: 240, simulation: 256, scale: 1 },
  { id: "underwater-optimized-b", mode: "optimized", view: "underwater", mesh: 240, simulation: 256, scale: 1 },
  { id: "underwater-reference", mode: "reference", view: "underwater", mesh: 240, simulation: 256, scale: 1 },
  { id: "dense-water", mode: "optimized", view: "surface", mesh: 240, simulation: 512, scale: 1.25 },
  { id: "wide-ocean", mode: "optimized", view: "surface", mesh: 256, simulation: 384, scale: 0.9 },
  { id: "shore-dense", mode: "optimized", view: "surface", scene: "shore", mesh: 320, simulation: 512, scale: 1.25 },
];

const server = process.env.WATER_WEBGPU_BENCH_URL ? null : spawn(
  "npm",
  ["run", "start", "--", "--hostname", "127.0.0.1", "--port", String(port)],
  { cwd: repoRoot, env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1" }, stdio: "ignore" },
);

function hostLoad() {
  const [one, five, fifteen] = os.loadavg();
  return {
    cpuCount: os.cpus().length,
    loadAverage: [one, five, fifteen].map((value) => Number(value.toFixed(3))),
    normalizedOneMinuteLoad: Number((one / os.cpus().length).toFixed(3)),
  };
}

function caseUrl(testCase) {
  const url = new URL(baseUrl);
  url.searchParams.set("benchmark", "1");
  for (const key of ["mode", "view", "scene", "mesh", "simulation", "scale"]) {
    if (testCase[key] !== undefined) url.searchParams.set(key, String(testCase[key]));
  }
  return url.toString();
}

async function waitForServer() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { if ((await fetch(baseUrl)).ok) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Timed out waiting for the WebGPU water benchmark server");
}

function averageGpu(entries, key) {
  const values = entries.map((entry) => entry.metrics[key]).filter((value) => typeof value === "number");
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function renderHtml(report) {
  const rows = report.cases.map((entry) => `<tr><td>${entry.id}</td><td>${entry.mode}</td><td>${entry.view}</td><td>${entry.metrics.meshResolution}²</td><td>${entry.metrics.simulationResolution}²</td><td>${entry.frame.meanMs}</td><td>${entry.frame.p95Ms}</td><td>${entry.metrics.gpuSimulationMeanMs?.toFixed(3) ?? "—"}</td><td>${entry.metrics.gpuRenderMeanMs?.toFixed(3) ?? "—"}</td><td>${entry.frame.hitchesOver50Ms}</td></tr>`).join("");
  return `<!doctype html><meta charset="utf-8"><title>Inkwell WebGPU water benchmark</title><style>body{margin:40px;background:#0a2026;color:#e8f1e3;font:14px system-ui}main{max-width:1180px;margin:auto}h1{font:34px Georgia}.pass{color:#91deab}.fail{color:#ff9393}table{width:100%;border-collapse:collapse;background:#123039}th,td{padding:11px;border:1px solid #31535b;text-align:right}th:first-child,td:first-child{text-align:left}code{color:#9fd1df}</style><main><h1>Inkwell WebGPU Tethys benchmark</h1><p class="${report.pass ? "pass" : "fail"}">${report.pass ? "PASS" : "FAIL"} — spectral far field, nonlinear nearshore state, clipmap water geometry, and scene-integrated shore refraction.</p><p><code>${report.capturedAt}</code> · ${report.viewport} · ${report.adapter}</p><table><thead><tr><th>Case</th><th>Path</th><th>View</th><th>Mesh</th><th>Nearshore</th><th>Mean</th><th>p95</th><th>GPU simulation</th><th>GPU render</th><th>&gt;50ms</th></tr></thead><tbody>${rows}</tbody></table><p>${report.frameSampleCount} steady-state rAF intervals per case after warmup. Surface and underwater optimized production profiles are each confirmed twice.</p></main>`;
}

let browser;
try {
  const loadAtStart = hostLoad();
  await waitForServer();
  browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--use-gl=angle", "--disable-dawn-features=disallow_unsafe_apis"] });
  const results = [];
  for (const testCase of cases) {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const consoleErrors = [];
    const pageErrors = [];
    page.on("console", (message) => { if (message.type() === "error" && consoleErrors.length < 20) consoleErrors.push(message.text()); });
    page.on("pageerror", (error) => { if (pageErrors.length < 20) pageErrors.push(error.message); });
    const navigationStarted = performance.now();
    await page.goto(caseUrl(testCase), { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForFunction(() => window.__WEBGPU_WATER_LAB__?.ready === true && window.__WEBGPU_WATER_LAB__?.getMetrics()?.error === null, null, { timeout: 60_000 });
    await page.waitForFunction(() => (window.__WEBGPU_WATER_LAB__?.getMetrics()?.gpuTimestampSamples ?? 0) >= 8, null, { timeout: 60_000 });
    const readyMs = Number((performance.now() - navigationStarted).toFixed(2));
    await page.waitForTimeout(1_200);
    await page.evaluate(() => window.__WEBGPU_WATER_LAB__?.resetMetrics());
    const measurement = await page.evaluate(async (samples) => {
      const intervals = await new Promise((resolve) => {
        const values = [];
        let previous = performance.now();
        const sample = (now) => {
          values.push(now - previous);
          previous = now;
          if (values.length >= samples + 1) resolve(values.slice(1));
          else requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
      });
      const sorted = [...intervals].sort((a, b) => a - b);
      const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
      const at = (fraction) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
      return {
        metrics: window.__WEBGPU_WATER_LAB__?.getMetrics(),
        frame: {
          fps: Number((1000 / mean).toFixed(2)), meanMs: Number(mean.toFixed(3)), p95Ms: Number(at(0.95).toFixed(3)), p99Ms: Number(at(0.99).toFixed(3)), maxMs: Number(sorted.at(-1).toFixed(3)),
          hitchesOver33Ms: sorted.filter((value) => value > 33.34).length, hitchesOver50Ms: sorted.filter((value) => value > 50).length,
        },
      };
    }, sampleCount);
    results.push({ ...testCase, readyMs, ...measurement, consoleErrors, pageErrors });
    await page.close();
  }

  const surfaceOptimized = results.filter((entry) => entry.id.startsWith("surface-optimized"));
  const underwaterOptimized = results.filter((entry) => entry.id.startsWith("underwater-optimized"));
  const surfaceReference = results.find((entry) => entry.id === "surface-reference");
  const underwaterReference = results.find((entry) => entry.id === "underwater-reference");
  const surfaceOptimizedTotal = (averageGpu(surfaceOptimized, "gpuSimulationMeanMs") ?? 0) + (averageGpu(surfaceOptimized, "gpuRenderMeanMs") ?? 0);
  const underwaterOptimizedTotal = (averageGpu(underwaterOptimized, "gpuSimulationMeanMs") ?? 0) + (averageGpu(underwaterOptimized, "gpuRenderMeanMs") ?? 0);
  const surfaceReferenceTotal = (surfaceReference?.metrics.gpuSimulationMeanMs ?? 0) + (surfaceReference?.metrics.gpuRenderMeanMs ?? 0);
  const underwaterReferenceTotal = (underwaterReference?.metrics.gpuSimulationMeanMs ?? 0) + (underwaterReference?.metrics.gpuRenderMeanMs ?? 0);
  const optimizedConfirmations = [...surfaceOptimized, ...underwaterOptimized];
  const shoreDense = results.find((entry) => entry.id === "shore-dense");
  const gates = {
    minimumFps: 58,
    maximumMeanFrameMs: 17.5,
    maximumP95FrameMs: 20,
    maximumHitchesOver50Ms: 0,
    maximumSimulationBytes: 4_194_304,
    surfaceOptimizedGpuMs: Number(surfaceOptimizedTotal.toFixed(3)),
    surfaceReferenceGpuMs: Number(surfaceReferenceTotal.toFixed(3)),
    underwaterOptimizedGpuMs: Number(underwaterOptimizedTotal.toFixed(3)),
    underwaterReferenceGpuMs: Number(underwaterReferenceTotal.toFixed(3)),
    surfaceGpuImprovementPercent: Number(((1 - surfaceOptimizedTotal / surfaceReferenceTotal) * 100).toFixed(2)),
    underwaterGpuImprovementPercent: Number(((1 - underwaterOptimizedTotal / underwaterReferenceTotal) * 100).toFixed(2)),
    shoreDenseGpuMs: Number((((shoreDense?.metrics.gpuSimulationMeanMs ?? 0) + (shoreDense?.metrics.gpuRenderMeanMs ?? 0)).toFixed(3))),
    maximumShoreDenseGpuMs: 12,
    noConsoleOrPageErrors: results.every((entry) => entry.consoleErrors.length === 0 && entry.pageErrors.length === 0),
  };
  const pass = surfaceOptimized.length === 2 && underwaterOptimized.length === 2
    && optimizedConfirmations.every((entry) => entry.frame.fps >= gates.minimumFps && entry.frame.meanMs <= gates.maximumMeanFrameMs && entry.frame.p95Ms <= gates.maximumP95FrameMs && entry.frame.hitchesOver50Ms <= gates.maximumHitchesOver50Ms && entry.metrics.simulationBytes <= gates.maximumSimulationBytes)
    && surfaceReference && underwaterReference
    && surfaceOptimizedTotal < surfaceReferenceTotal && underwaterOptimizedTotal < underwaterReferenceTotal
    && shoreDense && shoreDense.frame.fps >= gates.minimumFps && shoreDense.frame.p95Ms <= gates.maximumP95FrameMs
    && shoreDense.frame.hitchesOver50Ms <= gates.maximumHitchesOver50Ms && gates.shoreDenseGpuMs <= gates.maximumShoreDenseGpuMs
    && gates.noConsoleOrPageErrors;
  const report = {
    benchmark: "inkwell-webgpu-water",
    capturedAt: new Date().toISOString(),
    viewport: "1440×900; fixed DPR 1",
    adapter: results[0]?.metrics.adapter ?? "unknown",
    rendererArchitecture: "Raw WebGPU/WGSL; three independent 128² TMA/JONSWAP spectral cascades with Stockham inverse FFTs; nonlinear 256² nearshore height/momentum/foam state with hydrostatic wet/dry reconstruction and spectral boundary forcing; persistent instability-triggered breaker events and backtraced foam; camera-snapped 4-level water clipmap plus a local 256×48 open-ocean breaker patch; 513² procedural terrain; shared atmosphere/sun/water lighting; optional one-pass scene-color/depth refraction capture in the island scene; no image textures, particles, splash pass, Three.js, R3F, or React-owned world simulation.",
    frameSampleCount: sampleCount,
    hostLoad: { started: loadAtStart, completed: hostLoad() },
    methodology: "Production Next.js server, isolated page per case, hardware WebGPU through Dawn/Metal, 1.2s warmup, steady-state rAF samples, timestamp queries split between simulation and rendering, two optimized confirmations per camera medium, and reference-path A/B cases.",
    cases: results,
    gates,
    pass: Boolean(pass),
  };
  await fs.mkdir(outputDir, { recursive: true });
  const json = `${JSON.stringify(report, null, 2)}\n`;
  await Promise.all([
    fs.writeFile(path.join(outputDir, `webgpu-water-${stamp}.json`), json),
    fs.writeFile(path.join(outputDir, `webgpu-water-${stamp}.html`), renderHtml(report)),
    fs.writeFile(path.join(outputDir, "webgpu-water-latest.json"), json),
    fs.writeFile(path.join(outputDir, "webgpu-water-latest.html"), renderHtml(report)),
  ]);
  process.stdout.write(`${json}\nSaved webgpu-water-${stamp}.{json,html}\n`);
  if (!pass) process.exitCode = 1;
} finally {
  await browser?.close();
  server?.kill("SIGTERM");
}
