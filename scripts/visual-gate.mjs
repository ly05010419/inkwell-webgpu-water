#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";
import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";

const repoRoot = path.resolve(import.meta.dirname, "..");
const port = Number(process.env.WATER_VISUAL_PORT || 3144);
const baseUrl = `http://127.0.0.1:${port}`;
const outputDir = path.join(repoRoot, "benchmarks", "visual");
const baselineMode = process.env.WATER_VISUAL_BASELINE === "1";
const limits = {
  mismatchRatio: Number(process.env.WATER_VISUAL_MAX_MISMATCH || 0.08),
  meanAbsoluteChannelError: Number(process.env.WATER_VISUAL_MAX_MAE || 3.5),
  p95ChannelError: Number(process.env.WATER_VISUAL_MAX_P95 || 9),
};
const cases = [
  { id: "surface", view: "surface" },
  { id: "underwater", view: "underwater" },
];

const server = spawn("npm", ["run", "start", "--", "--hostname", "127.0.0.1", "--port", String(port)], { cwd: repoRoot, env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1" }, stdio: "ignore" });

async function waitForServer() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { if ((await fetch(baseUrl)).ok) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Timed out waiting for the visual parity server");
}

function compareImages(reference, candidate) {
  const diff = new PNG({ width: reference.width, height: reference.height });
  const mismatchedPixels = pixelmatch(reference.data, candidate.data, diff.data, reference.width, reference.height, { threshold: 0.08, includeAA: false });
  const errors = [];
  let total = 0;
  for (let index = 0; index < reference.data.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const error = Math.abs(reference.data[index + channel] - candidate.data[index + channel]);
      errors.push(error);
      total += error;
    }
  }
  errors.sort((a, b) => a - b);
  return {
    diff,
    mismatchedPixels,
    mismatchRatio: mismatchedPixels / (reference.width * reference.height),
    meanAbsoluteChannelError: total / errors.length,
    p95ChannelError: errors[Math.floor(errors.length * 0.95)],
  };
}

let browser;
try {
  await waitForServer();
  await fs.mkdir(outputDir, { recursive: true });
  browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--use-gl=angle", "--disable-dawn-features=disallow_unsafe_apis"] });
  const results = [];
  for (const testCase of cases) {
    const captures = {};
    for (const mode of ["reference", "optimized"]) {
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const url = `${baseUrl}/?benchmark=1&mode=${mode}&view=${testCase.view}&mesh=240&simulation=256&scale=1&fixedTime=8.25`;
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await page.waitForFunction(() => window.__WEBGPU_WATER_LAB__?.ready === true && window.__WEBGPU_WATER_LAB__?.getMetrics()?.error === null, null, { timeout: 60_000 });
      await page.waitForTimeout(700);
      await page.addStyleTag({ content: "aside { display: none !important; }" });
      captures[mode] = await page.screenshot();
      await fs.writeFile(path.join(outputDir, `${testCase.id}-${mode}.png`), captures[mode]);
      await page.close();
    }
    const comparison = compareImages(PNG.sync.read(captures.reference), PNG.sync.read(captures.optimized));
    const diffPath = path.join(outputDir, `${testCase.id}-diff.png`);
    await fs.writeFile(diffPath, PNG.sync.write(comparison.diff));
    const result = {
      ...testCase,
      ...comparison,
      diff: undefined,
      diffPath,
      pass: baselineMode || (comparison.mismatchRatio <= limits.mismatchRatio && comparison.meanAbsoluteChannelError <= limits.meanAbsoluteChannelError && comparison.p95ChannelError <= limits.p95ChannelError),
    };
    results.push(result);
  }
  const report = { capturedAt: new Date().toISOString(), comparison: "optimized against reference A/B", viewport: "1440×900 DPR1", fixedTime: 8.25, limits, results, pass: results.every((result) => result.pass) };
  await fs.writeFile(path.join(outputDir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!report.pass) process.exitCode = 1;
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}
