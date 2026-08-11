#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";

const repoRoot = path.resolve(import.meta.dirname, "..");
const port = Number(process.env.WATER_SCREENSHOT_PORT || 3143);
const baseUrl = `http://127.0.0.1:${port}`;
const outputDir = path.join(repoRoot, "docs", "screenshots");
const views = [
  { name: "tethys-surface-optimized", query: "mode=optimized&view=surface&mesh=240&simulation=256&scale=1&fixedTime=18.25" },
  { name: "tethys-underwater-optimized", query: "mode=optimized&view=underwater&mesh=240&simulation=256&scale=1&fixedTime=8.25" },
  { name: "tethys-surface-reference", query: "mode=reference&view=surface&mesh=240&simulation=256&scale=1&fixedTime=18.25" },
  { name: "tethys-breaker-profile", query: "mode=optimized&view=surface&mesh=240&simulation=256&scale=1&fixedTime=18.25&yaw=2.05&pitch=-0.02" },
  { name: "tethys-breaker-reset-before", query: "mode=optimized&view=surface&mesh=240&simulation=256&scale=1&fixedTime=24.5" },
  { name: "tethys-breaker-reset-after", query: "mode=optimized&view=surface&mesh=240&simulation=256&scale=1&fixedTime=25.5" },
  { name: "tethys-island-shore", query: "mode=optimized&view=surface&scene=shore&simulation=256&scale=1&fixedTime=18.25" },
];

const server = spawn("npm", ["run", "start", "--", "--hostname", "127.0.0.1", "--port", String(port)], { cwd: repoRoot, env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1" }, stdio: "ignore" });

async function waitForServer() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { if ((await fetch(baseUrl)).ok) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Timed out waiting for the screenshot server");
}

let browser;
try {
  await waitForServer();
  await fs.mkdir(outputDir, { recursive: true });
  browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--use-gl=angle", "--disable-dawn-features=disallow_unsafe_apis"] });
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  for (const view of views) {
    const page = await context.newPage();
    await page.goto(`${baseUrl}/?${view.query}`, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForFunction(() => window.__WEBGPU_WATER_LAB__?.ready === true && window.__WEBGPU_WATER_LAB__?.getMetrics()?.error === null, null, { timeout: 60_000 });
    await page.waitForTimeout(700);
    await page.screenshot({ path: path.join(outputDir, `${view.name}.png`) });
    await page.addStyleTag({ content: "aside { display: none !important; }" });
    await page.screenshot({ path: path.join(outputDir, `${view.name}-clean.png`) });
    await page.close();
  }
  process.stdout.write(`Saved ${views.length * 2} screenshots to ${outputDir}\n`);
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}
