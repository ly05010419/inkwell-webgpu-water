"use client";

import { useEffect, useRef, useState } from "react";

import styles from "@/components/webgpu-water-lab-experience.module.css";
import {
  WebGpuWaterEngine,
  type WaterLabMetrics,
} from "@/lib/webgpu-water-engine";
import {
  WATER_PROFILES,
  type WaterRenderMode,
  type WaterScene,
  type WaterView,
} from "@/lib/water-profiles";

type BenchmarkBridge = {
  ready: boolean;
  getMetrics: () => WaterLabMetrics;
  setMode: (mode: WaterRenderMode) => void;
  setView: (view: WaterView) => void;
  setScene: (scene: WaterScene) => void;
  setMeshResolution: (resolution: number) => void;
  setSimulationResolution: (resolution: number) => void;
  setRenderScale: (scale: number) => void;
  resetMetrics: () => void;
};

declare global {
  interface Window {
    __WEBGPU_WATER_LAB__?: BenchmarkBridge;
  }
}

const EMPTY_METRICS: WaterLabMetrics = {
  ready: false,
  mode: "optimized",
  view: "surface",
  meshResolution: 240,
  simulationResolution: 256,
  triangles: 230_400,
  simulationBytes: 1_048_576,
  simulationSubsteps: 1,
  sceneCapturePasses: 0,
  disturbanceCount: 0,
  particleCount: 0,
  frameMeanMs: 0,
  frameP95Ms: 0,
  frameP99Ms: 0,
  frameMaxMs: 0,
  fps: 0,
  hitchFrames: 0,
  submitMeanMs: 0,
  gpuSimulationMeanMs: null,
  gpuSimulationP95Ms: null,
  gpuRenderMeanMs: null,
  gpuRenderP95Ms: null,
  gpuTimestampSamples: 0,
  adapter: "Requesting WebGPU adapter…",
  error: null,
};

function formatCount(value: number) {
  return new Intl.NumberFormat("en", { notation: value >= 1_000_000 ? "compact" : "standard", maximumFractionDigits: 2 }).format(value);
}

export function WebGpuWaterLabExperience() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<WebGpuWaterEngine | null>(null);
  const [mode, setModeState] = useState<WaterRenderMode>("optimized");
  const [view, setViewState] = useState<WaterView>("surface");
  const [scene, setSceneState] = useState<WaterScene>("open");
  const [meshResolution, setMeshResolutionState] = useState(240);
  const [simulationResolution, setSimulationResolutionState] = useState(256);
  const [renderScale, setRenderScaleState] = useState(1);
  const [metrics, setMetrics] = useState<WaterLabMetrics>(EMPTY_METRICS);
  const [starting, setStarting] = useState(true);
  const [showUi, setShowUi] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const query = new URLSearchParams(window.location.search);
    const requestedMode: WaterRenderMode = query.get("mode") === "reference" ? "reference" : "optimized";
    const requestedView: WaterView = query.get("view") === "underwater" ? "underwater" : "surface";
    const requestedScene: WaterScene = query.get("scene") === "shore" ? "shore" : "open";
    const requestedMesh = Math.max(96, Math.min(320, Number(query.get("mesh")) || 240));
    const requestedSimulation = Math.max(64, Math.min(512, Number(query.get("simulation")) || 256));
    const requestedScale = Math.max(0.5, Math.min(1.25, Number(query.get("scale")) || 1));
    const requestedFixedTime = Number(query.get("fixedTime"));
    const fixedTime = query.has("fixedTime") && Number.isFinite(requestedFixedTime) ? requestedFixedTime : undefined;
    const requestedYaw = Number(query.get("yaw"));
    const cameraYaw = query.has("yaw") && Number.isFinite(requestedYaw) ? requestedYaw : undefined;
    const requestedPitch = Number(query.get("pitch"));
    const cameraPitch = query.has("pitch") && Number.isFinite(requestedPitch) ? requestedPitch : undefined;
    const benchmark = query.get("benchmark") === "1";
    setShowUi(query.get("ui") !== "0");
    setModeState(requestedMode);
    setViewState(requestedView);
    setSceneState(requestedScene);
    setMeshResolutionState(requestedMesh);
    setSimulationResolutionState(requestedSimulation);
    setRenderScaleState(requestedScale);
    const engine = new WebGpuWaterEngine(canvas, {
      mode: requestedMode,
      view: requestedView,
      scene: requestedScene,
      meshResolution: requestedMesh,
      simulationResolution: requestedSimulation,
      renderScale: requestedScale,
      fixedTime,
      benchmark,
      cameraYaw,
      cameraPitch,
    });
    engineRef.current = engine;
    window.__WEBGPU_WATER_LAB__ = {
      ready: false,
      getMetrics: () => engine.getMetrics(),
      setMode: (value) => { engine.setMode(value); setModeState(value); },
      setView: (value) => { engine.setView(value); setViewState(value); },
      setScene: (value) => { engine.setScene(value); setSceneState(value); },
      setMeshResolution: (value) => { engine.setMeshResolution(value); setMeshResolutionState(value); },
      setSimulationResolution: (value) => { engine.setSimulationResolution(value); setSimulationResolutionState(value); },
      setRenderScale: (value) => { engine.setRenderScale(value); setRenderScaleState(value); },
      resetMetrics: () => engine.resetMetrics(),
    };
    void engine.init().then(() => {
      setStarting(false);
      if (window.__WEBGPU_WATER_LAB__) window.__WEBGPU_WATER_LAB__.ready = true;
    }).catch((error: unknown) => {
      setStarting(false);
      setMetrics((current) => ({ ...current, error: error instanceof Error ? error.message : String(error) }));
    });
    const telemetry = window.setInterval(() => setMetrics(engine.getMetrics()), 250);
    return () => {
      window.clearInterval(telemetry);
      engine.dispose();
      engineRef.current = null;
      delete window.__WEBGPU_WATER_LAB__;
    };
  }, []);

  const setMode = (value: WaterRenderMode) => { setModeState(value); engineRef.current?.setMode(value); };
  const setView = (value: WaterView) => { setViewState(value); engineRef.current?.setView(value); };
  const setScene = (value: WaterScene) => { setSceneState(value); engineRef.current?.setScene(value); };
  const applyProfile = (profile: (typeof WATER_PROFILES)[number]) => {
    setMeshResolutionState(profile.meshResolution);
    setSimulationResolutionState(profile.simulationResolution);
    setRenderScaleState(profile.renderScale);
    engineRef.current?.setMeshResolution(profile.meshResolution);
    engineRef.current?.setSimulationResolution(profile.simulationResolution);
    engineRef.current?.setRenderScale(profile.renderScale);
  };

  return (
    <main className={`${styles.shell} ${view === "underwater" ? styles.underwaterShell : ""}`}>
      <canvas ref={canvasRef} className={styles.canvas} aria-label="Raw WebGPU Tethys water biome" />
      {showUi && <aside className={styles.panel}>
        <p className={styles.eyebrow}>Inkwell renderer experiment</p>
        <h1>WebGPU Tethys</h1>
        <p className={styles.intro}>Tethys water, submerged sand, propagated wakes, depth-aware waves, refraction, reflections and caustics in one framework-independent WebGPU renderer.</p>

        <section>
          <span className={styles.label}>Water path</span>
          <div className={styles.segmented}>
            <button className={mode === "optimized" ? styles.active : ""} onClick={() => setMode("optimized")}>Optimized</button>
            <button className={mode === "reference" ? styles.active : ""} onClick={() => setMode("reference")}>Reference A/B</button>
          </div>
          <p className={styles.hint}>{mode === "optimized" ? "One compute propagation step and analytic scene refraction/reflection; no duplicate scene captures." : "Two production-style propagation substeps and the wider reflection sampling reference."}</p>
        </section>

        <section>
          <span className={styles.label}>Camera medium</span>
          <div className={styles.segmented}>
            <button className={view === "surface" ? styles.active : ""} onClick={() => setView("surface")}>Surface</button>
            <button className={view === "underwater" ? styles.active : ""} onClick={() => setView("underwater")}>Underwater</button>
          </div>
        </section>

        <section>
          <span className={styles.label}>Validation scene</span>
          <div className={styles.segmented}>
            <button className={scene === "open" ? styles.active : ""} onClick={() => setScene("open")}>Open water</button>
            <button className={scene === "shore" ? styles.active : ""} onClick={() => setScene("shore")}>Island shore</button>
          </div>
        </section>

        <section>
          <span className={styles.label}>Quality profile</span>
          <div className={styles.profilePresets}>
            {WATER_PROFILES.map((profile) => <button key={profile.id} onClick={() => applyProfile(profile)}>{profile.label}</button>)}
          </div>
        </section>

        <section className={styles.sliders}>
          <label>
            <span>Water mesh <output>{meshResolution}²</output></span>
            <input type="range" min="96" max="320" step="8" value={meshResolution} onChange={(event) => { const value = Number(event.target.value); setMeshResolutionState(value); engineRef.current?.setMeshResolution(value); }} />
          </label>
          <label>
            <span>Nearshore field <output>{simulationResolution}²</output></span>
            <input type="range" min="128" max="512" step="64" value={simulationResolution} onChange={(event) => { const value = Number(event.target.value); setSimulationResolutionState(value); engineRef.current?.setSimulationResolution(value); }} />
          </label>
          <label>
            <span>Render scale <output>{renderScale.toFixed(2)}×</output></span>
            <input type="range" min="0.5" max="1.25" step="0.05" value={renderScale} onChange={(event) => { const value = Number(event.target.value); setRenderScaleState(value); engineRef.current?.setRenderScale(value); }} />
          </label>
        </section>

        <dl className={styles.metrics}>
          <div><dt>Triangles</dt><dd>{formatCount(metrics.triangles)}</dd></div>
          <div><dt>Nearshore state</dt><dd>{(metrics.simulationBytes / 1_048_576).toFixed(2)} MiB</dd></div>
          <div><dt>Simulation</dt><dd>{metrics.simulationSubsteps} compute step{metrics.simulationSubsteps === 1 ? "" : "s"}</dd></div>
          <div><dt>Scene capture</dt><dd>{metrics.sceneCapturePasses} shared</dd></div>
          <div><dt>Disturbances</dt><dd>{formatCount(metrics.disturbanceCount)}</dd></div>
          <div><dt>FPS / mean</dt><dd>{metrics.fps.toFixed(0)} · {metrics.frameMeanMs.toFixed(2)}ms</dd></div>
          <div><dt>p95 / p99</dt><dd>{metrics.frameP95Ms.toFixed(2)} · {metrics.frameP99Ms.toFixed(2)}ms</dd></div>
          <div><dt>Max / hitches</dt><dd>{metrics.frameMaxMs.toFixed(2)} · {metrics.hitchFrames}</dd></div>
          <div><dt>JS submit</dt><dd>{metrics.submitMeanMs.toFixed(3)}ms</dd></div>
          <div><dt>GPU simulation</dt><dd>{metrics.gpuSimulationMeanMs === null ? "—" : `${metrics.gpuSimulationMeanMs.toFixed(3)}ms`}</dd></div>
          <div><dt>GPU render</dt><dd>{metrics.gpuRenderMeanMs === null ? "—" : `${metrics.gpuRenderMeanMs.toFixed(2)}ms`}</dd></div>
        </dl>

        <p className={styles.adapter}>{metrics.adapter}</p>
        <p className={styles.controls}>Drag to orbit · wheel to zoom · React owns only this panel</p>
      </aside>}

      {(starting || metrics.error) && <div className={styles.status}>{metrics.error ?? "Building Tethys compute fields…"}</div>}
      <output id="webgpu-water-lab-qa" data-ready={metrics.ready ? "true" : "false"} hidden>{JSON.stringify(metrics)}</output>
    </main>
  );
}
