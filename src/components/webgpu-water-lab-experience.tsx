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
  adapter: "正在请求 WebGPU 适配器…",
  error: null,
};

function formatCount(value: number) {
  return new Intl.NumberFormat("zh-CN", { notation: value >= 1_000_000 ? "compact" : "standard", maximumFractionDigits: 2 }).format(value);
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
      <canvas ref={canvasRef} className={styles.canvas} aria-label="原生 WebGPU 特提斯水体生物群系" />
      {showUi && <aside className={styles.panel}>
        <p className={styles.eyebrow}>Inkwell 渲染器实验</p>
        <h1>WebGPU 特提斯</h1>
        <p className={styles.intro}>特提斯水体、水下沙床、传播尾迹、深度感知波浪、折射、反射与焦散，全部集成于一个框架无关的 WebGPU 渲染器中。</p>

        <section>
          <span className={styles.label}>水体路径</span>
          <div className={styles.segmented}>
            <button className={mode === "optimized" ? styles.active : ""} onClick={() => setMode("optimized")}>优化路径</button>
            <button className={mode === "reference" ? styles.active : ""} onClick={() => setMode("reference")}>参考对照 A/B</button>
          </div>
          <p className={styles.hint}>{mode === "optimized" ? "单次计算传播步 + 解析式场景折射/反射；无重复场景捕获。" : "两个生产级传播子步，以及更宽的反射采样参考实现。"}</p>
        </section>

        <section>
          <span className={styles.label}>相机介质</span>
          <div className={styles.segmented}>
            <button className={view === "surface" ? styles.active : ""} onClick={() => setView("surface")}>水面上</button>
            <button className={view === "underwater" ? styles.active : ""} onClick={() => setView("underwater")}>水面下</button>
          </div>
        </section>

        <section>
          <span className={styles.label}>验证场景</span>
          <div className={styles.segmented}>
            <button className={scene === "open" ? styles.active : ""} onClick={() => setScene("open")}>开阔水域</button>
            <button className={scene === "shore" ? styles.active : ""} onClick={() => setScene("shore")}>岛屿海岸</button>
          </div>
        </section>

        <section>
          <span className={styles.label}>质量档位</span>
          <div className={styles.profilePresets}>
            {WATER_PROFILES.map((profile) => <button key={profile.id} onClick={() => applyProfile(profile)}>{profile.label}</button>)}
          </div>
        </section>

        <section className={styles.sliders}>
          <label>
            <span>水面网格 <output>{meshResolution}²</output></span>
            <input type="range" min="96" max="320" step="8" value={meshResolution} onChange={(event) => { const value = Number(event.target.value); setMeshResolutionState(value); engineRef.current?.setMeshResolution(value); }} />
          </label>
          <label>
            <span>近岸场 <output>{simulationResolution}²</output></span>
            <input type="range" min="128" max="512" step="64" value={simulationResolution} onChange={(event) => { const value = Number(event.target.value); setSimulationResolutionState(value); engineRef.current?.setSimulationResolution(value); }} />
          </label>
          <label>
            <span>渲染缩放 <output>{renderScale.toFixed(2)}×</output></span>
            <input type="range" min="0.5" max="1.25" step="0.05" value={renderScale} onChange={(event) => { const value = Number(event.target.value); setRenderScaleState(value); engineRef.current?.setRenderScale(value); }} />
          </label>
        </section>

        <dl className={styles.metrics}>
          <div><dt>三角形数</dt><dd>{formatCount(metrics.triangles)}</dd></div>
          <div><dt>近岸状态</dt><dd>{(metrics.simulationBytes / 1_048_576).toFixed(2)} MiB</dd></div>
          <div><dt>模拟</dt><dd>{metrics.simulationSubsteps} 个计算步</dd></div>
          <div><dt>场景捕获</dt><dd>{metrics.sceneCapturePasses} 次共享</dd></div>
          <div><dt>扰动数</dt><dd>{formatCount(metrics.disturbanceCount)}</dd></div>
          <div><dt>帧率 / 平均</dt><dd>{metrics.fps.toFixed(0)} · {metrics.frameMeanMs.toFixed(2)}ms</dd></div>
          <div><dt>p95 / p99</dt><dd>{metrics.frameP95Ms.toFixed(2)} · {metrics.frameP99Ms.toFixed(2)}ms</dd></div>
          <div><dt>最大 / 卡顿</dt><dd>{metrics.frameMaxMs.toFixed(2)} · {metrics.hitchFrames}</dd></div>
          <div><dt>JS 提交</dt><dd>{metrics.submitMeanMs.toFixed(3)}ms</dd></div>
          <div><dt>GPU 模拟</dt><dd>{metrics.gpuSimulationMeanMs === null ? "—" : `${metrics.gpuSimulationMeanMs.toFixed(3)}ms`}</dd></div>
          <div><dt>GPU 渲染</dt><dd>{metrics.gpuRenderMeanMs === null ? "—" : `${metrics.gpuRenderMeanMs.toFixed(2)}ms`}</dd></div>
        </dl>

        <p className={styles.adapter}>{metrics.adapter}</p>
        <p className={styles.controls}>拖拽旋转视角 · 滚轮缩放 · React 仅负责此面板</p>
      </aside>}

      {(starting || metrics.error) && <div className={styles.status}>{metrics.error ?? "正在构建特提斯计算场…"}</div>}
      <output id="webgpu-water-lab-qa" data-ready={metrics.ready ? "true" : "false"} hidden>{JSON.stringify(metrics)}</output>
    </main>
  );
}
