import { describe, expect, it } from "vitest";

import {
  DEFAULT_WATER_LAB_OPTIONS,
  TETHYS_WATER_FIELD_SIZE,
  WebGpuWaterEngine,
  waterSimulationBytes,
} from "../src";

describe("public package API", () => {
  it("exports a browser engine with asset-free defaults", () => {
    expect(WebGpuWaterEngine).toBeTypeOf("function");
    expect(DEFAULT_WATER_LAB_OPTIONS.mode).toBe("optimized");
    expect(DEFAULT_WATER_LAB_OPTIONS.shipModelUrl).toBeNull();
  });

  it("exports stable water profile helpers", () => {
    expect(TETHYS_WATER_FIELD_SIZE).toBe(192);
    expect(waterSimulationBytes(256)).toBe(1_048_576);
  });
});
