import { Texture } from "three";
import { describe, expect, it } from "vitest";

import {
  createBuoyancySampler,
  createThreeGlobeWater,
} from "../src/three/index";

describe("Three.js compatibility entry point", () => {
  it("creates a shared spherical water controller without owning the renderer", async () => {
    const heightTexture = new Texture();
    const patchTexture = new Texture();
    const renderer = {} as never;
    const water = createThreeGlobeWater({
      renderer,
      radius: 100,
      heightField: { texture: heightTexture, metersPerPixel: 2 },
      projection: {
        worldSize: 4096,
        metersPerRadianLon: 6371000,
        metersPerRadianLat: 6371000,
      },
    });

    await water.init();
    expect(water.mesh.name).toBe("Three globe water");
    expect(water.waves.cascades.map((cascade) => cascade.lengthScale)).toEqual([240, 64, 12]);

    water.setPatch({
      texture: patchTexture,
      bounds: { lonMin: -0.2, latMin: -0.1, lonMax: 0.2, latMax: 0.1 },
    });
    water.setPatch(null);

    const buoyancy = createBuoyancySampler(water.waves);
    buoyancy.update(renderer, {
      planarX: 0,
      planarZ: 0,
      forwardX: 0,
      forwardZ: 1,
      effLong: 5,
      effMedium: 3,
      heaveMin: -2,
    });
    expect(buoyancy.poseBuffer.array).toHaveLength(4);
    expect(Number.isFinite((buoyancy.poseBuffer.array as Float32Array)[0])).toBe(true);

    water.dispose();
    expect(heightTexture.source.data).toBeNull();
    expect(patchTexture.source.data).toBeNull();
    buoyancy.dispose();
  });
});
