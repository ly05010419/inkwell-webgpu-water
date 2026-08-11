import { loadGltf, type GltfMaterial, type GltfPrimitive } from "@/lib/gltf-loader";
import { SHIP_RENDER_SHADER, SHIP_TRANSFORM_SHADER } from "@/lib/ship-shaders";

export type ShipPlacement = {
  /** World-space XZ of the waterline centre. */
  readonly centre: readonly [number, number];
  /** Heading in radians about +Y; 0 points the bow along +X. */
  readonly heading: number;
  /** Vertical offset from the water surface, tuning how deep the hull sits. */
  readonly draft: number;
};

export type ShipRendererOptions = {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  readonly depthFormat: GPUTextureFormat;
  readonly worldUniformBuffer: GPUBuffer;
  readonly longField: GPUTextureView;
  readonly mediumField: GPUTextureView;
  readonly spectrumSampler: GPUSampler;
  /** Metres per tile for the long and medium spectral cascades. */
  readonly cascadeScales: readonly [number, number];
  readonly modelUrl: string;
  readonly placement: ShipPlacement;
};

type ShipPart = {
  readonly vertices: GPUBuffer;
  readonly indices: GPUBuffer;
  readonly indexCount: number;
  readonly bindGroup: GPUBindGroup;
  readonly material: GltfMaterial;
};

const TRANSFORM_BYTES = 128; // two mat4x4<f32>
const MATERIAL_BYTES = 16;
const VERTEX_STRIDE = 32; // vec3 position + vec3 normal + vec2 uv, tightly packed

/** A 1x1 stand-in so a material missing a map still binds a valid texture. */
function createFallbackTexture(device: GPUDevice, srgb: boolean, rgba: readonly [number, number, number, number]) {
  const texture = device.createTexture({
    label: "ship fallback texture",
    size: [1, 1],
    format: srgb ? "rgba8unorm-srgb" : "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, new Uint8Array(rgba), {}, [1, 1]);
  return texture;
}

async function uploadTexture(device: GPUDevice, url: string, srgb: boolean) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ship: could not load texture (${response.status}) from ${url}`);
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none" });
  try {
    const texture = device.createTexture({
      label: `ship texture ${url.split("/").pop()}`,
      size: [bitmap.width, bitmap.height],
      format: srgb ? "rgba8unorm-srgb" : "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
    return texture;
  } finally {
    bitmap.close();
  }
}

/** glTF stores position/normal/uv in separate accessors; interleave for one vertex buffer. */
function interleave(primitive: GltfPrimitive) {
  const vertexCount = primitive.positions.length / 3;
  const data = new Float32Array(vertexCount * 8);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const out = vertex * 8;
    data[out] = primitive.positions[vertex * 3];
    data[out + 1] = primitive.positions[vertex * 3 + 1];
    data[out + 2] = primitive.positions[vertex * 3 + 2];
    data[out + 3] = primitive.normals[vertex * 3];
    data[out + 4] = primitive.normals[vertex * 3 + 1];
    data[out + 5] = primitive.normals[vertex * 3 + 2];
    data[out + 6] = primitive.uvs[vertex * 2];
    data[out + 7] = primitive.uvs[vertex * 2 + 1];
  }
  return data;
}

function createBuffer(device: GPUDevice, label: string, data: ArrayBufferView, usage: GPUBufferUsageFlags) {
  const buffer = device.createBuffer({ label, size: Math.ceil(data.byteLength / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
  return buffer;
}

/**
 * Draws the glTF hull riding the simulated surface. It is recorded into the
 * scene pass alongside the terrain -- that pass owns depth, so the water
 * composited afterwards submerges the hull below its waterline for free.
 */
export class ShipRenderer {
  private transformBindGroup: GPUBindGroup;

  private constructor(
    private readonly device: GPUDevice,
    private readonly parts: readonly ShipPart[],
    private readonly hullHalves: readonly [number, number],
    private readonly transformPipeline: GPUComputePipeline,
    private readonly transformInputs: {
      readonly worldUniformBuffer: GPUBuffer;
      readonly placementBuffer: GPUBuffer;
      readonly hullSpanBuffer: GPUBuffer;
      readonly transformBuffer: GPUBuffer;
    },
    initialFields: { long: GPUTextureView; medium: GPUTextureView; sampler: GPUSampler },
    private readonly renderPipeline: GPURenderPipeline,
    private readonly sceneBindGroup: GPUBindGroup,
    private readonly buffers: readonly GPUBuffer[],
    private readonly textures: readonly GPUTexture[],
  ) {
    this.transformBindGroup = this.createTransformBindGroup(initialFields);
  }

  private createTransformBindGroup(fields: { long: GPUTextureView; medium: GPUTextureView; sampler: GPUSampler }) {
    return this.device.createBindGroup({
      label: "ship transform inputs",
      layout: this.transformPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.transformInputs.worldUniformBuffer } },
        { binding: 1, resource: fields.long },
        { binding: 2, resource: fields.medium },
        { binding: 3, resource: fields.sampler },
        { binding: 4, resource: { buffer: this.transformInputs.placementBuffer } },
        { binding: 5, resource: { buffer: this.transformInputs.hullSpanBuffer } },
        { binding: 6, resource: { buffer: this.transformInputs.transformBuffer } },
      ],
    });
  }

  /**
   * Re-points the buoyancy probes at freshly allocated cascade textures. The
   * engine reallocates those whenever the scene or simulation resolution
   * changes, which would otherwise leave this holding destroyed views.
   */
  bindSpectralFields(long: GPUTextureView, medium: GPUTextureView, sampler: GPUSampler) {
    this.transformBindGroup = this.createTransformBindGroup({ long, medium, sampler });
  }

  /**
   * Moves the hull. The scenes do not share a seabed, so a spot that is open
   * water in one can be dry dune in the other.
   */
  /** Keeps the buoyancy probes tiling-consistent with the live water surface. */
  setCascadeScales(longScale: number, mediumScale: number) {
    this.device.queue.writeBuffer(this.transformInputs.hullSpanBuffer, 0, new Float32Array([
      this.hullHalves[0], this.hullHalves[1], longScale, mediumScale,
    ]));
  }

  setPlacement(placement: ShipPlacement) {
    this.device.queue.writeBuffer(this.transformInputs.placementBuffer, 0, new Float32Array([
      placement.centre[0], placement.centre[1], placement.heading, placement.draft,
    ]));
  }

  static async create(options: ShipRendererOptions): Promise<ShipRenderer> {
    const { device, placement } = options;
    const model = await loadGltf(options.modelUrl);

    const transformBuffer = device.createBuffer({
      label: "ship rigid-body transform",
      size: TRANSFORM_BYTES,
      usage: GPUBufferUsage.STORAGE,
    });

    // Half-length and half-beam come from the model's own bounds, so the
    // buoyancy probes follow the hull rather than hard-coded dimensions.
    const halfLength = Math.max(model.max[0] - model.min[0], 1) * 0.5;
    const halfBeam = Math.max(model.max[2] - model.min[2], 1) * 0.5;
    const placementBuffer = createBuffer(device, "ship placement", new Float32Array([
      placement.centre[0], placement.centre[1], placement.heading, placement.draft,
    ]), GPUBufferUsage.UNIFORM);
    const hullSpanBuffer = createBuffer(device, "ship hull span", new Float32Array([
      halfLength, halfBeam, options.cascadeScales[0], options.cascadeScales[1],
    ]), GPUBufferUsage.UNIFORM);

    const transformModule = device.createShaderModule({ label: "ship transform", code: SHIP_TRANSFORM_SHADER });
    const transformPipeline = await device.createComputePipelineAsync({
      label: "ship rigid-body transform",
      layout: "auto",
      compute: { module: transformModule, entryPoint: "buildShipTransform" },
    });
    const renderModule = device.createShaderModule({ label: "ship render", code: SHIP_RENDER_SHADER });
    const renderPipeline = await device.createRenderPipelineAsync({
      label: "ship physically-based hull",
      layout: "auto",
      vertex: {
        module: renderModule,
        entryPoint: "shipVertex",
        buffers: [{
          arrayStride: VERTEX_STRIDE,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: { module: renderModule, entryPoint: "shipFragment", targets: [{ format: options.format }] },
      // Every material on this asset is double-sided (sails, rigging, planking).
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: options.depthFormat, depthWriteEnabled: true, depthCompare: "less" },
    });

    const sceneBindGroup = device.createBindGroup({
      label: "ship scene resources",
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: options.worldUniformBuffer } },
        { binding: 1, resource: { buffer: transformBuffer } },
      ],
    });

    const materialSampler = device.createSampler({
      label: "ship material sampler",
      addressModeU: "repeat",
      addressModeV: "repeat",
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      maxAnisotropy: 8,
    });

    const buffers: GPUBuffer[] = [transformBuffer, placementBuffer, hullSpanBuffer];
    const textures: GPUTexture[] = [];
    const fallbackBase = createFallbackTexture(device, true, [200, 200, 200, 255]);
    const fallbackNormal = createFallbackTexture(device, false, [128, 128, 255, 255]);
    const fallbackArm = createFallbackTexture(device, false, [255, 200, 0, 255]);
    textures.push(fallbackBase, fallbackNormal, fallbackArm);

    // Distinct URIs are uploaded once each even when materials share a map.
    const textureCache = new Map<string, GPUTexture>();
    const loadShared = async (uri: string | null, srgb: boolean, fallback: GPUTexture) => {
      if (!uri) return fallback;
      const cached = textureCache.get(uri);
      if (cached) return cached;
      const texture = await uploadTexture(device, uri, srgb);
      textureCache.set(uri, texture);
      textures.push(texture);
      return texture;
    };

    const parts: ShipPart[] = [];
    for (const primitive of model.primitives) {
      const { material } = primitive;
      const [base, normal, arm] = await Promise.all([
        loadShared(material.baseColorUri, true, fallbackBase),
        loadShared(material.normalUri, false, fallbackNormal),
        loadShared(material.metallicRoughnessUri, false, fallbackArm),
      ]);
      const materialBuffer = device.createBuffer({
        label: `ship material ${material.name}`,
        size: MATERIAL_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(materialBuffer, 0, new Float32Array([
        material.alphaCutoff,
        material.alphaMode === "MASK" ? 1 : 0,
        material.metallicFactor,
        material.roughnessFactor,
      ]));
      buffers.push(materialBuffer);

      const vertices = createBuffer(device, `ship vertices ${material.name}`, interleave(primitive), GPUBufferUsage.VERTEX);
      const indices = createBuffer(device, `ship indices ${material.name}`, primitive.indices, GPUBufferUsage.INDEX);
      buffers.push(vertices, indices);

      parts.push({
        vertices,
        indices,
        indexCount: primitive.indices.length,
        material,
        bindGroup: device.createBindGroup({
          label: `ship material bindings ${material.name}`,
          layout: renderPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: base.createView() },
            { binding: 1, resource: normal.createView() },
            { binding: 2, resource: arm.createView() },
            { binding: 3, resource: materialSampler },
            { binding: 4, resource: { buffer: materialBuffer } },
          ],
        }),
      });
    }

    return new ShipRenderer(
      device,
      parts,
      [halfLength, halfBeam],
      transformPipeline,
      { worldUniformBuffer: options.worldUniformBuffer, placementBuffer, hullSpanBuffer, transformBuffer },
      { long: options.longField, medium: options.mediumField, sampler: options.spectrumSampler },
      renderPipeline,
      sceneBindGroup,
      buffers,
      textures,
    );
  }

  get triangleCount() {
    return this.parts.reduce((total, part) => total + part.indexCount / 3, 0);
  }

  /** Must run before the scene pass that draws the hull. */
  updateTransform(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.transformPipeline);
    pass.setBindGroup(0, this.transformBindGroup);
    pass.dispatchWorkgroups(1);
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.sceneBindGroup);
    for (const part of this.parts) {
      pass.setBindGroup(1, part.bindGroup);
      pass.setVertexBuffer(0, part.vertices);
      pass.setIndexBuffer(part.indices, "uint32");
      pass.drawIndexed(part.indexCount);
    }
  }

  dispose() {
    for (const buffer of this.buffers) buffer.destroy();
    for (const texture of this.textures) texture.destroy();
    void this.device;
  }
}
