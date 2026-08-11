// A minimal glTF 2.0 reader for this lab's single static model. It handles
// exactly what the asset needs -- non-skinned triangle primitives with float
// POSITION/NORMAL/TEXCOORD_0 and scalar indices -- and rejects anything else
// loudly rather than importing a general-purpose loader into a renderer that
// deliberately has no 3D-library dependencies.

export type GltfAlphaMode = "OPAQUE" | "MASK" | "BLEND";

export type GltfMaterial = {
  readonly name: string;
  readonly baseColorUri: string | null;
  readonly normalUri: string | null;
  readonly metallicRoughnessUri: string | null;
  readonly alphaMode: GltfAlphaMode;
  readonly alphaCutoff: number;
  readonly metallicFactor: number;
  readonly roughnessFactor: number;
};

export type GltfPrimitive = {
  readonly name: string;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly indices: Uint32Array;
  readonly material: GltfMaterial;
};

export type GltfModel = {
  readonly primitives: readonly GltfPrimitive[];
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
};

const COMPONENT_READERS = {
  5120: { bytes: 1, read: (v: DataView, o: number) => v.getInt8(o) },
  5121: { bytes: 1, read: (v: DataView, o: number) => v.getUint8(o) },
  5122: { bytes: 2, read: (v: DataView, o: number) => v.getInt16(o, true) },
  5123: { bytes: 2, read: (v: DataView, o: number) => v.getUint16(o, true) },
  5125: { bytes: 4, read: (v: DataView, o: number) => v.getUint32(o, true) },
  5126: { bytes: 4, read: (v: DataView, o: number) => v.getFloat32(o, true) },
} as const;

const TYPE_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 } as const;

type ComponentType = keyof typeof COMPONENT_READERS;
type AccessorType = keyof typeof TYPE_COMPONENTS;

// The spec's structures, narrowed to the fields this reader consumes.
type RawAccessor = { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string };
type RawBufferView = { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number };
type RawTextureRef = { index: number };
type RawMaterial = {
  name?: string;
  alphaMode?: string;
  alphaCutoff?: number;
  normalTexture?: RawTextureRef;
  pbrMetallicRoughness?: {
    baseColorTexture?: RawTextureRef;
    metallicRoughnessTexture?: RawTextureRef;
    metallicFactor?: number;
    roughnessFactor?: number;
  };
};
type RawGltf = {
  asset?: { version?: string };
  accessors?: RawAccessor[];
  bufferViews?: RawBufferView[];
  buffers?: { uri?: string; byteLength: number }[];
  images?: { uri?: string }[];
  textures?: { source?: number }[];
  materials?: RawMaterial[];
  meshes?: { name?: string; primitives: { attributes: Record<string, number>; indices?: number; material?: number; mode?: number }[] }[];
};

function fail(message: string): never {
  throw new Error(`glTF: ${message}`);
}

function isComponentType(value: number): value is ComponentType {
  return value in COMPONENT_READERS;
}

function isAccessorType(value: string): value is AccessorType {
  return value in TYPE_COMPONENTS;
}

/**
 * Copies one accessor out into a flat array. Honours the bufferView stride so
 * interleaved sources still read correctly, even though this asset is not.
 */
function readAccessor(gltf: RawGltf, buffer: ArrayBuffer, accessorIndex: number): Float64Array {
  const accessor = gltf.accessors?.[accessorIndex] ?? fail(`accessor ${accessorIndex} is missing`);
  if (!isComponentType(accessor.componentType)) fail(`unsupported componentType ${accessor.componentType}`);
  if (!isAccessorType(accessor.type)) fail(`unsupported accessor type ${accessor.type}`);
  if (accessor.bufferView === undefined) fail(`accessor ${accessorIndex} has no bufferView; sparse data is not supported`);

  const view = gltf.bufferViews?.[accessor.bufferView] ?? fail(`bufferView ${accessor.bufferView} is missing`);
  const reader = COMPONENT_READERS[accessor.componentType];
  const components = TYPE_COMPONENTS[accessor.type];
  const elementBytes = reader.bytes * components;
  const stride = view.byteStride ?? elementBytes;
  const base = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);

  const required = base + stride * (accessor.count - 1) + elementBytes;
  if (required > buffer.byteLength) fail(`accessor ${accessorIndex} reads ${required} bytes past a ${buffer.byteLength}-byte buffer`);

  const data = new DataView(buffer);
  const out = new Float64Array(accessor.count * components);
  for (let element = 0; element < accessor.count; element += 1) {
    const offset = base + element * stride;
    for (let component = 0; component < components; component += 1) {
      out[element * components + component] = reader.read(data, offset + component * reader.bytes);
    }
  }
  return out;
}

function resolveTextureUri(gltf: RawGltf, ref: RawTextureRef | undefined): string | null {
  if (!ref) return null;
  const texture = gltf.textures?.[ref.index];
  if (!texture || texture.source === undefined) return null;
  return gltf.images?.[texture.source]?.uri ?? null;
}

function readMaterial(gltf: RawGltf, index: number | undefined): GltfMaterial {
  const raw: RawMaterial = (index === undefined ? undefined : gltf.materials?.[index]) ?? {};
  const pbr = raw.pbrMetallicRoughness ?? {};
  const alphaMode = raw.alphaMode === "MASK" || raw.alphaMode === "BLEND" ? raw.alphaMode : "OPAQUE";
  return Object.freeze({
    name: raw.name ?? `material_${index ?? "default"}`,
    baseColorUri: resolveTextureUri(gltf, pbr.baseColorTexture),
    normalUri: resolveTextureUri(gltf, raw.normalTexture),
    metallicRoughnessUri: resolveTextureUri(gltf, pbr.metallicRoughnessTexture),
    alphaMode,
    alphaCutoff: raw.alphaCutoff ?? 0.5,
    metallicFactor: pbr.metallicFactor ?? 1,
    roughnessFactor: pbr.roughnessFactor ?? 1,
  });
}

function baseDirectory(url: string) {
  const cut = url.lastIndexOf("/");
  return cut < 0 ? "" : url.slice(0, cut + 1);
}

async function fetchOrThrow(url: string, what: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`glTF: could not load ${what} (${response.status} ${response.statusText}) from ${url}`);
  return response;
}

/**
 * Loads a .gltf document plus its external binary buffer. Texture images are
 * left as URIs for the caller to upload, so this module stays free of any
 * rendering-API types.
 */
export async function loadGltf(url: string): Promise<GltfModel> {
  const gltf = (await (await fetchOrThrow(url, "document")).json()) as RawGltf;
  const version = gltf.asset?.version;
  if (version !== "2.0") fail(`unsupported asset version ${version ?? "(none)"}; expected 2.0`);

  const buffers = gltf.buffers ?? [];
  if (buffers.length !== 1) fail(`expected exactly one buffer, found ${buffers.length}`);
  const bufferUri = buffers[0].uri;
  if (!bufferUri) fail("embedded (GLB / data-URI) buffers are not supported");
  const binary = await (await fetchOrThrow(baseDirectory(url) + bufferUri, "buffer")).arrayBuffer();

  const directory = baseDirectory(url);
  const primitives: GltfPrimitive[] = [];
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  for (const mesh of gltf.meshes ?? []) {
    for (const primitive of mesh.primitives) {
      // Mode 4 is TRIANGLES. Strips/fans would need a different draw topology.
      if ((primitive.mode ?? 4) !== 4) fail(`primitive mode ${primitive.mode} is not TRIANGLES`);
      const { POSITION, NORMAL, TEXCOORD_0 } = primitive.attributes;
      if (POSITION === undefined) fail(`mesh "${mesh.name ?? "?"}" has a primitive without POSITION`);
      if (primitive.indices === undefined) fail(`mesh "${mesh.name ?? "?"}" has a non-indexed primitive`);

      const positions = Float32Array.from(readAccessor(gltf, binary, POSITION));
      const vertexCount = positions.length / 3;
      const normals = NORMAL === undefined
        ? new Float32Array(positions.length)
        : Float32Array.from(readAccessor(gltf, binary, NORMAL));
      const uvs = TEXCOORD_0 === undefined
        ? new Float32Array(vertexCount * 2)
        : Float32Array.from(readAccessor(gltf, binary, TEXCOORD_0));
      const indices = Uint32Array.from(readAccessor(gltf, binary, primitive.indices));

      for (let vertex = 0; vertex < vertexCount; vertex += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          const value = positions[vertex * 3 + axis];
          if (value < min[axis]) min[axis] = value;
          if (value > max[axis]) max[axis] = value;
        }
      }

      const material = readMaterial(gltf, primitive.material);
      primitives.push(Object.freeze({
        name: mesh.name ?? material.name,
        positions,
        normals,
        uvs,
        indices,
        material: Object.freeze({
          ...material,
          baseColorUri: material.baseColorUri && directory + material.baseColorUri,
          normalUri: material.normalUri && directory + material.normalUri,
          metallicRoughnessUri: material.metallicRoughnessUri && directory + material.metallicRoughnessUri,
        }),
      }));
    }
  }

  if (primitives.length === 0) fail("document contains no drawable primitives");
  return Object.freeze({ primitives: Object.freeze(primitives), min: Object.freeze(min), max: Object.freeze(max) });
}
