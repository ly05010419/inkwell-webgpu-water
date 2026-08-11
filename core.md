# core.md — 水体渲染核心实现文档

> 面向 AI/开发者的代码导读。描述本仓库 WebGPU 水体的完整实现：数据流、着色器职责、分块（clipmap）渲染结构与帧管线。
> 行号基于 commit `bd7b979`，代码变动后请以符号名为准。

## 0. 一句话总览

**内容层**：三层 FFT 频谱海浪（远场深水）+ 守恒型浅水方程求解器（近岸），二者在顶点着色器里按水深与覆盖范围混合。
**几何层**：整个工程没有任何顶点缓冲——水面与地形网格全部由顶点着色器根据 `vertex_index` 程序化生成；开阔海面用 **10 级实例化几何 clipmap（同心方环，逐级格距翻倍）** 一次 draw 画完，这就是"一块一块"渲染的来源。

---

## 1. 技术栈与文件地图

- Next.js（App Router）+ React，仅负责 UI 面板；渲染器本身框架无关，直接使用原生 WebGPU API（`@webgpu/types`）。
- 所有 WGSL 以模板字符串内嵌在 TypeScript 里，部分常量（cascade 波长、choppiness 等）在构建字符串时插值进 shader。

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/lib/webgpu-water-engine.ts` | ~2370 | **核心**。全部水体 WGSL + 引擎类 `WebGpuWaterEngine`（资源分配、帧循环、相机、指标） |
| `src/lib/shared-wgsl.ts` | 114 | 引擎与船共享的 WGSL：`WORLD_UNIFORMS` 布局、ACES/sRGB、程序化天空 `skyColor`、雾 `tethysAerialColor`（独立成文件是为了避免 import 环） |
| `src/lib/water-profiles.ts` | 24 | 场景/模式类型、水位等常量、质量预设 |
| `src/lib/ship-renderer.ts` | 326 | glTF 帆船：加载、浮力 compute、PBR 渲染 |
| `src/lib/ship-shaders.ts` | 232 | 船的两个 WGSL：刚体变换 compute + PBR 片元 |
| `src/lib/gltf-loader.ts` | 219 | 最小 glTF 解析（position/normal/uv/indices/材质贴图 URI） |
| `src/lib/nearshore-reference.ts` | 99 | 浅水求解器单格更新的 **CPU 镜像**，供测试当数值契约用 |
| `src/components/webgpu-water-lab-experience.tsx` | 277 | React 面板：滑块、指标、URL 参数解析、`window.__WEBGPU_WATER_LAB__` 基准桥 |
| `tests/` | — | `nearshore-reference.test.ts`（数值契约）、`water-profiles.test.ts` |

两个验证场景（`WaterScene`）：
- **`open` 开阔海面**：clipmap 水面 + 淹没的海底沙床，视野被放大 100×（`OPEN_WATER_VIEW_SCALE`），远平面 50 km。
- **`shore` 岛屿海岸**：固定 512² 单网格水面 + 露出水面的沙丘岛屿，用于干湿边界/爬坡验证，远平面 560 m。

两个渲染模式（`WaterRenderMode`）：`optimized`（1 个模拟子步）/ `reference`（2 个子步 + 更宽的反射采样，作 A/B 对照）。

---

## 2. WorldUniforms 布局（所有 pass 共享，256 B）

定义在 `shared-wgsl.ts:6`，由 `writeUniforms()`（engine ~2114 行）每帧写入。**改任何字段都要同时核对 CPU 写入偏移与全部 shader 读取处。**

| 字段 | 内容 |
|---|---|
| `viewProj` | 视图×投影矩阵（列主序） |
| `cameraTime` | xyz=相机世界位置，w=elapsed 秒 |
| `cameraRight/Up` | xyz=相机基向量，w=tan(fov/2)（Up.w）与其×宽高比（Right.w），供天空 ray 重建与像素尺寸计算 |
| `cameraForward` | xyz=前向 |
| `sunWater` | xyz=太阳方向，w=**水位** `TETHYS_WATER_LEVEL = 1.4` |
| `terrain` | x=地形场覆盖范围 `TERRAIN_EXTENT = 390` m，y=meshResolution，z=simulationResolution，w=是否水下视角 |
| `simulation` | xy=模拟域中心 (0, -12)，z=模拟域边长 `TETHYS_WATER_FIELD_SIZE = 192` m，w=1/simulationResolution（texel） |
| `player` | xy=游泳者位置，zw=速度（驱动尾迹脉冲方向） |
| `interaction` | x=速度模长，y=1，zw=画布像素宽高 |
| `environment` | x=**是否 shore 场景**（0/1，shader 里大量用它分支），y/z=验证网格分辨率，w=**worldScale**（shore=1，open=100） |
| `waves` | x=浪高倍率，y=其平方（约束谐波项是波高二次方），z=远景粗糙度回收量，w=细节距离倍率 |
| `atmosphere` | x=雾距倍率（0=关闭雾墙；shore 场景强制 1），y=远景平滑强度（0=关闭），zw=长浪/中浪 cascade 的**实时平铺尺度**（米，默认 240/64；所有消费方必须除以它而非烤死常量） |

---

## 3. 水面内容：三套系统

### 3.1 频谱海浪（远场，FFT 海洋）

**配置** `SPECTRAL_CASCADES`（engine:173）——三个平铺周期不同的 cascade：

| # | 平铺周期 | 波数窗口 | 用途 |
|---|---|---|---|
| 0 long | 240 m（默认，滑块 80–480 可调） | 0.024–0.36 | 涌浪，**位移网格** |
| 1 medium | 64 m（默认，滑块 24–128 可调） | 0.30–1.42 | 风浪，**位移网格** |
| 2 short | 12 m | 1.22–24.0 | 分米级毛细-重力波，**只参与着色**（进网格会走样成棱纹） |

每个 cascade 分辨率 `SPECTRAL_RESOLUTION = 128`（`SPECTRAL_LOG_SIZE = 7` 级 FFT）。

**CPU 初始化** `buildSpectralOceanData()`（engine:1442）：
- JONSWAP 谱 × TMA 浅水修正（深度 54 m）+ 方向扩散（聚焦+宽泛两项混合）+ 可选次级涌浪谱（`secondaryScale`）。
- 用确定性 PRNG（`deterministicRandom`，各 cascade 固定 seed）生成高斯振幅 → `initialSpectrum` 纹理（含共轭镜像项，保证 IFFT 结果为实数）。
- 同时生成 `waveData` 纹理（kx, 1/|k|, kz, ω）与 Stockham FFT 的 twiddle 表。

**GPU 每帧**：
1. `SPECTRUM_EVOLUTION_SHADER`（engine:577）：按色散相位 `e^{iωt}` 推进，一次算出高度/两方向水平位移及其全部导数，打包进两张 rgba16float：
   - `field0 = (dx, dz, dy, dz∂x)`（近似记法：水平位移 xy、高度、交叉导数）
   - `field1 = (dy∂x, dy∂z, dx∂x, dz∂z)`（坡度 rg、水平导数 ba）
2. `SPECTRAL_IFFT_SHADER`（engine:617）：Stockham 逆 FFT，7 级 × 2 轴 = **14 个 pass/cascade**，ping-pong 于 `spectralFields[cascade][ping][channel]`；最后一个 pass 乘 checkerboard `(-1)^{x+y}` 完成中心化。

**消费方读取约定**（多处复用，需保持一致）：
- 高度 = `field0.b`；水平 choppy 位移 = `field0.rg × choppiness`；坡度 = `field1.rg`；
- Jacobian 压缩量 = `(1+∂x)(1+∂z) − cross²`，`<1` 即波峰挤压，驱动白浪/泡沫/卷浪判定；
- 约束谐波（bound harmonics）：`高度 + 0.14(h_long² − 0.08·waves.y) + 0.32(h_med² − 0.03·waves.y)`，让波峰尖、波谷平。**这条公式在水面顶点、浅水边界条件、船体浮力三处逐字重复，改一处必须同步三处。**

### 3.2 近岸浅水求解器（局部非线性域）

`WATER_SIMULATION_SHADER`（engine:291）。一张 64–512²（默认 256²）rgba16float 状态纹理，覆盖以 (0, −12) 为中心、边长 192 m 的固定区域。通道：`r=η`（相对水位）、`gb=q`（单位宽度动量）、`a=泡沫`。双缓冲 ping-pong（`activeSimulationIndex`）。

数值格式（与 `nearshore-reference.ts` 的 CPU 镜像逐项对应，测试保证契约）：
- 有限体积 + **Rusanov（局部 Lax-Friedrichs）通量**；
- **静水压重构**（`hydrostaticPair`：界面底高取两侧 max，重构水深）+ 侧压修正 `sidePressureCorrection` → 井平衡（well-balanced），静水不产生虚假流；
- 干湿处理：`depth ≤ MIN_DEPTH(0.035)` 时动量清零；
- Manning 摩擦（n=0.018）隐式衰减动量；
- dt 固定 1/60（reference 模式 1/120 × 2 子步）。

**与 FFT 的耦合**（engine:470 附近，关键设计）：域边缘 8.5% 宽的**海绵层（sponge）**把 η 和 q 强松弛到 `spectralBoundaryState()` 给出的入射海况（q = c·η 线性浅水传输，方向取频谱主方向的扰动）；深水内部只有微弱 warm-up 源；浅水完全交给求解器自治 → 频谱浪"流入"域内后自然发生折射、浅化、爬坡。

其它职责：泡沫的半拉格朗日回溯平流 + 生灭（Froude 数破碎、岸线、频谱 Jacobian、尾迹环四个出生项）；游泳者尾迹高斯脉冲（每 0.1 s 一次，`writeSimulationParams`）。

### 3.3 表面混合 `evaluateWaterSurface()`（engine:1014）

顶点着色器对每个网格点 p 调用：
1. 采样地形场得水深，`shallowAttenuation = smoothstep(0.14, 2.7, depth)` 让频谱浪在极浅处归零；
2. 采样 cascade 0/1 → 频谱高度（含约束谐波）、水平位移、坡度；
3. 采样模拟状态 + 中心差分求模拟坡度；
4. `nearshoreOwnership = 模拟域覆盖 × (1 − smoothstep(3.8, 5.55, depth))` —— 浅水区**替换**（不是叠加）频谱高度：`wave = mix(spectralHeight·att, sim.r, ownership)`；
5. 输出位移后世界坐标 + 两个切向量（含全部导数链），法线在顶点末尾 `cross(tangentZ, tangentX)`。

### 3.4 Breaker 卷浪系统（**当前整体禁用**）

`BREAKER_ENABLED = false`（engine:148）。一条沿海面横扫的局部化破碎浪前锋，包含五个必须同开同关的耦合点（自适应顶点重分布 warp、主表面位移、256×48 附着 crest patch、主表面对应带的 discard、patch draw call）——**只开其中一部分会在水面撕出透明洞**。所有 shader 端由编译期常量 `BREAKER_SHADER_GATE`（"0.0"）归零，CPU 端由 `drawBreakerPatch` 跳过 draw。另有 `BREAKER_EVENT_SHADER`：256×1 的沿岸破碎"事件史"纹理（快攻慢放衰减），仍每帧更新但结果被门控。

---

## 4. "一块一块"渲染：几何 clipmap（本仓库最有特点的部分）

### 4.1 无顶点缓冲的程序化网格

水面、地形网格都没有 vertex/index buffer。顶点着色器里（`waterVertex`，engine:1077）：

```
每 6 个顶点 = 1 个四边形（两个三角形，corners 查表）
cellId = vertexId / 6
cell   = (cellId % resolution, cellId / resolution)
uv     = (cell + corner) / resolution
```

CPU 只需 `draw(resolution² × 6, instanceCount)`。

### 4.2 开阔海面：10 级实例化 clipmap

一次 draw：`renderPass.draw(64·64·6, 10)`（engine:2250）。`instance_index` = clipmap 层级：

- **层级尺寸**：`halfExtent = 32 · 2^level` 米（engine:1092）。第 0 层覆盖 ±32 m，第 9 层 ±16384 m。每层分辨率恒为 `WATER_CLIPMAP_RESOLUTION = 64`，因此**格距逐层翻倍**：相机脚下 1 m/格，最外层 512 m/格。层数选 10 是为了在 1450 m 变焦极限处仍能从偏心原点覆盖 1950 m 地形半径（见 engine:150 注释）。
- **随相机移动 + 按格吸附**（engine:1094）：`snappedCamera = floor(camera.xz / cellSize) · cellSize`。每层按**自己的格距**取整——相机移动时顶点只整格跳变，波形采样点在世界空间不动，杜绝"顶点游泳"。
- **粗层挖心**（engine:1096–1104）：`instanceId > 0` 时，格中心落在内半径 `halfExtent/2 − cellSize` 以内的格子把 `baseP` 扔到 10000 → 零面积三角形被光栅化丢弃。每个粗层只画细层盖不住的"环"。内半径**故意少一格（underlap）**，让细粗两层重叠一格来遮住 T 型接缝——没有做真正的 T-junction 缝合。
- **地平线裙边 skirt**（engine:1111–1133）：最外层最外圈顶点沿径向甩到 `WATER_HORIZON_REACH = 20000` m，用 **w=0 方向投影**（且先把方向的 y 分量压平——否则高空视角下 `atan(相机高度/20 km)` 会把裙边方形的边缘压到真地平线以下露出方角），并在裁剪空间强制 `z = w · 0.99999`——刚好压在天空写的 0.999999 之前。作用：雾墙默认关闭后，没有裙边会看到有限水面方形边缘的切口。约束：20 km 必须大于最外环 16384 m，否则裙边会把几何往里拉。
- **每层之间没有几何缝**：所有层共用同一个连续的 `evaluateWaterSurface(p)` 世界空间波场函数，层只是采样密度不同。

三角形统计（`getMetrics`，engine:2306）：最内层全画 = 64²·2；外 9 层各挖掉约一半 = ×1.5。

### 4.3 岛屿海岸：单块大网格

`shore` 场景不用 clipmap：一张 512×512、铺满 `TERRAIN_EXTENT` 的单实例网格（`environment.x > 0.5` 分支 + engine:2248），因为相机被钉在 96 m 轨道上，不需要远景 LOD。

### 4.4 地形网格

`terrainVertex`（engine:724）同样程序化生成，采样预计算的地形场纹理（513²，rgba16float：高度 + 法线 xz），场由 `TERRAIN_FIELD_SHADER` compute 在首帧一次性生成（解析函数 `terrainHeight`：海底沙丘 + 4 个 `tethysCoastalShelf` 岛屿；`shoreMix`=environment.x 控制岛露不露出水面）。**注意**：地形场固定覆盖 390 m、始终以世界原点为中心，不随 open 场景的 100× 视野缩放（会毁掉海底沙丘的采样密度，engine:70 注释）；场外按 clamp 采样得到平坦 −8.5 m 海底，被水体吸收遮住。

---

## 5. 帧管线（`render()`，engine:2158）

每帧一个 command encoder，顺序：

```
[首帧一次] compute: 地形场 513² 生成
compute pass（计时点 0-1）:
  ├─ cascade × 3: 频谱推进 1 dispatch + IFFT 14 dispatch
  ├─ 浅水模拟 1 步（reference 模式再来 1 步，用 calm 参数）→ ping-pong 翻转
  ├─ breaker 事件史 1 dispatch → ping-pong 翻转
  └─ 船体刚体变换 1×1 dispatch（浮力采样必须在 scene pass 前）
scene pass（计时点 2；shore→离屏 sceneColorTexture，open→直接画布）:
  ├─ 天空：全屏三角形，depth 不写、z=0.999999
  ├─ 地形：程序化网格（shore 512²，open 用 meshResolution）
  └─ 帆船：写深度
water pass（计时点 3；画布；depthReadOnly + alpha 混合）:
  ├─ [shore] 先把离屏场景 blit 回画布
  ├─ 水面 draw：shore = 512²·6 单实例；open = 64²·6 × 10 实例 clipmap
  └─ [breaker patch draw，当前禁用]
```

关键点：
- **水 pass 深度只读**：水不写深度，靠读 scene pass 的深度做遮挡与折射合成；水面本身用 `src-alpha` 混合（片元 alpha = shorelineCoverage，用于岸线羽化）。
- **shore 场景的"场景捕获"**：地形先渲到离屏纹理，水的折射项用 group(1) 采样这张**真实水下场景**（含深度判定 `capturedGeometry`），代替解析沙床色。open 场景跳过捕获（`sceneCapturePasses = 0`），折射用解析沙床。
- **双缓冲索引**：模拟与 breaker 各自 ping-pong；渲染绑定组按 `[simulationIndex][breakerEventIndex]` 预建成 2×2 矩阵，避免每帧建绑定组。
- **GPU 计时**：`timestamp-query` 可用时每 8 帧测一次（compute 段 + render 段），异步 readback 进滑动窗口，供指标面板。

---

## 6. 水面片元着色（`waterFragment`，engine:1220）

按顺序：
1. **岸线覆盖**：用顶点插值来的位移后世界高度减地形高度得水柱厚度，`fwidth` 抗锯齿 smoothstep → alpha；`< 0.01` discard。阈值 shore=0.28（留湿沙边）、open=0.018。地形采样越界**clamp 不 discard**（engine:1226 注释：discard 会在场边界切出锯齿）。
1.5. **主表面法线逐片元重建（2026-08 修复）**：cascade 0/1 的坡度与近岸模拟坡度混合（`nearshoreOwnership` 那套）在片元里按位移前参数 `surfaceParam` 重新采样重建法线——参数由未 clamp 的 `simulationUv` 仿射反解。此前法线只在顶点上算、片元插值，clipmap 格距逐环翻倍导致几十米外欠采样 64 m cascade，呈现格子大小的菱形面片和环边界"换质感"分界线。几何位移仍是顶点级；breaker patch（surfaceKind>0.5）保留其顶点法线直通。代价 +9 次纹理采样/片元，实测 GPU 渲染 2.18→2.27 ms。**逐片元采样引出的次生问题**：远处一个像素跨多个中浪波长，坡度混叠成地平线下的密集闪光带（顶点插值时代被"意外"抹平）——因此 cascade 0/1 各带一个与毛细波同规格的屏幕采样率淡出（每波长 14 px 起淡、3 px 淡完，代表波长 8 m / 48 m，`detailRange` 同样生效），淡掉的坡度并入 `fadedSlope`→远景粗糙度方差回收路径。**新增每级 cascade 的着色贡献时必须同时给它配采样率淡出，否则远景必闪。**
2. **毛细细节（cascade 2）与屏幕空间淡出**：判据是 **pixels-per-wavelength**（`(12/12) / 像素世界尺寸`），不是世界距离——1/d 的透视映射让固定距离带在屏幕上塌缩成几十个像素、细节像被"开关"。`detailRange`（细节距离滑块）乘在判据上。淡出的坡度不丢弃：其**方差**按 `waves.z`（远景粗糙度滑块）回收进 Cox-Munk 分布与反射展宽——法线变平滑但 BRDF 保住能量，否则远处水面塌成镜子。
3. **BRDF**：精确介电 Fresnel（n=1.333，s/p 偏振平均）；太阳闪光 `oceanSunGlitter` 用 **Cox-Munk 净海面坡度分布**（11.5 m/s 风）+ Smith 可见性，`extraVariance` 即回收的毛细方差。
4. **折射/水体**：法线偏移的屏幕空间折射 UV；Beer-Lambert 吸收 `exp(−(0.37,0.125,0.054)·光程)` + Henyey-Greenstein 相函数的低能量体散射色。shore 浅水处透射直接混向捕获的沙床色（厘米级水不当海洋处理）。
5. **反射**：程序化 `skyColor`（含三倍频 value-noise 云）；`recoveredVariance > 0` 时额外 3 tap 展宽反射锥。reference 模式固定再加 2 tap 模糊。
6. **泡沫/白浪**：白浪 = 波峰高度 × Jacobian 压缩⁴ × 时变噪声碎化 × 距离淡出（95–188 m × detailRange）；持久泡沫 = 模拟 `a` 通道（open）；岸边冲刷 = 模拟泡沫+动量+深度窗（shore）；尾迹。全部 max 合成后 mix 一个低饱和白。
7. **收尾**：水下视角翻法线、压反射、加体积绿；open 场景加与地形一致的 `tethysAerialColor` 雾（含可调径向雾墙，默认 0=关闭）；ACES → sRGB 输出。

---

## 7. 帆船（浮力与水面的一致性契约）

- `SHIP_TRANSFORM_SHADER`：**1×1 compute** 每帧重建刚体矩阵（放顶点级会对 6.4 万顶点重复 5 次纹理采样）。在船中心/艏/艉/左/右舷 5 点采样 cascade 0/1 高度——**必须乘同样的 `waves.x` 并加同样的约束谐波修正**，否则船浮的高度和实际画出的海面对不上。heave 取水线面加权平均，trim/heel 用 atan2 并阻尼（0.55/0.45），24 m 船身不响应毛细波。
- 渲染：标准 PBR（GGX + 金属度/粗糙度 ARM 贴图），模型无切线属性→片元里用屏幕空间导数重建切线框架；索具处 UV footprint 爆炸时按 `tangentTrust` 退回几何法线（否则出黑点）。半球环境光的天空/水反色调与海面 shader 对齐。
- 船在 scene pass 中写深度 → water pass 的深度只读合成让吃水线以下自动被水淹没，无需特殊处理。
- 加载失败不致命：`init()` 里 try/catch，错误串进 adapter 标签展示。
- cascade 纹理在 `allocateFields()`（场景切换、近岸场分辨率调整）中销毁重建，其末尾会调用 `bindSpectralFields()` 重绑船体浮力采样——曾因重绑只写在 `resize()` 里，切场景后每帧报 "Destroyed texture used in a submit"（2026-08 修复）。

---

## 8. React / UI 桥

- 组件仅做面板；引擎实例挂在 `window.__WEBGPU_WATER_LAB__`（基准脚本用），每 250 ms 拉一次 `getMetrics()`。
- URL 参数：`mode / view / scene / mesh / simulation / scale / waves / farRough / detail / smooth / longScale / mediumScale / fog / fixedTime / yaw / pitch / benchmark / ui`。调尺度时引擎在 CPU 重新生成该 cascade 的初始频谱并原地覆写纹理（毫秒级，管线不重编译），船体浮力采样尺度同步更新（`ShipRenderer.setCascadeScales`）。`fixedTime` 冻结时间用于截图对比；`benchmark=1` 把 DPR 上限压到 1。
- 滑块与引擎 setter 一一对应，全部 clamp；`setSimulationResolution` 触发 `allocateFields()` 全量重建纹理与绑定组。
- 隐藏的 `#webgpu-water-lab-qa` output 元素输出 JSON 指标，`data-ready` 供自动化等待。

## 9. 测试与数值契约

- `nearshore-reference.ts` 是 WGSL 浅水单格更新的逐行 CPU 镜像（Rusanov、静水压重构、侧压修正、干湿清零）。`tests/nearshore-reference.test.ts` 用它验证守恒性/井平衡等性质。**改 WGSL 求解器必须同步改镜像，反之亦然**——这是刻意维护的双实现契约。
- `tests/water-profiles.test.ts` 锁定预设与字节计算。

## 10. 修改代码时的高频陷阱

1. **约束谐波公式三处重复**（水面顶点、浅水边界、船浮力）——必须同步。
2. **WorldUniforms 偏移**手写在 `writeUniforms()`，无反射校验——增删字段要同时改 CPU 偏移和所有 shader。
3. **breaker 的六个耦合点**必须同开同关，否则水面出透明洞（engine:140 注释；第 6 点是 2026-08 新增的片元法线路径——重新启用 breaker 时必须把其位移导数并进 `waterFragment` 的逐片元法线）。
4. **相机拉远上限**：开阔海旧上限 250 m 是给 tanh NaN 海床露出打的补丁（该 bug 修复后已放开到 12000 m，下限 6 m）；上限必须留在 clipmap 覆盖半径 16384 m 以内，且过大时 f32 世界坐标会引入频谱 UV 抖动。
5. **clipmap 常量互相约束**：`WATER_HORIZON_REACH(20000) > 最外环(16384)`；skirt 深度必须 < 天空的 0.999999；层数决定的覆盖必须 ≥ 变焦极限 + 地形半径。
6. **地形场不随 open 场景缩放**（0.76 m/texel 的刻意选择）；场外靠 clamp + 水体吸收兜底。
7. 片元里对地形**clamp 不 discard**（engine:1226），discard 会切出锯齿边。
8. 细节淡出用**屏幕空间采样率**不是世界距离（engine:1253 注释），改回距离制会产生"细节开关"突变。
9. 泡沫淡出的坡度要**回收成 BRDF 方差**，直接丢弃会让远景变镜面。
10. 双缓冲索引翻转（`activeSimulationIndex`/`activeBreakerEventIndex`）与 2×2 预建绑定组矩阵强耦合，调整子步数时注意翻转次数。
11. `uncapturederror` 只保留**第一条**错误（后续多是连锁反应，engine:2335）。
12. **tanh 溢出陷阱（2026-08 实测确认并已修复）**：`adaptiveBreakerCoordinates` 里的 `tanh((across - front) / bandWidth)` 对每个水面顶点执行，clipmap 外圈顶点的 `across` 可达 ±16384，参数 |x| 超过 ~89 时 Metal 的 tanh 通过 `e^x` 溢出产生 `Inf/Inf = NaN`；虽然 breaker 关闭注入了 `× 0.0` 门控，但 **`0 × NaN = NaN`，编译期置零关不住它**。曾经的后果：顺风向约 1.1 km 外的半平面顶点坐标变 NaN → GPU 丢弃三角形 → 远景出现按各层格距量化的阶梯缺口；0.55 m 差分切线采样跨过 NaN 边界 → 法线 NaN → 缺口边缘渲染纯黑方块。现已通过 `tanh(clamp(x, -30.0, 30.0))` 修复。通用教训：**永远不要给 shader 里的 tanh/exp 喂无界参数；`× 0` 不是禁用一段 NaN 风险代码的可靠手段**。
13. **init() 的 StrictMode 契约**：React dev StrictMode 会挂载→卸载→重挂载组件，被 dispose 的旧引擎的 `init()` 仍在异步执行，会与新引擎争抢同一个 canvas 的 WebGPU context（曾导致 dev 下渲染循环停摆、fps=0）。因此 `init()`/`createResources()` 在**每个 await 之后**都检查 `this.disposed` 并提前返回，组件 effect 用 `cancelled` 标志防止旧 init 的 `.then` 误标 ready。新增 await 时必须在其后补上同样的检查。
