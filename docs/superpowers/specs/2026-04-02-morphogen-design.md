# Morphogen — Design Spec

GPU-accelerated growth simulation in Zig. Cells follow local rules to self-organize into branching, tree-like structures. Starts with simple cellular automata and progresses to learned neural growth rules, with real-time volumetric rendering of organic 3D structures growing from a seed.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| GPU API | WebGPU (wgpu-native via `@cImport`) | Cross-platform (Metal on macOS, Vulkan on Linux), far less boilerplate than raw Vulkan, compute + graphics in one API |
| Shader language | WGSL | Native to wgpu, no GLSL→SPIR-V compile step |
| Windowing | GLFW via Zig C interop | Battle-tested, simple, works on macOS |
| Zig + wgpu binding | Raw wgpu-native C API | mach ecosystem is in flux; raw C API is stable and transparent |
| Architecture | Decoupled sim/render with shared double-buffered grid | Smooth 60fps rendering independent of simulation rate, temporal interpolation for smooth growth |
| Phase 3 training | Offline in Python (PyTorch/JAX), weights exported as flat f32 binary | Avoids building custom autograd on GPU; inference in WGSL is simple |
| Development platform | macOS (M4 Pro, 48GB) | Metal backend via wgpu; cross-platform by default, no platform-specific code |
| Visual aesthetic | Bioluminescent deep-sea + electron microscopy | Dark backgrounds, self-illuminated structures, emission-heavy lighting, bloom |
| Development approach | Incremental learning steps | Each step is a small deliverable you can see, debug, and experiment with before moving on |

## Learning Steps

This is a learning project. Development is organized as small incremental steps, each producing a visible, runnable result. No step should require more than ~1-2 sessions to complete. Each builds on the last.

### Foundation (build + window + GPU)

1. ~~**Hello Zig** — `build.zig` compiles and runs, prints to stdout. Verify toolchain works.~~ ✅
2. ~~**GLFW window** — blank window opens, responds to close. Learn Zig C interop with GLFW.~~ ✅
3. ~~**wgpu clear screen** — initialize wgpu device + surface, clear to a solid color each frame. Learn wgpu lifecycle.~~ ✅
4. ~~**Fullscreen triangle** — vertex + fragment shader, render a colored triangle. Learn wgpu render pipeline, WGSL basics.~~ ✅

### Compute (GPU data + compute shaders)

5. ~~**Compute shader: buffer write** — compute shader writes values to a storage buffer, read back to CPU and print. Learn wgpu compute pipeline, bind groups.~~ ✅
6. ~~**3D grid buffer** — allocate grid SSBO, seed center cell from CPU, read back and verify. Learn buffer sizing, grid indexing.~~ ✅
7. ~~**Compute step on grid** — compute shader reads grid, applies a trivial rule (e.g., "spread alive to all neighbors"), writes to second buffer. Verify by reading back.~~ ✅

### First Visuals (raymarch + automaton)

8. ~~**Raymarch a static grid** — fullscreen quad + fragment shader, cast rays through the grid SSBO, render seeded cell as a colored voxel. First visual output.~~ ✅
9. ~~**Orbit camera** — mouse drag orbits, scroll zooms. See the voxel from all angles.~~ ✅
10. ~~**Cellular automaton: single step** — run one CA step (birth/survival rules), see neighbors appear. Pause/step with keyboard.~~ ✅
11. ~~**CA continuous growth** — run CA in a loop, watch structure grow from seed in real time. Tweak birth/survival rules via params.~~ ✅

### Signal & Branching (Phase 2)

12. ~~**Signal diffusion** — add signal channel, place a source, visualize signal field as faint color overlay. Verify diffusion spreads correctly.~~ ✅
13. ~~**Growth tips + chemotaxis** — growth tips follow signal gradient. See a single tendril grow toward the source.~~ ✅
14. ~~**Branching** — add branching logic, see tree-like structures emerge. Color by cell type.~~ ✅
15. ~~**Interactive signals** — place/move signal sources with mouse. Watch growth respond in real time.~~ ✅

### NCA Inference (Phase 3)

16. **Load weights from file** — read a flat f32 binary into a GPU buffer. Verify contents.
17. **NCA forward pass** — compute shader runs MLP per cell using loaded weights. Verify output by reading back.
18. **NCA growth** — full NCA loop: perceive → MLP → apply. See learned growth patterns from seed.

### Beautiful Rendering (Phase 4)

19. **HDR + emission** — render to HDR offscreen texture, growth tips emit light. See bright structures on dark background.
20. **Bloom** — add bloom post-process pass. Bioluminescent glow effect.
21. **Tonemapping** — ACES tonemap, dark gradient background. Deep-sea aesthetic.
22. **Subsurface scattering** — translucent structures, light passing through thin branches.
23. **Ambient occlusion** — depth cues from neighbor density.
24. **Polish** — color palettes, smooth animation (temporal interpolation), screenshot support.

## System Architecture

### Overview

Everything revolves around the double-buffered 3D grid. Simulation writes one buffer, renderer reads both and interpolates. wgpu handles synchronization on Metal's single queue — no manual barriers.

```
main loop
├── poll input (GLFW) → update camera, simulation commands
├── simulation step (if frame budget allows)
│   └── compute dispatch: read buffer A → write buffer B → swap
├── render (always at display refresh)
│   ├── raymarch: read buffer A + B, interpolate by t
│   ├── bloom pass (Phase 4)
│   └── tonemap + present to swapchain
└── present
```

### Frame Budget

Render runs at 60fps always. Simulation steps are adaptive:
- 64³ grid: multiple sim steps per frame
- 128³ grid: ~1 step per frame
- 256³ grid: 1 step every few frames

The renderer interpolates between buffer A and B using a fractional `t` value, so growth appears smooth regardless of simulation rate.

### Synchronization

wgpu on Metal uses a single queue. Compute dispatch and render pass are submitted sequentially within the same command encoder. wgpu's resource usage tracking handles barriers automatically.

## Module Structure

```
morphogen/
├── build.zig                  # build system, links wgpu-native + GLFW
├── build.zig.zon              # Zig package manifest
├── shaders/
│   ├── automaton.wgsl         # phase 1: cellular automata rules
│   ├── diffusion.wgsl         # phase 2: signal diffusion
│   ├── growth.wgsl            # phase 2: chemotaxis growth
│   ├── nca_step.wgsl          # phase 3: NCA inference (MLP)
│   ├── raymarch.wgsl          # volumetric renderer (evolves per phase)
│   ├── bloom.wgsl             # phase 4: bloom post-process
│   ├── tonemap.wgsl           # phase 4: HDR → SDR tonemapping
│   └── fullscreen.wgsl        # passthrough vertex shader
├── src/
│   ├── main.zig               # entry point, main loop
│   ├── gpu.zig                # wgpu bootstrap: instance, device, surface, queue
│   ├── grid.zig               # 3D grid SSBOs, double-buffer swap
│   ├── simulation.zig         # compute dispatch per phase
│   ├── renderer.zig           # raymarch + post-processing chain
│   ├── camera.zig             # orbit camera, view/proj matrices
│   ├── input.zig              # GLFW keyboard/mouse handling
│   └── params.zig             # comptime config: grid size, channels, phase
└── lib/                       # vendored C deps (wgpu-native, GLFW)
```

### Module Responsibilities

**main.zig** — Owns the loop: init GPU → create grid → enter frame loop (poll input → sim step → render → present). Holds top-level state, no logic of its own.

**gpu.zig** — wgpu lifecycle: request adapter, create device, configure surface for swapchain, expose queue. Thin wrapper — one init function, holds handles other modules need.

**grid.zig** — Creates and manages the two grid SSBOs (buffer A and B). Knows the cell stride (bytes per cell) and grid dimensions. Provides `swap()` and `seedCenter()`. Doesn't know what the cell data means — that's the simulation's job.

**simulation.zig** — Holds compute pipelines and bind groups for the active phase. Dispatches the right shader(s) per step. Phase 1: one automaton pass. Phase 2: diffusion + growth. Phase 3: NCA forward. Exposes `step(encoder)`.

**renderer.zig** — Fullscreen quad + raymarch fragment shader. Reads both grid buffers + interpolation `t`. Later phases add bloom + tonemap as additional render passes. Exposes `draw(encoder, swapchain_view)`.

**camera.zig** — Orbit camera: spherical coords (theta, phi, radius) → view matrix. Projection matrix from FOV + aspect ratio. Updated by input, consumed by renderer as a uniform buffer.

**input.zig** — GLFW callbacks: mouse drag → orbit, scroll → zoom, keyboard → pause/reset/place signal. Translates raw input into camera updates and simulation commands.

**params.zig** — Comptime constants: grid dimensions, cell channel count, phase selection. Changing these and rebuilding reconfigures the whole pipeline.

## Grid & Cell State

### Storage

Flat 1D storage buffer (SSBO), indexed as `grid[z * W * H + y * W + x]`. Each cell is a struct of f32s. Two identical buffers (A and B) for double-buffering.

**Why flat SSBO over 3D texture:**
- No format restrictions — arbitrary struct per cell
- Compute shaders can read/write freely (3D storage textures have format limitations in wgpu)
- Easy to double-buffer (two buffers, swap bind group indices)
- No hardware filtering needed — simulation reads exact neighbors, raymarch does its own sampling

### Cell State Per Phase

| Phase | Fields | Floats/cell | 64³ × 2 | 128³ × 2 | 256³ × 2 |
|-------|--------|-------------|----------|-----------|-----------|
| 1 | alive, r, g, b, a | 5 | 5 MB | 40 MB | 320 MB |
| 2 | type, signal, r, g, b, a | 6 | 6 MB | 48 MB | 384 MB |
| 3 | type, signal, r, g, b, a, hidden[8] | 14 | 14 MB | 112 MB | 896 MB |

All within budget on 48GB unified memory.

### Addressing (WGSL)

```wgsl
fn cell_index(pos: vec3u) -> u32 {
    return pos.z * params.grid_w * params.grid_h + pos.y * params.grid_w + pos.x;
}
```

### Double-Buffer Swap

No data copy. Swap which buffer is bound as "read" vs "write" in the bind group. The renderer always reads both and interpolates by `t`.

## Simulation Phases

### Phase 1 — Cellular Automaton

Single compute dispatch over the full grid.

- Each thread reads 26 Moore neighbors from buffer A, counts alive cells
- Applies birth/survival rules, writes to buffer B
- Workgroup size: 4×4×4 (64 threads)
- Rules are uniforms (birth count, survival min/max) — tweakable at runtime
- Known interesting 3D rules: 4/4/4 (crystal), 5-7/6/6 (organic blobs)
- Seed: single alive cell at grid center

**Exit criteria:** a seed cell grows into a visible 3D structure, rendered in real time with camera controls.

### Phase 2 — Signal Diffusion + Chemotaxis

Two compute dispatches per step:

1. **Diffusion pass:** every cell updates signal via discrete Laplacian: `signal_new = signal + D * (avg_neighbors - signal) - decay * signal`. Source points inject signal at fixed or user-placed positions. Reads signal from buffer A, writes full cell state (including updated signal) to buffer B. No intermediate buffer needed — diffusion and growth are separate dispatches with a buffer swap between them.

2. **Growth pass:** reads from buffer B (post-diffusion), writes to buffer A. Only growth-tip cells are active. Read signal gradient (central differences in 3D), grow toward strongest gradient with stochastic noise. Branch if gradient is ambiguous (multiple strong directions). Die if signal below threshold. After this pass, buffer A is the "current" state for rendering.

Cell types: empty (0), axon body (1), growth tip (2), branch point (3).

**Exit criteria:** a seed grows into a branching tree that follows signal sources. Moving sources changes growth direction.

### Phase 3 — NCA Inference (Learned Rules)

Pre-trained weights loaded from binary file into a storage buffer.

Compute shader per cell:
1. **Perceive:** 3D Sobel on all channels → perception vector
2. **MLP forward:** perception → dense(64) → ReLU → dense(state_channels)
3. **Apply:** residual update with stochastic mask, clamp type/signal/rgba

Training happens offline in Python (PyTorch/JAX). Export weights as flat f32 binary. Zig loads at startup.

**Exit criteria:** network grows branching trees from seed without hardcoded rules. Different weight files produce different morphologies.

## Rendering Pipeline

### Phase 1 — Basic Raymarch

- Fullscreen quad, fragment shader casts rays through the volume
- Ray-AABB intersection for grid bounding box entry/exit
- Fixed step size (~0.5 voxels), sample grid at each point
- Binary opacity: alive = opaque, dead = skip
- Directional lighting: central difference normal estimation, dot with light direction
- Front-to-back alpha compositing, early exit on alpha saturation
- Orbit camera with mouse drag (theta/phi), scroll zoom

### Phase 2 — Enhanced

- Color per cell type: growth tip = bright cyan, axon body = muted blue, branch point = white
- Variable opacity based on local neighbor count (thinner branches more transparent)
- Signal field as faint volumetric glow (low opacity additive)

### Phase 4 — Full Visual Pipeline

Three render passes:

**Pass 1 — Raymarch → HDR offscreen texture + depth:**
- Temporal interpolation: lerp between buffer A and B per sample
- Subsurface scattering approximation: secondary short ray toward light, accumulate thickness
- Emission: growth tips and signal-rich cells emit light (added to HDR)
- Ambient occlusion: neighbor density as cheap AO estimate
- Color palette via uniform: bioluminescent blue-green for structure, bright cyan for tips, faint purple for signal

**Pass 2 — Bloom:**
- Threshold bright pixels from HDR buffer
- Downsample → separable Gaussian blur (horizontal + vertical) → upsample
- Additive blend back — gives bioluminescent glow

**Pass 3 — Tonemap + present:**
- ACES or Reinhard tonemapping (HDR → SDR)
- Optional FXAA
- Dark gradient background (deep ocean feel)
- Output to swapchain

## Controls

| Key | Action |
|-----|--------|
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause/resume simulation |
| R | Reset to seed |
| S | Place signal source at cursor (Phase 2+) |
| T | Toggle cell type visualization |
| 1/2/3/4 | Switch render mode (solid/transparent/volumetric lit/debug) |
| P | Screenshot to PNG |

## Build Dependencies

- Zig (latest stable)
- wgpu-native (C headers + dylib, vendored or fetched)
- GLFW (Zig C interop)
- Python + PyTorch (offline, Phase 3 training only — not part of the Zig build)
