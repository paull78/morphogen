# morphogen

GPU-accelerated growth simulation in Zig. Cells follow local rules to self-organize into branching, tree-like structures — like nervous systems, root networks, or vascular trees.

Starts with simple cellular automata and progresses to learned neural growth rules. The end goal is beautiful real-time volumetric rendering of organic 3D structures growing from a seed.

## Inspiration

Biological morphogenesis: neurons extend axons that branch and navigate via local chemical gradients, competitive resource signals, and stochastic exploration. The tree shape is not a blueprint — it emerges from simple local rules. This project recreates that process on the GPU.

## Tech Stack

- **Zig** — host language, build system
- **WebGPU** (wgpu-native) — compute shaders for simulation, graphics pipeline for rendering
- **WGSL** — shader language
- **GLFW** — windowing and input

## Phases

### Phase 1 — Cellular Automaton
3D cellular automaton with Moore neighborhood rules. A seed cell grows into a visible structure, rendered via volumetric raymarching with an orbit camera.

### Phase 2 — Signal Diffusion + Chemotaxis
A diffusing chemical signal guides growth. Cells extend toward signal sources, branching into tree-like structures. Interactive signal placement.

### Phase 3 — Neural Cellular Automata
Replace handcrafted rules with a small neural network (trained offline, inference on GPU). Different trained weights produce different morphologies.

### Phase 4 — Beautiful Rendering
Volumetric raymarching with HDR rendering, bloom, subsurface scattering, ambient occlusion, and bioluminescent color palettes. Deep-sea / electron microscopy aesthetic.

## Controls

| Key | Action |
|-----|--------|
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause/resume |
| R | Reset to seed |
| S | Place signal source (Phase 2+) |
| T | Toggle cell type view |
| 1/2/3/4 | Render mode |
| P | Screenshot |

## Building

Requires:
- [Zig](https://ziglang.org/download/) (0.14.x)
- wgpu-native (fetched by build system)

```bash
zig build run
```

## License

MIT
