# morphogen

GPU-accelerated growth simulation in Zig. Cells follow local rules to self-organize
into branching, tree-like structures — like nervous systems, root networks, or
vascular trees.

<!-- TODO: drop a 5–10s GIF here — structure growing + right-click signal placement.
     Capture it with the in-app screenshot key (P) / a screen recorder. -->
![morphogen growing a branching structure](docs/assets/demo.gif)

Starts with simple cellular automata and progresses toward learned neural growth
rules. The end goal is beautiful real-time volumetric rendering of organic 3D
structures growing from a seed.

## About this project

This is a **learning project**. I'm using it to teach myself Zig, GPU compute, and
the WebGPU/WGSL stack from the ground up — deliberately hand-writing the simulation
and rendering instead of reaching for an engine, so I actually understand each layer.

The commit history is meant to be read as a step-by-step learning log: each commit is
a small, self-contained, working increment (window opens → screen clears → first
triangle → compute shader → 3D automaton → raymarched render → camera → signal
diffusion). If you want to see how it was built, `git log` tells the story.

## Status

**Phase 2 complete.** A 3D cellular automaton runs entirely on the GPU, with a
diffusing chemical signal that guides growth via chemotaxis, branching into tree-like
structures, plus interactive signal placement. Next up: Phase 3 (neural cellular
automata).

- [x] **Phase 1 — Cellular Automaton.** 3D automaton with Moore-neighborhood rules.
      A seed cell grows into a visible structure, rendered with volumetric raymarching
      and an orbit camera.
- [x] **Phase 2 — Signal Diffusion + Chemotaxis.** A diffusing chemical signal guides
      growth; cells extend toward sources and branch. Interactive signal placement.
- [ ] **Phase 3 — Neural Cellular Automata.** Replace handcrafted rules with a small
      neural network (trained offline, inference on GPU). Different weights → different
      morphologies.
- [ ] **Phase 4 — Beautiful Rendering.** HDR volumetric raymarching with bloom,
      subsurface scattering, ambient occlusion, and bioluminescent palettes.
      Deep-sea / electron-microscopy aesthetic.

## Inspiration

Biological morphogenesis: neurons extend axons that branch and navigate via local
chemical gradients, competitive resource signals, and stochastic exploration. The
tree shape is not a blueprint — it emerges from simple local rules. This project
recreates that process on the GPU.

## Tech Stack

- **Zig** — host language and build system
- **WebGPU** (wgpu-native) — compute shaders for simulation, graphics pipeline for rendering
- **WGSL** — shader language
- **GLFW** — windowing and input
- **Objective-C runtime interop** — to attach a `CAMetalLayer` for the macOS surface

## What I learned (so far)

A few things this project taught me that I didn't know going in:

- **Driving wgpu-native's async callbacks from Zig.** Device/queue work reports back
  through callbacks; I initially blocked on `WaitAny` and nothing fired. Switching to
  pumping `ProcessEvents` was the fix (commit `4177a8d`).
- **GPU↔CPU readback as a debugging tool.** Before trusting the compute path, I read
  the storage buffer back to the CPU and verified it by hand — the only way to know a
  shader is actually doing what you think.
- **Volumetric raymarching with DDA voxel traversal**, double-buffered simulation
  state, and stochastic cellular-automaton rules with age-based coloring.

## Controls

| Key | Action |
|-----|--------|
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause / resume |
| N or → | Step once (while paused) |
| R | Reset to seed |
| Right-click | Place a signal source |
| Esc | Quit |

## Building

Requires:
- [Zig](https://ziglang.org/download/) (0.15.x)
- wgpu-native (fetched by the build system)

```bash
zig build run
```

> **Platform:** Developed and tested on **macOS (Apple Silicon)**. The surface-creation
> path uses Metal via Objective-C runtime calls, so other platforms aren't wired up yet.

## Design docs

I plan before I build. The design spec and per-phase implementation plans live in
[`docs/`](docs/):

- [Design spec](docs/specs/2026-04-02-morphogen-design.md)
- [Phase plans](docs/plans/) — foundation, compute, first visuals, signal & branching

## License

MIT
