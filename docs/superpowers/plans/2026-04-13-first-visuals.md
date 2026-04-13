# First Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** See the 3D simulation on screen — raymarch the grid, add an orbit camera, implement real CA rules, and watch structures grow in real time.

**Architecture:** The existing fullscreen triangle shader is replaced with a volumetric raymarcher that reads the grid SSBO. A new camera.zig provides orbit camera matrices passed as a uniform buffer. The simulation runs inside the render loop with adaptive stepping. input.zig handles GLFW callbacks for camera control and pause/step.

**Tech Stack:** Same as before. New: raymarching in WGSL, uniform buffers for camera/params, GLFW input callbacks.

**Prerequisites:** Compute block complete (steps 5-7). Grid double-buffering and simulation step working.

---

## File Structure

```
src/
├── main.zig          # modified: integrate camera, input, sim loop
├── gpu.zig           # modified: accept grid + camera uniforms in render pipeline
├── grid.zig          # unchanged
├── simulation.zig    # modified: real CA rules (Moore neighborhood)
├── camera.zig        # NEW: orbit camera, view/proj matrices
└── input.zig         # NEW: GLFW callbacks, camera control, pause/step
```

---

## Task 8: Raymarch a static grid

**Files:**
- Modify: `src/gpu.zig` (new render pipeline that reads grid SSBO, camera uniform)
- Modify: `src/main.zig` (pass grid buffers to renderer, remove test prints, scale grid to 32³)

The raymarch fragment shader:
- Receives camera position + inverse view-projection matrix as uniforms
- Casts a ray per pixel through the 3D volume
- Ray-AABB intersection to find entry/exit of the grid bounding box (unit cube centered at origin)
- Steps through at fixed intervals (~0.5 voxel spacing)
- At each step, samples the grid SSBO: if alive > 0.5, use the cell's RGBA
- Simple directional lighting: estimate normal via central differences on the alive field
- Front-to-back alpha compositing, early exit when alpha > 0.95
- Dark background

For this first step, use a hardcoded camera looking at the grid from a diagonal, no mouse interaction yet.

- [ ] **Step 1: Create camera uniform struct**

Add to `src/gpu.zig` (or create a shared types file):

Camera uniform (passed to fragment shader):
```
struct CameraUniforms {
    inv_view_proj: mat4x4<f32>,  // 64 bytes
    camera_pos: vec3<f32>,        // 12 bytes
    _padding: f32,                // 4 bytes (alignment)
    resolution: vec2<f32>,        // 8 bytes
    _padding2: vec2<f32>,         // 8 bytes
}
// Total: 96 bytes
```

- [ ] **Step 2: Create grid params uniform for the raymarcher**

```
struct GridParams {
    grid_size: vec3<u32>,       // 12 bytes
    floats_per_cell: u32,       // 4 bytes
}
// Total: 16 bytes
```

- [ ] **Step 3: Write the raymarch WGSL shader**

Replace the fullscreen triangle shader with a raymarcher. The vertex shader stays the same (fullscreen triangle). The fragment shader:

```wgsl
struct CameraUniforms {
    inv_view_proj: mat4x4f,
    camera_pos: vec3f,
    _pad: f32,
    resolution: vec2f,
    _pad2: vec2f,
}

struct GridParams {
    grid_size: vec3u,
    floats_per_cell: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> grid_params: GridParams;
@group(0) @binding(2) var<storage, read> grid: array<f32>;

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return (iz * grid_params.grid_size.y * grid_params.grid_size.x 
          + iy * grid_params.grid_size.x + ix) * grid_params.floats_per_cell;
}

fn sample_alive(pos: vec3f) -> f32 {
    let grid_pos = pos * vec3f(grid_params.grid_size);
    let ix = u32(clamp(grid_pos.x, 0.0, f32(grid_params.grid_size.x - 1)));
    let iy = u32(clamp(grid_pos.y, 0.0, f32(grid_params.grid_size.y - 1)));
    let iz = u32(clamp(grid_pos.z, 0.0, f32(grid_params.grid_size.z - 1)));
    return grid[cell_index(ix, iy, iz)];
}

fn sample_color(pos: vec3f) -> vec4f {
    let grid_pos = pos * vec3f(grid_params.grid_size);
    let ix = u32(clamp(grid_pos.x, 0.0, f32(grid_params.grid_size.x - 1)));
    let iy = u32(clamp(grid_pos.y, 0.0, f32(grid_params.grid_size.y - 1)));
    let iz = u32(clamp(grid_pos.z, 0.0, f32(grid_params.grid_size.z - 1)));
    let idx = cell_index(ix, iy, iz);
    return vec4f(grid[idx + 1], grid[idx + 2], grid[idx + 3], grid[idx + 4]);
}

fn intersect_aabb(ray_origin: vec3f, ray_dir: vec3f) -> vec2f {
    // AABB from (0,0,0) to (1,1,1)
    let inv_dir = 1.0 / ray_dir;
    let t0 = (vec3f(0.0) - ray_origin) * inv_dir;
    let t1 = (vec3f(1.0) - ray_origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(t_near, t_far);
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    return vec4f(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy / camera.resolution;
    // NDC: [-1, 1]
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    
    // Reconstruct world-space ray
    let clip_near = vec4f(ndc, 0.0, 1.0);
    let clip_far = vec4f(ndc, 1.0, 1.0);
    var world_near = camera.inv_view_proj * clip_near;
    var world_far = camera.inv_view_proj * clip_far;
    world_near /= world_near.w;
    world_far /= world_far.w;
    
    let ray_origin = world_near.xyz;
    let ray_dir = normalize(world_far.xyz - world_near.xyz);
    
    // Intersect with grid AABB [0,1]³
    let t_range = intersect_aabb(ray_origin, ray_dir);
    if (t_range.x > t_range.y || t_range.y < 0.0) {
        // Background: dark gradient
        return vec4f(0.02, 0.02, 0.05 + uv.y * 0.03, 1.0);
    }
    
    let t_start = max(t_range.x, 0.0);
    let t_end = t_range.y;
    let step_size = 1.0 / f32(max(grid_params.grid_size.x, max(grid_params.grid_size.y, grid_params.grid_size.z)));
    
    // Light direction
    let light_dir = normalize(vec3f(0.5, 1.0, 0.3));
    
    // Raymarch
    var accum_color = vec3f(0.0);
    var accum_alpha: f32 = 0.0;
    var t = t_start;
    
    while (t < t_end && accum_alpha < 0.95) {
        let pos = ray_origin + ray_dir * t;
        let alive = sample_alive(pos);
        
        if (alive > 0.5) {
            let cell_color = sample_color(pos);
            
            // Simple normal estimation via central differences
            let eps = step_size * 0.5;
            let nx = sample_alive(pos + vec3f(eps, 0.0, 0.0)) - sample_alive(pos - vec3f(eps, 0.0, 0.0));
            let ny = sample_alive(pos + vec3f(0.0, eps, 0.0)) - sample_alive(pos - vec3f(0.0, eps, 0.0));
            let nz = sample_alive(pos + vec3f(0.0, 0.0, eps)) - sample_alive(pos - vec3f(0.0, 0.0, eps));
            var normal = normalize(vec3f(nx, ny, nz));
            if (length(vec3f(nx, ny, nz)) < 0.001) {
                normal = -ray_dir; // fallback
            }
            
            let diffuse = max(dot(normal, light_dir), 0.1);
            let lit_color = cell_color.rgb * diffuse;
            
            // Front-to-back compositing
            let sample_alpha = cell_color.a * 0.8;
            accum_color += lit_color * sample_alpha * (1.0 - accum_alpha);
            accum_alpha += sample_alpha * (1.0 - accum_alpha);
        }
        
        t += step_size;
    }
    
    // Blend with background
    let bg = vec3f(0.02, 0.02, 0.05 + uv.y * 0.03);
    let final_color = accum_color + bg * (1.0 - accum_alpha);
    
    return vec4f(final_color, 1.0);
}
```

- [ ] **Step 4: Update gpu.zig render pipeline**

The render pipeline now needs a bind group with:
- binding 0: camera uniform buffer (96 bytes)
- binding 1: grid params uniform buffer (16 bytes)  
- binding 2: grid storage buffer (read-only in fragment)

Create the camera and grid_params uniform buffers in Gpu.init. Add a new method `renderFrameWithGrid(grid, camera_data)` that:
1. Updates camera uniform buffer via wgpuQueueWriteBuffer
2. Updates grid params uniform
3. Creates bind group with camera + grid_params + grid.readBuffer()
4. Draws the fullscreen triangle with this bind group

For the hardcoded camera (no input yet), compute from Zig:
- Camera position: `(1.5, 1.5, 1.5)` looking at `(0.5, 0.5, 0.5)` (center of unit cube)
- FOV: 60 degrees
- Build view matrix (lookAt), projection matrix (perspective), multiply, invert

- [ ] **Step 5: Update main.zig**

- Remove the simulation test prints (keep grid + sim creation)
- Run a few sim steps to grow the structure before entering the render loop
- In the render loop, call `gpu.renderFrameWithGrid(grid, camera_data)` instead of `gpu.renderFrame(0,0,0)`

- [ ] **Step 6: Build and run**

Expected: a window showing the 3D grid from a fixed diagonal angle. Alive cells should be visible as colored voxels with basic lighting against a dark background.

- [ ] **Step 7: Commit**

```bash
git add src/gpu.zig src/main.zig
git commit -m "feat: raymarch 3D grid with directional lighting

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Orbit camera

**Files:**
- Create: `src/camera.zig` (orbit camera with matrix math)
- Create: `src/input.zig` (GLFW callbacks for mouse/keyboard)
- Modify: `src/main.zig` (integrate camera + input)

- [ ] **Step 1: Create src/camera.zig**

The Camera struct holds:
- `theta`, `phi`: spherical coordinates (radians)
- `radius`: distance from target
- `target`: vec3 center of orbit (grid center: 0.5, 0.5, 0.5)
- `fov`: field of view in radians
- `aspect`: width/height

Methods:
- `position()` → vec3: compute camera world position from spherical coords
- `viewMatrix()` → mat4: lookAt from position toward target
- `projMatrix()` → mat4: perspective projection
- `viewProjMatrix()` → mat4: proj * view
- `invViewProjMatrix()` → mat4: inverse of viewProj (needed by shader)
- `orbit(dx, dy)`: adjust theta/phi by delta (from mouse drag)
- `zoom(delta)`: adjust radius (from scroll)

Implement matrix math manually in Zig (no external deps):
- 4x4 matrix as `[4][4]f32` or `[16]f32`
- lookAt, perspective, multiply, inverse functions
- Keep it simple and correct, not optimized

- [ ] **Step 2: Create src/input.zig**

Wraps GLFW callbacks:
- Mouse button callback: track left button press/release
- Cursor position callback: when dragging, compute delta and call camera.orbit()
- Scroll callback: call camera.zoom()
- Key callback: Space = toggle pause, R = reset, Escape = close

Store state in a struct that's accessible from callbacks (use GLFW user pointer or a global).

- [ ] **Step 3: Integrate in main.zig**

- Create Camera with default orbit position
- Set up Input with GLFW callbacks
- In render loop: update camera from input, pass camera uniforms to renderer

- [ ] **Step 4: Build and run**

Expected: drag mouse to orbit around the structure, scroll to zoom. The 3D growth should be visible from all angles.

- [ ] **Step 5: Commit**

```bash
git add src/camera.zig src/input.zig src/main.zig
git commit -m "feat: orbit camera with mouse drag and scroll zoom

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Cellular automaton with real rules

**Files:**
- Modify: `src/simulation.zig` (Moore neighborhood, birth/survival rules as uniforms)
- Modify: `src/main.zig` (configurable rules)

- [ ] **Step 1: Update the WGSL compute shader**

Replace the simple "spread to neighbors" rule with a proper 3D cellular automaton:
- Count alive cells in the 26-cell Moore neighborhood (all neighbors including diagonals)
- Birth: dead cell with exactly N alive neighbors becomes alive
- Survival: alive cell with M to K alive neighbors stays alive, otherwise dies
- N, M, K are passed as uniforms so they can be tuned at runtime

Extend the Params uniform:
```
struct Params {
    width: u32,
    height: u32,
    depth: u32,
    floats_per_cell: u32,
    birth_min: u32,    // new
    birth_max: u32,    // new
    survival_min: u32, // new
    survival_max: u32, // new
}
```

Default rules to try: birth=4, survival=4-4 (crystal-like growth).

- [ ] **Step 2: Color based on neighbor count**

Give newly born cells a color based on their neighbor count — more neighbors = brighter. This creates visual variety in the structure.

- [ ] **Step 3: Run in the render loop**

Move simulation stepping into the main loop:
- If not paused, run one sim step per frame
- Space toggles pause
- The structure should grow visibly over time

- [ ] **Step 4: Build and run**

Expected: a crystal-like structure growing from a single seed, visible in 3D with orbit camera. Try different rule sets.

- [ ] **Step 5: Commit**

```bash
git add src/simulation.zig src/main.zig
git commit -m "feat: 3D cellular automaton with configurable birth/survival rules

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Continuous growth with pause/step controls

**Files:**
- Modify: `src/main.zig` (sim loop integration, controls)
- Modify: `src/input.zig` (add R for reset, step-by-step mode)

- [ ] **Step 1: Integrate simulation into render loop**

The main loop should:
- Poll input
- If not paused: run one sim step per frame (or multiple for small grids)
- Always render the current grid state
- Display step count in window title

- [ ] **Step 2: Add controls**

- Space: toggle pause/resume
- R: reset grid to seed, restart simulation
- Right arrow (or N): single step when paused
- Escape: close

- [ ] **Step 3: Scale up to 32³ or 64³**

Increase grid size from 8³ (test) to 32³ or 64³. Adjust camera distance accordingly. Verify performance stays interactive.

- [ ] **Step 4: Build and run**

Expected: watch a structure grow from a seed in real time. Pause, step through, reset, orbit around it. This is the Phase 1 exit criteria from the design spec.

- [ ] **Step 5: Commit**

```bash
git add src/main.zig src/input.zig src/simulation.zig
git commit -m "feat: real-time growth with pause/step/reset controls

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Notes for Next Plan

After completing these 4 tasks, Phase 1 exit criteria are met:
- A seed cell grows into a visible 3D structure
- Rendered in real time via volumetric raymarching
- Camera controls (orbit, zoom)
- Simulation controls (pause, step, reset)
- Configurable CA rules

The next plan covers **Tasks 12-15 (Signal & Branching)**: signal diffusion, chemotaxis, branching trees, interactive signal placement.
