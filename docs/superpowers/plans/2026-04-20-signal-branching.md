# Phase 2: Signal & Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cellular automaton with directed growth: a signal field diffuses through the grid, growth tips follow the gradient, and branching produces tree-like structures.

**Architecture:** Two compute dispatches per simulation step — diffusion first (signal spreads from source), then growth (tips read gradient, advance, branch). Cell format expands from 5 to 6 floats: `[type, signal, r, g, b, a]`. Cell types: 0=empty, 1=axon body, 2=growth tip, 3=branch point. The renderer adds a faint signal overlay and colors cells by type.

**Tech Stack:** Zig 0.14, wgpu-native (WGSL compute shaders), GLFW

---

### Task 1: Expand cell format and add signal source (Step 12a)

**Files:**
- Modify: `src/main.zig` — change floats_per_cell from 5 to 6, update seed data
- Modify: `src/simulation.zig` — add signal source buffer, diffusion compute shader
- Modify: `src/grid.zig` — no structural changes, just receives 6 from main

**Context:** Currently cells are 5 floats: `[alive, age, r, g, b]`. Phase 2 cells are 6 floats: `[type, signal, r, g, b, a]`. Type values: 0=empty, 1=axon, 2=growth tip, 3=branch point. Channel 1 becomes signal strength (0.0–1.0) used by diffusion. RGB channels (2–4) are set by the growth shader based on cell type.

- [ ] **Step 1: Update cell format in main.zig**

In `src/main.zig`, change floats_per_cell from 5 to 6 and update seed data to use the new layout:

```zig
// Line 119: change 5 → 6
var grid = try Grid.init(gpu.device, gpu.queue, gpu.instance, 64, 64, 64, 6);

// Line 123: update seed cell to new layout: [type, signal, r, g, b, a]
// type=2 (growth tip), signal=0, bright cyan color, full opacity
const seed_cell = [_]f32{ 2.0, 0.0, 0.05, 0.9, 1.0, 1.0 };
```

- [ ] **Step 2: Build and verify it compiles**

Run: `zig build 2>&1 | head -20`
Expected: Successful build. The old simulation shader will have mismatched floats_per_cell but that's fine — we're replacing it next.

- [ ] **Step 3: Commit**

```bash
git add src/main.zig
git commit -m "refactor: expand cell format to 6 floats for Phase 2 signal+type layout"
```

---

### Task 2: Signal diffusion compute shader (Step 12b)

**Files:**
- Modify: `src/simulation.zig` — replace CA shader with diffusion + growth two-pass system

**Context:** The diffusion shader runs first each step. It reads signal from the input buffer, applies a discrete Laplacian to spread signal, injects signal at the source position (fixed at grid center, y=top), and writes the updated cell state to the output buffer. Growth logic comes in Task 4 — for now, diffusion just copies cell type/color through unchanged.

The discrete Laplacian for 3D diffusion:
`signal_new = signal + D * (avg_6_neighbors - signal) - decay * signal`

Where `D` (diffusion rate) ≈ 0.1 and `decay` ≈ 0.01. We use 6 face-neighbors (Von Neumann), not 26 Moore neighbors, for stable diffusion.

The signal source is injected by setting `signal = 1.0` at a fixed position each step (the source overrides diffusion at that cell).

- [ ] **Step 1: Write the Params struct with signal source position**

In `src/simulation.zig`, update the Params struct to include signal source coordinates:

```zig
const Params = extern struct {
    width: u32,
    height: u32,
    depth: u32,
    floats_per_cell: u32,
    // Signal diffusion parameters
    source_x: u32,
    source_y: u32,
    source_z: u32,
    step: u32,
    // Growth parameters
    birth_min: u32,   // kept for later, unused in diffusion
    birth_max: u32,
    survival_min: u32,
    survival_max: u32,
};
```

This is 48 bytes (12 × u32), already 16-byte aligned — no padding needed.

- [ ] **Step 2: Write the diffusion WGSL shader**

Replace the entire `shader_src` in `src/simulation.zig` with a two-entry-point shader. The diffusion entry point (`diffuse_main`) handles signal spreading. A second entry point (`growth_main`) will be added in Task 4.

```wgsl
struct Params {
    width: u32,
    height: u32,
    depth: u32,
    floats_per_cell: u32,
    source_x: u32,
    source_y: u32,
    source_z: u32,
    step: u32,
    birth_min: u32,
    birth_max: u32,
    survival_min: u32,
    survival_max: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_out: array<f32>;

fn cell_index(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.height * params.width + y * params.width + x) * params.floats_per_cell;
}

fn get_signal(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || y < 0 || z < 0 ||
        x >= i32(params.width) ||
        y >= i32(params.height) ||
        z >= i32(params.depth)) {
        return 0.0;
    }
    return grid_in[cell_index(u32(x), u32(y), u32(z)) + 1];
}

@compute @workgroup_size(4, 4, 4)
fn diffuse_main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= params.width || y >= params.height || z >= params.depth) {
        return;
    }

    let idx = cell_index(x, y, z);

    // Copy type and color through unchanged
    grid_out[idx]     = grid_in[idx];      // type
    grid_out[idx + 2] = grid_in[idx + 2];  // r
    grid_out[idx + 3] = grid_in[idx + 3];  // g
    grid_out[idx + 4] = grid_in[idx + 4];  // b
    grid_out[idx + 5] = grid_in[idx + 5];  // a

    // Signal diffusion: 6-neighbor Von Neumann Laplacian
    let ix = i32(x);
    let iy = i32(y);
    let iz = i32(z);
    let center = grid_in[idx + 1];

    let avg = (get_signal(ix-1, iy, iz) + get_signal(ix+1, iy, iz)
             + get_signal(ix, iy-1, iz) + get_signal(ix, iy+1, iz)
             + get_signal(ix, iy, iz-1) + get_signal(ix, iy, iz+1)) / 6.0;

    let D = 0.1;     // diffusion rate
    let decay = 0.01; // signal decay

    var new_signal = center + D * (avg - center) - decay * center;
    new_signal = clamp(new_signal, 0.0, 1.0);

    // Inject signal at source position
    if (x == params.source_x && y == params.source_y && z == params.source_z) {
        new_signal = 1.0;
    }

    grid_out[idx + 1] = new_signal;
}
```

- [ ] **Step 3: Update Simulation struct to create diffusion pipeline**

Replace the `init` function in `src/simulation.zig`. The key changes: the shader module now uses the new diffusion WGSL, and we create the compute pipeline with entry point `"diffuse_main"`. Store source position. Remove birth/survival params from the constructor for now — we'll re-add growth params in Task 4.

```zig
pub const Simulation = struct {
    device: c.WGPUDevice,
    diffuse_pipeline: c.WGPUComputePipeline,
    bgl: c.WGPUBindGroupLayout,
    params_buf: c.WGPUBuffer,
    source_x: u32,
    source_y: u32,
    source_z: u32,
    step_count: u32,

    pub fn init(
        device: c.WGPUDevice,
        grid: *const Grid,
        source_x: u32,
        source_y: u32,
        source_z: u32,
    ) !Simulation {
        // Params uniform buffer
        var params_desc = std.mem.zeroes(c.WGPUBufferDescriptor);
        params_desc.label = c.WGPUStringView{ .data = "sim_params", .length = 10 };
        params_desc.size = @sizeOf(Params);
        params_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        const params_buf = c.wgpuDeviceCreateBuffer(device, &params_desc) orelse
            return error.ParamsBufferFailed;

        // Shader module
        var wgsl_source = std.mem.zeroes(c.WGPUShaderSourceWGSL);
        wgsl_source.chain.sType = c.WGPUSType_ShaderSourceWGSL;
        wgsl_source.code = c.WGPUStringView{ .data = shader_src.ptr, .length = shader_src.len };
        var shader_desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
        shader_desc.nextInChain = @ptrCast(&wgsl_source.chain);
        const shader_module = c.wgpuDeviceCreateShaderModule(device, &shader_desc) orelse
            return error.ShaderCreateFailed;
        defer c.wgpuShaderModuleRelease(shader_module);

        // Bind group layout (same 3 bindings: uniform, read storage, write storage)
        var bgl_entries: [3]c.WGPUBindGroupLayoutEntry = undefined;

        bgl_entries[0] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[0].binding = 0;
        bgl_entries[0].visibility = c.WGPUShaderStage_Compute;
        bgl_entries[0].buffer.type = c.WGPUBufferBindingType_Uniform;
        bgl_entries[0].buffer.minBindingSize = @sizeOf(Params);

        bgl_entries[1] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[1].binding = 1;
        bgl_entries[1].visibility = c.WGPUShaderStage_Compute;
        bgl_entries[1].buffer.type = c.WGPUBufferBindingType_ReadOnlyStorage;
        bgl_entries[1].buffer.minBindingSize = grid.buffer_size;

        bgl_entries[2] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[2].binding = 2;
        bgl_entries[2].visibility = c.WGPUShaderStage_Compute;
        bgl_entries[2].buffer.type = c.WGPUBufferBindingType_Storage;
        bgl_entries[2].buffer.minBindingSize = grid.buffer_size;

        var bgl_desc = std.mem.zeroes(c.WGPUBindGroupLayoutDescriptor);
        bgl_desc.entryCount = 3;
        bgl_desc.entries = &bgl_entries[0];
        const bgl = c.wgpuDeviceCreateBindGroupLayout(device, &bgl_desc) orelse
            return error.BindGroupLayoutFailed;

        // Pipeline layout
        var pl_desc = std.mem.zeroes(c.WGPUPipelineLayoutDescriptor);
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &bgl;
        const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device, &pl_desc) orelse
            return error.PipelineLayoutFailed;
        defer c.wgpuPipelineLayoutRelease(pipeline_layout);

        // Diffusion compute pipeline
        var cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        cp_desc.layout = pipeline_layout;
        cp_desc.compute.module = shader_module;
        cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "diffuse_main", .length = 12 };
        const diffuse_pipeline = c.wgpuDeviceCreateComputePipeline(device, &cp_desc) orelse
            return error.ComputePipelineFailed;

        std.debug.print("simulation: diffusion pipeline created (source at {d},{d},{d})\n", .{
            source_x, source_y, source_z,
        });

        return Simulation{
            .device = device,
            .diffuse_pipeline = diffuse_pipeline,
            .bgl = bgl,
            .params_buf = params_buf,
            .source_x = source_x,
            .source_y = source_y,
            .source_z = source_z,
            .step_count = 0,
        };
    }

    pub fn deinit(self: *Simulation) void {
        c.wgpuComputePipelineRelease(self.diffuse_pipeline);
        c.wgpuBindGroupLayoutRelease(self.bgl);
        c.wgpuBufferRelease(self.params_buf);
    }
```

- [ ] **Step 4: Write the diffusion step() function**

Replace the `step` function. For now, it only runs the diffusion pass (no growth yet). The buffer swap happens after diffusion.

```zig
    pub fn step(self: *Simulation, grid: *Grid) void {
        const queue = c.wgpuDeviceGetQueue(self.device);

        // Upload params
        const params = Params{
            .width = grid.width,
            .height = grid.height,
            .depth = grid.depth,
            .floats_per_cell = grid.floats_per_cell,
            .source_x = self.source_x,
            .source_y = self.source_y,
            .source_z = self.source_z,
            .step = self.step_count,
            .birth_min = 0,
            .birth_max = 0,
            .survival_min = 0,
            .survival_max = 0,
        };
        self.step_count += 1;
        c.wgpuQueueWriteBuffer(queue, self.params_buf, 0, &params, @sizeOf(Params));

        // Create bind group
        var bg_entries: [3]c.WGPUBindGroupEntry = undefined;

        bg_entries[0] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[0].binding = 0;
        bg_entries[0].buffer = self.params_buf;
        bg_entries[0].size = @sizeOf(Params);

        bg_entries[1] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[1].binding = 1;
        bg_entries[1].buffer = grid.readBuffer();
        bg_entries[1].size = grid.buffer_size;

        bg_entries[2] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[2].binding = 2;
        bg_entries[2].buffer = grid.writeBuffer();
        bg_entries[2].size = grid.buffer_size;

        var bg_desc = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bg_desc.layout = self.bgl;
        bg_desc.entryCount = 3;
        bg_desc.entries = &bg_entries[0];
        const bind_group = c.wgpuDeviceCreateBindGroup(self.device, &bg_desc);
        defer c.wgpuBindGroupRelease(bind_group);

        // Dispatch diffusion
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        const compute_pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
        c.wgpuComputePassEncoderSetPipeline(compute_pass, self.diffuse_pipeline);
        c.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, null);

        const wx = (grid.width + 3) / 4;
        const wy = (grid.height + 3) / 4;
        const wz = (grid.depth + 3) / 4;
        c.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, wx, wy, wz);
        c.wgpuComputePassEncoderEnd(compute_pass);
        c.wgpuComputePassEncoderRelease(compute_pass);

        const cmd = c.wgpuCommandEncoderFinish(encoder, null);
        c.wgpuCommandEncoderRelease(encoder);
        c.wgpuQueueSubmit(queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);

        grid.swap();
    }
};
```

- [ ] **Step 5: Update main.zig to use new Simulation API**

In `src/main.zig`, update the simulation initialization. Signal source is placed above grid center (x=32, y=60, z=32 for a 64³ grid):

```zig
// Replace the old Simulation.init call (around line 129):
// OLD: var sim = try Simulation.init(gpu.device, &grid, 3, 4, 3, 6);
// NEW:
var sim = try Simulation.init(gpu.device, &grid, 32, 60, 32);
```

Remove the auto-pause at boundary logic (lines 167-170) — it was specific to the CA:

```zig
// DELETE these lines:
            // Auto-pause near grid boundary to avoid flickering
            if (sim_step >= grid.width / 2 - 2) {
                input.paused = true;
                std.debug.print("simulation: auto-paused (reached boundary)\n", .{});
            }
```

Also remove `seedBlock` usage — replace with single growth-tip seed at center:

```zig
// Replace seedBlock call with single center cell seed:
// The seed is a growth tip (type=2) at grid center
grid.seedCenter(&seed_cell);
```

And in the reset handler, replace `seedBlock` with `seedCenter`:

```zig
        if (input.should_reset_sim) {
            grid.clear();
            grid.seedCenter(&seed_cell);
            sim_step = 0;
            input.should_reset_sim = false;
            input.paused = true;
            std.debug.print("simulation: reset to step 0\n", .{});
        }
```

The `seedBlock` function can be deleted entirely.

- [ ] **Step 6: Build and run to verify diffusion spreads**

Run: `zig build run 2>&1 | head -30`

Expected: Application launches. You'll see the seed cell at center. Press Space to unpause. The signal field diffuses outward from (32, 60, 32) but is invisible yet — only the seed voxel renders. This confirms the shader compiles and runs without crashes.

- [ ] **Step 7: Commit**

```bash
git add src/simulation.zig src/main.zig
git commit -m "feat: signal diffusion compute shader with fixed source above grid"
```

---

### Task 3: Visualize signal field as faint overlay (Step 12c)

**Files:**
- Modify: `src/gpu.zig` — update raymarch shader to show signal as faint glow and color by cell type

**Context:** The renderer currently reads `[alive, age, r, g, b]`. We need it to read `[type, signal, r, g, b, a]`. The signal field should render as a faint volumetric glow even in empty cells. Cell type colors: growth tip = bright cyan, axon = muted blue, branch point = white.

- [ ] **Step 1: Update the WGSL fragment shader in gpu.zig**

Replace the `is_alive`, `get_cell_color` functions and the DDA hit-coloring section. The new shader checks cell type (channel 0) for solid voxels and also samples signal (channel 1) for volumetric glow.

Replace the `is_alive` function:

```wgsl
fn get_cell_type(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || y < 0 || z < 0 ||
        x >= i32(grid_params.grid_size.x) ||
        y >= i32(grid_params.grid_size.y) ||
        z >= i32(grid_params.grid_size.z)) {
        return 0.0;
    }
    return grid_data[cell_index(x, y, z)];
}

fn get_signal(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || y < 0 || z < 0 ||
        x >= i32(grid_params.grid_size.x) ||
        y >= i32(grid_params.grid_size.y) ||
        z >= i32(grid_params.grid_size.z)) {
        return 0.0;
    }
    return grid_data[cell_index(x, y, z) + 1];
}
```

Keep `get_cell_color` as-is (reads channels 2-4).

In the DDA loop, replace the `is_alive` check with type-based rendering plus signal glow:

```wgsl
        // Check if this voxel has a living cell (type > 0)
        let cell_type = get_cell_type(voxel.x, voxel.y, voxel.z);
        if (cell_type > 0.5) {
            let cell_color = get_cell_color(voxel.x, voxel.y, voxel.z);
            let ndl = max(dot(face_normal, light_dir), 0.0);
            let lit = ambient + (1.0 - ambient) * ndl;
            let lit_color = cell_color * lit;

            // Front-to-back compositing
            let w = voxel_opacity * (1.0 - accum_alpha);
            accum_color += lit_color * w;
            accum_alpha += w;
        } else {
            // Empty cell: show signal as faint volumetric glow
            let sig = get_signal(voxel.x, voxel.y, voxel.z);
            if (sig > 0.01) {
                // Purple-blue glow, opacity proportional to signal strength
                let glow_color = vec3f(0.2, 0.1, 0.4) * sig;
                let glow_opacity = sig * 0.08; // very subtle
                let w = glow_opacity * (1.0 - accum_alpha);
                accum_color += glow_color * w;
                accum_alpha += w;
            }
        }
```

- [ ] **Step 2: Build and run**

Run: `zig build run 2>&1 | head -20`

Expected: Application launches. The seed voxel is visible at center. After unpausing, a faint purple glow spreads from the signal source at (32, 60, 32) — visible after ~20-30 steps as the diffusion reaches cells the camera ray passes through. The glow is subtle — look for a faint purple haze in the upper part of the grid.

- [ ] **Step 3: Commit**

```bash
git add src/gpu.zig
git commit -m "feat: visualize signal field as faint purple volumetric glow"
```

---

### Task 4: Growth tips + chemotaxis (Step 13)

**Files:**
- Modify: `src/simulation.zig` — add growth compute shader (second entry point + second pipeline + two-pass dispatch)

**Context:** This is where the animation becomes organic. Growth tips (type=2) read the signal gradient (central differences in 3D), identify the direction of strongest signal, and advance one cell in that direction. The old cell becomes an axon body (type=1). A stochastic element prevents perfectly straight lines. Only growth-tip cells are active — axon bodies and empty cells are inert in the growth pass.

The two-pass flow per step:
1. **Diffusion** (already done): read A → write B, swap → B is current
2. **Growth**: read B → write A, swap → A is current

Both passes use the same bind group layout but different pipelines and different read/write buffer assignments. After diffusion swaps once, growth reads from what is now the "read" buffer and writes to the "write" buffer, then swaps again.

- [ ] **Step 1: Add growth shader entry point to the WGSL**

Append the growth shader to `shader_src` in `src/simulation.zig`, after the `diffuse_main` function:

```wgsl
// Hash-based pseudo-random: returns 0.0..1.0
fn hash(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    var h = x * 374761393u + y * 668265263u + z * 1274126177u + seed * 1103515245u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFu) / 65535.0;
}

@compute @workgroup_size(4, 4, 4)
fn growth_main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= params.width || y >= params.height || z >= params.depth) {
        return;
    }

    let idx = cell_index(x, y, z);
    let cell_type = grid_in[idx];

    // Non-tip cells: copy through unchanged
    if (cell_type < 1.5 || cell_type > 2.5) {
        for (var i: u32 = 0; i < params.floats_per_cell; i = i + 1) {
            grid_out[idx + i] = grid_in[idx + i];
        }
        return;
    }

    // This is a growth tip (type == 2). Compute signal gradient.
    let ix = i32(x);
    let iy = i32(y);
    let iz = i32(z);

    let gx = get_signal(ix+1, iy, iz) - get_signal(ix-1, iy, iz);
    let gy = get_signal(ix, iy+1, iz) - get_signal(ix, iy-1, iz);
    let gz = get_signal(ix, iy, iz+1) - get_signal(ix, iy, iz-1);

    // Find dominant direction (largest gradient component)
    let agx = abs(gx);
    let agy = abs(gy);
    let agz = abs(gz);

    var dx: i32 = 0;
    var dy: i32 = 0;
    var dz: i32 = 0;

    if (agx >= agy && agx >= agz) {
        dx = select(-1, 1, gx > 0.0);
    } else if (agy >= agx && agy >= agz) {
        dy = select(-1, 1, gy > 0.0);
    } else {
        dz = select(-1, 1, gz > 0.0);
    }

    // Stochastic deviation: 20% chance to pick a random adjacent direction instead
    let rng = hash(x, y, z, params.step);
    if (rng < 0.2) {
        let dir_idx = u32(rng * 30.0) % 6u; // pick one of 6 face directions
        dx = 0; dy = 0; dz = 0;
        switch (dir_idx) {
            0u: { dx = 1; }
            1u: { dx = -1; }
            2u: { dy = 1; }
            3u: { dy = -1; }
            4u: { dz = 1; }
            default: { dz = -1; }
        }
    }

    let nx = ix + dx;
    let ny = iy + dy;
    let nz = iz + dz;

    // Convert current cell to axon body (type=1, muted blue)
    grid_out[idx]     = 1.0;  // axon body
    grid_out[idx + 1] = grid_in[idx + 1]; // keep signal
    grid_out[idx + 2] = 0.15; // muted blue
    grid_out[idx + 3] = 0.3;
    grid_out[idx + 4] = 0.6;
    grid_out[idx + 5] = 1.0;

    // Place new growth tip at target if in bounds and empty
    if (nx >= 0 && nx < i32(params.width) &&
        ny >= 0 && ny < i32(params.height) &&
        nz >= 0 && nz < i32(params.depth)) {
        let nidx = cell_index(u32(nx), u32(ny), u32(nz));
        let target_type = grid_in[nidx];
        if (target_type < 0.5) {
            // Empty cell → new growth tip (bright cyan)
            grid_out[nidx]     = 2.0;
            grid_out[nidx + 1] = grid_in[nidx + 1]; // keep signal
            grid_out[nidx + 2] = 0.05;
            grid_out[nidx + 3] = 0.9;
            grid_out[nidx + 4] = 1.0;
            grid_out[nidx + 5] = 1.0;
        }
    }
}
```

**Important race condition note:** Multiple growth tips could try to write to the same target cell. In GPU compute, this is a data race. For now, this is acceptable — the result is non-deterministic but visually fine (one tip "wins"). This actually creates interesting organic variation.

- [ ] **Step 2: Create the growth pipeline in init()**

Add a second pipeline to the Simulation struct. In `init()`, after creating `diffuse_pipeline`, create a second pipeline with entry point `"growth_main"` using the same layout:

Add field to struct:
```zig
    growth_pipeline: c.WGPUComputePipeline,
```

In `init()`, after diffuse pipeline creation:
```zig
        // Growth compute pipeline
        var growth_cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        growth_cp_desc.layout = pipeline_layout;
        growth_cp_desc.compute.module = shader_module;
        growth_cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "growth_main", .length = 11 };
        const growth_pipeline = c.wgpuDeviceCreateComputePipeline(device, &growth_cp_desc) orelse
            return error.ComputePipelineFailed;
```

Add to return struct:
```zig
            .growth_pipeline = growth_pipeline,
```

In `deinit()`:
```zig
        c.wgpuComputePipelineRelease(self.growth_pipeline);
```

- [ ] **Step 3: Update step() for two-pass dispatch**

Replace the `step` function with a two-pass version. After diffusion (read→write, swap), run growth (read→write, swap). Each pass needs its own bind group because the read/write buffers have swapped.

```zig
    pub fn step(self: *Simulation, grid: *Grid) void {
        const queue = c.wgpuDeviceGetQueue(self.device);

        // Upload params
        const params = Params{
            .width = grid.width,
            .height = grid.height,
            .depth = grid.depth,
            .floats_per_cell = grid.floats_per_cell,
            .source_x = self.source_x,
            .source_y = self.source_y,
            .source_z = self.source_z,
            .step = self.step_count,
            .birth_min = 0,
            .birth_max = 0,
            .survival_min = 0,
            .survival_max = 0,
        };
        self.step_count += 1;
        c.wgpuQueueWriteBuffer(queue, self.params_buf, 0, &params, @sizeOf(Params));

        const wx = (grid.width + 3) / 4;
        const wy = (grid.height + 3) / 4;
        const wz = (grid.depth + 3) / 4;

        // --- Pass 1: Diffusion ---
        const bg1 = self.createBindGroup(grid);
        defer c.wgpuBindGroupRelease(bg1);

        const enc1 = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        const pass1 = c.wgpuCommandEncoderBeginComputePass(enc1, null);
        c.wgpuComputePassEncoderSetPipeline(pass1, self.diffuse_pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass1, 0, bg1, 0, null);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass1, wx, wy, wz);
        c.wgpuComputePassEncoderEnd(pass1);
        c.wgpuComputePassEncoderRelease(pass1);

        const cmd1 = c.wgpuCommandEncoderFinish(enc1, null);
        c.wgpuCommandEncoderRelease(enc1);
        c.wgpuQueueSubmit(queue, 1, &cmd1);
        c.wgpuCommandBufferRelease(cmd1);

        grid.swap();

        // --- Pass 2: Growth ---
        const bg2 = self.createBindGroup(grid);
        defer c.wgpuBindGroupRelease(bg2);

        const enc2 = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        const pass2 = c.wgpuCommandEncoderBeginComputePass(enc2, null);
        c.wgpuComputePassEncoderSetPipeline(pass2, self.growth_pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass2, 0, bg2, 0, null);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass2, wx, wy, wz);
        c.wgpuComputePassEncoderEnd(pass2);
        c.wgpuComputePassEncoderRelease(pass2);

        const cmd2 = c.wgpuCommandEncoderFinish(enc2, null);
        c.wgpuCommandEncoderRelease(enc2);
        c.wgpuQueueSubmit(queue, 1, &cmd2);
        c.wgpuCommandBufferRelease(cmd2);

        grid.swap();
    }

    fn createBindGroup(self: *const Simulation, grid: *const Grid) c.WGPUBindGroup {
        var bg_entries: [3]c.WGPUBindGroupEntry = undefined;

        bg_entries[0] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[0].binding = 0;
        bg_entries[0].buffer = self.params_buf;
        bg_entries[0].size = @sizeOf(Params);

        bg_entries[1] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[1].binding = 1;
        bg_entries[1].buffer = grid.readBuffer();
        bg_entries[1].size = grid.buffer_size;

        bg_entries[2] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[2].binding = 2;
        bg_entries[2].buffer = grid.writeBuffer();
        bg_entries[2].size = grid.buffer_size;

        var bg_desc = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bg_desc.layout = self.bgl;
        bg_desc.entryCount = 3;
        bg_desc.entries = &bg_entries[0];
        return c.wgpuDeviceCreateBindGroup(self.device, &bg_desc);
    }
```

- [ ] **Step 4: Build and run — watch a tendril grow toward the signal**

Run: `zig build run 2>&1 | head -30`

Expected: The seed growth tip at center advances upward toward the signal source at (32, 60, 32). Each step, the bright cyan tip moves one cell and leaves a muted blue axon body behind. With 20% stochastic deviation, the path isn't perfectly straight — it wobbles slightly. The purple signal glow is visible ahead of the tendril.

- [ ] **Step 5: Commit**

```bash
git add src/simulation.zig
git commit -m "feat: growth tips follow signal gradient via chemotaxis"
```

---

### Task 5: Branching (Step 14)

**Files:**
- Modify: `src/simulation.zig` — add branching logic to growth shader

**Context:** When a growth tip's gradient is ambiguous (two directions have similar signal strength), it branches: the tip becomes a branch point (type=3, white), and two new growth tips are created. This produces tree-like structures.

- [ ] **Step 1: Add branching to the growth shader**

Replace the direction-selection section of `growth_main` (after the gradient computation, before the stochastic deviation). The new logic: if the ratio of second-strongest to strongest gradient component is above a threshold (0.7), branch into both directions.

Replace the direction-selection and cell-writing section of `growth_main` with:

```wgsl
    // Sort gradient components to find strongest and second-strongest
    let agx = abs(gx);
    let agy = abs(gy);
    let agz = abs(gz);

    // Find primary and secondary directions
    var dx1: i32 = 0; var dy1: i32 = 0; var dz1: i32 = 0;
    var dx2: i32 = 0; var dy2: i32 = 0; var dz2: i32 = 0;
    var strongest: f32 = 0.0;
    var second: f32 = 0.0;

    if (agx >= agy && agx >= agz) {
        dx1 = select(-1, 1, gx > 0.0);
        strongest = agx;
        if (agy >= agz) {
            dy2 = select(-1, 1, gy > 0.0);
            second = agy;
        } else {
            dz2 = select(-1, 1, gz > 0.0);
            second = agz;
        }
    } else if (agy >= agx && agy >= agz) {
        dy1 = select(-1, 1, gy > 0.0);
        strongest = agy;
        if (agx >= agz) {
            dx2 = select(-1, 1, gx > 0.0);
            second = agx;
        } else {
            dz2 = select(-1, 1, gz > 0.0);
            second = agz;
        }
    } else {
        dz1 = select(-1, 1, gz > 0.0);
        strongest = agz;
        if (agx >= agy) {
            dx2 = select(-1, 1, gx > 0.0);
            second = agx;
        } else {
            dy2 = select(-1, 1, gy > 0.0);
            second = agy;
        }
    }

    // Stochastic deviation on primary direction: 20% chance to go random
    let rng = hash(x, y, z, params.step);
    if (rng < 0.2) {
        let dir_idx = u32(rng * 30.0) % 6u;
        dx1 = 0; dy1 = 0; dz1 = 0;
        switch (dir_idx) {
            0u: { dx1 = 1; }
            1u: { dx1 = -1; }
            2u: { dy1 = 1; }
            3u: { dy1 = -1; }
            4u: { dz1 = 1; }
            default: { dz1 = -1; }
        }
    }

    // Decide: branch or single advance
    let branch_threshold = 0.7;
    let should_branch = strongest > 0.001 && second / strongest > branch_threshold && rng > 0.4;

    if (should_branch) {
        // Convert to branch point (type=3, white)
        grid_out[idx]     = 3.0;
        grid_out[idx + 1] = grid_in[idx + 1];
        grid_out[idx + 2] = 0.8;
        grid_out[idx + 3] = 0.8;
        grid_out[idx + 4] = 0.9;
        grid_out[idx + 5] = 1.0;

        // Primary branch
        let nx1 = ix + dx1;
        let ny1 = iy + dy1;
        let nz1 = iz + dz1;
        if (nx1 >= 0 && nx1 < i32(params.width) &&
            ny1 >= 0 && ny1 < i32(params.height) &&
            nz1 >= 0 && nz1 < i32(params.depth)) {
            let nidx1 = cell_index(u32(nx1), u32(ny1), u32(nz1));
            if (grid_in[nidx1] < 0.5) {
                grid_out[nidx1]     = 2.0;
                grid_out[nidx1 + 1] = grid_in[nidx1 + 1];
                grid_out[nidx1 + 2] = 0.05;
                grid_out[nidx1 + 3] = 0.9;
                grid_out[nidx1 + 4] = 1.0;
                grid_out[nidx1 + 5] = 1.0;
            }
        }

        // Secondary branch
        let nx2 = ix + dx2;
        let ny2 = iy + dy2;
        let nz2 = iz + dz2;
        if (nx2 >= 0 && nx2 < i32(params.width) &&
            ny2 >= 0 && ny2 < i32(params.height) &&
            nz2 >= 0 && nz2 < i32(params.depth)) {
            let nidx2 = cell_index(u32(nx2), u32(ny2), u32(nz2));
            if (grid_in[nidx2] < 0.5) {
                grid_out[nidx2]     = 2.0;
                grid_out[nidx2 + 1] = grid_in[nidx2 + 1];
                grid_out[nidx2 + 2] = 0.05;
                grid_out[nidx2 + 3] = 0.9;
                grid_out[nidx2 + 4] = 1.0;
                grid_out[nidx2 + 5] = 1.0;
            }
        }
    } else {
        // Single advance: convert to axon body
        grid_out[idx]     = 1.0;
        grid_out[idx + 1] = grid_in[idx + 1];
        grid_out[idx + 2] = 0.15;
        grid_out[idx + 3] = 0.3;
        grid_out[idx + 4] = 0.6;
        grid_out[idx + 5] = 1.0;

        // Place new tip
        let nx = ix + dx1;
        let ny = iy + dy1;
        let nz = iz + dz1;
        if (nx >= 0 && nx < i32(params.width) &&
            ny >= 0 && ny < i32(params.height) &&
            nz >= 0 && nz < i32(params.depth)) {
            let nidx = cell_index(u32(nx), u32(ny), u32(nz));
            if (grid_in[nidx] < 0.5) {
                grid_out[nidx]     = 2.0;
                grid_out[nidx + 1] = grid_in[nidx + 1];
                grid_out[nidx + 2] = 0.05;
                grid_out[nidx + 3] = 0.9;
                grid_out[nidx + 4] = 1.0;
                grid_out[nidx + 5] = 1.0;
            }
        }
    }
```

Remove the old direction-selection code, stochastic deviation, cell-writing, and tip placement that this replaces (everything from `// Find dominant direction` through the end of the function body before the closing `}`).

- [ ] **Step 2: Build and run — watch branching tree structures**

Run: `zig build run 2>&1 | head -20`

Expected: The tendril now branches when signal gradient is ambiguous (which happens away from the direct line to source). You should see: bright cyan tips advancing and splitting, muted blue axon bodies forming the trunk, white branch points where splits occurred. The overall shape grows upward toward the signal source with lateral branches.

- [ ] **Step 3: Tune parameters if needed**

If branching is too frequent (bushy mess) or too rare (just a single line), adjust:
- `branch_threshold`: lower (0.5) = more branching, higher (0.85) = less
- `rng > 0.4` in should_branch: higher = less branching
- Stochastic deviation `0.2`: lower = straighter paths

- [ ] **Step 4: Commit**

```bash
git add src/simulation.zig
git commit -m "feat: branching logic produces tree-like growth structures"
```

---

### Task 6: Interactive signal placement (Step 15)

**Files:**
- Modify: `src/input.zig` — add click-to-place-signal
- Modify: `src/simulation.zig` — expose `setSource()` method
- Modify: `src/main.zig` — wire click to source update

**Context:** The user clicks somewhere in the window. We raycast from the camera through the click point and intersect with the grid bounding box. The signal source moves to the intersection point (clamped to grid bounds). The growth tips will change direction to follow the new signal.

- [ ] **Step 1: Add setSource to Simulation**

In `src/simulation.zig`, add a method:

```zig
    pub fn setSource(self: *Simulation, x: u32, y: u32, z: u32) void {
        self.source_x = x;
        self.source_y = y;
        self.source_z = z;
        std.debug.print("simulation: signal source moved to ({d},{d},{d})\n", .{ x, y, z });
    }
```

- [ ] **Step 2: Add signal placement to Input**

In `src/input.zig`, add a field to store the requested signal position and a flag:

```zig
    should_place_signal: bool,
    signal_click_x: f64,
    signal_click_y: f64,
```

Initialize in `init()`:
```zig
            .should_place_signal = false,
            .signal_click_x = 0,
            .signal_click_y = 0,
```

In `update()`, add right-click detection (after the escape key handling):

```zig
        // Right-click: place signal source
        const right_button = window.getMouseButton(.right);
        if (right_button == .press) {
            self.should_place_signal = true;
            self.signal_click_x = cursor_pos.xpos;
            self.signal_click_y = cursor_pos.ypos;
        }
```

- [ ] **Step 3: Add raycast helper to camera.zig**

In `src/camera.zig`, add a function that takes a click position (pixel coords) and window size, and returns the grid voxel coordinate where the ray enters the grid:

```zig
    pub fn clickToGridPos(self: *const Camera, click_x: f32, click_y: f32, win_w: u32, win_h: u32, grid_w: u32, grid_h: u32, grid_d: u32) ?[3]u32 {
        const w = @as(f32, @floatFromInt(win_w));
        const h = @as(f32, @floatFromInt(win_h));
        const aspect = w / h;

        // NDC coordinates
        const ndc_x = (click_x / w) * 2.0 - 1.0;
        const ndc_y = -((click_y / h) * 2.0 - 1.0); // flip Y

        const view = gpu_mod.mat4LookAt(self.position(), self.target, .{ 0, 1, 0 });
        const proj = gpu_mod.mat4Perspective(self.fov, aspect, 0.01, 100.0);
        const view_proj = gpu_mod.mat4Mul(proj, view);
        const inv_vp = gpu_mod.mat4Inverse(view_proj);

        // Unproject near and far
        const near_clip = mat4MulVec4(inv_vp, .{ ndc_x, ndc_y, 0.0, 1.0 });
        const far_clip = mat4MulVec4(inv_vp, .{ ndc_x, ndc_y, 1.0, 1.0 });

        const ro = [3]f32{
            near_clip[0] / near_clip[3],
            near_clip[1] / near_clip[3],
            near_clip[2] / near_clip[3],
        };
        const far_pt = [3]f32{
            far_clip[0] / far_clip[3],
            far_clip[1] / far_clip[3],
            far_clip[2] / far_clip[3],
        };

        // Ray direction
        const rd = gpu_mod.vec3Normalize(.{
            far_pt[0] - ro[0],
            far_pt[1] - ro[1],
            far_pt[2] - ro[2],
        });

        // Ray-AABB intersection with unit cube [0,1]³
        const inv_rd = [3]f32{
            1.0 / rd[0],
            1.0 / rd[1],
            1.0 / rd[2],
        };
        const t1 = [3]f32{ -ro[0] * inv_rd[0], -ro[1] * inv_rd[1], -ro[2] * inv_rd[2] };
        const t2 = [3]f32{ (1.0 - ro[0]) * inv_rd[0], (1.0 - ro[1]) * inv_rd[1], (1.0 - ro[2]) * inv_rd[2] };

        const tmin = [3]f32{ @min(t1[0], t2[0]), @min(t1[1], t2[1]), @min(t1[2], t2[2]) };
        const tmax = [3]f32{ @max(t1[0], t2[0]), @max(t1[1], t2[1]), @max(t1[2], t2[2]) };

        const t_near = @max(tmin[0], @max(tmin[1], tmin[2]));
        const t_far = @min(tmax[0], @min(tmax[1], tmax[2]));

        if (t_near > t_far or t_far < 0.0) return null;

        // Hit point on grid surface (or midpoint if inside)
        const t_hit = @max(t_near, 0.0) + 0.001;
        const hit = [3]f32{
            ro[0] + rd[0] * t_hit,
            ro[1] + rd[1] * t_hit,
            ro[2] + rd[2] * t_hit,
        };

        // Convert to voxel coords
        const gw = @as(f32, @floatFromInt(grid_w));
        const gh = @as(f32, @floatFromInt(grid_h));
        const gd = @as(f32, @floatFromInt(grid_d));

        const vx = @as(u32, @intFromFloat(std.math.clamp(hit[0] * gw, 0.0, gw - 1.0)));
        const vy = @as(u32, @intFromFloat(std.math.clamp(hit[1] * gh, 0.0, gh - 1.0)));
        const vz = @as(u32, @intFromFloat(std.math.clamp(hit[2] * gd, 0.0, gd - 1.0)));

        return .{ vx, vy, vz };
    }
```

Also add a helper `mat4MulVec4` in `camera.zig`:

```zig
fn mat4MulVec4(m: gpu_mod.Mat4, v: [4]f32) [4]f32 {
    var r: [4]f32 = undefined;
    for (0..4) |row| {
        r[row] = m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row] * v[3];
    }
    return r;
}
```

Note: This requires `vec3Normalize` to be `pub` in gpu.zig — it already is not. Make it pub:
In `src/gpu.zig`, change `fn vec3Normalize` to `pub fn vec3Normalize`.

- [ ] **Step 4: Wire it up in main.zig**

In the main loop in `src/main.zig`, after the input.update() call:

```zig
        // Handle signal source placement
        if (input.should_place_signal) {
            const fb = window.getFramebufferSize();
            if (camera.clickToGridPos(
                @floatCast(input.signal_click_x),
                @floatCast(input.signal_click_y),
                fb.width,
                fb.height,
                grid.width,
                grid.height,
                grid.depth,
            )) |pos| {
                sim.setSource(pos[0], pos[1], pos[2]);
            }
            input.should_place_signal = false;
        }
```

- [ ] **Step 5: Update controls print in main.zig**

Update the controls line:
```zig
    std.debug.print("controls: Space=pause/resume  N/Right=step  R=reset  Right-click=move signal  Escape=quit\n", .{});
```

- [ ] **Step 6: Build and run — test interactive signal placement**

Run: `zig build run 2>&1 | head -20`

Expected: Right-clicking somewhere on the grid moves the signal source. After a few steps, growth tips should change direction toward the new source. The signal glow should reposition accordingly. Try placing the signal to the side — the tree should start bending sideways.

- [ ] **Step 7: Commit**

```bash
git add src/input.zig src/camera.zig src/gpu.zig src/simulation.zig src/main.zig
git commit -m "feat: interactive signal placement via right-click raycast"
```

---

### Task 7: Faster simulation rate + update design spec (Step 15 polish)

**Files:**
- Modify: `src/main.zig` — increase simulation speed for real-time feel
- Modify: `docs/superpowers/specs/2026-04-02-morphogen-design.md` — mark steps 12-15 complete

- [ ] **Step 1: Increase simulation rate**

In `src/main.zig`, change the step rate from 1/sec to 5/sec for a more dynamic feel. Growth is now directional and continuous, so higher rates look good:

```zig
        const steps_per_second: u64 = 5;
```

- [ ] **Step 2: Mark steps 12-15 as complete in design spec**

In `docs/superpowers/specs/2026-04-02-morphogen-design.md`, mark the Phase 2 steps as complete:

```markdown
12. ~~**Signal diffusion** — add signal channel, place a source, visualize signal field as faint color overlay. Verify diffusion spreads correctly.~~ ✅
13. ~~**Growth tips + chemotaxis** — growth tips follow signal gradient. See a single tendril grow toward the source.~~ ✅
14. ~~**Branching** — add branching logic, see tree-like structures emerge. Color by cell type.~~ ✅
15. ~~**Interactive signals** — place/move signal sources with mouse. Watch growth respond in real time.~~ ✅
```

- [ ] **Step 3: Final build and test**

Run: `zig build run`

Expected: Tree grows from center toward signal source at 5 steps/sec. Branching visible. Right-click moves signal, growth redirects. Purple glow visible. Cell types colored: cyan tips, blue axon, white branch points.

- [ ] **Step 4: Commit**

```bash
git add src/main.zig docs/superpowers/specs/2026-04-02-morphogen-design.md
git commit -m "feat: complete Phase 2 — signal diffusion, chemotaxis, branching, interactive signals"
```
