const std = @import("std");
const c = @import("gpu.zig").c;
const Grid = @import("grid.zig").Grid;

const Params = extern struct {
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
};

const shader_src =
    \\struct Params {
    \\    width: u32,
    \\    height: u32,
    \\    depth: u32,
    \\    floats_per_cell: u32,
    \\    source_x: u32,
    \\    source_y: u32,
    \\    source_z: u32,
    \\    step: u32,
    \\    birth_min: u32,
    \\    birth_max: u32,
    \\    survival_min: u32,
    \\    survival_max: u32,
    \\}
    \\
    \\@group(0) @binding(0) var<uniform> params: Params;
    \\@group(0) @binding(1) var<storage, read> grid_in: array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> grid_out: array<f32>;
    \\
    \\fn cell_index(x: u32, y: u32, z: u32) -> u32 {
    \\    return (z * params.height * params.width + y * params.width + x) * params.floats_per_cell;
    \\}
    \\
    \\fn get_signal(x: i32, y: i32, z: i32) -> f32 {
    \\    if (x < 0 || x >= i32(params.width) ||
    \\        y < 0 || y >= i32(params.height) ||
    \\        z < 0 || z >= i32(params.depth)) {
    \\        return 0.0;
    \\    }
    \\    let idx = cell_index(u32(x), u32(y), u32(z));
    \\    return grid_in[idx + 1u];
    \\}
    \\
    \\@compute @workgroup_size(4, 4, 4)
    \\fn diffuse_main(@builtin(global_invocation_id) id: vec3u) {
    \\    let x = id.x;
    \\    let y = id.y;
    \\    let z = id.z;
    \\
    \\    if (x >= params.width || y >= params.height || z >= params.depth) {
    \\        return;
    \\    }
    \\
    \\    let idx = cell_index(x, y, z);
    \\
    \\    // Copy type (channel 0) and color (channels 2-5) unchanged
    \\    grid_out[idx]     = grid_in[idx];
    \\    grid_out[idx + 2u] = grid_in[idx + 2u];
    \\    grid_out[idx + 3u] = grid_in[idx + 3u];
    \\    grid_out[idx + 4u] = grid_in[idx + 4u];
    \\    grid_out[idx + 5u] = grid_in[idx + 5u];
    \\
    \\    // Signal diffusion: 6-neighbor Von Neumann Laplacian
    \\    let center = get_signal(i32(x), i32(y), i32(z));
    \\    let sum_neighbors =
    \\        get_signal(i32(x) - 1, i32(y), i32(z)) +
    \\        get_signal(i32(x) + 1, i32(y), i32(z)) +
    \\        get_signal(i32(x), i32(y) - 1, i32(z)) +
    \\        get_signal(i32(x), i32(y) + 1, i32(z)) +
    \\        get_signal(i32(x), i32(y), i32(z) - 1) +
    \\        get_signal(i32(x), i32(y), i32(z) + 1);
    \\    let avg_neighbors = sum_neighbors / 6.0;
    \\
    \\    let D = 0.1;
    \\    let decay = 0.01;
    \\    var new_signal = center + D * (avg_neighbors - center) - decay * center;
    \\    new_signal = clamp(new_signal, 0.0, 1.0);
    \\
    \\    // Inject signal at source position
    \\    if (x == params.source_x && y == params.source_y && z == params.source_z) {
    \\        new_signal = 1.0;
    \\    }
    \\
    \\    grid_out[idx + 1u] = new_signal;
    \\}
    \\
    \\fn hash(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    \\    var h = x * 374761393u + y * 668265263u + z * 1274126177u + seed * 1103515245u;
    \\    h = (h ^ (h >> 13u)) * 1274126177u;
    \\    h = h ^ (h >> 16u);
    \\    return f32(h & 0xFFFFu) / 65535.0;
    \\}
    \\
    \\// Check if a cell is occupied (type > 0)
    \\fn is_occupied(x: i32, y: i32, z: i32) -> f32 {
    \\    if (x < 0 || x >= i32(params.width) ||
    \\        y < 0 || y >= i32(params.height) ||
    \\        z < 0 || z >= i32(params.depth)) {
    \\        return 0.0;
    \\    }
    \\    let t = grid_in[cell_index(u32(x), u32(y), u32(z))];
    \\    return select(0.0, 1.0, t > 0.5);
    \\}
    \\
    \\// Count occupied cells in a radius-2 neighborhood (excluding self)
    \\fn neighbor_density(x: i32, y: i32, z: i32) -> f32 {
    \\    var count: f32 = 0.0;
    \\    for (var dz: i32 = -2; dz <= 2; dz = dz + 1) {
    \\        for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
    \\            for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
    \\                if (dx == 0 && dy == 0 && dz == 0) { continue; }
    \\                count += is_occupied(x + dx, y + dy, z + dz);
    \\            }
    \\        }
    \\    }
    \\    return count;
    \\}
    \\
    \\// Score a direction for a growth tip: prefer signal, avoid crowded areas
    \\fn score_direction(x: i32, y: i32, z: i32, dx: i32, dy: i32, dz: i32) -> f32 {
    \\    let nx = x + dx;
    \\    let ny = y + dy;
    \\    let nz = z + dz;
    \\    // Out of bounds: very bad
    \\    if (nx < 0 || nx >= i32(params.width) ||
    \\        ny < 0 || ny >= i32(params.height) ||
    \\        nz < 0 || nz >= i32(params.depth)) {
    \\        return -100.0;
    \\    }
    \\    // Occupied: can't go there
    \\    if (is_occupied(nx, ny, nz) > 0.5) {
    \\        return -100.0;
    \\    }
    \\    // Signal pull: higher signal = better
    \\    let sig = get_signal(nx, ny, nz);
    \\    // Self-avoidance: penalize crowded directions
    \\    let density = neighbor_density(nx, ny, nz);
    \\    return sig * 3.0 - density * 0.08;
    \\}
    \\
    \\@compute @workgroup_size(4, 4, 4)
    \\fn growth_main(@builtin(global_invocation_id) id: vec3u) {
    \\    let x = id.x;
    \\    let y = id.y;
    \\    let z = id.z;
    \\
    \\    if (x >= params.width || y >= params.height || z >= params.depth) {
    \\        return;
    \\    }
    \\
    \\    let idx = cell_index(x, y, z);
    \\    let cell_type = grid_in[idx];
    \\    let ix = i32(x);
    \\    let iy = i32(y);
    \\    let iz = i32(z);
    \\    let rng = hash(x, y, z, params.step);
    \\
    \\    // --- Interstitial branching: axon bodies can sprout new tips ---
    \\    if (cell_type > 0.5 && cell_type < 1.5) {
    \\        // Very rare: 0.02% chance, only isolated surface cells
    \\        let density = neighbor_density(ix, iy, iz);
    \\        if (rng < 0.0002 && density < 5.0) {
    \\            // Find an empty face-neighbor to sprout into
    \\            let rng2 = hash(x + 3u, y + 5u, z + 7u, params.step);
    \\            let start_dir = u32(rng2 * 6.0) % 6u;
    \\            // Try all 6 directions starting from a random one
    \\            for (var attempt: u32 = 0; attempt < 6u; attempt = attempt + 1) {
    \\                let dir = (start_dir + attempt) % 6u;
    \\                var sx: i32 = 0; var sy: i32 = 0; var sz: i32 = 0;
    \\                switch (dir) {
    \\                    case 0u: { sx = 1; }
    \\                    case 1u: { sx = -1; }
    \\                    case 2u: { sy = 1; }
    \\                    case 3u: { sy = -1; }
    \\                    case 4u: { sz = 1; }
    \\                    default: { sz = -1; }
    \\                }
    \\                let snx = ix + sx;
    \\                let sny = iy + sy;
    \\                let snz = iz + sz;
    \\                if (snx >= 0 && snx < i32(params.width) &&
    \\                    sny >= 0 && sny < i32(params.height) &&
    \\                    snz >= 0 && snz < i32(params.depth)) {
    \\                    if (is_occupied(snx, sny, snz) < 0.5) {
    \\                        // Sprout! Mark self as branch point, place new tip
    \\                        grid_out[idx]     = 3.0;
    \\                        grid_out[idx + 2u] = 1.0;
    \\                        grid_out[idx + 3u] = 0.95;
    \\                        grid_out[idx + 4u] = 0.7;
    \\                        let nidx = cell_index(u32(snx), u32(sny), u32(snz));
    \\                        grid_out[nidx]     = 2.0;
    \\                        grid_out[nidx + 1u] = grid_in[nidx + 1u];
    \\                        grid_out[nidx + 2u] = 0.05;
    \\                        grid_out[nidx + 3u] = 0.9;
    \\                        grid_out[nidx + 4u] = 1.0;
    \\                        grid_out[nidx + 5u] = 1.0;
    \\                        break;
    \\                    }
    \\                }
    \\            }
    \\        }
    \\        return;
    \\    }
    \\
    \\    // Only growth tips (type=2) do the rest
    \\    if (cell_type < 1.5 || cell_type > 2.5) {
    \\        return;
    \\    }
    \\
    \\    // --- Growth tip: pick the best face-neighbor to advance into ---
    \\    // score_direction prefers high signal (chemotaxis) and penalizes crowding;
    \\    // out-of-bounds / occupied neighbors score -100. A small per-direction jitter
    \\    // keeps tips wandering outward when the signal field is still flat, instead of
    \\    // defaulting backwards into the parent cell (which would dead-end growth).
    \\    var dx: i32 = 0; var dy: i32 = 0; var dz: i32 = 0;
    \\    var best_score: f32 = -1000.0;
    \\    for (var dir: u32 = 0u; dir < 6u; dir = dir + 1u) {
    \\        var ox: i32 = 0; var oy: i32 = 0; var oz: i32 = 0;
    \\        switch (dir) {
    \\            case 0u: { ox = 1; }
    \\            case 1u: { ox = -1; }
    \\            case 2u: { oy = 1; }
    \\            case 3u: { oy = -1; }
    \\            case 4u: { oz = 1; }
    \\            default: { oz = -1; }
    \\        }
    \\        let jitter = hash(x + dir * 13u, y + dir * 7u, z + dir * 5u, params.step) * 0.5;
    \\        let s = score_direction(ix, iy, iz, ox, oy, oz) + jitter;
    \\        if (s > best_score) {
    \\            best_score = s;
    \\            dx = ox; dy = oy; dz = oz;
    \\        }
    \\    }
    \\
    \\    // If every neighbor is out of bounds or occupied, the tip can't advance.
    \\    var nx = ix + dx;
    \\    var ny = iy + dy;
    \\    var nz = iz + dz;
    \\    let blocked = best_score < -50.0;
    \\
    \\    // Convert current cell to axon body (green)
    \\    grid_out[idx]     = 1.0;
    \\    grid_out[idx + 1u] = grid_in[idx + 1u];
    \\    grid_out[idx + 2u] = 0.1;
    \\    grid_out[idx + 3u] = 0.7;
    \\    grid_out[idx + 4u] = 0.2;
    \\    grid_out[idx + 5u] = 1.0;
    \\
    \\    // Place new tip if not blocked
    \\    if (!blocked) {
    \\        let nidx = cell_index(u32(nx), u32(ny), u32(nz));
    \\        grid_out[nidx]     = 2.0;
    \\        grid_out[nidx + 1u] = grid_in[nidx + 1u];
    \\        grid_out[nidx + 2u] = 0.05;
    \\        grid_out[nidx + 3u] = 0.9;
    \\        grid_out[nidx + 4u] = 1.0;
    \\        grid_out[nidx + 5u] = 1.0;
    \\    }
    \\
    \\    // 25% chance: also spawn a perpendicular branch tip
    \\    let rng2 = hash(x + 11u, y + 17u, z + 23u, params.step);
    \\    if (rng2 < 0.25 && !blocked) {
    \\        // Pick perpendicular direction
    \\        var bx: i32 = 0; var by: i32 = 0; var bz: i32 = 0;
    \\        let perp_rng = hash(x + 31u, y + 37u, z + 41u, params.step);
    \\        if (dx != 0) {
    \\            if (perp_rng < 0.5) { by = select(-1, 1, perp_rng < 0.25); }
    \\            else { bz = select(-1, 1, perp_rng < 0.75); }
    \\        } else if (dy != 0) {
    \\            if (perp_rng < 0.5) { bx = select(-1, 1, perp_rng < 0.25); }
    \\            else { bz = select(-1, 1, perp_rng < 0.75); }
    \\        } else {
    \\            if (perp_rng < 0.5) { bx = select(-1, 1, perp_rng < 0.25); }
    \\            else { by = select(-1, 1, perp_rng < 0.75); }
    \\        }
    \\        let bnx = ix + bx;
    \\        let bny = iy + by;
    \\        let bnz = iz + bz;
    \\        if (bnx >= 0 && bnx < i32(params.width) &&
    \\            bny >= 0 && bny < i32(params.height) &&
    \\            bnz >= 0 && bnz < i32(params.depth) &&
    \\            is_occupied(bnx, bny, bnz) < 0.5 &&
    \\            neighbor_density(bnx, bny, bnz) < 10.0) {
    \\            // Mark self as branch point instead of axon
    \\            grid_out[idx]     = 3.0;
    \\            grid_out[idx + 2u] = 1.0;
    \\            grid_out[idx + 3u] = 0.95;
    \\            grid_out[idx + 4u] = 0.7;
    \\            // Place branch tip
    \\            let bidx = cell_index(u32(bnx), u32(bny), u32(bnz));
    \\            grid_out[bidx]     = 2.0;
    \\            grid_out[bidx + 1u] = grid_in[bidx + 1u];
    \\            grid_out[bidx + 2u] = 0.05;
    \\            grid_out[bidx + 3u] = 0.9;
    \\            grid_out[bidx + 4u] = 1.0;
    \\            grid_out[bidx + 5u] = 1.0;
    \\        }
    \\    }
    \\}
;

pub const Simulation = struct {
    device: c.WGPUDevice,
    diffuse_pipeline: c.WGPUComputePipeline,
    growth_pipeline: c.WGPUComputePipeline,
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
        // Params uniform buffer (48 bytes)
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

        // Bind group layout: binding 0 = uniform, 1 = read-only storage, 2 = read_write storage
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

        // Compute pipeline with diffuse_main entry point
        var cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        cp_desc.layout = pipeline_layout;
        cp_desc.compute.module = shader_module;
        cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "diffuse_main", .length = 12 };
        const diffuse_pipeline = c.wgpuDeviceCreateComputePipeline(device, &cp_desc) orelse
            return error.ComputePipelineFailed;

        var growth_cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        growth_cp_desc.layout = pipeline_layout;
        growth_cp_desc.compute.module = shader_module;
        growth_cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "growth_main", .length = 11 };
        const growth_pipeline = c.wgpuDeviceCreateComputePipeline(device, &growth_cp_desc) orelse
            return error.ComputePipelineFailed;

        std.debug.print("simulation: diffuse + growth pipelines created (source at {d},{d},{d})\n", .{
            source_x, source_y, source_z,
        });

        return Simulation{
            .device = device,
            .diffuse_pipeline = diffuse_pipeline,
            .growth_pipeline = growth_pipeline,
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
        c.wgpuComputePipelineRelease(self.growth_pipeline);
        c.wgpuBindGroupLayoutRelease(self.bgl);
        c.wgpuBufferRelease(self.params_buf);
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

    pub fn setSource(self: *Simulation, x: u32, y: u32, z: u32) void {
        self.source_x = x;
        self.source_y = y;
        self.source_z = z;
        std.debug.print("simulation: signal source moved to ({d},{d},{d})\n", .{ x, y, z });
    }

    pub fn step(self: *Simulation, grid: *Grid) void {
        const queue = c.wgpuDeviceGetQueue(self.device);

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

        // Pass 1: Diffusion
        {
            const bg = self.createBindGroup(grid);
            defer c.wgpuBindGroupRelease(bg);
            const enc = c.wgpuDeviceCreateCommandEncoder(self.device, null);
            const pass = c.wgpuCommandEncoderBeginComputePass(enc, null);
            c.wgpuComputePassEncoderSetPipeline(pass, self.diffuse_pipeline);
            c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
            c.wgpuComputePassEncoderDispatchWorkgroups(pass, wx, wy, wz);
            c.wgpuComputePassEncoderEnd(pass);
            c.wgpuComputePassEncoderRelease(pass);
            const cmd = c.wgpuCommandEncoderFinish(enc, null);
            c.wgpuCommandEncoderRelease(enc);
            c.wgpuQueueSubmit(queue, 1, &cmd);
            c.wgpuCommandBufferRelease(cmd);
        }
        grid.swap();

        // Pre-copy read → write so growth shader only needs to modify tip cells
        {
            const enc = c.wgpuDeviceCreateCommandEncoder(self.device, null);
            c.wgpuCommandEncoderCopyBufferToBuffer(enc, grid.readBuffer(), 0, grid.writeBuffer(), 0, grid.buffer_size);
            const cmd = c.wgpuCommandEncoderFinish(enc, null);
            c.wgpuCommandEncoderRelease(enc);
            c.wgpuQueueSubmit(queue, 1, &cmd);
            c.wgpuCommandBufferRelease(cmd);
        }

        // Pass 2: Growth
        {
            const bg = self.createBindGroup(grid);
            defer c.wgpuBindGroupRelease(bg);
            const enc = c.wgpuDeviceCreateCommandEncoder(self.device, null);
            const pass = c.wgpuCommandEncoderBeginComputePass(enc, null);
            c.wgpuComputePassEncoderSetPipeline(pass, self.growth_pipeline);
            c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
            c.wgpuComputePassEncoderDispatchWorkgroups(pass, wx, wy, wz);
            c.wgpuComputePassEncoderEnd(pass);
            c.wgpuComputePassEncoderRelease(pass);
            const cmd = c.wgpuCommandEncoderFinish(enc, null);
            c.wgpuCommandEncoderRelease(enc);
            c.wgpuQueueSubmit(queue, 1, &cmd);
            c.wgpuCommandBufferRelease(cmd);
        }
        grid.swap();
    }
};
