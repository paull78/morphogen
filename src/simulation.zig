const std = @import("std");
const c = @import("gpu.zig").c;
const Grid = @import("grid.zig").Grid;

const Params = extern struct {
    width: u32,
    height: u32,
    depth: u32,
    floats_per_cell: u32,
    birth_min: u32,
    birth_max: u32,
    survival_min: u32,
    survival_max: u32,
    step: u32,        // incremented each step, used as RNG seed
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const shader_src =
    \\struct Params {
    \\    width: u32,
    \\    height: u32,
    \\    depth: u32,
    \\    floats_per_cell: u32,
    \\    birth_min: u32,
    \\    birth_max: u32,
    \\    survival_min: u32,
    \\    survival_max: u32,
    \\    step: u32,
    \\    _pad0: u32,
    \\    _pad1: u32,
    \\    _pad2: u32,
    \\}
    \\
    \\@group(0) @binding(0) var<uniform> params: Params;
    \\@group(0) @binding(1) var<storage, read> grid_in: array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> grid_out: array<f32>;
    \\
    \\// Hash-based pseudo-random: returns 0.0..1.0
    \\fn hash(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    \\    var h = x * 374761393u + y * 668265263u + z * 1274126177u + seed * 1103515245u;
    \\    h = (h ^ (h >> 13u)) * 1274126177u;
    \\    h = h ^ (h >> 16u);
    \\    return f32(h & 0xFFFFu) / 65535.0;
    \\}
    \\
    \\fn cell_index(x: u32, y: u32, z: u32) -> u32 {
    \\    return (z * params.height * params.width + y * params.width + x) * params.floats_per_cell;
    \\}
    \\
    \\@compute @workgroup_size(4, 4, 4)
    \\fn main(@builtin(global_invocation_id) id: vec3u) {
    \\    let x = id.x;
    \\    let y = id.y;
    \\    let z = id.z;
    \\
    \\    if (x >= params.width || y >= params.height || z >= params.depth) {
    \\        return;
    \\    }
    \\
    \\    let idx = cell_index(x, y, z);
    \\    let alive = grid_in[idx] > 0.5;
    \\    let rng = hash(x, y, z, params.step);
    \\
    \\    // Count Moore neighborhood (26 neighbors)
    \\    var count: u32 = 0;
    \\    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
    \\        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    \\            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
    \\                if (dx == 0 && dy == 0 && dz == 0) { continue; }
    \\                let nx = i32(x) + dx;
    \\                let ny = i32(y) + dy;
    \\                let nz = i32(z) + dz;
    \\                if (nx >= 0 && nx < i32(params.width) &&
    \\                    ny >= 0 && ny < i32(params.height) &&
    \\                    nz >= 0 && nz < i32(params.depth)) {
    \\                    if (grid_in[cell_index(u32(nx), u32(ny), u32(nz))] > 0.5) {
    \\                        count = count + 1;
    \\                    }
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    // r channel stores age (incremented each step a cell survives)
    \\    let age = grid_in[idx + 1];
    \\
    \\    if (alive) {
    \\        if (count >= params.survival_min && count <= params.survival_max) {
    \\            // Survive: increment age, color by age + neighbor pressure
    \\            let new_age = min(age + 1.0, 50.0);
    \\            let t = new_age / 50.0; // 0..1 over lifetime
    \\            let pressure = f32(count) / 26.0; // how crowded
    \\            grid_out[idx]     = 1.0;
    \\            grid_out[idx + 1] = new_age;
    \\            // Young: bright cyan. Old: muted teal. Crowded: warmer
    \\            grid_out[idx + 2] = 0.1 + pressure * 0.3;
    \\            grid_out[idx + 3] = 0.7 - t * 0.3;
    \\            grid_out[idx + 4] = 0.9 - t * 0.4;
    \\        } else {
    \\            // Dying: stay visible as red for one frame (age = -1 sentinel)
    \\            if (age >= 0.0) {
    \\                grid_out[idx]     = 1.0;  // still "alive" visually
    \\                grid_out[idx + 1] = -1.0; // sentinel: will be removed next step
    \\                grid_out[idx + 2] = 0.8;  // red flash
    \\                grid_out[idx + 3] = 0.1;
    \\                grid_out[idx + 4] = 0.1;
    \\            } else {
    \\                // Was already flashing red — now actually die
    \\                for (var i: u32 = 0; i < params.floats_per_cell; i = i + 1) {
    \\                    grid_out[idx + i] = 0.0;
    \\                }
    \\            }
    \\        }
    \\    } else {
    \\        if (count >= params.birth_min && count <= params.birth_max && rng < 0.5) {
    \\            // Birth: bright cyan, age=0
    \\            grid_out[idx]     = 1.0;
    \\            grid_out[idx + 1] = 0.0;  // age = 0
    \\            grid_out[idx + 2] = 0.05;
    \\            grid_out[idx + 3] = 0.9;
    \\            grid_out[idx + 4] = 1.0;
    \\        } else {
    \\            // Stay dead
    \\            for (var i: u32 = 0; i < params.floats_per_cell; i = i + 1) {
    \\                grid_out[idx + i] = 0.0;
    \\            }
    \\        }
    \\    }
    \\}
;

pub const Simulation = struct {
    device: c.WGPUDevice,
    pipeline: c.WGPUComputePipeline,
    bgl: c.WGPUBindGroupLayout,
    params_buf: c.WGPUBuffer,
    birth_min: u32,
    birth_max: u32,
    survival_min: u32,
    survival_max: u32,
    step_count: u32,

    pub fn init(
        device: c.WGPUDevice,
        grid: *const Grid,
        birth_min: u32,
        birth_max: u32,
        survival_min: u32,
        survival_max: u32,
    ) !Simulation {
        // Params uniform buffer (32 bytes)
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

        // Bind group layout: binding 0 = uniform, 1 = read-only storage, 2 = storage
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

        // Compute pipeline
        var cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        cp_desc.layout = pipeline_layout;
        cp_desc.compute.module = shader_module;
        cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "main", .length = 4 };
        const pipeline = c.wgpuDeviceCreateComputePipeline(device, &cp_desc) orelse
            return error.ComputePipelineFailed;

        std.debug.print("simulation: compute pipeline created (Moore neighborhood, birth={d}-{d} survival={d}-{d})\n", .{
            birth_min, birth_max, survival_min, survival_max,
        });

        return Simulation{
            .device = device,
            .pipeline = pipeline,
            .bgl = bgl,
            .params_buf = params_buf,
            .birth_min = birth_min,
            .birth_max = birth_max,
            .survival_min = survival_min,
            .survival_max = survival_max,
            .step_count = 0,
        };
    }

    pub fn deinit(self: *Simulation) void {
        c.wgpuComputePipelineRelease(self.pipeline);
        c.wgpuBindGroupLayoutRelease(self.bgl);
        c.wgpuBufferRelease(self.params_buf);
    }

    pub fn step(self: *Simulation, grid: *Grid) void {
        const queue = c.wgpuDeviceGetQueue(self.device);

        // Upload params
        const params = Params{
            .width = grid.width,
            .height = grid.height,
            .depth = grid.depth,
            .floats_per_cell = grid.floats_per_cell,
            .birth_min = self.birth_min,
            .birth_max = self.birth_max,
            .survival_min = self.survival_min,
            .survival_max = self.survival_max,
            .step = self.step_count,
        };
        self.step_count += 1;
        c.wgpuQueueWriteBuffer(queue, self.params_buf, 0, &params, @sizeOf(Params));

        // Create bind group (fresh each call, buffers may have swapped)
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

        // Dispatch
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        const compute_pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
        c.wgpuComputePassEncoderSetPipeline(compute_pass, self.pipeline);
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
