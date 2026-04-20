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
;

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

        std.debug.print("simulation: diffuse pipeline created (source at {d},{d},{d})\n", .{
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

        // Create bind group (fresh each call, buffers may have swapped)
        const bind_group = self.createBindGroup(grid);
        defer c.wgpuBindGroupRelease(bind_group);

        // Dispatch
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
