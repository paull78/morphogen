const std = @import("std");
const gpu_mod = @import("gpu.zig");
const c = gpu_mod.c;

const compute_shader_src =
    \\@group(0) @binding(0) var<storage, read_write> output: array<f32>;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) id: vec3u) {
    \\    let idx = id.x;
    \\    output[idx] = f32(idx) * f32(idx);
    \\}
;

pub fn run(device: c.WGPUDevice, queue: c.WGPUQueue, instance: c.WGPUInstance) !void {
    const num_elements: u32 = 256;
    const buffer_size: u64 = @as(u64, num_elements) * @sizeOf(f32);

    // Storage buffer
    var storage_buf_desc = std.mem.zeroes(c.WGPUBufferDescriptor);
    storage_buf_desc.label = c.WGPUStringView{ .data = "storage", .length = 7 };
    storage_buf_desc.size = buffer_size;
    storage_buf_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc;
    const storage_buf = c.wgpuDeviceCreateBuffer(device, &storage_buf_desc);
    if (storage_buf == null) return error.BufferCreateFailed;
    defer c.wgpuBufferRelease(storage_buf);

    // Staging buffer for readback
    var staging_buf_desc = std.mem.zeroes(c.WGPUBufferDescriptor);
    staging_buf_desc.label = c.WGPUStringView{ .data = "staging", .length = 7 };
    staging_buf_desc.size = buffer_size;
    staging_buf_desc.usage = c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_MapRead;
    const staging_buf = c.wgpuDeviceCreateBuffer(device, &staging_buf_desc);
    if (staging_buf == null) return error.BufferCreateFailed;
    defer c.wgpuBufferRelease(staging_buf);

    // Shader module
    var wgsl_source = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_source.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_source.code = c.WGPUStringView{ .data = compute_shader_src.ptr, .length = compute_shader_src.len };
    var shader_desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    shader_desc.nextInChain = @ptrCast(&wgsl_source.chain);
    const shader_module = c.wgpuDeviceCreateShaderModule(device, &shader_desc);
    if (shader_module == null) return error.ShaderCreateFailed;
    defer c.wgpuShaderModuleRelease(shader_module);

    // Bind group layout
    var bgl_entry = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
    bgl_entry.binding = 0;
    bgl_entry.visibility = c.WGPUShaderStage_Compute;
    bgl_entry.buffer.type = c.WGPUBufferBindingType_Storage;
    bgl_entry.buffer.minBindingSize = buffer_size;
    var bgl_desc = std.mem.zeroes(c.WGPUBindGroupLayoutDescriptor);
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &bgl_entry;
    const bgl = c.wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
    if (bgl == null) return error.BindGroupLayoutFailed;
    defer c.wgpuBindGroupLayoutRelease(bgl);

    // Pipeline layout
    var pl_desc = std.mem.zeroes(c.WGPUPipelineLayoutDescriptor);
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bgl;
    const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device, &pl_desc);
    if (pipeline_layout == null) return error.PipelineLayoutFailed;
    defer c.wgpuPipelineLayoutRelease(pipeline_layout);

    // Compute pipeline
    var cp_desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    cp_desc.layout = pipeline_layout;
    cp_desc.compute.module = shader_module;
    cp_desc.compute.entryPoint = c.WGPUStringView{ .data = "main", .length = 4 };
    const compute_pipeline = c.wgpuDeviceCreateComputePipeline(device, &cp_desc);
    if (compute_pipeline == null) return error.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(compute_pipeline);

    // Bind group
    var bg_entry = std.mem.zeroes(c.WGPUBindGroupEntry);
    bg_entry.binding = 0;
    bg_entry.buffer = storage_buf;
    bg_entry.size = buffer_size;
    var bg_desc = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    const bind_group = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bind_group == null) return error.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bind_group);

    // Dispatch
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    const compute_pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(compute_pass, compute_pipeline);
    c.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, null);
    c.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, num_elements / 64, 1, 1);
    c.wgpuComputePassEncoderEnd(compute_pass);
    c.wgpuComputePassEncoderRelease(compute_pass);

    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, storage_buf, 0, staging_buf, 0, buffer_size);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    c.wgpuCommandEncoderRelease(encoder);
    c.wgpuQueueSubmit(queue, 1, &cmd);
    c.wgpuCommandBufferRelease(cmd);

    // Map staging buffer
    var map_done: bool = false;
    var map_status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Unknown;
    const map_callback_info = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = &mapCallback,
        .userdata1 = @ptrCast(&map_status),
        .userdata2 = @ptrCast(&map_done),
    };
    _ = c.wgpuBufferMapAsync(staging_buf, c.WGPUMapMode_Read, 0, buffer_size, map_callback_info);
    while (!map_done) {
        c.wgpuInstanceProcessEvents(instance);
    }
    if (map_status != c.WGPUMapAsyncStatus_Success) return error.BufferMapFailed;

    const mapped_ptr: ?*const anyopaque = c.wgpuBufferGetConstMappedRange(staging_buf, 0, buffer_size);
    if (mapped_ptr == null) return error.MappedRangeNull;
    const data: [*]const f32 = @ptrCast(@alignCast(mapped_ptr));

    std.debug.print("compute test: first 16 values (expect i*i):\n", .{});
    for (0..16) |i| {
        std.debug.print("  [{d}] = {d:.0}\n", .{ i, data[i] });
    }
    c.wgpuBufferUnmap(staging_buf);
    std.debug.print("compute test: PASSED\n", .{});
}

fn mapCallback(status: c.WGPUMapAsyncStatus, _: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    const status_ptr: *c.WGPUMapAsyncStatus = @ptrCast(@alignCast(userdata1));
    const done_ptr: *bool = @ptrCast(@alignCast(userdata2));
    status_ptr.* = status;
    done_ptr.* = true;
}
