const std = @import("std");
const c = @cImport({
    @cInclude("webgpu.h");
    @cInclude("wgpu.h");
});

const wgsl_shader =
    \\@vertex
    \\fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    \\    var pos = array<vec2f, 3>(
    \\        vec2f(-1.0, -1.0),
    \\        vec2f( 3.0, -1.0),
    \\        vec2f(-1.0,  3.0),
    \\    );
    \\    return vec4f(pos[idx], 0.0, 1.0);
    \\}
    \\
    \\@fragment
    \\fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    \\    let uv = pos.xy / vec2f(800.0, 600.0);
    \\    let col = vec3f(uv.x * 0.1, uv.y * 0.15, 0.2 + uv.y * 0.15);
    \\    return vec4f(col, 1.0);
    \\}
;

pub const Gpu = struct {
    instance: c.WGPUInstance,
    surface: c.WGPUSurface,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    surface_config: c.WGPUSurfaceConfiguration,
    pipeline: c.WGPURenderPipeline,

    pub fn init(metal_layer: *anyopaque, width: u32, height: u32) !Gpu {
        // Create instance
        const instance_desc = std.mem.zeroes(c.WGPUInstanceDescriptor);
        const instance = c.wgpuCreateInstance(&instance_desc);
        if (instance == null) {
            std.debug.print("Failed to create WGPUInstance\n", .{});
            return error.WGPUInstanceFailed;
        }

        // Create surface from Metal layer
        var metal_source = std.mem.zeroes(c.WGPUSurfaceSourceMetalLayer);
        metal_source.chain.sType = c.WGPUSType_SurfaceSourceMetalLayer;
        metal_source.chain.next = null;
        metal_source.layer = metal_layer;

        var surface_desc = std.mem.zeroes(c.WGPUSurfaceDescriptor);
        surface_desc.nextInChain = @ptrCast(&metal_source.chain);
        surface_desc.label = c.WGPUStringView{ .data = "surface", .length = 7 };

        const surface = c.wgpuInstanceCreateSurface(instance, &surface_desc);
        if (surface == null) {
            std.debug.print("Failed to create WGPUSurface\n", .{});
            return error.WGPUSurfaceFailed;
        }

        // Request adapter
        var adapter_result: c.WGPUAdapter = null;
        var adapter_request_done: bool = false;

        var adapter_opts = std.mem.zeroes(c.WGPURequestAdapterOptions);
        adapter_opts.compatibleSurface = surface;
        adapter_opts.powerPreference = c.WGPUPowerPreference_HighPerformance;
        adapter_opts.featureLevel = c.WGPUFeatureLevel_Core;
        adapter_opts.backendType = c.WGPUBackendType_Undefined;

        const adapter_callback_info = c.WGPURequestAdapterCallbackInfo{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = &adapterCallback,
            .userdata1 = @ptrCast(&adapter_result),
            .userdata2 = @ptrCast(&adapter_request_done),
        };

        _ = c.wgpuInstanceRequestAdapter(instance, &adapter_opts, adapter_callback_info);

        // Poll until the callback fires
        while (!adapter_request_done) {
            c.wgpuInstanceProcessEvents(instance);
        }

        if (adapter_result == null) {
            std.debug.print("Failed to get WGPUAdapter\n", .{});
            return error.WGPUAdapterFailed;
        }

        // Request device
        var device_result: c.WGPUDevice = null;
        var device_request_done: bool = false;

        var device_desc = std.mem.zeroes(c.WGPUDeviceDescriptor);
        device_desc.label = c.WGPUStringView{ .data = "device", .length = 6 };

        const device_callback_info = c.WGPURequestDeviceCallbackInfo{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = &deviceCallback,
            .userdata1 = @ptrCast(&device_result),
            .userdata2 = @ptrCast(&device_request_done),
        };

        _ = c.wgpuAdapterRequestDevice(adapter_result, &device_desc, device_callback_info);

        // Poll until the callback fires
        while (!device_request_done) {
            c.wgpuInstanceProcessEvents(instance);
        }

        if (device_result == null) {
            std.debug.print("Failed to get WGPUDevice\n", .{});
            return error.WGPUDeviceFailed;
        }

        // Get queue
        const queue = c.wgpuDeviceGetQueue(device_result);
        if (queue == null) {
            std.debug.print("Failed to get WGPUQueue\n", .{});
            return error.WGPUQueueFailed;
        }

        // Configure surface
        var config = std.mem.zeroes(c.WGPUSurfaceConfiguration);
        config.device = device_result;
        config.format = c.WGPUTextureFormat_BGRA8Unorm;
        config.usage = c.WGPUTextureUsage_RenderAttachment;
        config.width = width;
        config.height = height;
        config.presentMode = c.WGPUPresentMode_Fifo;
        config.alphaMode = c.WGPUCompositeAlphaMode_Auto;
        config.viewFormatCount = 0;
        config.viewFormats = null;

        c.wgpuSurfaceConfigure(surface, &config);

        // Create shader module from WGSL source
        var wgsl_source = std.mem.zeroes(c.WGPUShaderSourceWGSL);
        wgsl_source.chain.sType = c.WGPUSType_ShaderSourceWGSL;
        wgsl_source.chain.next = null;
        wgsl_source.code = c.WGPUStringView{ .data = wgsl_shader.ptr, .length = wgsl_shader.len };

        var shader_desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
        shader_desc.nextInChain = @ptrCast(&wgsl_source.chain);
        shader_desc.label = c.WGPUStringView{ .data = "fullscreen_triangle", .length = 19 };

        const shader_module = c.wgpuDeviceCreateShaderModule(device_result, &shader_desc);
        defer c.wgpuShaderModuleRelease(shader_module);

        // Color target: BGRA8Unorm, no blending, write all
        const color_target = c.WGPUColorTargetState{
            .nextInChain = null,
            .format = c.WGPUTextureFormat_BGRA8Unorm,
            .blend = null,
            .writeMask = c.WGPUColorWriteMask_All,
        };

        // Fragment state
        var fragment_state = std.mem.zeroes(c.WGPUFragmentState);
        fragment_state.module = shader_module;
        fragment_state.entryPoint = c.WGPUStringView{ .data = "fs_main", .length = 7 };
        fragment_state.targetCount = 1;
        fragment_state.targets = &color_target;

        // Render pipeline descriptor
        var pipeline_desc = std.mem.zeroes(c.WGPURenderPipelineDescriptor);
        pipeline_desc.label = c.WGPUStringView{ .data = "fullscreen_pipeline", .length = 19 };
        pipeline_desc.layout = null; // auto layout
        pipeline_desc.vertex.module = shader_module;
        pipeline_desc.vertex.entryPoint = c.WGPUStringView{ .data = "vs_main", .length = 7 };
        pipeline_desc.vertex.bufferCount = 0;
        pipeline_desc.vertex.buffers = null;
        pipeline_desc.primitive.topology = c.WGPUPrimitiveTopology_TriangleList;
        pipeline_desc.primitive.stripIndexFormat = c.WGPUIndexFormat_Undefined;
        pipeline_desc.primitive.cullMode = c.WGPUCullMode_None;
        pipeline_desc.multisample.count = 1;
        pipeline_desc.multisample.mask = 0xFFFFFFFF;
        pipeline_desc.multisample.alphaToCoverageEnabled = 0;
        pipeline_desc.depthStencil = null;
        pipeline_desc.fragment = &fragment_state;

        const pipeline = c.wgpuDeviceCreateRenderPipeline(device_result, &pipeline_desc);
        if (pipeline == null) {
            std.debug.print("Failed to create render pipeline\n", .{});
            return error.WGPURenderPipelineFailed;
        }

        std.debug.print("morphogen: wgpu initialized ({}x{})\n", .{ width, height });

        return Gpu{
            .instance = instance,
            .surface = surface,
            .adapter = adapter_result,
            .device = device_result,
            .queue = queue,
            .surface_config = config,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *Gpu) void {
        c.wgpuRenderPipelineRelease(self.pipeline);
        c.wgpuSurfaceUnconfigure(self.surface);
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuSurfaceRelease(self.surface);
        c.wgpuInstanceRelease(self.instance);
    }

    pub fn resize(self: *Gpu, width: u32, height: u32) void {
        self.surface_config.width = width;
        self.surface_config.height = height;
        c.wgpuSurfaceConfigure(self.surface, &self.surface_config);
    }

    pub fn renderFrame(self: *Gpu, r: f64, g: f64, b: f64) void {
        // Get current surface texture
        var surface_texture: c.WGPUSurfaceTexture = undefined;
        c.wgpuSurfaceGetCurrentTexture(self.surface, &surface_texture);

        if (surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal and
            surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        {
            // Skip this frame
            return;
        }

        // Create texture view
        const view = c.wgpuTextureCreateView(surface_texture.texture, null);
        defer c.wgpuTextureViewRelease(view);
        defer {
            c.wgpuTextureRelease(surface_texture.texture);
        }

        // Create command encoder
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        defer c.wgpuCommandEncoderRelease(encoder);

        // Begin render pass
        const color_attachment = c.WGPURenderPassColorAttachment{
            .nextInChain = null,
            .view = view,
            .depthSlice = c.WGPU_DEPTH_SLICE_UNDEFINED,
            .resolveTarget = null,
            .loadOp = c.WGPULoadOp_Clear,
            .storeOp = c.WGPUStoreOp_Store,
            .clearValue = c.WGPUColor{ .r = r, .g = g, .b = b, .a = 1.0 },
        };

        var render_pass_desc = std.mem.zeroes(c.WGPURenderPassDescriptor);
        render_pass_desc.label = c.WGPUStringView{ .data = "render_pass", .length = 11 };
        render_pass_desc.colorAttachmentCount = 1;
        render_pass_desc.colorAttachments = &color_attachment;
        render_pass_desc.depthStencilAttachment = null;
        render_pass_desc.occlusionQuerySet = null;
        render_pass_desc.timestampWrites = null;

        const render_pass = c.wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
        c.wgpuRenderPassEncoderSetPipeline(render_pass, self.pipeline);
        c.wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
        c.wgpuRenderPassEncoderEnd(render_pass);
        c.wgpuRenderPassEncoderRelease(render_pass);

        // Finish and submit
        const command_buffer = c.wgpuCommandEncoderFinish(encoder, null);
        defer c.wgpuCommandBufferRelease(command_buffer);

        c.wgpuQueueSubmit(self.queue, 1, &command_buffer);

        // Present
        _ = c.wgpuSurfacePresent(self.surface);
    }
};

fn adapterCallback(status: c.WGPURequestAdapterStatus, adapter: c.WGPUAdapter, message: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    const result_ptr: *c.WGPUAdapter = @ptrCast(@alignCast(userdata1));
    const done_ptr: *bool = @ptrCast(@alignCast(userdata2));

    if (status == c.WGPURequestAdapterStatus_Success) {
        result_ptr.* = adapter;
    } else {
        const data: ?[*]const u8 = message.data;
        if (data != null and message.length > 0) {
            const msg_len = if (message.length == std.math.maxInt(usize)) strLen(data.?) else message.length;
            std.debug.print("Adapter request failed: {s}\n", .{data.?[0..msg_len]});
        } else {
            std.debug.print("Adapter request failed with status: {}\n", .{status});
        }
    }
    done_ptr.* = true;
}

fn deviceCallback(status: c.WGPURequestDeviceStatus, device: c.WGPUDevice, message: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    const result_ptr: *c.WGPUDevice = @ptrCast(@alignCast(userdata1));
    const done_ptr: *bool = @ptrCast(@alignCast(userdata2));

    if (status == c.WGPURequestDeviceStatus_Success) {
        result_ptr.* = device;
    } else {
        const data: ?[*]const u8 = message.data;
        if (data != null and message.length > 0) {
            const msg_len = if (message.length == std.math.maxInt(usize)) strLen(data.?) else message.length;
            std.debug.print("Device request failed: {s}\n", .{data.?[0..msg_len]});
        } else {
            std.debug.print("Device request failed with status: {}\n", .{status});
        }
    }
    done_ptr.* = true;
}

fn strLen(ptr: [*]const u8) usize {
    var len: usize = 0;
    while (ptr[len] != 0) : (len += 1) {}
    return len;
}
