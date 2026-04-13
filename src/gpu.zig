const std = @import("std");
const Grid = @import("grid.zig").Grid;
pub const c = @cImport({
    @cInclude("webgpu.h");
    @cInclude("wgpu.h");
});

const wgsl_shader =
    \\struct Camera {
    \\    inv_view_proj: mat4x4<f32>,
    \\    camera_pos: vec3<f32>,
    \\    _padding: f32,
    \\    resolution: vec2<f32>,
    \\    _padding2: vec2<f32>,
    \\}
    \\
    \\struct GridParams {
    \\    grid_size: vec3<u32>,
    \\    floats_per_cell: u32,
    \\}
    \\
    \\@group(0) @binding(0) var<uniform> camera: Camera;
    \\@group(0) @binding(1) var<uniform> grid_params: GridParams;
    \\@group(0) @binding(2) var<storage, read> grid_data: array<f32>;
    \\
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
    \\fn cell_index(x: u32, y: u32, z: u32) -> u32 {
    \\    return (z * grid_params.grid_size.y * grid_params.grid_size.x
    \\          + y * grid_params.grid_size.x + x) * grid_params.floats_per_cell;
    \\}
    \\
    \\fn sample_grid(pos: vec3f) -> vec4f {
    \\    // pos is in [0,1]^3, convert to grid coords
    \\    let gs = vec3f(f32(grid_params.grid_size.x), f32(grid_params.grid_size.y), f32(grid_params.grid_size.z));
    \\    let gc = pos * gs;
    \\    let ix = u32(floor(gc.x));
    \\    let iy = u32(floor(gc.y));
    \\    let iz = u32(floor(gc.z));
    \\    if (ix >= grid_params.grid_size.x || iy >= grid_params.grid_size.y || iz >= grid_params.grid_size.z) {
    \\        return vec4f(0.0);
    \\    }
    \\    let idx = cell_index(ix, iy, iz);
    \\    let alive = grid_data[idx];
    \\    if (alive < 0.5) {
    \\        return vec4f(0.0);
    \\    }
    \\    let r = grid_data[idx + 1];
    \\    let g = grid_data[idx + 2];
    \\    let b = grid_data[idx + 3];
    \\    let a = grid_data[idx + 4];
    \\    return vec4f(r, g, b, a);
    \\}
    \\
    \\fn sample_alive(pos: vec3f) -> f32 {
    \\    let gs = vec3f(f32(grid_params.grid_size.x), f32(grid_params.grid_size.y), f32(grid_params.grid_size.z));
    \\    let gc = pos * gs;
    \\    let ix = u32(floor(gc.x));
    \\    let iy = u32(floor(gc.y));
    \\    let iz = u32(floor(gc.z));
    \\    if (ix >= grid_params.grid_size.x || iy >= grid_params.grid_size.y || iz >= grid_params.grid_size.z) {
    \\        return 0.0;
    \\    }
    \\    let idx = cell_index(ix, iy, iz);
    \\    return grid_data[idx];
    \\}
    \\
    \\fn estimate_normal(pos: vec3f) -> vec3f {
    \\    let eps = 1.0 / max(f32(grid_params.grid_size.x), max(f32(grid_params.grid_size.y), f32(grid_params.grid_size.z)));
    \\    let dx = sample_alive(pos + vec3f(eps, 0.0, 0.0)) - sample_alive(pos - vec3f(eps, 0.0, 0.0));
    \\    let dy = sample_alive(pos + vec3f(0.0, eps, 0.0)) - sample_alive(pos - vec3f(0.0, eps, 0.0));
    \\    let dz = sample_alive(pos + vec3f(0.0, 0.0, eps)) - sample_alive(pos - vec3f(0.0, 0.0, eps));
    \\    let n = vec3f(dx, dy, dz);
    \\    let len = length(n);
    \\    if (len < 0.001) {
    \\        return vec3f(0.0, 1.0, 0.0);
    \\    }
    \\    return n / len;
    \\}
    \\
    \\fn ray_aabb(ro: vec3f, rd: vec3f, box_min: vec3f, box_max: vec3f) -> vec2f {
    \\    let inv_rd = 1.0 / rd;
    \\    let t1 = (box_min - ro) * inv_rd;
    \\    let t2 = (box_max - ro) * inv_rd;
    \\    let tmin = min(t1, t2);
    \\    let tmax = max(t1, t2);
    \\    let t_near = max(tmin.x, max(tmin.y, tmin.z));
    \\    let t_far = min(tmax.x, min(tmax.y, tmax.z));
    \\    return vec2f(t_near, t_far);
    \\}
    \\
    \\@fragment
    \\fn fs_main(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    \\    // Reconstruct ray from pixel coordinates
    \\    let uv = (frag_pos.xy / camera.resolution) * 2.0 - 1.0;
    \\    // Flip Y for correct orientation
    \\    let ndc = vec4f(uv.x, -uv.y, 0.0, 1.0);
    \\    let ndc_far = vec4f(uv.x, -uv.y, 1.0, 1.0);
    \\
    \\    var world_near = camera.inv_view_proj * ndc;
    \\    world_near = world_near / world_near.w;
    \\    var world_far = camera.inv_view_proj * ndc_far;
    \\    world_far = world_far / world_far.w;
    \\
    \\    let ro = world_near.xyz;
    \\    let rd = normalize(world_far.xyz - world_near.xyz);
    \\
    \\    // Ray-AABB for unit cube [0,1]^3
    \\    let hit = ray_aabb(ro, rd, vec3f(0.0), vec3f(1.0));
    \\    if (hit.x > hit.y || hit.y < 0.0) {
    \\        // Background: dark gradient
    \\        let bg_t = frag_pos.y / camera.resolution.y;
    \\        return vec4f(0.02, 0.02, 0.05 + bg_t * 0.08, 1.0);
    \\    }
    \\
    \\    let t_start = max(hit.x, 0.0);
    \\    let t_end = hit.y;
    \\
    \\    // Step size: roughly 1 voxel
    \\    let max_dim = f32(max(grid_params.grid_size.x, max(grid_params.grid_size.y, grid_params.grid_size.z)));
    \\    let step_size = 1.0 / (max_dim * 1.5);
    \\
    \\    // Light direction
    \\    let light_dir = normalize(vec3f(0.6, 1.0, 0.8));
    \\    let ambient = 0.25;
    \\
    \\    // Front-to-back compositing
    \\    var accum_color = vec3f(0.0);
    \\    var accum_alpha = 0.0;
    \\    var t = t_start + step_size * 0.5;
    \\
    \\    for (var i = 0u; i < 512u; i = i + 1u) {
    \\        if (t > t_end || accum_alpha > 0.95) {
    \\            break;
    \\        }
    \\        let pos = ro + rd * t;
    \\        let sample_val = sample_grid(pos);
    \\        if (sample_val.a > 0.01) {
    \\            // Compute lighting
    \\            let normal = -estimate_normal(pos);
    \\            let ndl = max(dot(normal, light_dir), 0.0);
    \\            let lit = ambient + (1.0 - ambient) * ndl;
    \\            let cell_color = sample_val.rgb * lit;
    \\            let cell_alpha = sample_val.a * 0.8; // soften a bit
    \\
    \\            // Front-to-back composite
    \\            let w = cell_alpha * (1.0 - accum_alpha);
    \\            accum_color = accum_color + cell_color * w;
    \\            accum_alpha = accum_alpha + w;
    \\        }
    \\        t = t + step_size;
    \\    }
    \\
    \\    if (accum_alpha < 0.01) {
    \\        let bg_t = frag_pos.y / camera.resolution.y;
    \\        return vec4f(0.02, 0.02, 0.05 + bg_t * 0.08, 1.0);
    \\    }
    \\
    \\    // Blend with background
    \\    let bg_t = frag_pos.y / camera.resolution.y;
    \\    let bg = vec3f(0.02, 0.02, 0.05 + bg_t * 0.08);
    \\    let final_color = accum_color + bg * (1.0 - accum_alpha);
    \\    return vec4f(final_color, 1.0);
    \\}
;

// -- Matrix math helpers --
pub const Vec3 = [3]f32;
const Vec4 = [4]f32;
pub const Mat4 = [16]f32; // column-major

fn vec3Sub(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn vec3Cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn vec3Dot(a: Vec3, b: Vec3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn vec3Normalize(v: Vec3) Vec3 {
    const len = @sqrt(vec3Dot(v, v));
    if (len < 1e-12) return .{ 0, 0, 0 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

fn mat4Identity() Mat4 {
    var m = [_]f32{0} ** 16;
    m[0] = 1;
    m[5] = 1;
    m[10] = 1;
    m[15] = 1;
    return m;
}

pub fn mat4LookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
    const f = vec3Normalize(vec3Sub(target, eye));
    const s = vec3Normalize(vec3Cross(f, up));
    const u = vec3Cross(s, f);

    var m = mat4Identity();
    // Column-major
    m[0] = s[0];
    m[4] = s[1];
    m[8] = s[2];
    m[1] = u[0];
    m[5] = u[1];
    m[9] = u[2];
    m[2] = -f[0];
    m[6] = -f[1];
    m[10] = -f[2];
    m[12] = -vec3Dot(s, eye);
    m[13] = -vec3Dot(u, eye);
    m[14] = vec3Dot(f, eye);
    return m;
}

pub fn mat4Perspective(fov_rad: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const t = @tan(fov_rad / 2.0);
    var m = [_]f32{0} ** 16;
    m[0] = 1.0 / (aspect * t);
    m[5] = 1.0 / t;
    m[10] = -(far + near) / (far - near);
    m[11] = -1.0;
    m[14] = -(2.0 * far * near) / (far - near);
    return m;
}

pub fn mat4Mul(a: Mat4, b: Mat4) Mat4 {
    var r = [_]f32{0} ** 16;
    for (0..4) |col| {
        for (0..4) |row| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            r[col * 4 + row] = sum;
        }
    }
    return r;
}

pub fn mat4Inverse(m: Mat4) Mat4 {
    // Inline 4x4 matrix inverse (column-major)
    var inv: [16]f32 = undefined;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    var det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (@abs(det) < 1e-12) {
        return mat4Identity();
    }
    det = 1.0 / det;

    var result: Mat4 = undefined;
    for (0..16) |i| {
        result[i] = inv[i] * det;
    }
    return result;
}

pub fn buildCameraData(width: u32, height: u32) [24]f32 {
    const eye = Vec3{ 1.8, 1.5, 1.8 };
    const target = Vec3{ 0.5, 0.5, 0.5 };
    const up = Vec3{ 0, 1, 0 };

    const fov = std.math.degreesToRadians(60.0);
    const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));

    const view = mat4LookAt(eye, target, up);
    const proj = mat4Perspective(fov, aspect, 0.01, 100.0);
    const view_proj = mat4Mul(proj, view);
    const inv_vp = mat4Inverse(view_proj);

    var data: [24]f32 = undefined;
    // inv_view_proj: 16 floats
    for (0..16) |i| {
        data[i] = inv_vp[i];
    }
    // camera_pos: 3 floats + padding
    data[16] = eye[0];
    data[17] = eye[1];
    data[18] = eye[2];
    data[19] = 0; // padding
    // resolution: 2 floats + padding
    data[20] = @floatFromInt(width);
    data[21] = @floatFromInt(height);
    data[22] = 0; // padding
    data[23] = 0; // padding
    return data;
}

pub const Gpu = struct {
    instance: c.WGPUInstance,
    surface: c.WGPUSurface,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    surface_config: c.WGPUSurfaceConfiguration,
    pipeline: c.WGPURenderPipeline,
    bind_group_layout: c.WGPUBindGroupLayout,
    camera_buf: c.WGPUBuffer,
    grid_params_buf: c.WGPUBuffer,

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

        // Create uniform buffers
        var cam_buf_desc = std.mem.zeroes(c.WGPUBufferDescriptor);
        cam_buf_desc.label = c.WGPUStringView{ .data = "camera_buf", .length = 10 };
        cam_buf_desc.size = 96;
        cam_buf_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        const camera_buf = c.wgpuDeviceCreateBuffer(device_result, &cam_buf_desc) orelse
            return error.CameraBufferFailed;

        var gp_buf_desc = std.mem.zeroes(c.WGPUBufferDescriptor);
        gp_buf_desc.label = c.WGPUStringView{ .data = "grid_params_buf", .length = 15 };
        gp_buf_desc.size = 16;
        gp_buf_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        const grid_params_buf = c.wgpuDeviceCreateBuffer(device_result, &gp_buf_desc) orelse
            return error.GridParamsBufferFailed;

        // Bind group layout
        var bgl_entries: [3]c.WGPUBindGroupLayoutEntry = undefined;

        bgl_entries[0] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[0].binding = 0;
        bgl_entries[0].visibility = c.WGPUShaderStage_Fragment;
        bgl_entries[0].buffer.type = c.WGPUBufferBindingType_Uniform;
        bgl_entries[0].buffer.minBindingSize = 96;

        bgl_entries[1] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[1].binding = 1;
        bgl_entries[1].visibility = c.WGPUShaderStage_Fragment;
        bgl_entries[1].buffer.type = c.WGPUBufferBindingType_Uniform;
        bgl_entries[1].buffer.minBindingSize = 16;

        bgl_entries[2] = std.mem.zeroes(c.WGPUBindGroupLayoutEntry);
        bgl_entries[2].binding = 2;
        bgl_entries[2].visibility = c.WGPUShaderStage_Fragment;
        bgl_entries[2].buffer.type = c.WGPUBufferBindingType_ReadOnlyStorage;
        bgl_entries[2].buffer.minBindingSize = 0;

        var bgl_desc = std.mem.zeroes(c.WGPUBindGroupLayoutDescriptor);
        bgl_desc.entryCount = 3;
        bgl_desc.entries = &bgl_entries[0];
        const bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device_result, &bgl_desc) orelse
            return error.BindGroupLayoutFailed;

        // Pipeline layout
        var pl_desc = std.mem.zeroes(c.WGPUPipelineLayoutDescriptor);
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &bind_group_layout;
        const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device_result, &pl_desc) orelse
            return error.PipelineLayoutFailed;
        defer c.wgpuPipelineLayoutRelease(pipeline_layout);

        // Create shader module from WGSL source
        var wgsl_source = std.mem.zeroes(c.WGPUShaderSourceWGSL);
        wgsl_source.chain.sType = c.WGPUSType_ShaderSourceWGSL;
        wgsl_source.chain.next = null;
        wgsl_source.code = c.WGPUStringView{ .data = wgsl_shader.ptr, .length = wgsl_shader.len };

        var shader_desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
        shader_desc.nextInChain = @ptrCast(&wgsl_source.chain);
        shader_desc.label = c.WGPUStringView{ .data = "raymarch_shader", .length = 15 };

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
        pipeline_desc.label = c.WGPUStringView{ .data = "raymarch_pipeline", .length = 17 };
        pipeline_desc.layout = pipeline_layout;
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
            .bind_group_layout = bind_group_layout,
            .camera_buf = camera_buf,
            .grid_params_buf = grid_params_buf,
        };
    }

    pub fn deinit(self: *Gpu) void {
        c.wgpuBufferRelease(self.grid_params_buf);
        c.wgpuBufferRelease(self.camera_buf);
        c.wgpuBindGroupLayoutRelease(self.bind_group_layout);
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

    pub fn renderFrameWithGrid(self: *Gpu, grid: *const Grid, camera_data: []const f32) void {
        // Upload camera data
        c.wgpuQueueWriteBuffer(self.queue, self.camera_buf, 0, camera_data.ptr, camera_data.len * @sizeOf(f32));

        // Upload grid params as u32 array
        const grid_params = [4]u32{ grid.width, grid.height, grid.depth, grid.floats_per_cell };
        c.wgpuQueueWriteBuffer(self.queue, self.grid_params_buf, 0, &grid_params, @sizeOf([4]u32));

        // Create bind group
        var bg_entries: [3]c.WGPUBindGroupEntry = undefined;

        bg_entries[0] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[0].binding = 0;
        bg_entries[0].buffer = self.camera_buf;
        bg_entries[0].size = 96;

        bg_entries[1] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[1].binding = 1;
        bg_entries[1].buffer = self.grid_params_buf;
        bg_entries[1].size = 16;

        bg_entries[2] = std.mem.zeroes(c.WGPUBindGroupEntry);
        bg_entries[2].binding = 2;
        bg_entries[2].buffer = grid.readBuffer();
        bg_entries[2].size = grid.buffer_size;

        var bg_desc = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bg_desc.layout = self.bind_group_layout;
        bg_desc.entryCount = 3;
        bg_desc.entries = &bg_entries[0];
        const bind_group = c.wgpuDeviceCreateBindGroup(self.device, &bg_desc);
        defer c.wgpuBindGroupRelease(bind_group);

        // Get current surface texture
        var surface_texture: c.WGPUSurfaceTexture = undefined;
        c.wgpuSurfaceGetCurrentTexture(self.surface, &surface_texture);

        if (surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal and
            surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        {
            return;
        }

        // Create texture view
        const view = c.wgpuTextureCreateView(surface_texture.texture, null);
        defer c.wgpuTextureViewRelease(view);
        defer c.wgpuTextureRelease(surface_texture.texture);

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
            .clearValue = c.WGPUColor{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 },
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
        c.wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_group, 0, null);
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
