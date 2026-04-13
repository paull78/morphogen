# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a working Zig + GLFW + wgpu-native pipeline on macOS that can clear the screen to a color and render a fullscreen triangle with a WGSL shader.

**Architecture:** Zig executable links GLFW (via zig package) and wgpu-native (vendored prebuilt static lib). GLFW creates a window and provides a Metal surface. wgpu-native handles all GPU work. Build system compiles everything in one `zig build run` command.

**Tech Stack:** Zig 0.14.x, wgpu-native (prebuilt aarch64-macos), GLFW (zig-glfw package), WGSL shaders (embedded as strings initially)

---

## File Structure

```
morphogen/
├── build.zig                  # build system: links wgpu + GLFW, defines run step
├── build.zig.zon              # package manifest with zig-glfw dependency
├── vendor/
│   └── wgpu/
│       ├── include/
│       │   ├── webgpu.h       # WebGPU C header
│       │   └── wgpu.h         # wgpu-native extensions header
│       └── lib/
│           └── libwgpu_native.a  # prebuilt static library (aarch64-macos)
├── src/
│   ├── main.zig               # entry point, main loop
│   └── gpu.zig                # wgpu initialization, device, surface
```

---

## Task 1: Install Zig and verify toolchain

**Files:**
- Create: `build.zig`
- Create: `build.zig.zon`
- Create: `src/main.zig`

- [x] **Step 1: Install Zig 0.14.x**

```bash
brew install zig@0.14
```

Brew will print the keg path (e.g., `/opt/homebrew/Cellar/zig@0.14/0.14.1`). Add it to your PATH:

```bash
export PATH="/opt/homebrew/opt/zig@0.14/bin:$PATH"
```

Add this line to your `~/.zshrc` to make it permanent. Verify:

```bash
zig version
```

Expected: `0.14.1` (or similar 0.14.x)

- [x] **Step 2: Create build.zig.zon**

```zig
.{
    .name = .@"morphogen",
    .version = "0.0.1",
    .minimum_zig_version = "0.14.0",
    .dependencies = .{},
    .paths = .{"."},
}
```

- [x] **Step 3: Create build.zig**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "morphogen",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run morphogen");
    run_step.dependOn(&run_cmd.step);
}
```

- [x] **Step 4: Create src/main.zig**

```zig
const std = @import("std");

pub fn main() void {
    std.debug.print("morphogen: hello from zig\n", .{});
}
```

- [x] **Step 5: Build and run**

```bash
zig build run
```

Expected output: `morphogen: hello from zig`

- [x] **Step 6: Commit**

```bash
git add build.zig build.zig.zon src/main.zig
git commit -m "feat: minimal Zig project, builds and runs"
```

---

## Task 2: Add GLFW and open a window

**Files:**
- Modify: `build.zig.zon` (add zig-glfw dependency)
- Modify: `build.zig` (link GLFW)
- Modify: `src/main.zig` (create window)

- [x] **Step 1: Fetch zig-glfw dependency**

```bash
zig fetch --save git+https://github.com/falsepattern/zig-glfw
```

This updates `build.zig.zon` with the dependency hash automatically.

- [x] **Step 2: Update build.zig to link GLFW**

Replace `build.zig` contents with:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "morphogen",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // GLFW
    const glfw_dep = b.dependency("zig_glfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("glfw", glfw_dep.module("zig-glfw"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run morphogen");
    run_step.dependOn(&run_cmd.step);
}
```

- [x] **Step 3: Update main.zig to open a window**

Replace `src/main.zig` contents with:

```zig
const std = @import("std");
const glfw = @import("glfw");

pub fn main() !void {
    // Initialize GLFW
    if (!glfw.init(.{})) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return error.GLFWInitFailed;
    }
    defer glfw.terminate();

    // Create window — no OpenGL context (we'll use wgpu)
    const window = glfw.Window.create(.{
        .width = 800,
        .height = 600,
        .title = "morphogen",
        .client_api = .no_api,
    }, null) orelse {
        std.debug.print("Failed to create GLFW window\n", .{});
        return error.WindowCreateFailed;
    };
    defer window.destroy();

    std.debug.print("morphogen: window open, close it to exit\n", .{});

    // Main loop — just poll events until window is closed
    while (!window.shouldClose()) {
        glfw.pollEvents();
    }

    std.debug.print("morphogen: goodbye\n", .{});
}
```

- [x] **Step 4: Build and run**

```bash
zig build run
```

Expected: A blank window titled "morphogen" appears. Closing it prints "morphogen: goodbye" and exits cleanly.

**Experiment:** Try resizing the window. Try pressing Escape (it won't close yet — that's fine).

- [x] **Step 5: Commit**

```bash
git add build.zig build.zig.zon src/main.zig
git commit -m "feat: GLFW window opens and closes cleanly"
```

---

## Task 3: Vendor wgpu-native and clear screen to a color

**Files:**
- Create: `vendor/wgpu/include/webgpu.h` (from wgpu-native release)
- Create: `vendor/wgpu/include/wgpu.h` (from wgpu-native release)
- Create: `vendor/wgpu/lib/libwgpu_native.a` (from wgpu-native release)
- Modify: `build.zig` (link wgpu-native)
- Create: `src/gpu.zig` (wgpu initialization)
- Modify: `src/main.zig` (integrate gpu.zig, clear screen)

- [x] **Step 1: Download wgpu-native prebuilt binary**

```bash
mkdir -p vendor/wgpu/include vendor/wgpu/lib
cd vendor/wgpu

# Download the latest release for macOS aarch64
curl -L -o wgpu-release.zip \
  https://github.com/gfx-rs/wgpu-native/releases/download/v24.0.3.1/wgpu-macos-aarch64-release.zip

unzip wgpu-release.zip
cp release/libwgpu_native.a lib/
cp release/webgpu.h include/
cp release/wgpu.h include/
rm -rf release wgpu-release.zip

cd ../..
```

Verify:

```bash
ls vendor/wgpu/include/
# Expected: webgpu.h  wgpu.h
ls vendor/wgpu/lib/
# Expected: libwgpu_native.a
```

Note: The exact release tag may differ. Check https://github.com/gfx-rs/wgpu-native/releases for the latest macOS aarch64 release if the URL above 404s. Adjust the download URL accordingly.

- [x] **Step 2: Update .gitignore to track vendor headers but not the binary**

Add to `.gitignore`:

```
# wgpu prebuilt binary (too large for git, re-download with setup script)
vendor/wgpu/lib/*.a
```

- [x] **Step 3: Update build.zig to link wgpu-native**

Replace `build.zig` contents with:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "morphogen",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // GLFW
    const glfw_dep = b.dependency("zig_glfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("glfw", glfw_dep.module("zig-glfw"));

    // wgpu-native (vendored)
    exe.addIncludePath(b.path("vendor/wgpu/include"));
    exe.addObjectFile(b.path("vendor/wgpu/lib/libwgpu_native.a"));
    exe.linkFramework("Metal");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Foundation");
    exe.linkFramework("CoreGraphics");
    exe.linkLibC();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run morphogen");
    run_step.dependOn(&run_cmd.step);
}
```

- [x] **Step 4: Create src/gpu.zig**

```zig
const std = @import("std");
const c = @cImport({
    @cInclude("webgpu.h");
    @cInclude("wgpu.h");
});

pub const Gpu = struct {
    instance: c.WGPUInstance,
    surface: c.WGPUSurface,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    surface_config: c.WGPUSurfaceConfiguration,

    pub fn init(metal_layer: *anyopaque, width: u32, height: u32) !Gpu {
        // Create instance
        const instance_desc = c.WGPUInstanceDescriptor{
            .nextInChain = null,
        };
        const instance = c.wgpuCreateInstance(&instance_desc) orelse
            return error.WGPUInstanceFailed;

        // Create surface from Metal layer
        const metal_surface_desc = c.WGPUSurfaceSourceMetalLayer{
            .chain = .{
                .sType = c.WGPUSType_SurfaceSourceMetalLayer,
                .next = null,
            },
            .layer = metal_layer,
        };
        const surface_desc = c.WGPUSurfaceDescriptor{
            .nextInChain = @ptrCast(&metal_surface_desc),
            .label = c.WGPUStringView{ .data = "surface", .length = 7 },
        };
        const surface = c.wgpuInstanceCreateSurface(instance, &surface_desc) orelse
            return error.WGPUSurfaceFailed;

        // Request adapter
        const adapter = try requestAdapter(instance, surface);

        // Request device
        const device = try requestDevice(adapter);

        // Get queue
        const queue = c.wgpuDeviceGetQueue(device) orelse
            return error.WGPUQueueFailed;

        // Configure surface
        var surface_config = c.WGPUSurfaceConfiguration{
            .nextInChain = null,
            .device = device,
            .format = c.WGPUTextureFormat_BGRA8UnormSrgb,
            .usage = c.WGPUTextureUsage_RenderAttachment,
            .width = width,
            .height = height,
            .presentMode = c.WGPUPresentMode_Fifo,
            .alphaMode = c.WGPUCompositeAlphaMode_Auto,
            .viewFormatCount = 0,
            .viewFormats = null,
        };

        // Query preferred format
        var caps: c.WGPUSurfaceCapabilities = std.mem.zeroes(c.WGPUSurfaceCapabilities);
        c.wgpuSurfaceGetCapabilities(surface, adapter, &caps);
        if (caps.formatCount > 0 and caps.formats != null) {
            surface_config.format = caps.formats[0];
        }

        c.wgpuSurfaceConfigure(surface, &surface_config);

        return Gpu{
            .instance = instance,
            .surface = surface,
            .adapter = adapter,
            .device = device,
            .queue = queue,
            .surface_config = surface_config,
        };
    }

    pub fn deinit(self: *Gpu) void {
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuSurfaceRelease(self.surface);
        c.wgpuInstanceRelease(self.instance);
    }

    pub fn renderFrame(self: *Gpu, r: f64, g: f64, b: f64) void {
        var surface_texture: c.WGPUSurfaceTexture = undefined;
        c.wgpuSurfaceGetCurrentTexture(self.surface, &surface_texture);

        if (surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal and
            surface_texture.status != c.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
        {
            std.debug.print("Failed to get surface texture\n", .{});
            return;
        }

        const view_desc = c.WGPUTextureViewDescriptor{
            .nextInChain = null,
            .label = c.WGPUStringView{ .data = "view", .length = 4 },
            .format = self.surface_config.format,
            .dimension = c.WGPUTextureViewDimension_2D,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1,
            .aspect = c.WGPUTextureAspect_All,
            .usage = c.WGPUTextureUsage_RenderAttachment,
        };
        const view = c.wgpuTextureCreateView(surface_texture.texture, &view_desc);
        defer c.wgpuTextureViewRelease(view);

        const encoder_desc = c.WGPUCommandEncoderDescriptor{
            .nextInChain = null,
            .label = c.WGPUStringView{ .data = "encoder", .length = 7 },
        };
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, &encoder_desc);

        const color_attachment = c.WGPURenderPassColorAttachment{
            .view = view,
            .resolveTarget = null,
            .loadOp = c.WGPULoadOp_Clear,
            .storeOp = c.WGPUStoreOp_Store,
            .clearValue = c.WGPUColor{ .r = r, .g = g, .b = b, .a = 1.0 },
            .depthSlice = c.WGPU_DEPTH_SLICE_UNDEFINED,
        };

        const render_pass_desc = c.WGPURenderPassDescriptor{
            .nextInChain = null,
            .label = c.WGPUStringView{ .data = "pass", .length = 4 },
            .colorAttachmentCount = 1,
            .colorAttachments = &color_attachment,
            .depthStencilAttachment = null,
            .occlusionQuerySet = null,
            .timestampWrites = null,
        };

        const render_pass = c.wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
        c.wgpuRenderPassEncoderEnd(render_pass);
        c.wgpuRenderPassEncoderRelease(render_pass);

        const cmd_desc = c.WGPUCommandBufferDescriptor{
            .nextInChain = null,
            .label = c.WGPUStringView{ .data = "cmd", .length = 3 },
        };
        const cmd_buffer = c.wgpuCommandEncoderFinish(encoder, &cmd_desc);
        c.wgpuCommandEncoderRelease(encoder);

        c.wgpuQueueSubmit(self.queue, 1, &cmd_buffer);
        c.wgpuCommandBufferRelease(cmd_buffer);

        c.wgpuSurfacePresent(self.surface);
        c.wgpuTextureRelease(surface_texture.texture);
    }

    pub fn resize(self: *Gpu, width: u32, height: u32) void {
        self.surface_config.width = width;
        self.surface_config.height = height;
        c.wgpuSurfaceConfigure(self.surface, &self.surface_config);
    }
};

fn requestAdapter(instance: c.WGPUInstance, surface: c.WGPUSurface) !c.WGPUAdapter {
    const State = struct {
        adapter: c.WGPUAdapter = null,
        done: bool = false,
    };
    var state = State{};

    const callback = struct {
        fn cb(
            status: c.WGPURequestAdapterStatus,
            adapter: c.WGPUAdapter,
            _: c.WGPUStringView,
            userdata1: ?*anyopaque,
            _: ?*anyopaque,
        ) callconv(.c) void {
            const s: *State = @ptrCast(@alignCast(userdata1));
            if (status == c.WGPURequestAdapterStatus_Success) {
                s.adapter = adapter;
            }
            s.done = true;
        }
    }.cb;

    const options = c.WGPURequestAdapterOptions{
        .nextInChain = null,
        .compatibleSurface = surface,
        .powerPreference = c.WGPUPowerPreference_HighPerformance,
        .backendType = c.WGPUBackendType_Metal,
        .forceFallbackAdapter = 0,
    };

    const callback_info = c.WGPURequestAdapterCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_WaitAnyOnly,
        .callback = callback,
        .userdata1 = @ptrCast(&state),
        .userdata2 = null,
    };

    const future = c.wgpuInstanceRequestAdapter(instance, &options, callback_info);

    const wait_info = c.WGPUFutureWaitInfo{
        .future = future,
        .completed = 0,
    };
    _ = c.wgpuInstanceWaitAny(instance, 1, @constCast(&wait_info), std.math.maxInt(u64));

    if (state.adapter) |adapter| return adapter;
    return error.WGPUAdapterFailed;
}

fn requestDevice(adapter: c.WGPUAdapter) !c.WGPUDevice {
    const State = struct {
        device: c.WGPUDevice = null,
        done: bool = false,
    };
    var state = State{};

    const callback = struct {
        fn cb(
            status: c.WGPURequestDeviceStatus,
            device: c.WGPUDevice,
            _: c.WGPUStringView,
            userdata1: ?*anyopaque,
            _: ?*anyopaque,
        ) callconv(.c) void {
            const s: *State = @ptrCast(@alignCast(userdata1));
            if (status == c.WGPURequestDeviceStatus_Success) {
                s.device = device;
            }
            s.done = true;
        }
    }.cb;

    const device_desc = c.WGPUDeviceDescriptor{
        .nextInChain = null,
        .label = c.WGPUStringView{ .data = "device", .length = 6 },
        .requiredFeatureCount = 0,
        .requiredFeatures = null,
        .requiredLimits = null,
        .defaultQueue = c.WGPUQueueDescriptor{
            .nextInChain = null,
            .label = c.WGPUStringView{ .data = "queue", .length = 5 },
        },
        .deviceLostCallbackInfo = std.mem.zeroes(c.WGPUDeviceLostCallbackInfo),
        .uncapturedErrorCallbackInfo = std.mem.zeroes(c.WGPUUncapturedErrorCallbackInfo),
    };

    const callback_info = c.WGPURequestDeviceCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_WaitAnyOnly,
        .callback = callback,
        .userdata1 = @ptrCast(&state),
        .userdata2 = null,
    };

    const future = c.wgpuAdapterRequestDevice(adapter, &device_desc, callback_info);

    const wait_info = c.WGPUFutureWaitInfo{
        .future = future,
        .completed = 0,
    };
    _ = c.wgpuInstanceWaitAny(c.wgpuAdapterGetInstance(adapter), 1, @constCast(&wait_info), std.math.maxInt(u64));

    if (state.device) |device| return device;
    return error.WGPUDeviceFailed;
}
```

- [x] **Step 5: Update main.zig to use gpu.zig and clear screen**

Replace `src/main.zig` contents with:

```zig
const std = @import("std");
const glfw = @import("glfw");
const Gpu = @import("gpu.zig").Gpu;

pub fn main() !void {
    // Initialize GLFW
    if (!glfw.init(.{})) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return error.GLFWInitFailed;
    }
    defer glfw.terminate();

    // Create window — no OpenGL context (we use wgpu)
    const window = glfw.Window.create(.{
        .width = 800,
        .height = 600,
        .title = "morphogen",
        .client_api = .no_api,
    }, null) orelse {
        std.debug.print("Failed to create GLFW window\n", .{});
        return error.WindowCreateFailed;
    };
    defer window.destroy();

    // Get Metal layer from GLFW
    const metal_layer = glfw.native.getCocoaMetalLayer(window) orelse {
        std.debug.print("Failed to get Metal layer\n", .{});
        return error.MetalLayerFailed;
    };

    // Initialize wgpu
    const fb_size = window.getFramebufferSize();
    var gpu = try Gpu.init(metal_layer, fb_size[0], fb_size[1]);
    defer gpu.deinit();

    std.debug.print("morphogen: GPU initialized, rendering...\n", .{});

    var frame_count: u64 = 0;

    // Main loop
    while (!window.shouldClose()) {
        glfw.pollEvents();

        // Handle resize
        const size = window.getFramebufferSize();
        if (size[0] != gpu.surface_config.width or size[1] != gpu.surface_config.height) {
            if (size[0] > 0 and size[1] > 0) {
                gpu.resize(size[0], size[1]);
            }
        }

        // Animate clear color: slowly cycle hue
        const t: f64 = @as(f64, @floatFromInt(frame_count)) * 0.005;
        const r = (@sin(t) + 1.0) * 0.15;
        const g = (@sin(t + 2.094) + 1.0) * 0.15;
        const b = (@sin(t + 4.189) + 1.0) * 0.25;

        gpu.renderFrame(r, g, b);
        frame_count += 1;
    }

    std.debug.print("morphogen: goodbye\n", .{});
}
```

- [x] **Step 6: Build and run**

```bash
zig build run
```

Expected: A window with a slowly cycling dark color (deep blues/purples — bioluminescent preview!). Resizing works. Closing exits cleanly.

**If you get linker errors** about missing frameworks, check `build.zig` links Metal, QuartzCore, Foundation, CoreGraphics. If you see wgpu-native symbol errors, verify `libwgpu_native.a` is the correct architecture (`file vendor/wgpu/lib/libwgpu_native.a` should show `arm64`).

- [x] **Step 7: Commit**

```bash
git add build.zig build.zig.zon src/main.zig src/gpu.zig vendor/wgpu/include/ .gitignore
git commit -m "feat: wgpu-native initializes and clears screen to animated color"
```

---

## Task 4: Render a fullscreen triangle with a WGSL shader

**Files:**
- Modify: `src/gpu.zig` (add render pipeline, shader)
- Modify: `src/main.zig` (call new render method)

- [x] **Step 1: Add shader source and pipeline to gpu.zig**

Add these constants and the pipeline setup to `src/gpu.zig`. At the top of the file, after the imports:

```zig
const triangle_shader_src =
    \\@vertex
    \\fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    \\    // Fullscreen triangle: 3 vertices covering the entire screen
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
    \\    // Gradient based on normalized screen position (works at any resolution)
    \\    // Note: hardcoded resolution for now — will use a uniform buffer later
    \\    let uv = pos.xy / vec2f(800.0, 600.0);
    \\    let col = vec3f(uv.x * 0.1, uv.y * 0.15, 0.2 + uv.y * 0.15);
    \\    return vec4f(col, 1.0);
    \\}
;
```

Add a `pipeline` field to the `Gpu` struct:

```zig
pipeline: c.WGPURenderPipeline,
```

- [x] **Step 2: Create the render pipeline in Gpu.init**

Add after the `wgpuSurfaceConfigure` call, before the `return Gpu{...}`:

```zig
// Create shader module
const shader_desc = c.WGPUShaderModuleWGSLDescriptor{
    .chain = .{
        .sType = c.WGPUSType_ShaderSourceWGSL,
        .next = null,
    },
    .code = c.WGPUStringView{
        .data = triangle_shader_src.ptr,
        .length = triangle_shader_src.len,
    },
};
const shader_module_desc = c.WGPUShaderModuleDescriptor{
    .nextInChain = @ptrCast(&shader_desc),
    .label = c.WGPUStringView{ .data = "shader", .length = 6 },
};
const shader_module = c.wgpuDeviceCreateShaderModule(device, &shader_module_desc) orelse
    return error.WGPUShaderFailed;
defer c.wgpuShaderModuleRelease(shader_module);

// Create render pipeline
const color_target = c.WGPUColorTargetState{
    .nextInChain = null,
    .format = surface_config.format,
    .blend = &c.WGPUBlendState{
        .color = c.WGPUBlendComponent{
            .srcFactor = c.WGPUBlendFactor_One,
            .dstFactor = c.WGPUBlendFactor_Zero,
            .operation = c.WGPUBlendOperation_Add,
        },
        .alpha = c.WGPUBlendComponent{
            .srcFactor = c.WGPUBlendFactor_One,
            .dstFactor = c.WGPUBlendFactor_Zero,
            .operation = c.WGPUBlendOperation_Add,
        },
    },
    .writeMask = c.WGPUColorWriteMask_All,
};

const pipeline_desc = c.WGPURenderPipelineDescriptor{
    .nextInChain = null,
    .label = c.WGPUStringView{ .data = "pipeline", .length = 8 },
    .layout = null, // auto layout
    .vertex = c.WGPUVertexState{
        .nextInChain = null,
        .module = shader_module,
        .entryPoint = c.WGPUStringView{ .data = "vs_main", .length = 7 },
        .constantCount = 0,
        .constants = null,
        .bufferCount = 0,
        .buffers = null,
    },
    .primitive = c.WGPUPrimitiveState{
        .nextInChain = null,
        .topology = c.WGPUPrimitiveTopology_TriangleList,
        .stripIndexFormat = c.WGPUIndexFormat_Undefined,
        .frontFace = c.WGPUFrontFace_CCW,
        .cullMode = c.WGPUCullMode_None,
        .unclippedDepth = 0,
    },
    .depthStencil = null,
    .multisample = c.WGPUMultisampleState{
        .nextInChain = null,
        .count = 1,
        .mask = 0xFFFFFFFF,
        .alphaToCoverageEnabled = 0,
    },
    .fragment = &c.WGPUFragmentState{
        .nextInChain = null,
        .module = shader_module,
        .entryPoint = c.WGPUStringView{ .data = "fs_main", .length = 7 },
        .constantCount = 0,
        .constants = null,
        .targetCount = 1,
        .targets = &color_target,
    },
};

const pipeline = c.wgpuDeviceCreateRenderPipeline(device, &pipeline_desc) orelse
    return error.WGPUPipelineFailed;
```

Add `pipeline` to the return struct:

```zig
.pipeline = pipeline,
```

- [x] **Step 3: Update renderFrame to draw the triangle**

In the `renderFrame` method, after `wgpuCommandEncoderBeginRenderPass`, add the draw call before `wgpuRenderPassEncoderEnd`:

```zig
const render_pass = c.wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
c.wgpuRenderPassEncoderSetPipeline(render_pass, self.pipeline);
c.wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
c.wgpuRenderPassEncoderEnd(render_pass);
```

- [x] **Step 4: Add pipeline release to deinit**

```zig
pub fn deinit(self: *Gpu) void {
    c.wgpuRenderPipelineRelease(self.pipeline);
    c.wgpuQueueRelease(self.queue);
    // ... rest unchanged
}
```

- [x] **Step 5: Simplify main.zig renderFrame call**

In `main.zig`, change the render call to just use a dark background (the shader now provides the visuals):

```zig
gpu.renderFrame(0.0, 0.0, 0.0);
```

- [x] **Step 6: Build and run**

```bash
zig build run
```

Expected: A fullscreen dark gradient — dark blue/teal at the bottom, nearly black at the top. This is the fragment shader running on every pixel. The gradient proves the shader pipeline works end to end.

**Experiment:** Edit the shader source in `gpu.zig` — change the color math, try different patterns. Rebuild and see the result immediately.

- [x] **Step 7: Commit**

```bash
git add src/gpu.zig src/main.zig
git commit -m "feat: fullscreen triangle with WGSL fragment shader gradient"
```

---

## Notes for Next Plan

After completing these 4 tasks, the foundation is in place:
- Zig builds and runs
- GLFW window with input
- wgpu-native initialized with Metal backend
- Render pipeline with WGSL shaders
- Clear → draw → present loop at 60fps

The next plan will cover **Tasks 5-7 (Compute)**: creating a compute pipeline, writing to a storage buffer from a compute shader, and setting up the 3D grid with double-buffering.
