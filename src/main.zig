const std = @import("std");
const glfw = @import("glfw");
const Gpu = @import("gpu.zig").Gpu;
const Grid = @import("grid.zig").Grid;
const Simulation = @import("simulation.zig").Simulation;
const Camera = @import("camera.zig").Camera;
const Input = @import("input.zig").Input;

const objc = @cImport({
    @cInclude("objc/message.h");
    @cInclude("objc/runtime.h");
});

fn msgSend(obj: anytype, s: objc.SEL) ?*anyopaque {
    const f: *const fn (@TypeOf(obj), objc.SEL) callconv(.c) ?*anyopaque = @ptrCast(&objc.objc_msgSend);
    return f(obj, s);
}

fn msgSendWithObj(obj: anytype, s: objc.SEL, arg: ?*anyopaque) void {
    const f: *const fn (@TypeOf(obj), objc.SEL, ?*anyopaque) callconv(.c) void = @ptrCast(&objc.objc_msgSend);
    f(obj, s, arg);
}

fn msgSendBool(obj: anytype, s: objc.SEL, arg: bool) void {
    const f: *const fn (@TypeOf(obj), objc.SEL, bool) callconv(.c) void = @ptrCast(&objc.objc_msgSend);
    f(obj, s, arg);
}

fn selName(name: [*:0]const u8) objc.SEL {
    return objc.sel_registerName(name);
}

fn getClass(name: [*:0]const u8) objc.Class {
    return objc.objc_getClass(name).?;
}

fn createMetalLayer(ns_window: *anyopaque) ?*anyopaque {
    // Get contentView from NSWindow
    const content_view = msgSend(ns_window, selName("contentView")) orelse return null;

    // Set wantsLayer = YES
    msgSendBool(content_view, selName("setWantsLayer:"), true);

    // Create CAMetalLayer
    const metal_layer_class = getClass("CAMetalLayer");
    const metal_layer = msgSend(metal_layer_class, selName("layer")) orelse return null;

    // Set layer on contentView
    msgSendWithObj(content_view, selName("setLayer:"), metal_layer);

    return metal_layer;
}


pub fn main() !void {
    // Initialize GLFW
    if (!glfw.init(.{})) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return error.GLFWInitFailed;
    }
    defer glfw.terminate();

    // Create window -- no OpenGL context (we'll use wgpu)
    const window = glfw.Window.create(800, 600, "morphogen", null, null, .{
        .client_api = .no_api,
    }) orelse {
        std.debug.print("Failed to create GLFW window\n", .{});
        return error.WindowCreateFailed;
    };
    defer window.destroy();

    // Get NSWindow via zig-glfw native API
    const native = glfw.Native(.{ .cocoa = true });
    const ns_window = native.getCocoaWindow(window) orelse {
        std.debug.print("Failed to get Cocoa window\n", .{});
        return error.CocoaWindowFailed;
    };

    // Create Metal layer on the window's content view
    const metal_layer = createMetalLayer(ns_window) orelse {
        std.debug.print("Failed to create Metal layer\n", .{});
        return error.MetalLayerFailed;
    };

    const fb_size = window.getFramebufferSize();
    var gpu = try Gpu.init(metal_layer, fb_size.width, fb_size.height);
    defer gpu.deinit();

    std.debug.print("morphogen: GPU initialized, rendering...\n", .{});

    // Create grid: 64x64x64, 6 floats/cell [type, signal, r, g, b, a]
    var grid = try Grid.init(gpu.device, gpu.queue, gpu.instance, 64, 64, 64, 6);
    defer grid.deinit();

    // Seed single growth tip at center: type=2, signal=0, bright cyan, full opacity
    const seed_cell = [_]f32{ 2.0, 0.0, 0.05, 0.9, 1.0, 1.0 };
    grid.seedCenter(&seed_cell);

    // Signal source above grid center at (32, 60, 32)
    var sim = try Simulation.init(gpu.device, &grid, 32, 60, 32);
    defer sim.deinit();

    // Set up orbit camera and input handling
    var camera = Camera.init();
    var input = Input.init(&camera);
    input.setupCallbacks(window);

    var sim_step: u32 = 0;
    var frame_count: u64 = 0;

    std.debug.print("controls: Space=pause/resume  N/Right=step  R=reset  Right-click=move signal  Escape=quit\n", .{});
    std.debug.print("simulation: starting paused at step 0 (seed visible)\n", .{});

    while (!window.shouldClose()) {
        glfw.pollEvents();
        input.update(window);

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

        // Handle simulation reset: clear grid, re-seed, restart
        if (input.should_reset_sim) {
            grid.clear();
            grid.seedCenter(&seed_cell);
            sim_step = 0;
            input.should_reset_sim = false;
            input.paused = true;
            std.debug.print("simulation: reset to step 0\n", .{});
        }

        // Run one simulation step every N frames (or single-step when paused)
        const steps_per_second: u64 = 1;
        const frames_per_step: u64 = 60 / steps_per_second;
        if (input.should_step or (!input.paused and frame_count % frames_per_step == 0)) {
            sim.step(&grid);
            sim_step += 1;
            input.should_step = false;
            std.debug.print("simulation: step {d}\n", .{sim_step});
        }

        // Handle resize
        const size = window.getFramebufferSize();
        if (size.width != gpu.surface_config.width or size.height != gpu.surface_config.height) {
            if (size.width > 0 and size.height > 0) {
                gpu.resize(size.width, size.height);
            }
        }

        const camera_data = camera.buildUniformData(size.width, size.height);
        gpu.renderFrameWithGrid(&grid, &camera_data);
        frame_count += 1;
    }

    std.debug.print("morphogen: {d} simulation steps, goodbye\n", .{sim_step});
}
