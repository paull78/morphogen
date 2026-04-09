const std = @import("std");
const glfw = @import("glfw");
const Gpu = @import("gpu.zig").Gpu;

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

    var frame_count: u64 = 0;

    while (!window.shouldClose()) {
        glfw.pollEvents();

        const size = window.getFramebufferSize();
        if (size.width != gpu.surface_config.width or size.height != gpu.surface_config.height) {
            if (size.width > 0 and size.height > 0) {
                gpu.resize(size.width, size.height);
            }
        }

        gpu.renderFrame(0.0, 0.0, 0.0);
        frame_count += 1;
    }

    std.debug.print("morphogen: goodbye\n", .{});
}
