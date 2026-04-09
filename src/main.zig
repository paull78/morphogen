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
    const window = glfw.Window.create(800, 600, "morphogen", null, null, .{
        .client_api = .no_api,
    }) orelse {
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
