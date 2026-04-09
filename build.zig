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
    const glfw_dep = b.dependency("glfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("glfw", glfw_dep.module("glfw"));

    // wgpu-native (vendored)
    exe.addIncludePath(b.path("vendor/wgpu/include"));
    exe.addObjectFile(b.path("vendor/wgpu/lib/libwgpu_native.a"));
    exe.linkFramework("Metal");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Foundation");
    exe.linkFramework("CoreGraphics");
    exe.linkFramework("IOKit");
    exe.linkFramework("IOSurface");
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
