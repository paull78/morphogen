const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "morphogen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // GLFW (zig-gamedev/zglfw — bundles GLFW C source, builds it from scratch)
    const zglfw_dep = b.dependency("zglfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zglfw", zglfw_dep.module("root"));
    exe.linkLibrary(zglfw_dep.artifact("glfw"));

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

    // Unit tests for the pure-logic modules (camera math, ray casting).
    // These need the wgpu headers for @cImport but not the static library,
    // so `zig build test` runs without the vendored binary or a GPU.
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/camera.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addIncludePath(b.path("vendor/wgpu/include"));
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
