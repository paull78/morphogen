const std = @import("std");
const gpu_mod = @import("gpu.zig");

pub const Camera = struct {
    theta: f32, // horizontal angle (radians)
    phi: f32, // vertical angle (radians), clamped to avoid gimbal lock
    radius: f32, // distance from target
    target: [3]f32, // orbit center
    fov: f32, // field of view in radians

    pub fn init() Camera {
        return .{
            .theta = std.math.pi / 4.0, // 45 degrees
            .phi = std.math.pi / 4.0, // 45 degrees above horizon
            .radius = 2.0,
            .target = .{ 0.5, 0.5, 0.5 }, // grid center (unit cube)
            .fov = std.math.degreesToRadians(60.0),
        };
    }

    pub fn position(self: *const Camera) [3]f32 {
        const x = self.target[0] + self.radius * @cos(self.phi) * @cos(self.theta);
        const y = self.target[1] + self.radius * @sin(self.phi);
        const z = self.target[2] + self.radius * @cos(self.phi) * @sin(self.theta);
        return .{ x, y, z };
    }

    pub fn orbit(self: *Camera, dx: f32, dy: f32) void {
        self.theta -= dx * 0.01;
        self.phi += dy * 0.01;
        // Clamp phi to avoid flipping
        self.phi = std.math.clamp(self.phi, -std.math.pi / 2.0 + 0.01, std.math.pi / 2.0 - 0.01);
    }

    pub fn zoom(self: *Camera, delta: f32) void {
        self.radius *= 1.0 - delta * 0.1;
        self.radius = std.math.clamp(self.radius, 0.5, 10.0);
    }

    /// Build the 24-float camera uniform data for the shader.
    pub fn buildUniformData(self: *const Camera, width: u32, height: u32) [24]f32 {
        const eye = self.position();
        const up = [3]f32{ 0, 1, 0 };
        const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));

        const view = gpu_mod.mat4LookAt(eye, self.target, up);
        const proj = gpu_mod.mat4Perspective(self.fov, aspect, 0.01, 100.0);
        const view_proj = gpu_mod.mat4Mul(proj, view);
        const inv_vp = gpu_mod.mat4Inverse(view_proj);

        var data: [24]f32 = undefined;
        for (0..16) |i| {
            data[i] = inv_vp[i];
        }
        data[16] = eye[0];
        data[17] = eye[1];
        data[18] = eye[2];
        data[19] = 0;
        data[20] = @floatFromInt(width);
        data[21] = @floatFromInt(height);
        data[22] = 0;
        data[23] = 0;
        return data;
    }
};
