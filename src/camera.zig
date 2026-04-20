const std = @import("std");
const gpu_mod = @import("gpu.zig");

fn mat4MulVec4(m: gpu_mod.Mat4, v: [4]f32) [4]f32 {
    var r: [4]f32 = undefined;
    for (0..4) |row| {
        r[row] = m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row] * v[3];
    }
    return r;
}

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

    pub fn clickToGridPos(self: *const Camera, click_x: f32, click_y: f32, win_w: u32, win_h: u32, grid_w: u32, grid_h: u32, grid_d: u32) ?[3]u32 {
        const w = @as(f32, @floatFromInt(win_w));
        const h = @as(f32, @floatFromInt(win_h));
        const aspect = w / h;

        const ndc_x = (click_x / w) * 2.0 - 1.0;
        const ndc_y = -((click_y / h) * 2.0 - 1.0);

        const view = gpu_mod.mat4LookAt(self.position(), self.target, .{ 0, 1, 0 });
        const proj = gpu_mod.mat4Perspective(self.fov, aspect, 0.01, 100.0);
        const view_proj = gpu_mod.mat4Mul(proj, view);
        const inv_vp = gpu_mod.mat4Inverse(view_proj);

        const near_clip = mat4MulVec4(inv_vp, .{ ndc_x, ndc_y, 0.0, 1.0 });
        const far_clip = mat4MulVec4(inv_vp, .{ ndc_x, ndc_y, 1.0, 1.0 });

        const ro = [3]f32{
            near_clip[0] / near_clip[3],
            near_clip[1] / near_clip[3],
            near_clip[2] / near_clip[3],
        };
        const far_pt = [3]f32{
            far_clip[0] / far_clip[3],
            far_clip[1] / far_clip[3],
            far_clip[2] / far_clip[3],
        };

        const rd = gpu_mod.vec3Normalize(.{
            far_pt[0] - ro[0],
            far_pt[1] - ro[1],
            far_pt[2] - ro[2],
        });

        // Ray-AABB intersection with unit cube [0,1]^3
        const inv_rd = [3]f32{ 1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2] };
        const t1 = [3]f32{ -ro[0] * inv_rd[0], -ro[1] * inv_rd[1], -ro[2] * inv_rd[2] };
        const t2 = [3]f32{ (1.0 - ro[0]) * inv_rd[0], (1.0 - ro[1]) * inv_rd[1], (1.0 - ro[2]) * inv_rd[2] };

        const tmin = [3]f32{ @min(t1[0], t2[0]), @min(t1[1], t2[1]), @min(t1[2], t2[2]) };
        const tmax = [3]f32{ @max(t1[0], t2[0]), @max(t1[1], t2[1]), @max(t1[2], t2[2]) };

        const t_near = @max(tmin[0], @max(tmin[1], tmin[2]));
        const t_far = @min(tmax[0], @min(tmax[1], tmax[2]));

        if (t_near > t_far or t_far < 0.0) return null;

        const t_hit = @max(t_near, 0.0) + 0.001;
        const hit = [3]f32{
            ro[0] + rd[0] * t_hit,
            ro[1] + rd[1] * t_hit,
            ro[2] + rd[2] * t_hit,
        };

        const gw = @as(f32, @floatFromInt(grid_w));
        const gh = @as(f32, @floatFromInt(grid_h));
        const gd = @as(f32, @floatFromInt(grid_d));

        const vx = @as(u32, @intFromFloat(std.math.clamp(hit[0] * gw, 0.0, gw - 1.0)));
        const vy = @as(u32, @intFromFloat(std.math.clamp(hit[1] * gh, 0.0, gh - 1.0)));
        const vz = @as(u32, @intFromFloat(std.math.clamp(hit[2] * gd, 0.0, gd - 1.0)));

        return .{ vx, vy, vz };
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
