const std = @import("std");
const gpu_mod = @import("gpu.zig");
const c = gpu_mod.c;

pub const Grid = struct {
    width: u32,
    height: u32,
    depth: u32,
    floats_per_cell: u32,

    buffer_a: c.WGPUBuffer,
    buffer_b: c.WGPUBuffer,
    staging: c.WGPUBuffer,
    buffer_size: u64,

    current_is_a: bool,

    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    instance: c.WGPUInstance,

    pub fn init(
        device: c.WGPUDevice,
        queue: c.WGPUQueue,
        instance: c.WGPUInstance,
        width: u32,
        height: u32,
        depth: u32,
        floats_per_cell: u32,
    ) !Grid {
        const total_cells = @as(u64, width) * height * depth;
        const buffer_size = total_cells * floats_per_cell * @sizeOf(f32);

        const buffer_a = createStorageBuffer(device, "grid_a", buffer_size) orelse
            return error.GridBufferFailed;
        const buffer_b = createStorageBuffer(device, "grid_b", buffer_size) orelse
            return error.GridBufferFailed;
        const staging = createStagingBuffer(device, "grid_staging", buffer_size) orelse
            return error.GridBufferFailed;

        std.debug.print("grid: allocated {d}x{d}x{d} ({d} floats/cell, {d} bytes per buffer)\n", .{
            width, height, depth, floats_per_cell, buffer_size,
        });

        return Grid{
            .width = width,
            .height = height,
            .depth = depth,
            .floats_per_cell = floats_per_cell,
            .buffer_a = buffer_a,
            .buffer_b = buffer_b,
            .staging = staging,
            .buffer_size = buffer_size,
            .current_is_a = true,
            .device = device,
            .queue = queue,
            .instance = instance,
        };
    }

    pub fn deinit(self: *Grid) void {
        c.wgpuBufferRelease(self.staging);
        c.wgpuBufferRelease(self.buffer_b);
        c.wgpuBufferRelease(self.buffer_a);
    }

    pub fn readBuffer(self: *const Grid) c.WGPUBuffer {
        return if (self.current_is_a) self.buffer_a else self.buffer_b;
    }

    pub fn writeBuffer(self: *const Grid) c.WGPUBuffer {
        return if (self.current_is_a) self.buffer_b else self.buffer_a;
    }

    pub fn swap(self: *Grid) void {
        self.current_is_a = !self.current_is_a;
    }

    pub fn cellIndex(self: *const Grid, x: u32, y: u32, z: u32) u64 {
        return (@as(u64, z) * self.height * self.width + @as(u64, y) * self.width + x) * self.floats_per_cell;
    }

    /// Zero out both buffers and reset to initial state (no seed).
    pub fn clear(self: *Grid) void {
        const chunk_size = 4096;
        const zeros = std.mem.zeroes([chunk_size]u8);
        var offset: u64 = 0;
        while (offset < self.buffer_size) {
            const remaining = self.buffer_size - offset;
            const write_size = @min(remaining, chunk_size);
            c.wgpuQueueWriteBuffer(self.queue, self.buffer_a, offset, &zeros, write_size);
            c.wgpuQueueWriteBuffer(self.queue, self.buffer_b, offset, &zeros, write_size);
            offset += write_size;
        }
        self.current_is_a = true;
    }

    pub fn seedCenter(self: *Grid, cell_data: []const f32) void {
        const cx = self.width / 2;
        const cy = self.height / 2;
        const cz = self.depth / 2;
        const offset = self.cellIndex(cx, cy, cz) * @sizeOf(f32);

        c.wgpuQueueWriteBuffer(
            self.queue,
            self.readBuffer(),
            offset,
            cell_data.ptr,
            cell_data.len * @sizeOf(f32),
        );

        std.debug.print("grid: seeded center cell at ({d}, {d}, {d})\n", .{ cx, cy, cz });
    }

    pub fn readBack(self: *Grid) ![]const f32 {
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        c.wgpuCommandEncoderCopyBufferToBuffer(encoder, self.readBuffer(), 0, self.staging, 0, self.buffer_size);
        const cmd = c.wgpuCommandEncoderFinish(encoder, null);
        c.wgpuCommandEncoderRelease(encoder);
        c.wgpuQueueSubmit(self.queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);

        var map_done: bool = false;
        var map_status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Unknown;

        const map_callback_info = c.WGPUBufferMapCallbackInfo{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = &mapCallback,
            .userdata1 = @ptrCast(&map_status),
            .userdata2 = @ptrCast(&map_done),
        };

        _ = c.wgpuBufferMapAsync(self.staging, c.WGPUMapMode_Read, 0, self.buffer_size, map_callback_info);

        while (!map_done) {
            c.wgpuInstanceProcessEvents(self.instance);
        }

        if (map_status != c.WGPUMapAsyncStatus_Success) {
            return error.BufferMapFailed;
        }

        const mapped_ptr: ?*const anyopaque = c.wgpuBufferGetConstMappedRange(self.staging, 0, self.buffer_size);
        if (mapped_ptr == null) return error.MappedRangeNull;

        const total_floats = @as(usize, @intCast(self.buffer_size / @sizeOf(f32)));
        const data: [*]const f32 = @ptrCast(@alignCast(mapped_ptr));
        return data[0..total_floats];
    }

    pub fn unmapStaging(self: *Grid) void {
        c.wgpuBufferUnmap(self.staging);
    }

    pub fn totalCells(self: *const Grid) u64 {
        return @as(u64, self.width) * self.height * self.depth;
    }
};

fn createStorageBuffer(device: c.WGPUDevice, label: [*:0]const u8, size: u64) ?c.WGPUBuffer {
    var desc = std.mem.zeroes(c.WGPUBufferDescriptor);
    desc.label = c.WGPUStringView{ .data = label, .length = std.mem.len(label) };
    desc.size = size;
    desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
    return c.wgpuDeviceCreateBuffer(device, &desc);
}

fn createStagingBuffer(device: c.WGPUDevice, label: [*:0]const u8, size: u64) ?c.WGPUBuffer {
    var desc = std.mem.zeroes(c.WGPUBufferDescriptor);
    desc.label = c.WGPUStringView{ .data = label, .length = std.mem.len(label) };
    desc.size = size;
    desc.usage = c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_MapRead;
    return c.wgpuDeviceCreateBuffer(device, &desc);
}

fn mapCallback(status: c.WGPUMapAsyncStatus, _: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    const status_ptr: *c.WGPUMapAsyncStatus = @ptrCast(@alignCast(userdata1));
    const done_ptr: *bool = @ptrCast(@alignCast(userdata2));
    status_ptr.* = status;
    done_ptr.* = true;
}
