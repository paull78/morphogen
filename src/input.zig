const std = @import("std");
const glfw = @import("glfw");
const Camera = @import("camera.zig").Camera;

pub const Input = struct {
    camera: *Camera,
    dragging: bool,
    last_x: f64,
    last_y: f64,
    paused: bool,
    should_step: bool,
    should_reset_sim: bool,
    space_was_pressed: bool,
    r_was_pressed: bool,
    n_was_pressed: bool,
    should_place_signal: bool,
    signal_click_x: f64,
    signal_click_y: f64,

    pub fn init(camera: *Camera) Input {
        return .{
            .camera = camera,
            .dragging = false,
            .last_x = 0,
            .last_y = 0,
            .paused = true, // start paused so the seed is visible
            .should_step = false,
            .should_reset_sim = false,
            .space_was_pressed = false,
            .r_was_pressed = false,
            .n_was_pressed = false,
            .should_place_signal = false,
            .signal_click_x = 0,
            .signal_click_y = 0,
        };
    }

    /// Set up GLFW callbacks that require user pointer (scroll).
    pub fn setupCallbacks(self: *Input, window: glfw.Window) void {
        window.setUserPointer(self);
        window.setScrollCallback(scrollCallback);
    }

    fn scrollCallback(window: glfw.Window, _: f64, yoffset: f64) void {
        const self = window.getUserPointer(Input) orelse return;
        self.camera.zoom(@floatCast(yoffset));
    }

    /// Poll mouse and keyboard state. Call once per frame.
    pub fn update(self: *Input, window: glfw.Window) void {
        // Mouse drag (orbit)
        const cursor_pos = window.getCursorPos();
        const left_button = window.getMouseButton(.left);

        if (left_button == .press) {
            if (self.dragging) {
                const dx = cursor_pos.xpos - self.last_x;
                const dy = cursor_pos.ypos - self.last_y;
                self.camera.orbit(@floatCast(dx), @floatCast(dy));
            }
            self.dragging = true;
        } else {
            self.dragging = false;
        }
        self.last_x = cursor_pos.xpos;
        self.last_y = cursor_pos.ypos;

        // Space: toggle pause (rising edge)
        const space_pressed = window.getKey(.space) == .press;
        if (space_pressed and !self.space_was_pressed) {
            self.paused = !self.paused;
        }
        self.space_was_pressed = space_pressed;

        // R: reset simulation (rising edge)
        const r_pressed = window.getKey(.r) == .press;
        if (r_pressed and !self.r_was_pressed) {
            self.should_reset_sim = true;
        }
        self.r_was_pressed = r_pressed;

        // N or Right arrow: single step when paused (rising edge)
        const n_pressed = window.getKey(.n) == .press;
        const right_pressed = window.getKey(.right) == .press;
        if ((n_pressed and !self.n_was_pressed) and self.paused) {
            self.should_step = true;
        }
        if (right_pressed and self.paused) {
            self.should_step = true;
        }
        self.n_was_pressed = n_pressed;

        // Escape: close window
        if (window.getKey(.escape) == .press) {
            window.setShouldClose(true);
        }

        // Right-click: place signal source
        const right_button = window.getMouseButton(.right);
        if (right_button == .press) {
            self.should_place_signal = true;
            self.signal_click_x = cursor_pos.xpos;
            self.signal_click_y = cursor_pos.ypos;
        }
    }
};
