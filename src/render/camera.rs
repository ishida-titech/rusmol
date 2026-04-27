use glam::{Mat4, Quat, Vec2, Vec3};

#[derive(Debug)]
pub struct Camera {
    /// Point the camera orbits around
    pub center: Vec3,
    /// Distance from center to eye
    pub distance: f32,
    /// Arcball rotation (quaternion)
    pub rotation: Quat,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    /// Viewport size for aspect ratio
    pub viewport: Vec2,
}

impl Camera {
    pub fn new(center: Vec3, distance: f32, viewport: Vec2) -> Self {
        Self {
            center,
            distance,
            rotation: Quat::IDENTITY,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            viewport,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.center + self.rotation * Vec3::new(0.0, 0.0, self.distance);
        let up  = self.rotation * Vec3::Y;
        Mat4::look_at_rh(eye, self.center, up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        let aspect = self.viewport.x / self.viewport.y;
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    pub fn eye_position(&self) -> Vec3 {
        self.center + self.rotation * Vec3::new(0.0, 0.0, self.distance)
    }

    /// Arcball drag: convert 2D delta (pixels) to rotation quaternion
    pub fn arcball_rotate(&mut self, delta: Vec2) {
        let sensitivity = 0.005;
        let yaw   = Quat::from_rotation_y(-delta.x * sensitivity);
        let pitch = Quat::from_rotation_x(-delta.y * sensitivity);
        // Apply: yaw in world space, pitch in camera space
        self.rotation = (yaw * self.rotation * pitch).normalize();
    }

    /// Pan: move center perpendicular to view direction
    pub fn pan(&mut self, delta: Vec2) {
        let sensitivity = self.distance * 0.001;
        let right = self.rotation * Vec3::X;
        let up    = self.rotation * Vec3::Y;
        self.center -= right * delta.x * sensitivity;
        self.center += up    * delta.y * sensitivity;
    }

    /// Zoom: change distance multiplicatively
    pub fn zoom(&mut self, scroll: f32) {
        self.distance *= 1.0 - scroll * 0.1;
        self.distance = self.distance.clamp(0.5, 500.0);
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.viewport = Vec2::new(width, height);
    }
}
