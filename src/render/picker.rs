use wgpu::util::DeviceExt;

use crate::render::ball_stick::{SphereInstance, Vertex};

/// Pixel radius for nearest-neighbor ghost-sphere search (Ribbon / Surface picking).
pub const GHOST_PICK_RADIUS: u32 = 20;

/// Offscreen color-ID picking pass.
/// Renders sphere instances to a R32Uint texture, then reads back the pixel
/// under the cursor to identify which sphere was clicked.
pub struct Picker {
    pipeline: wgpu::RenderPipeline,
    pick_texture: wgpu::Texture,
    pick_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// CPU-side readback buffer (256 bytes) — used for exact single-pixel readback.
    readback_buf: wgpu::Buffer,
    /// Larger readback buffer for region nearest-search (Ribbon/Surface ghost spheres).
    region_readback_buf: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl Picker {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PickShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pick.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PickPipelineLayout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PickPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc(), SphereInstance::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None, // integer formats don't support blending
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (pick_texture, pick_view) = create_pick_texture(device, width, height);
        let (depth_texture, depth_view) = create_pick_depth(device, width, height);
        let readback_buf = create_readback_buf(device);
        let region_readback_buf = create_region_readback_buf(device);

        Self {
            pipeline,
            pick_texture,
            pick_view,
            depth_texture,
            depth_view,
            readback_buf,
            region_readback_buf,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        let (t, v) = create_pick_texture(device, width, height);
        self.pick_texture = t;
        self.pick_view = v;
        let (dt, dv) = create_pick_depth(device, width, height);
        self.depth_texture = dt;
        self.depth_view = dv;
    }

    /// Render the picking pass and return the 1-based sphere instance index at pixel (px, py).
    /// Returns `None` if the background was clicked or there are no spheres.
    pub fn pick_at(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform_bind_group: &wgpu::BindGroup,
        sphere_vb: &wgpu::Buffer,
        sphere_ib: &wgpu::Buffer,
        sphere_index_count: u32,
        sphere_instances: &wgpu::Buffer,
        sphere_instance_count: u32,
        px: u32,
        py: u32,
    ) -> Option<u32> {
        if px >= self.width || py >= self.height {
            return None;
        }

        let mut encoder = device.create_command_encoder(&Default::default());

        // ── Picking render pass ──────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, sphere_vb.slice(..));
            pass.set_vertex_buffer(1, sphere_instances.slice(..));
            pass.set_index_buffer(sphere_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..sphere_index_count, 0, 0..sphere_instance_count);
        }

        // ── Copy single pixel to readback buffer ─────────────────────────────
        // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pick_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        queue.submit([encoder.finish()]);

        // ── Read back ────────────────────────────────────────────────────────
        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let id = {
            let view = slice.get_mapped_range();
            let bytes: [u8; 4] = view[..4].try_into().unwrap();
            u32::from_ne_bytes(bytes)
        };
        self.readback_buf.unmap();

        if id == 0 { None } else { Some(id - 1) }
    }

    /// Render the picking pass for ghost spheres and return the nearest 1-based sphere
    /// instance index within `GHOST_PICK_RADIUS` pixels of (px, py).
    /// Returns `None` if no ghost sphere is found within the search radius.
    pub fn pick_nearest(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform_bind_group: &wgpu::BindGroup,
        sphere_vb: &wgpu::Buffer,
        sphere_ib: &wgpu::Buffer,
        sphere_index_count: u32,
        sphere_instances: &wgpu::Buffer,
        sphere_instance_count: u32,
        px: u32,
        py: u32,
    ) -> Option<u32> {
        if px >= self.width || py >= self.height {
            return None;
        }

        // Compute clamped region bounds.
        let r = GHOST_PICK_RADIUS;
        let x0 = px.saturating_sub(r);
        let y0 = py.saturating_sub(r);
        let x1 = (px + r + 1).min(self.width);
        let y1 = (py + r + 1).min(self.height);
        let region_w = x1 - x0;
        let region_h = y1 - y0;

        let mut encoder = device.create_command_encoder(&Default::default());

        // ── Picking render pass ──────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GhostPickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, sphere_vb.slice(..));
            pass.set_vertex_buffer(1, sphere_instances.slice(..));
            pass.set_index_buffer(sphere_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..sphere_index_count, 0, 0..sphere_instance_count);
        }

        // ── Copy region to region_readback_buf ───────────────────────────────
        // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        // region_w * 4 bytes ≤ (2*GHOST_PICK_RADIUS+1) * 4 = 164 ≤ 256, so 256 suffices.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pick_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.region_readback_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: region_w, height: region_h, depth_or_array_layers: 1 },
        );

        queue.submit([encoder.finish()]);

        // ── Scan region for nearest non-zero pixel ───────────────────────────
        let slice = self.region_readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let result = {
            let view = slice.get_mapped_range();
            let row_stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize; // 256 bytes = 64 u32s
            let mut best_id: u32 = 0;
            let mut best_dist_sq: i32 = i32::MAX;

            for dy in 0..region_h as usize {
                for dx in 0..region_w as usize {
                    let offset = dy * row_stride + dx * 4;
                    let bytes: [u8; 4] = view[offset..offset + 4].try_into().unwrap();
                    let id = u32::from_ne_bytes(bytes);
                    if id == 0 {
                        continue;
                    }
                    let sx = (x0 as i32 + dx as i32) - px as i32;
                    let sy = (y0 as i32 + dy as i32) - py as i32;
                    let dist_sq = sx * sx + sy * sy;
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_id = id;
                    }
                }
            }

            // Circular radius check: reject corners of the bounding rectangle.
            if best_id != 0 && best_dist_sq <= (r * r) as i32 {
                Some(best_id - 1) // 0-based instance index
            } else {
                None
            }
        };
        self.region_readback_buf.unmap();
        result
    }
}

fn create_region_readback_buf(device: &wgpu::Device) -> wgpu::Buffer {
    // Holds (2*GHOST_PICK_RADIUS+1) rows, each padded to COPY_BYTES_PER_ROW_ALIGNMENT.
    let rows = (2 * GHOST_PICK_RADIUS + 1) as u64;
    let size = rows * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PickRegionReadback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn create_pick_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("PickTexture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_pick_depth(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("PickDepth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_readback_buf(device: &wgpu::Device) -> wgpu::Buffer {
    // Must be at least COPY_BYTES_PER_ROW_ALIGNMENT bytes for copy_texture_to_buffer.
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PickReadback"),
        size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
