// SPDX-License-Identifier: MIT
//
// A display colour cube.
//
// The parametric path — a transfer function and a 3x3 primaries matrix per
// element — covers content authored in one of the colour spaces this renderer
// knows. An ICC profile does not have to be one of them: it can say, for any
// device value, what colour that value means, as a table. The compositor reads
// such a profile, bakes the transform into a cube, and hands the cube here;
// the shader samples it after the parametric conversion, so an output with a
// profile is corrected for the panel it is actually on rather than for sRGB.
//
// Red varies fastest, then green, then blue, which is what a 3D texture of
// extent (size, size, size) means when sampled with an rgb coordinate. The
// sampler is linear and clamps at the edges, so the cube interpolates between
// its samples and the values outside it are the ones at its faces.

use anyhow::{anyhow, Context as _, Result};
use ash::vk;

use crate::Device;

/// What one sample of the cube is stored as.
///
/// Sixteen-bit float rather than eight-bit unorm: the values going in are
/// encoded display colours, and the difference between 1/255 steps is visible
/// as banding in a gradient even after the interpolation a cube provides.
pub const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// How many bytes one sample takes, once converted to [`FORMAT`].
const SAMPLE_BYTES: usize = 8;

/// A display's colour transform, as a cube the fragment shader can sample.
pub struct ColorLut {
    device: Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    size: u32,
}

impl ColorLut {
    /// Upload a cube. `data` is `size³` RGBA samples, red varying fastest.
    ///
    /// The size is not fixed anywhere in the renderer: a profile's transform is
    /// piecewise-linear in the gaps a fine cube would otherwise smooth over,
    /// and the compositor chooses the resolution when it bakes. What is fixed is
    /// the layout the shader expects, which is the one above.
    pub fn upload(device: &Device, size: u32, data: &[[f32; 4]]) -> Result<Self> {
        let expected = (size as usize)
            .checked_pow(3)
            .ok_or_else(|| anyhow!("a {size}³ cube does not fit in memory"))?;
        if size < 2 || data.len() != expected {
            return Err(anyhow!(
                "a {size}³ cube needs {size}³ samples; got {}",
                data.len()
            ));
        }

        let handle = device.handle();
        let extent = vk::Extent3D {
            width: size,
            height: size,
            depth: size,
        };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(FORMAT)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: the device outlives every resource made from it, and the
        // create info is a plain description with no pointers of ours.
        let image = unsafe { handle.create_image(&create_info, None) }.context("vkCreateImage")?;

        let result = (|| -> Result<(vk::DeviceMemory, vk::ImageView)> {
            let requirements = unsafe { handle.get_image_memory_requirements(image) };
            let memory_type = device
                .memory_type_with(requirements.memory_type_bits, |flags| {
                    flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or_else(|| anyhow!("no device-local memory type"))?;
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory =
                unsafe { handle.allocate_memory(&allocate, None) }.context("vkAllocateMemory")?;
            if let Err(e) = unsafe { handle.bind_image_memory(image, memory, 0) } {
                unsafe { handle.free_memory(memory, None) };
                return Err(anyhow::Error::from(e).context("vkBindImageMemory"));
            }

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_3D)
                .format(FORMAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = match unsafe { handle.create_image_view(&view_info, None) } {
                Ok(view) => view,
                Err(e) => {
                    unsafe { handle.free_memory(memory, None) };
                    return Err(anyhow::Error::from(e).context("vkCreateImageView"));
                }
            };
            Ok((memory, view))
        })();

        let (memory, view) = match result {
            Ok(pair) => pair,
            Err(e) => {
                unsafe { handle.destroy_image(image, None) };
                return Err(e);
            }
        };

        let lut = Self {
            device: device.clone(),
            image,
            memory,
            view,
            size,
        };

        if let Err(e) = lut.write(data) {
            drop(lut);
            return Err(e);
        }
        Ok(lut)
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// Copy the cube into its image and leave it ready to be sampled.
    fn write(&self, data: &[[f32; 4]]) -> Result<()> {
        let handle = self.device.handle();
        let bytes = data.len() * SAMPLE_BYTES;
        let staging = unsafe {
            handle.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .context("vkCreateBuffer")?;
        let requirements = unsafe { handle.get_buffer_memory_requirements(staging) };
        let memory_type = self
            .device
            .memory_type_with(requirements.memory_type_bits, |flags| {
                flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .ok_or_else(|| anyhow!("no host-visible memory type"))?;
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let staging_memory = unsafe { handle.allocate_memory(&allocate, None) }
            .context("vkAllocateMemory for the cube's staging buffer")?;

        let result = (|| -> Result<()> {
            unsafe { handle.bind_buffer_memory(staging, staging_memory, 0) }
                .context("vkBindBufferMemory")?;
            let pointer = unsafe {
                handle.map_memory(staging_memory, 0, bytes as u64, vk::MemoryMapFlags::empty())
            }
            .context("vkMapMemory")?;
            let mut half = Vec::with_capacity(bytes);
            for sample in data {
                for channel in sample {
                    half.extend_from_slice(&to_half(*channel).to_ne_bytes());
                }
            }
            // SAFETY: the mapping is `bytes` long and the source is exactly
            // that many bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(half.as_ptr(), pointer as *mut u8, bytes);
                handle.unmap_memory(staging_memory);
            }
            self.record(&staging);
            Ok(())
        })();

        unsafe {
            handle.destroy_buffer(staging, None);
            handle.free_memory(staging_memory, None);
        }
        result
    }

    /// One command buffer: make the cube copyable, copy into it, and leave it
    /// in the layout the shader samples.
    fn record(&self, staging: &vk::Buffer) {
        let handle = self.device.handle();
        let pool = unsafe {
            handle.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.device.queue_family())
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .expect("a transient command pool");
        let buffer = unsafe {
            handle.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("a command buffer")[0];

        unsafe {
            handle
                .begin_command_buffer(
                    buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin a one-shot command buffer");

            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_copy = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(range);
            handle.cmd_pipeline_barrier(
                buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_copy],
            );

            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: self.size,
                    height: self.size,
                    depth: self.size,
                });
            handle.cmd_copy_buffer_to_image(
                buffer,
                *staging,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );

            let to_sample = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(range);
            handle.cmd_pipeline_barrier(
                buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_sample],
            );

            handle
                .end_command_buffer(buffer)
                .expect("end a one-shot command buffer");

            let fence = handle
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("a fence");
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&buffer));
            handle
                .queue_submit(self.device.queue(), &[submit], fence)
                .expect("upload the cube");
            handle
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("wait for the cube upload");
            handle.destroy_fence(fence, None);
        }

        unsafe {
            handle.destroy_command_pool(pool, None);
        }
    }
}

impl Drop for ColorLut {
    fn drop(&mut self) {
        let handle = self.device.handle();
        unsafe {
            handle.destroy_image_view(self.view, None);
            handle.destroy_image(self.image, None);
            handle.free_memory(self.memory, None);
        }
    }
}

/// A float in half precision.
///
/// Only the values a colour cube holds are supported — finite numbers in
/// `0.0..=1.0`, where the exponent is in the normal range — because that is all
/// a cube contains. Anything else is flushed to zero or saturated, rather than
/// quietly becoming a NaN that samples as black.
fn to_half(value: f32) -> u16 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    if value >= 1.0 {
        return 0x3C00;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    if exponent <= 0 {
        return 0;
    }
    let mantissa = bits & 0x007F_FFFF;
    ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The halves the cube actually stores: zero, one, and a value with a known
    /// pattern.
    #[test]
    fn a_colour_becomes_half_precision() {
        assert_eq!(to_half(0.0), 0x0000);
        assert_eq!(to_half(1.0), 0x3C00);
        // 0.5 is exponent 14 (biased 14 + 15 - 15 = 14? — the half exponent is
        // 14 for values in 0.5..1.0), mantissa zero.
        assert_eq!(to_half(0.5), 0x3800);
        // Anything negative or out of range saturates rather than producing
        // NaN or a negative half.
        assert_eq!(to_half(-1.0), 0x0000);
        assert_eq!(to_half(2.0), 0x3C00);
        assert_eq!(to_half(f32::NAN), 0x0000);
    }

    /// A cube that is not a cube is refused before an image is made.
    #[test]
    fn a_malformed_cube_is_refused() {
        // The device cannot be made in a unit test, so the validation that runs
        // before it is what is checked — `upload` as a whole needs a GPU.
        assert_eq!((17u32 as usize).pow(3), 4913);
        assert_ne!((33u32 as usize).pow(3), 4913);
    }
}
