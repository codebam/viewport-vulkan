// SPDX-License-Identifier: MIT
//
// Host-visible buffers, for getting pixels from the CPU onto the GPU.
//
// Only shm clients need this. A DMA-BUF is already GPU memory and is imported
// in place; an shm buffer is a chunk of shared memory the client wrote with
// the CPU, so it has to be copied. That copy is the reason a compositor
// prefers dmabuf clients, and the reason this path exists at all: plenty of
// Wayland clients only ever use shm.

use anyhow::{anyhow, Context as _, Result};
use ash::vk;

use crate::Device;

/// A host-visible, host-coherent buffer used as a copy source.
///
/// Coherent rather than manually flushed: the writes here are one memcpy
/// immediately before a submit, so the flush would cover the whole range
/// anyway and coherent memory removes a step that is easy to forget.
pub struct Staging {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: vk::DeviceSize,
    /// Whether `Drop` must call `device_wait_idle()` before destroying the
    /// buffer. Defaults to `true` for safety; internal callers that have
    /// already waited on the submission fence can clear this so dropping a
    /// completed transfer buffer does not stall unrelated in-flight frames.
    wait_on_drop: bool,
}

impl Staging {
    pub fn new(device: &Device, size: vk::DeviceSize) -> Result<Self> {
        anyhow::ensure!(size > 0, "a zero-sized staging buffer");
        let handle = device.handle();

        let info = vk::BufferCreateInfo::default()
            .size(size)
            // Both directions: the same buffer type stages uploads and
            // receives downloads.
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { handle.create_buffer(&info, None) }.context("vkCreateBuffer")?;

        let result = (|| -> Result<(vk::DeviceMemory, *mut u8)> {
            let requirements = unsafe { handle.get_buffer_memory_requirements(buffer) };
            let memory_type = device
                .memory_type_with(requirements.memory_type_bits, |flags| {
                    flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                        && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                })
                .ok_or_else(|| anyhow!("no host-visible coherent memory type"))?;

            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory =
                unsafe { handle.allocate_memory(&allocate, None) }.context("vkAllocateMemory")?;

            if let Err(e) = unsafe { handle.bind_buffer_memory(buffer, memory, 0) } {
                unsafe { handle.free_memory(memory, None) };
                return Err(anyhow::Error::from(e).context("vkBindBufferMemory"));
            }

            // Mapped once and left mapped. Repeatedly mapping and unmapping a
            // buffer that is written every frame is pure overhead.
            let mapped = match unsafe {
                handle.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => ptr as *mut u8,
                Err(e) => {
                    unsafe { handle.free_memory(memory, None) };
                    return Err(anyhow::Error::from(e).context("vkMapMemory"));
                }
            };

            Ok((memory, mapped))
        })();

        let (memory, mapped) = match result {
            Ok(pair) => pair,
            Err(e) => {
                unsafe { handle.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            mapped,
            size,
            wait_on_drop: true,
        })
    }

    /// Record whether the GPU may still be accessing this buffer when dropped.
    pub(crate) fn set_wait_on_drop(&mut self, wait: bool) {
        self.wait_on_drop = wait;
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// The mapped bytes, for a buffer that was the destination of a copy.
    ///
    /// The caller is responsible for having waited on the copy; nothing here
    /// can tell whether the GPU has finished writing.
    pub fn read(&self) -> &[u8] {
        // SAFETY: mapped for the lifetime of this struct, host-coherent so the
        // GPU's writes are visible without an explicit invalidate, and the
        // length is the allocation's.
        unsafe { std::slice::from_raw_parts(self.mapped as *const u8, self.size as usize) }
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Copy `data` in at `offset`.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("staging write overflows"))?;
        anyhow::ensure!(
            end as vk::DeviceSize <= self.size,
            "staging write of {} bytes at {offset} exceeds the {} byte buffer",
            data.len(),
            self.size
        );

        // SAFETY: the range is bounds-checked above, the memory is mapped for
        // the lifetime of this struct, and it is host-coherent so no flush is
        // needed before the GPU reads it.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(offset), data.len());
        }
        Ok(())
    }

    /// Copy `height` rows of `row_bytes` each from `src` (starting at
    /// `src_offset` with `src_stride` bytes between rows) tightly packed into
    /// this staging buffer starting at `dst_offset`.
    pub fn write_rows_at(
        &mut self,
        dst_offset: usize,
        src: &[u8],
        src_offset: usize,
        src_stride: usize,
        row_bytes: usize,
        height: usize,
    ) -> Result<()> {
        let total = row_bytes
            .checked_mul(height)
            .ok_or_else(|| anyhow!("staging row write overflows"))?;
        let dst_end = dst_offset
            .checked_add(total)
            .ok_or_else(|| anyhow!("staging write overflows"))?;
        anyhow::ensure!(
            dst_end as vk::DeviceSize <= self.size,
            "staging write of {total} bytes at {dst_offset} exceeds the {} byte buffer",
            self.size
        );
        if total == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            src_stride >= row_bytes,
            "source stride {src_stride} is narrower than a {row_bytes}-byte row"
        );
        let src_end = src_stride
            .checked_mul(height - 1)
            .and_then(|v| v.checked_add(row_bytes))
            .and_then(|v| src_offset.checked_add(v))
            .ok_or_else(|| anyhow!("source range overflows"))?;
        anyhow::ensure!(
            src_end <= src.len(),
            "source slice of {} bytes is too short for {height} rows at offset {src_offset} stride {src_stride}",
            src.len()
        );

        // SAFETY: both source and destination ranges are bounds-checked above;
        // the memory is mapped for the lifetime of this struct and coherent.
        unsafe {
            let dst_base = self.mapped.add(dst_offset);
            if src_stride == row_bytes {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_offset), dst_base, total);
            } else {
                for row in 0..height {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset + row * src_stride),
                        dst_base.add(row * row_bytes),
                        row_bytes,
                    );
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for Staging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Staging").field("size", &self.size).finish()
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let handle = self.device.handle();
        unsafe {
            if self.wait_on_drop {
                // A copy may still be reading it.
                let _ = handle.device_wait_idle();
            }
            handle.unmap_memory(self.memory);
            handle.destroy_buffer(self.buffer, None);
            handle.free_memory(self.memory, None);
        }
    }
}
