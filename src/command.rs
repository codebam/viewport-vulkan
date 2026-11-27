// SPDX-License-Identifier: MIT
//
// Command pools and submission.
//
// A compositor records one command buffer per output per frame and throws it
// away, so the pool is created with RESET_COMMAND_BUFFER and buffers are
// re-recorded rather than freed and reallocated. Everything here is
// single-queue: this renderer composites, it does not run async compute, and a
// second queue would only add ownership transfers to pay for.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Context as _, Result};
use ash::vk;

use crate::Device;

/// Convert a `Duration` to Vulkan's nanosecond timeout without truncating
/// durations longer than `u64::MAX` nanoseconds.
fn timeout_nanos(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX)
}

fn create_exportable_semaphore(device: &Device) -> Result<vk::Semaphore> {
    // Created up front as exportable; a semaphore cannot be made
    // exportable after the fact.
    let mut export = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
    unsafe { device.handle().create_semaphore(&semaphore_info, None) }.context("vkCreateSemaphore")
}

/// A command pool and one reusable primary command buffer.
pub struct Commands {
    device: Device,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Signalled by exportable submissions and exported as a sync_file, so the
    /// caller can hand the fd to KMS instead of waiting here.
    ///
    /// Exporting a SYNC_FD transfers the payload out, which leaves the
    /// semaphore unsignalled and ready for the next submit. If a caller waits
    /// instead of exporting — or if export fails — `signal_pending` stays true
    /// so the semaphore is drained or recreated before it is signalled again.
    signal: vk::Semaphore,
    /// Whether `signal` currently holds an unconsumed signal from a previous
    /// submission. Re-signalling a binary semaphore that is already signalled
    /// violates VUID-vkQueueSubmit-pSignalSemaphores-00067.
    signal_pending: AtomicBool,
    /// Semaphores the next submission must wait on, imported from fences
    /// handed to us by clients or by another renderer.
    waits: Vec<vk::Semaphore>,
    /// Semaphores belonging to the submission in flight. They cannot be
    /// destroyed until it completes, so they are held here and freed once the
    /// fence signals.
    retired: Vec<vk::Semaphore>,
    /// Whether the fence has been submitted and not yet waited on. Resetting
    /// an unsignalled fence, or waiting on one that was never submitted, both
    /// hang rather than fail.
    pending: bool,
}

impl Commands {
    pub fn new(device: &Device) -> Result<Self> {
        let handle = device.handle();

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { handle.create_command_pool(&pool_info, None) }
            .context("vkCreateCommandPool")?;

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffer = match unsafe { handle.allocate_command_buffers(&allocate) } {
            Ok(buffers) => buffers[0],
            Err(e) => {
                unsafe { handle.destroy_command_pool(pool, None) };
                return Err(anyhow::Error::from(e).context("vkAllocateCommandBuffers"));
            }
        };

        // Unsignalled: nothing has been submitted yet.
        let fence_info = vk::FenceCreateInfo::default();
        let fence = match unsafe { handle.create_fence(&fence_info, None) } {
            Ok(fence) => fence,
            Err(e) => {
                unsafe { handle.destroy_command_pool(pool, None) };
                return Err(anyhow::Error::from(e).context("vkCreateFence"));
            }
        };

        let signal = match create_exportable_semaphore(device) {
            Ok(semaphore) => semaphore,
            Err(e) => {
                unsafe {
                    handle.destroy_fence(fence, None);
                    handle.destroy_command_pool(pool, None);
                }
                return Err(e);
            }
        };

        Ok(Self {
            device: device.clone(),
            pool,
            buffer,
            fence,
            signal,
            signal_pending: AtomicBool::new(false),
            waits: Vec::new(),
            retired: Vec::new(),
            pending: false,
        })
    }

    pub fn buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }

    /// Start recording, waiting for the previous submission first.
    ///
    /// Re-recording a command buffer the GPU is still executing is undefined
    /// behaviour, and one of the few Vulkan mistakes that usually appears to
    /// work.
    pub fn begin(&mut self) -> Result<vk::CommandBuffer> {
        self.wait(Duration::from_secs(5))?;

        let handle = self.device.handle();
        unsafe {
            handle
                .reset_command_buffer(self.buffer, vk::CommandBufferResetFlags::empty())
                .context("vkResetCommandBuffer")?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            handle
                .begin_command_buffer(self.buffer, &begin)
                .context("vkBeginCommandBuffer")?;
        }
        Ok(self.buffer)
    }

    /// Finish recording and submit, signalling the exportable semaphore.
    pub fn submit(&mut self) -> Result<()> {
        self.submit_internal(true)
    }

    /// Finish recording and submit without signalling the exportable semaphore.
    ///
    /// Used by internal work whose completion is waited on via the fence
    /// (`upload`, `download`, `acquire_now`) or discarded (`VulkanFrame::drop`)
    /// and never exported as a `sync_file`.
    pub(crate) fn submit_unsignaled(&mut self) -> Result<()> {
        self.submit_internal(false)
    }

    /// Ensure `self.signal` is unsignalled before a submission signals it.
    ///
    /// By the time this runs, `begin()` has already waited for the previous
    /// submission's fence, so the GPU is done with `self.signal`.
    fn reset_signal_if_needed(&mut self) -> Result<()> {
        if !self.signal_pending.load(Ordering::Acquire) {
            return Ok(());
        }
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.signal)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        match unsafe { self.device.external_semaphore_fd().get_semaphore_fd(&info) } {
            Ok(raw) => {
                if raw > 0 {
                    // Close the drained sync_file immediately.
                    let _ = unsafe { OwnedFd::from_raw_fd(raw) };
                }
            }
            Err(_) => {
                // If export is unsupported on this driver, replace the
                // semaphore with a fresh unsignalled one.
                let fresh = create_exportable_semaphore(&self.device)?;
                unsafe { self.device.handle().destroy_semaphore(self.signal, None) };
                self.signal = fresh;
            }
        }
        self.signal_pending.store(false, Ordering::Release);
        Ok(())
    }

    fn submit_internal(&mut self, signal: bool) -> Result<()> {
        if signal {
            self.reset_signal_if_needed()?;
        }
        let handle = self.device.handle();
        unsafe {
            handle
                .end_command_buffer(self.buffer)
                .context("vkEndCommandBuffer")?;
            handle
                .reset_fences(&[self.fence])
                .context("vkResetFences")?;

            let buffers = [self.buffer];
            let signals = [self.signal];
            // ALL_COMMANDS rather than a narrower stage: the imported fence
            // guards the buffer's contents, and narrowing this to, say, the
            // fragment stage would be a claim about where it is read that this
            // layer cannot make.
            let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; self.waits.len()];
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(&buffers)
                .wait_semaphores(&self.waits)
                .wait_dst_stage_mask(&stages);
            if signal {
                submit = submit.signal_semaphores(&signals);
            }
            handle
                .queue_submit(self.device.queue(), &[submit], self.fence)
                .context("vkQueueSubmit")?;
        }
        // Held until the submission completes; destroying a semaphore the
        // queue is still waiting on is a use-after-free inside the driver.
        self.retired.append(&mut self.waits);
        self.pending = true;
        if signal {
            self.signal_pending.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Make the next submission wait on `fd` before it runs.
    ///
    /// This is the other half of explicit sync. A client that hands over an
    /// acquire fence is saying "do not read this buffer yet"; importing that
    /// fd into a semaphore and letting the queue wait on it means the GPU
    /// blocks, not us.
    ///
    /// The import is TEMPORARY, which is required for SYNC_FD and means the
    /// payload is consumed by the wait — so a semaphore is used once and then
    /// retired.
    pub fn wait_on(&mut self, fd: OwnedFd) -> Result<()> {
        let handle = self.device.handle();
        let semaphore =
            unsafe { handle.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .context("vkCreateSemaphore")?;

        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .fd(fd.as_raw_fd());

        if let Err(e) = unsafe {
            self.device
                .external_semaphore_fd()
                .import_semaphore_fd(&info)
        } {
            unsafe { handle.destroy_semaphore(semaphore, None) };
            return Err(anyhow::Error::from(e).context("vkImportSemaphoreFdKHR"));
        }

        // Vulkan took ownership of the fd on a successful import.
        std::mem::forget(fd);

        self.waits.push(semaphore);
        Ok(())
    }

    /// How many fences the next submission will wait on.
    pub fn pending_waits(&self) -> usize {
        self.waits.len()
    }

    /// Export the last submission's completion as a `sync_file` fd.
    ///
    /// Must be called after [`Commands::submit`] and before the next one: the
    /// payload belongs to that submission, and exporting transfers it out.
    ///
    /// Returns `None` where the driver cannot export, which is not an error —
    /// the caller falls back to waiting.
    pub fn export_fence(&self) -> Result<Option<OwnedFd>> {
        if !self.pending || !self.signal_pending.load(Ordering::Acquire) {
            return Ok(None);
        }
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.signal)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

        let raw = unsafe { self.device.external_semaphore_fd().get_semaphore_fd(&info) }
            .context("vkGetSemaphoreFdKHR")?;
        // Exporting a SYNC_FD transfers the payload out, resetting the binary
        // semaphore to unsignalled even when raw == -1 (already complete).
        self.signal_pending.store(false, Ordering::Release);

        // -1 is legal here: it means the work was already complete, so there
        // is nothing to wait on.
        if raw < 0 {
            return Ok(None);
        }
        if raw == 0 {
            return Err(anyhow!("vkGetSemaphoreFdKHR returned fd 0"));
        }
        Ok(Some(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// Block until the last submission completes.
    ///
    /// A real frame loop should not call this — the whole point of explicit
    /// sync is that the compositor hands a fence onward instead of waiting.
    /// It is here for teardown and for tests that need to read results back.
    pub fn wait(&mut self, timeout: Duration) -> Result<()> {
        if !self.pending {
            return Ok(());
        }
        let handle = self.device.handle();
        unsafe {
            handle
                .wait_for_fences(&[self.fence], true, timeout_nanos(timeout))
                .context("vkWaitForFences")?;
        }
        self.pending = false;
        self.free_retired();
        Ok(())
    }

    fn free_retired(&mut self) {
        let handle = self.device.handle();
        for semaphore in self.retired.drain(..) {
            unsafe { handle.destroy_semaphore(semaphore, None) };
        }
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // The pool cannot be destroyed while its buffer is still executing.
        let _ = self.wait(Duration::from_secs(5));

        let device = self.device.clone();
        let handle = device.handle();
        self.free_retired();
        unsafe {
            for semaphore in self.waits.drain(..) {
                handle.destroy_semaphore(semaphore, None);
            }
            handle.destroy_semaphore(self.signal, None);
            handle.destroy_fence(self.fence, None);
            handle.destroy_command_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{require_gpu, TestGpu};

    #[test]
    fn large_wait_timeouts_saturate_rather_than_truncate() {
        assert_eq!(timeout_nanos(Duration::from_secs(5)), 5_000_000_000);
        assert_eq!(timeout_nanos(Duration::MAX), u64::MAX);
    }

    #[test]
    fn consecutive_submits_without_export_drain_the_signal_semaphore() {
        let Some(TestGpu { device, .. }) = require_gpu() else {
            return;
        };
        let mut commands = Commands::new(&device).expect("commands");

        // Submit without exporting: signal_pending becomes true.
        commands.begin().expect("begin 1");
        commands.submit().expect("submit 1");
        assert!(commands.signal_pending.load(Ordering::Acquire));

        // Waiting clears pending without consuming the signal semaphore.
        commands.wait(Duration::from_secs(5)).expect("wait 1");
        assert!(commands.signal_pending.load(Ordering::Acquire));

        // The next submit must drain/reset before re-signalling, and exporting
        // afterwards clears signal_pending cleanly.
        commands.begin().expect("begin 2");
        commands.submit().expect("submit 2");
        let _ = commands.export_fence().expect("export 2");
        assert!(!commands.signal_pending.load(Ordering::Acquire));
        commands.wait(Duration::from_secs(5)).expect("wait 2");

        // An unsignalled submit leaves signal_pending false and export_fence
        // returns None.
        commands.begin().expect("begin 3");
        commands.submit_unsignaled().expect("submit 3");
        assert!(!commands.signal_pending.load(Ordering::Acquire));
        assert!(commands.export_fence().expect("export 3").is_none());
        commands.wait(Duration::from_secs(5)).expect("wait 3");
    }
}
