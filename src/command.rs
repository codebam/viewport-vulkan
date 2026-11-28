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

/// Number of command-buffer slots in the submission ring.
///
/// Multiple slots let frame N+1 record while frame N is still executing on the
/// GPU instead of stalling `begin()` on the immediately preceding frame's fence.
pub(crate) const SLOTS: usize = 3;

struct Slot {
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
    /// Semaphores belonging to this slot's in-flight submission. They cannot be
    /// destroyed until its fence signals, so they are held here and freed once
    /// that happens.
    retired: Vec<vk::Semaphore>,
    /// Whether this slot's fence has been submitted and not yet waited on.
    /// Resetting an unsignalled fence, or waiting on one that was never
    /// submitted, both hang rather than fail.
    pending: bool,
}

/// A command pool and a ring of reusable primary command buffers.
pub struct Commands {
    device: Device,
    pool: vk::CommandPool,
    slots: [Slot; SLOTS],
    /// Slot currently being recorded (after [`Commands::begin`]) or most
    /// recently recorded.
    current: usize,
    /// Slot most recently submitted to the queue, used by
    /// [`Commands::export_fence`] and [`Commands::wait`].
    last_submitted: Option<usize>,
    /// Semaphores the next submission must wait on, imported from fences
    /// handed to us by clients or by another renderer.
    waits: Vec<vk::Semaphore>,
    /// Reusable stage-mask scratch storage for `vkQueueSubmit` so submissions
    /// do not heap-allocate on the frame path.
    wait_stages: Vec<vk::PipelineStageFlags>,
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
            .command_buffer_count(SLOTS as u32);
        let buffers = match unsafe { handle.allocate_command_buffers(&allocate) } {
            Ok(buffers) => buffers,
            Err(e) => {
                unsafe { handle.destroy_command_pool(pool, None) };
                return Err(anyhow::Error::from(e).context("vkAllocateCommandBuffers"));
            }
        };

        let mut built = Vec::with_capacity(SLOTS);
        let fence_info = vk::FenceCreateInfo::default();
        for buffer in buffers {
            // Unsignalled: nothing has been submitted on this slot yet.
            let fence = match unsafe { handle.create_fence(&fence_info, None) } {
                Ok(fence) => fence,
                Err(e) => {
                    Self::destroy_slots(handle, built);
                    unsafe { handle.destroy_command_pool(pool, None) };
                    return Err(anyhow::Error::from(e).context("vkCreateFence"));
                }
            };
            let signal = match create_exportable_semaphore(device) {
                Ok(semaphore) => semaphore,
                Err(e) => {
                    unsafe { handle.destroy_fence(fence, None) };
                    Self::destroy_slots(handle, built);
                    unsafe { handle.destroy_command_pool(pool, None) };
                    return Err(e);
                }
            };
            built.push(Slot {
                buffer,
                fence,
                signal,
                signal_pending: AtomicBool::new(false),
                retired: Vec::new(),
                pending: false,
            });
        }

        let slots: [Slot; SLOTS] = built
            .try_into()
            .unwrap_or_else(|_| unreachable!("allocated exactly SLOTS command buffers"));

        Ok(Self {
            device: device.clone(),
            pool,
            slots,
            // Start at the last slot so the first `begin()` selects slot 0.
            current: SLOTS - 1,
            last_submitted: None,
            waits: Vec::new(),
            wait_stages: Vec::new(),
        })
    }

    fn destroy_slots(handle: &ash::Device, slots: impl IntoIterator<Item = Slot>) {
        unsafe {
            for mut slot in slots {
                for semaphore in slot.retired.drain(..) {
                    handle.destroy_semaphore(semaphore, None);
                }
                handle.destroy_semaphore(slot.signal, None);
                handle.destroy_fence(slot.fence, None);
            }
        }
    }

    pub fn buffer(&self) -> vk::CommandBuffer {
        self.slots[self.current].buffer
    }

    /// Start recording into the next slot in the ring, waiting only for that
    /// slot's previous submission if it is still in flight.
    ///
    /// Re-recording a command buffer the GPU is still executing is undefined
    /// behaviour, and one of the few Vulkan mistakes that usually appears to
    /// work. Advancing through a multi-slot ring preserves that guarantee while
    /// allowing frame N+1 to be recorded before frame N finishes executing.
    pub fn begin(&mut self) -> Result<vk::CommandBuffer> {
        let next = (self.current + 1) % self.slots.len();
        self.wait_slot(next, Duration::from_secs(5))?;

        let handle = self.device.handle();
        let buffer = self.slots[next].buffer;
        unsafe {
            handle
                .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())
                .context("vkResetCommandBuffer")?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            handle
                .begin_command_buffer(buffer, &begin)
                .context("vkBeginCommandBuffer")?;
        }
        self.current = next;
        Ok(buffer)
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

    /// Ensure `slots[index].signal` is unsignalled before a submission signals
    /// it.
    ///
    /// By the time this runs, `begin()` has already waited for this slot's
    /// previous submission fence, so the GPU is done with its semaphore.
    fn reset_signal_if_needed(&mut self, index: usize) -> Result<()> {
        if !self.slots[index].signal_pending.load(Ordering::Acquire) {
            return Ok(());
        }
        let signal = self.slots[index].signal;
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(signal)
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
                unsafe { self.device.handle().destroy_semaphore(signal, None) };
                self.slots[index].signal = fresh;
            }
        }
        self.slots[index]
            .signal_pending
            .store(false, Ordering::Release);
        Ok(())
    }

    fn submit_internal(&mut self, signal: bool) -> Result<()> {
        let index = self.current;
        if signal {
            self.reset_signal_if_needed(index)?;
        }
        // ALL_COMMANDS rather than a narrower stage: the imported fence
        // guards the buffer's contents, and narrowing this to, say, the
        // fragment stage would be a claim about where it is read that this
        // layer cannot make.
        if self.wait_stages.len() < self.waits.len() {
            self.wait_stages
                .resize(self.waits.len(), vk::PipelineStageFlags::ALL_COMMANDS);
        }
        let handle = self.device.handle();
        let buffer = self.slots[index].buffer;
        let fence = self.slots[index].fence;
        let signal_sem = self.slots[index].signal;
        unsafe {
            handle
                .end_command_buffer(buffer)
                .context("vkEndCommandBuffer")?;
            handle.reset_fences(&[fence]).context("vkResetFences")?;

            let buffers = [buffer];
            let signals = [signal_sem];
            let stages = &self.wait_stages[..self.waits.len()];
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(&buffers)
                .wait_semaphores(&self.waits)
                .wait_dst_stage_mask(stages);
            if signal {
                submit = submit.signal_semaphores(&signals);
            }
            if let Err(e) = handle.queue_submit(self.device.queue(), &[submit], fence) {
                for semaphore in self.waits.drain(..) {
                    handle.destroy_semaphore(semaphore, None);
                }
                return Err(anyhow::Error::from(e).context("vkQueueSubmit"));
            }
        }
        // Held until this slot's submission completes; destroying a semaphore
        // the queue is still waiting on is a use-after-free inside the driver.
        let slot = &mut self.slots[index];
        slot.retired.append(&mut self.waits);
        slot.pending = true;
        if signal {
            slot.signal_pending.store(true, Ordering::Release);
        }
        self.last_submitted = Some(index);
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
        let Some(index) = self.last_submitted else {
            return Ok(None);
        };
        let slot = &self.slots[index];
        if !slot.pending || !slot.signal_pending.load(Ordering::Acquire) {
            return Ok(None);
        }
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(slot.signal)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

        let raw = unsafe { self.device.external_semaphore_fd().get_semaphore_fd(&info) }
            .context("vkGetSemaphoreFdKHR")?;
        // Exporting a SYNC_FD transfers the payload out, resetting the binary
        // semaphore to unsignalled even when raw == -1 (already complete).
        slot.signal_pending.store(false, Ordering::Release);

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

    /// Block until the most recently submitted slot completes.
    ///
    /// A real frame loop should not call this — the whole point of explicit
    /// sync is that the compositor hands a fence onward instead of waiting.
    /// Synchronous operations (`upload`, `download`) and tests use it to wait
    /// on the submission they just made.
    pub fn wait(&mut self, timeout: Duration) -> Result<()> {
        let Some(index) = self.last_submitted else {
            return Ok(());
        };
        self.wait_slot(index, timeout)
    }

    /// Block until every in-flight slot in the ring completes and retire all of
    /// their wait semaphores.
    ///
    /// Used before dropping shared GPU resources (caches, display colour cubes)
    /// and on teardown so no in-flight slot can still be referencing them.
    pub fn wait_all(&mut self, timeout: Duration) -> Result<()> {
        let mut fences = [vk::Fence::null(); SLOTS];
        let mut count = 0;
        for slot in &self.slots {
            if slot.pending {
                fences[count] = slot.fence;
                count += 1;
            }
        }
        if count == 0 {
            return Ok(());
        }
        let handle = self.device.handle();
        unsafe {
            handle
                .wait_for_fences(&fences[..count], true, timeout_nanos(timeout))
                .context("vkWaitForFences")?;
        }
        for index in 0..self.slots.len() {
            self.slots[index].pending = false;
            self.free_retired_slot(index);
        }
        Ok(())
    }

    fn wait_slot(&mut self, index: usize, timeout: Duration) -> Result<()> {
        if !self.slots[index].pending {
            return Ok(());
        }
        let handle = self.device.handle();
        let fence = self.slots[index].fence;
        unsafe {
            handle
                .wait_for_fences(&[fence], true, timeout_nanos(timeout))
                .context("vkWaitForFences")?;
        }
        self.slots[index].pending = false;
        self.free_retired_slot(index);
        Ok(())
    }

    fn free_retired_slot(&mut self, index: usize) {
        let handle = self.device.handle();
        for semaphore in self.slots[index].retired.drain(..) {
            unsafe { handle.destroy_semaphore(semaphore, None) };
        }
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // The pool cannot be destroyed while any of its buffers are executing.
        let _ = self.wait_all(Duration::from_secs(5));

        let device = self.device.clone();
        let handle = device.handle();
        unsafe {
            for semaphore in self.waits.drain(..) {
                handle.destroy_semaphore(semaphore, None);
            }
            for slot in &mut self.slots {
                for semaphore in slot.retired.drain(..) {
                    handle.destroy_semaphore(semaphore, None);
                }
                handle.destroy_semaphore(slot.signal, None);
                handle.destroy_fence(slot.fence, None);
            }
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

        // Fill every slot without exporting: each slot's signal_pending becomes true.
        for i in 0..SLOTS {
            commands.begin().expect("begin unexported");
            commands.submit().expect("submit unexported");
            assert_eq!(commands.current, i);
            assert!(commands.slots[i].signal_pending.load(Ordering::Acquire));
        }

        // Waiting on the most recent slot clears its pending flag without
        // consuming its signal semaphore.
        let last = commands.last_submitted.expect("last submitted");
        commands.wait(Duration::from_secs(5)).expect("wait last");
        assert!(!commands.slots[last].pending);
        assert!(commands.slots[last].signal_pending.load(Ordering::Acquire));

        // Wrapping around every slot must drain/reset each slot's unexported
        // semaphore before re-signalling it, and exporting afterwards clears
        // signal_pending cleanly.
        for i in 0..SLOTS {
            commands.begin().expect("begin wrap");
            assert_eq!(commands.current, i);
            commands.submit().expect("submit wrap");
            let _ = commands.export_fence().expect("export wrap");
            assert!(!commands.slots[i].signal_pending.load(Ordering::Acquire));
        }
        commands.wait_all(Duration::from_secs(5)).expect("wait_all");

        // An unsignalled submit leaves signal_pending false and export_fence
        // returns None.
        commands.begin().expect("begin unsignaled");
        commands.submit_unsignaled().expect("submit unsignaled");
        let idx = commands.current;
        assert!(!commands.slots[idx].signal_pending.load(Ordering::Acquire));
        assert!(commands
            .export_fence()
            .expect("export unsignaled")
            .is_none());
        commands
            .wait(Duration::from_secs(5))
            .expect("wait unsignaled");
    }

    #[test]
    fn multi_slot_pipelined_submits_consecutive_exports_and_wait_all() {
        let Some(TestGpu { device, .. }) = require_gpu() else {
            return;
        };
        let mut commands = Commands::new(&device).expect("commands");

        // Record and submit SLOTS frames back-to-back without waiting between
        // them: each gets a distinct command buffer and leaves previous slots
        // untouched until the ring wraps.
        let mut buffers = Vec::with_capacity(SLOTS);
        let mut exported = Vec::with_capacity(SLOTS);
        for expected_slot in 0..SLOTS {
            let cmd = commands.begin().expect("pipelined begin");
            assert_eq!(commands.current, expected_slot);
            assert_eq!(commands.buffer(), cmd);
            assert!(
                !buffers.contains(&cmd),
                "pipelined slot reused an in-flight command buffer"
            );
            buffers.push(cmd);

            commands.submit().expect("pipelined submit");
            assert_eq!(commands.last_submitted, Some(expected_slot));
            assert!(commands.slots[expected_slot].pending);

            let fd = commands.export_fence().expect("pipelined export");
            assert!(
                !commands.slots[expected_slot]
                    .signal_pending
                    .load(Ordering::Acquire),
                "export must transfer out signal_pending on the submitted slot"
            );
            // Exporting a second time for the same submission is a no-op.
            assert!(commands.export_fence().expect("second export").is_none());
            if let Some(fd) = fd {
                exported.push(fd);
            }
        }

        // All SLOTS submissions are tracked as pending simultaneously.
        assert!(commands.slots.iter().all(|s| s.pending));

        // Feed an exported fence back as a wait semaphore into the wrap-around
        // submission, exercising `wait_stages` reuse and per-slot retirement.
        if let Some(fd) = exported.pop() {
            commands.wait_on(fd).expect("wait_on");
            assert_eq!(commands.pending_waits(), 1);
        }

        // Wrapping around reuses slot 0 after waiting on slot 0's fence only,
        // while slots 1..SLOTS stay pending until their turn or `wait_all`.
        let wrapped = commands.begin().expect("wrap begin");
        assert_eq!(commands.current, 0);
        assert_eq!(wrapped, buffers[0]);
        assert!(commands.slots[1].pending);
        commands.submit().expect("wrap submit");
        assert_eq!(commands.pending_waits(), 0);
        let _ = commands.export_fence().expect("wrap export");

        // `wait_all` waits for every pending slot and retires all wait semaphores.
        commands.wait_all(Duration::from_secs(5)).expect("wait_all");
        assert!(commands.slots.iter().all(|s| !s.pending));
        assert!(commands.slots.iter().all(|s| s.retired.is_empty()));
    }
}
