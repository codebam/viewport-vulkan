// SPDX-License-Identifier: MIT
//
// The `wl_buffer` wrappers: ImportDmaWl and ImportMemWl.
//
// These are thin. A dmabuf-backed `wl_buffer` unwraps to a `Dmabuf` and goes
// through `ImportDma` unchanged, so Smithay's default body already does the
// right thing. An shm one needs real work, because the shared memory a client
// wrote is described by a stride and an offset that need not match how Vulkan
// wants the rows laid out.
//
// Behind the `wayland` feature so the renderer stays usable — and testable —
// without a Wayland display.

use smithay::backend::renderer::{ImportDmaWl, ImportMemWl, Texture as _};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::Resource;
use smithay::utils::{Buffer as BufferCoord, Rectangle};
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::shm::with_buffer_contents;

use crate::renderer::{Error, VulkanRenderer, VulkanTexture};

/// A dmabuf `wl_buffer` is just a `Dmabuf` underneath, so the import itself is
/// Smithay's default body. What is added is the one thing this layer knows and
/// `ImportDma` does not: which surface the buffer came from, and therefore what
/// the client said its pixels mean.
///
/// This is the only place the two meet. `import_dmabuf` is handed a `Dmabuf`
/// with no surface attached, so a description recorded against the surface can
/// never reach the texture from there — it would be stored by the protocol
/// code and silently dropped on the way to the shader, which is how a client
/// that correctly declared PQ still got decoded as sRGB.
impl ImportDmaWl for VulkanRenderer {
    fn import_dma_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        let dmabuf = smithay::wayland::dmabuf::get_dmabuf(buffer)
            .map_err(|e| Error::Unsupported(format!("not a dmabuf: {e}")))?;
        // The surface's declared representation, if any, goes with the buffer:
        // the YCbCr conversion is decided at import, and this is the only
        // point where the buffer and the surface that committed it are both
        // in hand. `import_dmabuf_rep` drops it again for an RGB buffer, so
        // the lookup pays for it only where it can change the result.
        let representation = surface.and_then(crate::color::representation_in);
        let texture = self.import_dmabuf_rep(dmabuf, representation)?;
        Ok(described(texture, surface, self.sdr))
    }
}

/// Stamp the surface's declared colour onto a texture.
///
/// A texture is a cheap handle over a shared image, so this is a second view
/// of the same pixels rather than a copy — which is also why the description
/// has to be applied on every import instead of once: the cached image is
/// shared with whatever else imported the same buffer.
fn described(
    texture: VulkanTexture,
    surface: Option<&SurfaceData>,
    sdr: crate::color::SdrEotf,
) -> VulkanTexture {
    match surface {
        Some(states) => texture
            .with_description(sdr.description(crate::color::declared_in(states)))
            .with_tonemap(crate::color::tonemap_in(states).unwrap_or_default()),
        // No surface: a cursor, a screencopy source, something the compositor
        // made itself. sRGB is what the texture already carries, and tone
        // mapping there is the identity until an output is dimmer than the
        // source.
        None => texture,
    }
}

/// Validate an shm buffer's slice bounds and return its tightly packed row
/// length in bytes.
fn validate_shm_layout(
    src_len: usize,
    offset: usize,
    stride: usize,
    width: usize,
    height: usize,
    bpp: usize,
) -> Result<usize, Error> {
    if width == 0 || height == 0 || bpp == 0 {
        return Err(Error::Unsupported("a zero-sized shm buffer".to_owned()));
    }
    let packed = width
        .checked_mul(bpp)
        .ok_or_else(|| Error::Unsupported("shm buffer dimensions overflow".to_owned()))?;
    if stride < packed {
        return Err(Error::Unsupported(format!(
            "shm stride {stride} is narrower than a {packed}-byte row"
        )));
    }
    let needed = stride
        .checked_mul(height - 1)
        .and_then(|v| v.checked_add(packed))
        .and_then(|v| offset.checked_add(v))
        .ok_or_else(|| Error::Unsupported("shm buffer dimensions overflow".to_owned()))?;
    if src_len < needed {
        return Err(Error::Unsupported(format!(
            "shm buffer is {src_len} bytes; {width}x{height} at stride {stride} needs {needed}"
        )));
    }
    Ok(packed)
}

/// Repack `src` into tightly packed rows.
///
/// A client's shm buffer has whatever stride it chose — often padded for
/// alignment — while tightly packed buffers use `width * bpp` per row.
#[cfg(test)]
pub(crate) fn repack(
    src: &[u8],
    offset: usize,
    stride: usize,
    width: usize,
    height: usize,
    bpp: usize,
) -> Result<std::borrow::Cow<'_, [u8]>, Error> {
    let packed = validate_shm_layout(src.len(), offset, stride, width, height, bpp)?;
    let total = packed
        .checked_mul(height)
        .ok_or_else(|| Error::Unsupported("shm buffer dimensions overflow".to_owned()))?;
    if stride == packed && offset == 0 {
        return Ok(std::borrow::Cow::Borrowed(&src[..total]));
    }

    let mut out = Vec::with_capacity(total);
    for row in 0..height {
        let start = offset + row * stride;
        out.extend_from_slice(&src[start..start + packed]);
    }
    Ok(std::borrow::Cow::Owned(out))
}

impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        let id = buffer.id();
        // Smithay's damage is relative to one surface's previous commit. A
        // renderer-wide cache also contains buffers from every other surface,
        // so only buffers belonging to this surface missed that damage.
        let surface_id = surface.map(|surface| std::ptr::from_ref(surface).addr());

        // Upload directly from the locked shm mapping into the reusable staging
        // buffer, packing only damaged rows and avoiding a full-frame heap copy.
        with_buffer_contents(buffer, |pointer, len, data| {
            let fourcc =
                smithay::wayland::shm::shm_format_to_fourcc(data.format).ok_or_else(|| {
                    Error::Unsupported(format!("no fourcc for shm format {:?}", data.format))
                })?;
            let bpp = crate::format::bytes_per_pixel(fourcc)
                .ok_or_else(|| Error::Unsupported(format!("unsupported shm format {fourcc:?}")))?;

            if data.width <= 0 || data.height <= 0 || data.offset < 0 || data.stride < 0 {
                return Err(Error::Unsupported(
                    "invalid shm buffer dimensions".to_owned(),
                ));
            }
            let (width, height) = (data.width as usize, data.height as usize);
            let offset = data.offset as usize;
            let stride = data.stride as usize;

            // SAFETY: with_buffer_contents guarantees the mapping is valid and
            // the pool is locked for the duration of this closure.
            let bytes = unsafe { std::slice::from_raw_parts(pointer, len) };
            validate_shm_layout(bytes.len(), offset, stride, width, height, bpp)?;

            // Every other cached buffer belonging to this surface has now
            // missed this commit. Surfaceless imports (`surface_id == None`) do
            // not share a commit history with each other.
            if surface_id.is_some() {
                for entry in self
                    .shm
                    .iter_mut()
                    .filter(|e| e.surface_id == surface_id && e.id != id)
                {
                    if entry.pending.len() < MAX_PENDING {
                        entry.pending.extend_from_slice(damage);
                    }
                }
            }

            let found = self.shm.iter().position(|e| {
                e.id == id
                    && e.surface_id == surface_id
                    && e.texture.width() as usize == width
                    && e.texture.height() as usize == height
                    && e.texture.image().fourcc() == fourcc
            });

            let whole = Rectangle::from_size((width as i32, height as i32).into());
            if let Some(index) = found {
                let texture = self.shm[index].texture.clone();
                let regions = plan_upload(damage, &self.shm[index].pending, whole);
                if !regions.is_empty() {
                    self.upload_regions(
                        &texture,
                        bytes,
                        offset,
                        stride,
                        &regions,
                        ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )?;
                }
                self.shm[index].pending.clear();
                return Ok(described(texture, surface, self.sdr));
            }

            // New, or a different shape: a full upload either way.
            let texture =
                self.allocate_upload_texture(width as u32, height as u32, fourcc, false)?;
            self.upload_regions(
                &texture,
                bytes,
                offset,
                stride,
                &[whole],
                ash::vk::ImageLayout::UNDEFINED,
            )?;
            self.shm
                .retain(|e| e.id != id || e.surface_id != surface_id);
            self.shm.push(ShmEntry {
                id,
                surface_id,
                texture: texture.clone(),
                pending: Vec::new(),
            });
            Ok(described(texture, surface, self.sdr))
        })
        .map_err(|e| Error::Unsupported(format!("not an shm buffer: {e}")))?
    }
}

impl VulkanRenderer {
    /// Forget the texture cached for a `wl_buffer`.
    ///
    /// Called when the buffer is destroyed; without it the cache grows for the
    /// lifetime of the compositor.
    pub fn forget_shm_buffer(&mut self, buffer: &WlBuffer) {
        let id = buffer.id();
        self.shm.retain(|e| e.id != id);
    }
}

/// One cached shm texture.
pub(crate) struct ShmEntry {
    pub id: smithay::reexports::wayland_server::backend::ObjectId,
    /// Identity of the surface whose commit history `pending` describes.
    pub surface_id: Option<usize>,
    pub texture: VulkanTexture,
    /// Damage that landed while some other buffer was the one being drawn.
    ///
    /// Smithay reports damage since this renderer last imported *for the
    /// surface*, and its own model is one texture per surface. A texture per
    /// `wl_buffer` is faster — a client that alternates two buffers uploads
    /// only what changed instead of everything — but it means a buffer coming
    /// back around has missed every commit made through the other one. Those
    /// regions are collected here and uploaded when it does.
    ///
    /// Without it the untouched parts of the returning buffer are two commits
    /// old: a terminal shows glyphs from an older frame in the cells that were
    /// not rewritten, and alternates as the buffers do.
    pub pending: Vec<Rectangle<i32, BufferCoord>>,
}

/// The cache entry type, kept here so `renderer.rs` does not need the Wayland
/// types when the feature is off.
pub(crate) type ShmCache = Vec<ShmEntry>;

/// Past this many outstanding regions a full upload is cheaper than tracking
/// them, and much cheaper than getting the bookkeeping wrong.
const MAX_PENDING: usize = 32;

/// Clip a damage rectangle to `whole` without overflowing on `i32::MAX` extents.
fn clip_to_buffer(
    rect: Rectangle<i32, BufferCoord>,
    whole: Rectangle<i32, BufferCoord>,
) -> Option<Rectangle<i32, BufferCoord>> {
    if rect.size.w <= 0 || rect.size.h <= 0 || whole.size.w <= 0 || whole.size.h <= 0 {
        return None;
    }
    let x0 = i64::from(rect.loc.x).max(i64::from(whole.loc.x));
    let y0 = i64::from(rect.loc.y).max(i64::from(whole.loc.y));
    let x1 = (i64::from(rect.loc.x) + i64::from(rect.size.w))
        .min(i64::from(whole.loc.x) + i64::from(whole.size.w));
    let y1 = (i64::from(rect.loc.y) + i64::from(rect.size.h))
        .min(i64::from(whole.loc.y) + i64::from(whole.size.h));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rectangle::new(
        (x0 as i32, y0 as i32).into(),
        ((x1 - x0) as i32, (y1 - y0) as i32).into(),
    ))
}

/// What to upload into a returning buffer: this commit's damage, and every
/// commit it missed while another buffer was in use.
///
/// An empty result means nothing changed and the upload can be skipped, which
/// the trait allows. A `pending` list at the limit means take the whole
/// buffer instead of replaying a long history. Every region is clipped to
/// `whole` because Wayland clients may pass oversized or out-of-bounds damage
/// (such as `0, 0, i32::MAX, i32::MAX`) and empty rectangles.
fn plan_upload(
    damage: &[Rectangle<i32, BufferCoord>],
    pending: &[Rectangle<i32, BufferCoord>],
    whole: Rectangle<i32, BufferCoord>,
) -> Vec<Rectangle<i32, BufferCoord>> {
    if pending.len() >= MAX_PENDING {
        return vec![whole];
    }
    damage
        .iter()
        .chain(pending.iter())
        .filter_map(|&rect| clip_to_buffer(rect, whole))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, BufferCoord> {
        Rectangle::new((x, y).into(), (w, h).into())
    }

    #[test]
    fn a_returning_buffer_is_given_the_commits_it_missed() {
        // A client alternating two buffers: this one was last drawn two
        // commits ago, so the damage from the commit in between is its to
        // catch up on. Uploading only the newest damage leaves the rest of it
        // a generation old — in a terminal, glyphs from an older frame in
        // every cell that was not rewritten.
        let whole = rect(0, 0, 100, 100);
        let now = [rect(0, 0, 10, 10)];
        let missed = [rect(50, 50, 10, 10)];
        let plan = plan_upload(&now, &missed, whole);
        assert_eq!(plan, vec![rect(0, 0, 10, 10), rect(50, 50, 10, 10)]);
    }

    #[test]
    fn nothing_to_do_stays_nothing_to_do() {
        // No damage and nothing missed: the client says the buffer is
        // unchanged and the upload is skipped entirely.
        assert!(plan_upload(&[], &[], rect(0, 0, 100, 100)).is_empty());
    }

    #[test]
    fn oversized_negative_and_empty_damage_are_clipped_to_the_buffer() {
        let whole = rect(0, 0, 100, 50);
        let plan = plan_upload(
            &[
                // Common Wayland "damage everything" sentinel with non-zero origin.
                rect(10, 5, i32::MAX, i32::MAX),
                // Partially negative origin.
                rect(-20, -10, 30, 25),
                // Completely outside or empty.
                rect(100, 0, 20, 20),
                rect(-50, 0, 50, 50),
                rect(0, 0, 0, 10),
            ],
            &[],
            whole,
        );
        assert_eq!(plan, vec![rect(10, 5, 90, 45), rect(0, 0, 10, 15)]);
    }

    #[test]
    fn a_long_history_is_replaced_by_the_whole_buffer() {
        let whole = rect(0, 0, 100, 100);
        let missed: Vec<_> = (0..MAX_PENDING as i32).map(|i| rect(i, 0, 1, 1)).collect();
        assert_eq!(
            plan_upload(&[rect(0, 0, 5, 5)], &missed, whole),
            vec![whole]
        );
    }

    #[test]
    fn a_tightly_packed_buffer_is_not_copied() {
        // 2x2, stride exactly the row length.
        let src: Vec<u8> = (0..16).collect();
        let out = repack(&src, 0, 8, 2, 2, 4).expect("repack");
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "needlessly copied"
        );
        assert_eq!(&*out, &src[..]);
    }

    #[test]
    fn a_padded_stride_is_repacked() {
        // 2x2 with 4 bytes of padding per row.
        let mut src = Vec::new();
        src.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        src.extend_from_slice(&[0, 0, 0, 0]);
        src.extend_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        src.extend_from_slice(&[0, 0, 0, 0]);

        let out = repack(&src, 0, 12, 2, 2, 4).expect("repack");
        assert!(
            matches!(out, std::borrow::Cow::Owned(_)),
            "should have copied"
        );
        assert_eq!(
            &*out,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn an_offset_is_honoured() {
        let mut src = vec![0xff; 4];
        src.extend_from_slice(&[1, 2, 3, 4]);
        let out = repack(&src, 4, 4, 1, 1, 4).expect("repack");
        assert_eq!(&*out, &[1, 2, 3, 4]);
    }

    #[test]
    fn a_short_buffer_is_an_error_rather_than_a_read_past_the_end() {
        // The last row is missing.
        let src = vec![0u8; 8];
        let error = repack(&src, 0, 8, 2, 2, 4).expect_err("a short buffer must be refused");
        assert!(error.to_string().contains("needs"), "{error}");
    }

    #[test]
    fn absurd_dimensions_do_not_overflow() {
        let src = vec![0u8; 16];
        let error = repack(&src, 0, usize::MAX, 2, 2, 4).expect_err("must not overflow");
        assert!(
            error.to_string().contains("overflow") || error.to_string().contains("needs"),
            "{error}"
        );
    }

    #[test]
    fn non_32bpp_formats_and_narrow_strides_are_handled_by_bpp() {
        // 2x2 half-float (8 bytes/pixel) with 4 bytes of row padding: stride 20.
        let mut src = Vec::new();
        src.extend_from_slice(&[1u8; 16]);
        src.extend_from_slice(&[0u8; 4]);
        src.extend_from_slice(&[2u8; 16]);
        src.extend_from_slice(&[0u8; 4]);
        let out = repack(&src, 0, 20, 2, 2, 8).expect("8bpp repack");
        assert_eq!(out.len(), 32);
        assert_eq!(&out[..16], &[1u8; 16]);
        assert_eq!(&out[16..], &[2u8; 16]);

        // Stride narrower than one row must be rejected instead of overlapping rows.
        let err = repack(&src, 0, 4, 2, 2, 4).expect_err("narrow stride");
        assert!(err.to_string().contains("narrower"), "{err}");
    }
}
