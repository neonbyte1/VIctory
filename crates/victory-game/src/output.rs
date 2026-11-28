//! Build signatures and bounded output-resolution patching.
//!
//! The profile is intentionally limited to the one verified ladder cap. A
//! caller must identify the executable build before planning or applying it.

use crate::KnownBuild;
use victory_core::Resolution;

const LADDER: [(u32, u32); 5] = [
    (1280, 720),
    (1664, 936),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];
const LADDER_BYTES: usize = LADDER.len() * 16;
const SELF_CHECK_BYTES: usize = 26;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchError {
    UnsupportedBuild,
    UnsupportedResolution,
    TruncatedImage,
    MissingLadder,
    AmbiguousLadder,
    MissingSelfCheck,
    AmbiguousSelfCheck,
    StalePlan,
}

impl std::fmt::Display for PatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PatchError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchPlan {
    ladder_at: usize,
    self_check_at: usize,
}

impl PatchPlan {
    pub fn self_check_offset(self) -> usize {
        self.self_check_at + SELF_CHECK_BYTES - 1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PatchWrite {
    offset: usize,
    bytes: [u8; 8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchWrites([PatchWrite; 2]);

impl PatchWrites {
    /// The smallest contiguous span covering both writes.
    pub fn range(self) -> std::ops::Range<usize> {
        self.0[0].offset..self.0[1].offset + self.0[1].bytes.len()
    }

    /// Writes only the two validated width/height pairs.
    ///
    /// # Safety
    /// `base` points to the mapped image validated by [`validate_writes`],
    /// and the pages covering [`Self::range`] are writable.
    pub unsafe fn write_to(self, base: *mut u8) {
        for write in self.0 {
            // SAFETY: guaranteed by this method's caller; each operation
            // writes exactly one validated eight-byte field.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    write.bytes.as_ptr(),
                    base.add(write.offset),
                    write.bytes.len(),
                );
            }
        }
    }
}

/// Finds exactly one ladder and exactly one associated self-check before any
/// title memory is made writable. Unknown or ambiguous images fail closed.
pub fn plan(
    bytes: &[u8],
    build: &KnownBuild,
    resolution: Resolution,
) -> Result<PatchPlan, PatchError> {
    if !build.supports_resolution_override {
        return Err(PatchError::UnsupportedBuild);
    }
    if resolution != Resolution::HD_1080 {
        return Err(PatchError::UnsupportedResolution);
    }
    if bytes.len() < LADDER_BYTES.max(SELF_CHECK_BYTES) {
        return Err(PatchError::TruncatedImage);
    }
    let ladder_at = unique_ladder(bytes)?;
    let self_check_at = unique_self_check(bytes)?;
    Ok(PatchPlan {
        ladder_at,
        self_check_at,
    })
}

/// Revalidates both signatures and returns the two exact writes. The caller
/// can perform them without creating a mutable slice over the full image.
pub fn validate_writes(
    bytes: &[u8],
    plan: PatchPlan,
    resolution: Resolution,
) -> Result<PatchWrites, PatchError> {
    if bytes.len() < LADDER_BYTES.max(SELF_CHECK_BYTES) {
        return Err(PatchError::TruncatedImage);
    }
    if resolution != Resolution::HD_1080
        || unique_ladder(bytes)? != plan.ladder_at
        || unique_self_check(bytes)? != plan.self_check_at
    {
        return Err(PatchError::StalePlan);
    }

    let mut encoded = [0; 8];
    encoded[..4].copy_from_slice(&resolution.width.to_le_bytes());
    encoded[4..].copy_from_slice(&resolution.height.to_le_bytes());
    Ok(PatchWrites([
        PatchWrite {
            offset: plan.ladder_at + 3 * 16,
            bytes: encoded,
        },
        PatchWrite {
            offset: plan.ladder_at + 4 * 16,
            bytes: encoded,
        },
    ]))
}

fn unique_ladder(bytes: &[u8]) -> Result<usize, PatchError> {
    let mut found = None;
    for (at, window) in bytes.windows(LADDER_BYTES).enumerate() {
        if is_ladder(window) {
            if found.is_some() {
                return Err(PatchError::AmbiguousLadder);
            }
            found = Some(at);
        }
    }
    found.ok_or(PatchError::MissingLadder)
}

fn is_ladder(bytes: &[u8]) -> bool {
    if bytes.len() < LADDER_BYTES {
        return false;
    }
    LADDER.iter().enumerate().all(|(index, &(width, height))| {
        let at = index * 16;
        bytes[at..at + 4] == width.to_le_bytes()
            && bytes[at + 4..at + 8] == height.to_le_bytes()
            && bytes[at + 8..at + 12] == 60u32.to_le_bytes()
            && bytes[at + 12..at + 16] == 1u32.to_le_bytes()
    })
}

fn unique_self_check(bytes: &[u8]) -> Result<usize, PatchError> {
    if bytes.len() < SELF_CHECK_BYTES {
        return Err(PatchError::TruncatedImage);
    }
    let mut found = None;
    for at in 0..=bytes.len().saturating_sub(SELF_CHECK_BYTES) {
        if is_self_check(&bytes[at..at + SELF_CHECK_BYTES]) {
            if found.is_some() {
                return Err(PatchError::AmbiguousSelfCheck);
            }
            found = Some(at);
        }
    }
    found.ok_or(PatchError::MissingSelfCheck)
}

fn is_self_check(bytes: &[u8]) -> bool {
    const PREFIX: [u8; 10] = [0x48, 0x8b, 0x44, 0x24, 0x78, 0x48, 0x2b, 0x44, 0x24, 0x70];
    const ADD: [u8; 3] = [0x48, 0x03, 0x05];
    const CMP: [u8; 6] = [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00];
    bytes.len() == SELF_CHECK_BYTES
        && bytes[..10] == PREFIX
        && bytes[10..13] == ADD
        && bytes[17..23] == CMP
        && bytes[23] == 0x76
        && bytes[24] == 1
        && bytes[25] == 0xcc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> KnownBuild {
        KnownBuild {
            label: "synthetic build",
            blake3: "synthetic",
            startup_archive: "synthetic",
            supports_resolution_override: true,
        }
    }

    fn image_with_signatures() -> Vec<u8> {
        let mut image = vec![0x90; 512];
        for (index, &(width, height)) in LADDER.iter().enumerate() {
            let at = 64 + index * 16;
            image[at..at + 4].copy_from_slice(&width.to_le_bytes());
            image[at + 4..at + 8].copy_from_slice(&height.to_le_bytes());
            image[at + 8..at + 12].copy_from_slice(&60u32.to_le_bytes());
            image[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        }
        image[256..282].copy_from_slice(&[
            0x48, 0x8b, 0x44, 0x24, 0x78, 0x48, 0x2b, 0x44, 0x24, 0x70, 0x48, 0x03, 0x05, 0, 0, 0,
            0, 0x48, 0x3d, 0xff, 0xff, 0x1f, 0, 0x76, 1, 0xcc,
        ]);
        image
    }

    #[test]
    fn plan_requires_unique_ladder_and_self_check_before_patch() {
        let mut image = image_with_signatures();
        let plan = plan(&image, &build(), Resolution::HD_1080).unwrap();
        assert_eq!(plan.self_check_offset(), 281);
        let before = image.clone();
        let writes = validate_writes(&image, plan, Resolution::HD_1080).unwrap();
        assert_eq!(writes.range(), 64 + 3 * 16..64 + 4 * 16 + 8);
        // SAFETY: synthetic `Vec` has the mapped-image length validated above.
        unsafe { writes.write_to(image.as_mut_ptr()) };
        assert_eq!(
            &image[64 + 3 * 16..64 + 3 * 16 + 8],
            &[0x80, 0x07, 0, 0, 0x38, 0x04, 0, 0]
        );
        assert_eq!(
            &image[64 + 4 * 16..64 + 4 * 16 + 8],
            &[0x80, 0x07, 0, 0, 0x38, 0x04, 0, 0]
        );
        for (at, (&actual, &before)) in image.iter().zip(&before).enumerate() {
            let is_write = (64 + 3 * 16..64 + 3 * 16 + 8).contains(&at)
                || (64 + 4 * 16..64 + 4 * 16 + 8).contains(&at);
            assert!(is_write || actual == before, "unexpected mutation at {at}");
        }
    }

    #[test]
    fn truncated_images_fail_without_panicking() {
        for length in [0, 25, SELF_CHECK_BYTES - 1, LADDER_BYTES - 1] {
            assert_eq!(
                plan(&vec![0; length], &build(), Resolution::HD_1080),
                Err(PatchError::TruncatedImage)
            );
        }
    }

    #[test]
    fn missing_or_ambiguous_signatures_fail_without_mutation() {
        let mut missing = image_with_signatures();
        missing[64..64 + LADDER_BYTES].fill(0x90);
        let before = missing.clone();
        assert_eq!(
            plan(&missing, &build(), Resolution::HD_1080),
            Err(PatchError::MissingLadder)
        );
        assert_eq!(missing, before);

        let mut duplicate = image_with_signatures();
        let ladder = duplicate[64..64 + LADDER_BYTES].to_vec();
        duplicate[320..320 + LADDER_BYTES].copy_from_slice(&ladder);
        let before = duplicate.clone();
        assert_eq!(
            plan(&duplicate, &build(), Resolution::HD_1080),
            Err(PatchError::AmbiguousLadder)
        );
        assert_eq!(duplicate, before);
    }

    #[test]
    fn unknown_build_and_resolution_are_rejected() {
        let mut unknown = build();
        unknown.supports_resolution_override = false;
        assert_eq!(
            plan(&image_with_signatures(), &unknown, Resolution::HD_1080),
            Err(PatchError::UnsupportedBuild)
        );
        assert_eq!(
            plan(
                &image_with_signatures(),
                &build(),
                Resolution {
                    width: 1280,
                    height: 720
                }
            ),
            Err(PatchError::UnsupportedResolution)
        );
    }

    #[test]
    fn stale_or_ambiguous_self_check_rejects_patch_without_mutation() {
        let mut image = image_with_signatures();
        let plan = plan(&image, &build(), Resolution::HD_1080).unwrap();
        let duplicate = image[256..282].to_vec();
        image[300..326].copy_from_slice(&duplicate);
        let before = image.clone();
        assert_eq!(
            validate_writes(&image, plan, Resolution::HD_1080),
            Err(PatchError::AmbiguousSelfCheck)
        );
        assert_eq!(image, before);
    }
}
