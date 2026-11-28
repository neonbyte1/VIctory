/// A render profile that the title runtime knows how to apply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    /// The only reduced profile currently supported by the title signature gate.
    pub const HD_1080: Self = Self {
        width: 1920,
        height: 1080,
    };

    /// Parses a supported `WIDTHxHEIGHT` profile. New sizes must be added
    /// only after their title ladder and presentation behavior are verified.
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        match value.trim() {
            "1920x1080" => Ok(Self::HD_1080),
            _ => Err("supported resolution is 1920x1080"),
        }
    }

    /// The source rectangle the host may present at this resolution. The
    /// known title paths are a native target or a 2x backing frame; other
    /// geometry is rejected until verified for that build.
    pub fn picture_size(self, frame_width: u32, frame_height: u32) -> Option<(u32, u32)> {
        let native = (self.width, self.height);
        let double = (self.width.checked_mul(2)?, self.height.checked_mul(2)?);
        if (frame_width, frame_height) == native || (frame_width, frame_height) == double {
            Some(native)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Resolution;

    #[test]
    fn accepts_only_the_tested_resolution_profile() {
        assert_eq!(Resolution::parse("1920x1080"), Ok(Resolution::HD_1080));
        for invalid in ["1080p", "0x0", "3840x2160", "1280x720", "1920", "1920X1080"] {
            assert!(Resolution::parse(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn computes_the_title_picture_crop_from_a_frame_size() {
        assert_eq!(
            Resolution::HD_1080.picture_size(1920, 1080),
            Some((1920, 1080))
        );
        assert_eq!(
            Resolution::HD_1080.picture_size(3840, 2160),
            Some((1920, 1080))
        );
        assert_eq!(Resolution::HD_1080.picture_size(0, 1080), None);
        assert_eq!(Resolution::HD_1080.picture_size(1920, 0), None);
        assert_eq!(Resolution::HD_1080.picture_size(2560, 1440), None);
    }
}
