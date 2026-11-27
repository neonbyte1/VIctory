//! Hash sets and maps for keys we make ourselves (host object addresses,
//! title addresses and lengths), hashed with FxHash's multiply-rotate step
//! instead of the standard library's SipHash: a draw adds to several of
//! them, and SipHash was most of what the draw's refresh bookkeeping cost.
//! None of these keys come from outside, so SipHash's resistance to chosen
//! collisions buys nothing here. (`victory_shader::structure` keeps the
//! same hasher for its own tables.)

use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type Set<K> = std::collections::HashSet<K, BuildHasherDefault<Fx>>;
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fx>>;

/// FxHash's multiply-rotate step (finished as rustc-hash 2 does).
#[derive(Default)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    /// Rotated, so that the bits the multiply mixed best pick the bucket:
    /// without it, aligned addresses (low bits zero) land in few buckets.
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}
