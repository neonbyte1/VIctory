//! Synthetic reverse-index regressions; no host device or title data.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expected {
    Texture(usize),
    Left((usize, u64, u64)),
}

fn check(inner: &Inner, reference: &BTreeMap<u32, Expected>) {
    let actual: BTreeMap<_, _> = inner
        .slots
        .iter()
        .map(|(&slot, seen)| {
            let value = match seen {
                Seen::Texture(r) => Expected::Texture(*r),
                Seen::Left(v) => Expected::Left(*v),
                Seen::Buffer(..) => panic!("unexpected buffer"),
            };
            (slot, value)
        })
        .collect();
    assert_eq!(&actual, reference);
    let mut reverse = BTreeMap::<usize, BTreeSet<u32>>::new();
    for (&slot, &value) in reference {
        if let Expected::Texture(r) = value {
            reverse.entry(r).or_default().insert(slot);
        }
    }
    assert_eq!(inner.textures, reverse);
    assert!(inner.ranges.is_empty());
    assert!(inner.ends.is_empty());
    assert!(inner.hulls.is_empty());
}

#[test]
fn release_preserves_overwritten_and_unrelated_slots() {
    let mut inner = Inner::default();
    inner.replace(0, Some(Seen::Texture(1)));
    inner.replace(u32::MAX, Some(Seen::Texture(1)));
    inner.replace(2, Some(Seen::Texture(2)));
    inner.replace(0, Some(Seen::Left((2, 4096, 128))));
    inner.forget_texture(1);
    inner.forget_texture(1);
    check(
        &inner,
        &BTreeMap::from([
            (0, Expected::Left((2, 4096, 128))),
            (2, Expected::Texture(2)),
        ]),
    );
    inner.replace(u32::MAX, Some(Seen::Texture(1)));
    inner.replace(u32::MAX, Some(Seen::Texture(1)));
    inner.replace(u32::MAX, None);
    assert!(!inner.textures.contains_key(&1));
}

#[test]
fn seeded_slot_updates_match_full_scan() {
    for seed in 1..=200u64 {
        let mut random = seed;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut inner = Inner::default();
        let mut reference = BTreeMap::new();
        let span = [8, 32, 128, 1024][seed as usize % 4];
        let resources = [2, 8, 32, 128][(seed as usize / 4) % 4];
        for _ in 0..256 {
            let offset = (next() % span) as u32;
            let slot = if seed.is_multiple_of(2) {
                u32::MAX - offset
            } else {
                offset
            };
            let r = (next() % resources) as usize;
            match next() % 8 {
                0 => {
                    inner.forget_texture(r);
                    reference.retain(|_, seen| *seen != Expected::Texture(r));
                }
                1 => {
                    inner.replace(slot, None);
                    reference.remove(&slot);
                }
                2 => {
                    let left = (r % 4, next() % 4096, next() % 512);
                    inner.replace(slot, Some(Seen::Left(left)));
                    reference.insert(slot, Expected::Left(left));
                }
                _ => {
                    inner.replace(slot, Some(Seen::Texture(r)));
                    reference.insert(slot, Expected::Texture(r));
                }
            }
            check(&inner, &reference);
        }
    }
}
