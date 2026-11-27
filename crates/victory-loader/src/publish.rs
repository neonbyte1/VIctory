//! Portable publication of derived files into a shared cache directory.
//!
//! Several copies of the runner can prepare the same title at once. Each
//! writer stages into its own file, so two of them cannot truncate each
//! other's bytes. The final rename is atomic within the directory. If
//! another writer has already published identical bytes, a failed rename
//! still counts as success.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` unless it is already there, using a staging file
/// unique to this process and call.
pub fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let sequence = NEXT_STAGING.fetch_add(1, Ordering::Relaxed);
    let staging = path.with_file_name(format!(".{name}.{}.{sequence}.tmp", std::process::id()));
    if let Err(error) = std::fs::write(&staging, bytes) {
        let _ = std::fs::remove_file(&staging);
        return Err(error);
    }
    publish_staged(&staging, path, bytes)
}

fn publish_staged(staging: &Path, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Err(error) = std::fs::rename(staging, path) {
        let _ = std::fs::remove_file(staging);
        if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(windows)]
    use std::os::windows::fs::OpenOptionsExt;

    /// Two publishers staging the same destination at once both succeed,
    /// and neither leaves a staging file behind.
    #[test]
    fn concurrent_publishers_do_not_share_a_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let bytes = b"a prepared image".to_vec();
        let (first, second) = (path.clone(), path.clone());
        let (first_bytes, second_bytes) = (bytes.clone(), bytes.clone());
        let (one, two) = (
            std::thread::spawn(move || write_if_changed(&first, &first_bytes)),
            std::thread::spawn(move || write_if_changed(&second, &second_bytes)),
        );
        one.join().unwrap().unwrap();
        two.join().unwrap().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != path)
            .collect();
        assert!(
            leftovers.is_empty(),
            "staging files were removed: {leftovers:?}"
        );
    }

    #[test]
    fn a_failed_publication_removes_its_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing-parent").join("image.bin");
        assert!(write_if_changed(&path, b"bytes").is_err());
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert!(leftovers.is_empty(), "no staging file survives an error");
    }

    #[cfg(windows)]
    #[test]
    fn concurrent_rename_accepts_identical_locked_destination() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let staging = dir.path().join("staging.tmp");
        let bytes = b"a prepared image";
        std::fs::write(&path, bytes).unwrap();
        let _reader = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(&path)
            .unwrap();
        std::fs::write(&staging, bytes).unwrap();

        publish_staged(&staging, &path, bytes).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(!staging.exists());
    }

    #[cfg(windows)]
    #[test]
    fn failed_rename_with_different_locked_destination_still_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        std::fs::write(&path, b"other").unwrap();
        let _reader = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(&path)
            .unwrap();

        assert!(write_if_changed(&path, b"wanted").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"other");
        let mut entries = std::fs::read_dir(dir.path()).unwrap();
        assert_eq!(entries.next().unwrap().unwrap().path(), path);
        assert!(entries.next().is_none());
    }
}
