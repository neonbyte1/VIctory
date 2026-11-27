use std::process::Command;

#[test]
fn shader_effects_has_its_own_summary() {
    let dir = std::env::temp_dir().join(format!("victory-shader-effects-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let empty = dir.join("empty.txt");
    let fixture = dir.join("synthetic-effect.sgd5");
    let mut database = b"SGD5".to_vec();
    database.extend([0; 8]);
    database.extend([0; (2 + 6) * 4]); // No shared blocks, strings, or shaders.
    database.extend(1u32.to_le_bytes());
    database.push(5);
    database.extend_from_slice(b"test\0");
    database.extend(48u32.to_le_bytes()); // Six empty stage lists in the unpacked effect.
    database.extend([0; 6 * 4 + 0x50 + 12 * 4]);
    std::fs::write(&fixture, database).unwrap();
    std::fs::write(&empty, b"ordinary test input").unwrap();

    let run = |path: &std::path::Path, effects: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_victory"));
        command.arg("shader-info");
        if effects {
            command.arg("--effects");
        }
        command.arg(path).output().unwrap()
    };
    let found = run(&fixture, true);
    let missing = run(&empty, true);
    let regular = run(&empty, false);
    std::fs::remove_file(fixture).unwrap();
    std::fs::remove_file(empty).unwrap();
    std::fs::remove_dir(dir).unwrap();

    let text = String::from_utf8_lossy(&found.stdout);
    assert!(found.status.success(), "{found:?}");
    assert!(text.contains("effect test"), "{text}");
    assert!(text.contains("shader effect databases found: 1"), "{text}");
    assert!(!text.contains("containers found") && !text.contains("(none found:"), "{text}");
    let text = String::from_utf8_lossy(&missing.stdout);
    assert!(missing.status.success(), "{missing:?}");
    assert!(text.contains("shader effect databases found: 0"), "{text}");
    assert!(!text.contains("containers found"), "{text}");
    assert!(regular.status.success(), "{regular:?}");
    assert!(String::from_utf8_lossy(&regular.stdout).contains("containers found"));
}
