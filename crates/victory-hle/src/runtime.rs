//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

impl Title {
    /// Id, store id and storage quota from the manifest (zero or empty
    /// where it has none); the storage folder is left for the caller.
    pub fn from_manifest(text: &str) -> Title {
        let elements = elements(text);
        let get = |path: &str| {
            elements
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, text)| text.as_str())
        };
        let id = get("Game/TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb = get("Game/PersistentLocalStorage/SizeMB").and_then(|v| v.parse().ok());
        Title {
            id: id.unwrap_or(0),
            store_id: get("Game/StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        }
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

/// The trimmed text of one element: comments inside it are skipped, not
/// treated as the end of its value. A manifest generator can write
/// `<TitleId><!-- generated -->639AA8DD</TitleId>`, which must still give
/// the title's id rather than an empty field.
fn field_text(rest: &str) -> String {
    let mut out = String::new();
    let mut rest = rest;
    loop {
        let Some(open) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..open]);
        if rest[open..].starts_with("<!--") {
            let Some(end) = rest[open..].find("-->") else {
                break;
            };
            rest = &rest[open + end + 3..];
        } else {
            // A child element, a closing tag or anything else ends the text.
            break;
        }
    }
    out.trim().to_owned()
}

/// One predefined entity or XML 1.0 character reference, without `&` / `;`.
fn reference(text: &str) -> Option<char> {
    match text {
        "lt" => return Some('<'),
        "gt" => return Some('>'),
        "quot" => return Some('"'),
        "apos" => return Some('\''),
        "amp" => return Some('&'),
        _ => {}
    }
    let number = text.strip_prefix('#')?;
    let (digits, radix) = number
        .strip_prefix('x')
        .map_or((number, 10), |hex| (hex, 16));
    if digits.is_empty()
        || !digits.bytes().all(|b| {
            if radix == 16 {
                b.is_ascii_hexdigit()
            } else {
                b.is_ascii_digit()
            }
        })
    {
        return None;
    }
    let value = u32::from_str_radix(digits, radix).ok()?;
    // XML's Char production excludes NUL, most controls and U+FFFE/U+FFFF,
    // as well as surrogate code points and values beyond Unicode.
    match value {
        0x9 | 0xa | 0xd | 0x20..=0xd7ff | 0xe000..=0xfffd | 0x10000..=0x10ffff => {
            char::from_u32(value)
        }
        _ => None,
    }
}

/// Decode only references in the input, never the text they produce.
fn decode_references(text: &str) -> String {
    let mut parts = text.split('&');
    let mut out = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        if let Some((name, tail)) = part.split_once(';')
            && let Some(ch) = reference(name)
        {
            out.push(ch);
            out.push_str(tail);
        } else {
            out.push('&');
            out.push_str(part);
        }
    }
    out
}

/// Each element of an XML text as its path from the root (`Game/TitleId`)
/// and its trimmed text up to the first child, with predefined entities and
/// character references decoded; comments, CDATA sections, declarations and
/// attributes are skipped.
fn elements(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let end = if rest.starts_with("<!--") {
            rest.find("-->").map(|e| e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest.find("]]>").map(|e| e + 3)
        } else {
            tag_end(rest)
        };
        let Some(end) = end else { break };
        let tag = &rest[1..end - 1];
        rest = &rest[end..];
        if tag.starts_with(['!', '?']) {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if let Some(at) = path.iter().rposition(|n| *n == name.trim()) {
                path.truncate(at);
            }
            continue;
        }
        let closed = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        path.push(name);
        // Text up to the first child belongs to this element. A self-closing
        // element has none, so whatever follows it belongs to its parent.
        let body = if closed {
            String::new()
        } else {
            decode_references(&field_text(rest)).trim().to_owned()
        };
        out.push((path.join("/"), body));
        if closed {
            path.pop();
        }
    }
    out
}

/// Where the tag that starts `text` ends, past its `>`; a `>` in a quoted
/// attribute value is part of the value (XML allows it there unescaped).
fn tag_end(text: &str) -> Option<usize> {
    let mut quote = None;
    for (at, c) in text.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(open), _) if c == open => quote = None,
            (None, '>') => return Some(at + 1),
            _ => {}
        }
    }
    None
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (Grand Theft Auto VI's audio mixer sleeps
/// about 4 ms per 5.3 ms block) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
        timeBeginPeriod(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The body recorded for `path`, if any element has it.
    fn body<'a>(elements: &'a [(String, String)], path: &str) -> Option<&'a str> {
        elements
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, text)| text.as_str())
    }

    #[test]
    fn a_quoted_angle_bracket_does_not_end_its_tag() {
        let text = "<Game><ShellVisuals DefaultDisplayName=\"A > B\" Description='x>y'/>\
                    <TitleId>639AA8DD</TitleId>\
                    <PersistentLocalStorage><SizeMB>64</SizeMB></PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 64 << 20));
    }

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>639AA8DD</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert_eq!(Title::from_manifest("").storage_bytes, 0);
    }

    #[test]
    fn inline_comments_do_not_truncate_a_field() {
        let text = "<Game>\
            <TitleId><!-- generated -->639AA8DD</TitleId>\
            <StoreId>9&amp;<!-- keep -->C</StoreId>\
            <PersistentLocalStorage><SizeMB><!-- n -->1024</SizeMB></PersistentLocalStorage>\
            </Game>";
        let t = Title::from_manifest(text);
        assert_eq!(t.id, 0x639A_A8DD);
        assert_eq!(t.store_id, "9&C");
        assert_eq!(t.storage_bytes, 1024 << 20);
    }

    #[test]
    fn elements_record_the_path_of_each_nested_element() {
        let found = elements("<A><B><C>x</C></B></A>");
        for (path, text) in [("A", ""), ("A/B", ""), ("A/B/C", "x")] {
            assert_eq!(body(&found, path), Some(text), "{path}");
        }
        assert_eq!(found.len(), 3, "one entry per element");
    }

    #[test]
    fn a_self_closing_element_owns_no_text() {
        // The words after `/>` belong to the parent, not to `Flag`.
        let found = elements("<Game><Flag/>owned by the parent<Name>ok</Name></Game>");
        assert_eq!(body(&found, "Game/Flag"), Some(""));
        assert_eq!(body(&found, "Game/Name"), Some("ok"));
        // Neighbours with nothing between them are still empty.
        let neighbours = elements("<R><A/><B/></R>");
        assert_eq!(body(&neighbours, "R/A"), Some(""));
        assert_eq!(body(&neighbours, "R/B"), Some(""));
    }

    #[test]
    fn comments_cdata_and_declarations_are_skipped() {
        let found =
            elements("<?xml version=\"1.0\"?><G><!-- <X>c</X> --><![CDATA[<Y>d</Y>]]><Z>1</Z></G>");
        assert_eq!(body(&found, "G/X"), None, "inside a comment");
        assert_eq!(body(&found, "G/Y"), None, "inside CDATA");
        assert_eq!(body(&found, "G/Z"), Some("1"));
    }

    #[test]
    fn predefined_entities_are_decoded_once() {
        assert_eq!(
            body(&elements("<A>&lt;&gt;&amp;&quot;&apos;</A>"), "A"),
            Some("<>&\"'")
        );
        // One pass: `&amp;lt;` is the text `&lt;`, not a second `<`.
        assert_eq!(body(&elements("<A>&amp;lt;</A>"), "A"), Some("&lt;"));
    }

    #[test]
    fn unterminated_markup_is_ignored_without_panicking() {
        let found = elements("<A>x<B");
        assert_eq!(body(&found, "A"), Some("x"));
        assert_eq!(body(&found, "A/B"), None, "the <?B has no >");
        assert!(elements("<").is_empty());
    }

    #[test]
    fn missing_manifest_yields_defaults() {
        for text in [
            "",
            "<Game></Game>",
            "<NotGame><TitleId>1</TitleId></NotGame>",
        ] {
            let title = Title::from_manifest(text);
            assert_eq!(
                (title.id, title.store_id.as_str(), title.storage_bytes),
                (0, "", 0),
                "{text:?}"
            );
            assert_eq!(title.storage, PathBuf::new(), "{text:?}");
        }
    }

    #[test]
    fn lowercase_hex_id_and_quota_in_bytes() {
        let title = Title::from_manifest(
            "<Game><TitleId>deadbeef</TitleId>\
             <PersistentLocalStorage><SizeMB>2</SizeMB></PersistentLocalStorage></Game>",
        );
        assert_eq!(title.id, 0xDEAD_BEEF);
        assert_eq!(title.storage_bytes, 2 << 20);
    }

    #[test]
    fn invalid_id_and_missing_quota_are_zero() {
        let title = Title::from_manifest("<Game><TitleId>0x12</TitleId></Game>");
        assert_eq!((title.id, title.storage_bytes), (0, 0));
    }

    #[test]
    fn store_id_is_trimmed_and_entity_decoded() {
        let title = Title::from_manifest("<Game><StoreId>  a&amp;b  </StoreId></Game>");
        assert_eq!(title.store_id, "a&b");
    }

    #[test]
    fn commented_and_cdata_ids_are_ignored() {
        let title = Title::from_manifest(
            "<Game><!--<TitleId>1</TitleId>--><![CDATA[<StoreId>x</StoreId>]]>\
             <TitleId>2</TitleId></Game>",
        );
        assert_eq!(title.id, 2);
        assert_eq!(title.store_id, "");
    }

    #[test]
    fn the_first_element_at_a_path_wins() {
        let title = Title::from_manifest("<Game><TitleId>1</TitleId><TitleId>2</TitleId></Game>");
        assert_eq!(title.id, 1);
    }

    #[test]
    fn saves_live_beside_storage() {
        assert_eq!(Title::default().saves(), PathBuf::from("saves"));
        let title = Title {
            storage: PathBuf::from("data/0000/storage"),
            ..Title::default()
        };
        assert_eq!(title.saves(), PathBuf::from("data/0000/saves"));
    }

    #[test]
    fn image_registry_remembers_replaces_and_forgets() {
        let base = 0x7DFF_5A5A_0001;
        let first = Path::new(r"C:\victory-test\victory-runtime-image.dll");
        let second = Path::new(r"D:\victory-test\victory-renamed-image.dll");
        let name = |path: &Path| path.file_name().unwrap().to_owned();

        set_image_path(base, first);
        assert_eq!(image_path(base).map(|p| name(&p)), Some(name(first)));
        assert_eq!(image_named("VICTORY-RUNTIME-IMAGE.DLL"), Some(base));

        // A second file at the same base replaces the first.
        set_image_path(base, second);
        assert_eq!(image_path(base).map(|p| name(&p)), Some(name(second)));
        assert_eq!(image_named("victory-runtime-image.dll"), None);
        assert_eq!(image_named("victory-renamed-image.dll"), Some(base));

        forget_image(base);
        assert_eq!(image_path(base), None);
        assert_eq!(image_named("victory-renamed-image.dll"), None);
    }

    #[test]
    fn numeric_references_preserve_manifest_fields() {
        let plain = Title::from_manifest(
            "<Game><TitleId>0012ABCD</TitleId><StoreId>store&amp;Ω🙂</StoreId>\
             <PersistentLocalStorage><SizeMB>2</SizeMB></PersistentLocalStorage></Game>",
        );
        let encoded = Title::from_manifest(
            "<Game><TitleId>&#32;&#48;&#x30;12ABCD&#x20;</TitleId>\
             <StoreId>store&#38;&#x3A9;&#128578;</StoreId>\
             <PersistentLocalStorage><SizeMB>&#x9;&#50;<!-- quota -->&#10;</SizeMB>\
             </PersistentLocalStorage></Game>",
        );
        assert_eq!(plain.id, 0x0012_ABCD);
        assert_eq!(plain.storage_bytes, 2 << 20);
        assert_eq!(encoded, plain);
    }

    #[test]
    fn manifest_references_are_decoded_once() {
        let title = Title::from_manifest(
            "<Game><StoreId>&amp;#65; &#38;lt; &#x26;amp; &#60;child&#62; \
             &lt;&gt;&amp;&quot;&apos;</StoreId></Game>",
        );
        assert_eq!(title.store_id, "&#65; &lt; &amp; <child> <>&\"'");
    }

    #[test]
    fn invalid_manifest_references_stay_literal() {
        for reference in [
            "&#;",
            "&#x;",
            "&#X41;",
            "&#+65;",
            "&#x+41;",
            "&#-1;",
            "&# 65;",
            "&#x1_0;",
            "&#0;",
            "&#1;",
            "&#xD800;",
            "&#xDFFF;",
            "&#xFFFE;",
            "&#xFFFF;",
            "&#x110000;",
            "&#4294967296;",
            "&#99999999999999999999;",
            "&#65",
            "&unknown;",
            "&;",
            "&",
        ] {
            let manifest = format!("<Game><StoreId>before{reference}after&amp;</StoreId></Game>");
            assert_eq!(
                Title::from_manifest(&manifest).store_id,
                format!("before{reference}after&"),
                "{reference}"
            );
        }
    }

    #[test]
    fn xml_character_reference_boundaries() {
        for value in [
            0x9, 0xa, 0xd, 0x20, 0xd7ff, 0xe000, 0xfffd, 0x10000, 0x10ffff,
        ] {
            let manifest = format!("<Game><StoreId>x&#{value};&#x{value:X};y</StoreId></Game>");
            let ch = char::from_u32(value).unwrap();
            assert_eq!(
                Title::from_manifest(&manifest).store_id,
                format!("x{ch}{ch}y"),
                "U+{value:04X}"
            );
        }
    }

    /// Both intents at once: references are decoded in an element's text, and
    /// a self-closing sibling still owns none of the text after it.
    #[test]
    fn references_and_self_closing_elements_combine() {
        let found = elements("<G><Name>x&#38;y&lt;z</Name><Flag/>&#65;<Tail>&#x42;</Tail></G>");
        assert_eq!(body(&found, "G/Name"), Some("x&y<z"));
        assert_eq!(
            body(&found, "G/Flag"),
            Some(""),
            "the text after `/>` is not its own"
        );
        assert_eq!(body(&found, "G/Tail"), Some("B"));
    }
}
