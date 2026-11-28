use object::LittleEndian as LE;
use object::Object;
use object::pe::{self, ImageNtHeaders64};
use object::read::pe::{
    ImageNtHeaders, ImageOptionalHeader, Import, ImportTable, ImportThunkList, PeFile64,
};

use crate::layout::{self, dir};
use crate::{MACHINE_AMD64, PeError};

/// Everything we need to know about a PE32+ image, extracted in one pass.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub machine: u16,
    /// `IMAGE_FILE_HEADER::Characteristics`.
    pub characteristics: u16,
    pub timestamp: u32,
    pub subsystem: u16,
    /// `IMAGE_OPTIONAL_HEADER64::DllCharacteristics`.
    pub dll_characteristics: u16,
    pub image_base: u64,
    pub size_of_image: u32,
    pub entry_rva: u32,
    pub stack_reserve: u64,
    pub stack_commit: u64,
    pub sections: Vec<SectionInfo>,
    pub imports: Vec<ImportModule>,
    pub delay_imports: Vec<ImportModule>,
    pub exports: Vec<ExportInfo>,
    pub tls: Option<TlsInfo>,
    pub has_relocations: bool,
    pub has_exception_table: bool,
    pub has_load_config: bool,
    pub codeview: Option<CodeViewInfo>,
}

#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub name: String,
    pub rva: u32,
    pub virtual_size: u32,
    pub file_offset: u32,
    pub file_size: u32,
    pub characteristics: u32,
}

/// One imported DLL and the functions taken from it.
#[derive(Debug, Clone)]
pub struct ImportModule {
    /// DLL name exactly as written in the image (case preserved).
    pub dll: String,
    pub entries: Vec<ImportEntry>,
}

#[derive(Debug, Clone)]
pub struct ImportEntry {
    pub symbol: ImportSymbol,
    /// RVA of the 8-byte IAT slot the loader must fill in.
    pub iat_slot_rva: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImportSymbol {
    Name { hint: u16, name: String },
    Ordinal(u16),
}

impl std::fmt::Display for ImportSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportSymbol::Name { name, .. } => f.write_str(name),
            ImportSymbol::Ordinal(o) => write!(f, "#{o}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportInfo {
    pub name: Option<String>,
    pub ordinal: u16,
    /// `None` for forwarded exports.
    pub rva: Option<u32>,
    /// `"dll.name"` or `"dll.#ordinal"` for forwarded exports.
    pub forward: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TlsInfo {
    /// RVA of `IMAGE_TLS_DIRECTORY64`.
    pub directory_rva: u32,
    /// VA (at the preferred base) of the callback array, 0 if none.
    pub callbacks_va: u64,
    /// Callback RVAs, in call order.
    pub callback_rvas: Vec<u32>,
}

/// CodeView (RSDS) record: identifies the exact build that produced an image.
#[derive(Debug, Clone)]
pub struct CodeViewInfo {
    pub pdb_path: String,
    pub guid: [u8; 16],
    pub age: u32,
}

impl CodeViewInfo {
    /// The key symbol servers use: GUID (Windows byte order) + age.
    pub fn symbol_key(&self) -> String {
        let g = &self.guid;
        let d1 = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        let d2 = u16::from_le_bytes([g[4], g[5]]);
        let d3 = u16::from_le_bytes([g[6], g[7]]);
        let mut s = format!("{d1:08X}{d2:04X}{d3:04X}");
        for b in &g[8..] {
            s.push_str(&format!("{b:02X}"));
        }
        s.push_str(&format!("{:X}", self.age));
        s
    }
}

/// Upper bound on TLS callbacks we are willing to walk (defends against garbage).
const MAX_TLS_CALLBACKS: usize = 1024;

/// Size of `IMAGE_TLS_DIRECTORY64`.
const TLS_DIRECTORY_SIZE: u32 = size_of::<pe::ImageTlsDirectory64>() as u32;

impl ImageInfo {
    /// Parses a complete PE32+ x86-64 image from its file bytes.
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let kind = object::FileKind::parse(data)?;
        if kind != object::FileKind::Pe64 {
            return Err(PeError::Unsupported(format!(
                "expected PE32+, found {kind:?}"
            )));
        }
        let file = PeFile64::parse(data)?;
        let nt = file.nt_headers();
        let fh = nt.file_header();
        let opt = nt.optional_header();

        let machine = fh.machine.get(LE).0;
        if machine != MACHINE_AMD64 {
            return Err(PeError::Unsupported(format!(
                "machine 0x{machine:04x} is not AMD64"
            )));
        }

        let sections: Vec<SectionInfo> = file
            .section_table()
            .iter()
            .map(|s| SectionInfo {
                name: String::from_utf8_lossy(s.raw_name()).into_owned(),
                rva: s.virtual_address.get(LE),
                virtual_size: s.virtual_size.get(LE),
                file_offset: s.pointer_to_raw_data.get(LE),
                file_size: s.size_of_raw_data.get(LE),
                characteristics: s.characteristics.get(LE).0,
            })
            .collect();

        // Validate section raw-data boundaries before accepting
        // the PE image.
        for section in &sections {
            if section.file_size == 0 {
                continue;
            }

            let start = u64::from(section.file_offset);
            let size = u64::from(section.file_size);

            let Some(end) = start.checked_add(size) else {
                return Err(PeError::Malformed(format!(
                    "section '{}' has invalid raw-data bounds",
                    section.name
                )));
            };

            if end > data.len() as u64 {
                return Err(PeError::Malformed(format!(
                    "section '{}' extends beyond file bounds",
                    section.name
                )));
            }
        }

        let dir_present = |index: usize| {
            file.data_directory(index)
                .is_some_and(|d| d.virtual_address.get(LE) != 0 && d.size.get(LE) != 0)
        };

        let imports = match file.import_table()? {
            Some(table) => parse_imports(&table, &sections, opt.size_of_image())?,
            None => Vec::new(),
        };
        let delay_imports = parse_delay_imports(&file, &sections, opt.size_of_image())?;
        let exports = parse_exports(&file, opt.image_base())?;
        let tls = parse_tls(&file, data, opt.image_base(), opt.size_of_image())?;
        let codeview = file.pdb_info()?.map(|cv| CodeViewInfo {
            pdb_path: String::from_utf8_lossy(cv.path()).into_owned(),
            guid: cv.guid(),
            age: cv.age(),
        });

        Ok(ImageInfo {
            machine,
            characteristics: fh.characteristics.get(LE).0,
            timestamp: fh.time_date_stamp.get(LE),
            subsystem: opt.subsystem().0,
            dll_characteristics: opt.dll_characteristics().0,
            image_base: opt.image_base(),
            size_of_image: opt.size_of_image(),
            entry_rva: opt.address_of_entry_point(),
            stack_reserve: opt.size_of_stack_reserve(),
            stack_commit: opt.size_of_stack_commit(),
            sections,
            imports,
            delay_imports,
            exports,
            tls,
            has_relocations: dir_present(dir::BASERELOC)
                && !has_flag(
                    fh.characteristics.get(LE).0,
                    layout::file_flags::RELOCS_STRIPPED,
                ),
            has_exception_table: dir_present(dir::EXCEPTION),
            has_load_config: dir_present(dir::LOAD_CONFIG),
            codeview,
        })
    }

    pub fn is_dll(&self) -> bool {
        has_flag(self.characteristics, layout::file_flags::DLL)
    }

    /// Maps an RVA to an offset in the file, if it is backed by file data.
    pub fn rva_to_file_offset(&self, rva: u32) -> Option<u32> {
        self.sections.iter().find_map(|s| {
            let delta = rva.checked_sub(s.rva)?;
            // Raw data may be padded past the virtual size; that padding is
            // not part of the section in memory.
            let backed = match s.virtual_size {
                0 => s.file_size,
                vs => vs.min(s.file_size),
            };
            // Header values are untrusted: an offset near 4 GiB overflows.
            (delta < backed).then(|| s.file_offset.checked_add(delta))?
        })
    }

    pub fn import_count(&self) -> usize {
        self.imports.iter().map(|m| m.entries.len()).sum()
    }

    pub fn subsystem_name(&self) -> &'static str {
        use layout::subsystem::*;
        match self.subsystem {
            WINDOWS_GUI => "windows-gui",
            WINDOWS_CUI => "windows-cui",
            XBOX => "xbox",
            XBOX_CODE_CATALOG => "xbox-code-catalog",
            _ => "other",
        }
    }
}

fn has_flag(value: u16, flag: u16) -> bool {
    value & flag != 0
}

fn utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn convert_import(import: Import<'_>) -> ImportSymbol {
    match import {
        Import::Ordinal(o) => ImportSymbol::Ordinal(o),
        Import::Name(hint, name) => ImportSymbol::Name {
            hint,
            name: utf8(name),
        },
    }
}

fn parse_imports(
    table: &ImportTable<'_>,
    sections: &[SectionInfo],
    size_of_image: u32,
) -> Result<Vec<ImportModule>, PeError> {
    let mut modules = Vec::new();
    let mut descriptors = table.descriptors()?;
    while let Some(desc) = descriptors.next()? {
        let iat_rva = desc.first_thunk.get(LE);
        // Prefer the unbound name table; fall back to the IAT for old linkers.
        let int_rva = match desc.original_first_thunk.get(LE) {
            0 => iat_rva,
            rva => rva,
        };
        modules.push(ImportModule {
            dll: utf8(table.name(desc.name.get(LE))?),
            entries: entries(
                table.thunks(int_rva)?,
                iat_rva,
                sections,
                size_of_image,
                |t| table.import::<ImageNtHeaders64>(t),
            )?,
        });
    }
    Ok(modules)
}

fn parse_delay_imports(
    file: &PeFile64<'_>,
    sections: &[SectionInfo],
    size_of_image: u32,
) -> Result<Vec<ImportModule>, PeError> {
    let Some(table) = file.delay_load_import_table()? else {
        return Ok(Vec::new());
    };
    let mut modules = Vec::new();
    let mut descriptors = table.descriptors()?;
    while let Some(desc) = descriptors.next()? {
        modules.push(ImportModule {
            dll: utf8(table.name(desc.dll_name_rva.get(LE))?),
            entries: entries(
                table.thunks(desc.import_name_table_rva.get(LE))?,
                desc.import_address_table_rva.get(LE),
                sections,
                size_of_image,
                |t| table.import::<ImageNtHeaders64>(t),
            )?,
        });
    }
    Ok(modules)
}

/// Returns whether the 8-byte IAT slot at `rva` lies inside one file-backed
/// section.
fn iat_slot_in_section(sections: &[SectionInfo], rva: u32) -> bool {
    sections.iter().any(|s| {
        let Some(delta) = rva.checked_sub(s.rva) else {
            return false;
        };
        let backed = match s.virtual_size {
            0 => s.file_size,
            vs => vs.min(s.file_size),
        };
        delta.checked_add(8).is_some_and(|end| end <= backed)
            && s.file_offset
                .checked_add(delta)
                .and_then(|offset| offset.checked_add(7))
                .is_some()
    })
}

/// Pairs each name-table entry with its IAT slot. The IAT RVA is not checked
/// by `object`, so every slot is checked to lie inside a file-backed section
/// within the image here.
fn entries<'data>(
    mut thunks: ImportThunkList<'data>,
    iat_rva: u32,
    sections: &[SectionInfo],
    size_of_image: u32,
    import: impl Fn(pe::ImageThunkData64) -> object::read::Result<Import<'data>>,
) -> Result<Vec<ImportEntry>, PeError> {
    let mut entries = Vec::new();
    while let Some(thunk) = thunks.next::<ImageNtHeaders64>()? {
        let iat_slot_rva = u32::try_from(entries.len() * 8)
            .ok()
            .and_then(|offset| iat_rva.checked_add(offset))
            .filter(|&rva| {
                rva.checked_add(8).is_some_and(|end| end <= size_of_image)
                    && iat_slot_in_section(sections, rva)
            })
            .ok_or_else(|| {
                PeError::Malformed(format!(
                    "IAT at {iat_rva:#x} runs outside a file-backed section"
                ))
            })?;
        entries.push(ImportEntry {
            symbol: convert_import(import(thunk)?),
            iat_slot_rva,
        });
    }
    Ok(entries)
}

fn parse_exports(file: &PeFile64<'_>, image_base: u64) -> Result<Vec<ExportInfo>, PeError> {
    use object::read::{ExportFlags, ExportTarget as ReadTarget, NameOrOrdinal};

    // The PE-specific table keeps one name per address-table slot, so two
    // names sharing an ordinal lose all but the last one. The generic
    // iterator yields every name, then the ordinal-only entries.
    file.exports()
        .map_err(|e| PeError::Malformed(format!("export table: {e}")))?
        .map(|e| {
            let e = e.map_err(|e| PeError::Malformed(format!("export: {e}")))?;
            let ordinal = match e.flags() {
                ExportFlags::Pe { ordinal } => ordinal.0,
                _ => 0,
            };
            let name = match e.name() {
                NameOrOrdinal::Name(name) => Some(utf8(name)),
                NameOrOrdinal::Ordinal(_) => None,
            };
            let (rva, forward) = match e.target() {
                // The generic iterator adds the image base to an address.
                ReadTarget::Address { address } => {
                    (Some(address.wrapping_sub(image_base) as u32), None)
                }
                ReadTarget::Reexport { library, name } => {
                    let symbol = match name {
                        NameOrOrdinal::Name(name) => utf8(name),
                        NameOrOrdinal::Ordinal(ordinal) => format!("#{ordinal}"),
                    };
                    (None, Some(format!("{}.{}", utf8(library), symbol)))
                }
                _ => (None, None),
            };
            Ok(ExportInfo {
                name,
                ordinal,
                rva,
                forward,
            })
        })
        .collect()
}

fn parse_tls(
    file: &PeFile64<'_>,
    data: &[u8],
    image_base: u64,
    size_of_image: u32,
) -> Result<Option<TlsInfo>, PeError> {
    let Some(dd) = file.data_directory(pe::IMAGE_DIRECTORY_ENTRY_TLS) else {
        return Ok(None);
    };
    let directory_rva = dd.virtual_address.get(LE);
    if directory_rva == 0 {
        return Ok(None);
    }
    // Header values are untrusted. The loader reads the directory's index
    // field from the mapped image, so all of the directory must lie inside it.
    if directory_rva
        .checked_add(TLS_DIRECTORY_SIZE)
        .is_none_or(|end| end > size_of_image)
    {
        return Err(PeError::Malformed(format!(
            "TLS directory at {directory_rva:#x} runs outside the image"
        )));
    }
    let sections = file.section_table();
    let tls = sections
        .pe_data_at(data, directory_rva)
        .ok_or_else(|| PeError::Malformed("TLS directory outside of sections".into()))?;
    let callbacks_va = layout::read_u64(tls, layout::TLS64_ADDRESS_OF_CALLBACKS)
        .ok_or_else(|| PeError::Malformed("truncated TLS directory".into()))?;

    let mut callback_rvas = Vec::new();
    if callbacks_va != 0 {
        let array_rva = va_to_rva(callbacks_va, image_base, size_of_image)?;
        let array = sections
            .pe_data_at(data, array_rva)
            .ok_or_else(|| PeError::Malformed("TLS callback array outside of sections".into()))?;
        for slot in array.as_chunks::<8>().0.iter().take(MAX_TLS_CALLBACKS) {
            let va = u64::from_le_bytes(*slot);
            if va == 0 {
                break;
            }
            callback_rvas.push(va_to_rva(va, image_base, size_of_image)?);
        }
    }
    Ok(Some(TlsInfo {
        directory_rva,
        callbacks_va,
        callback_rvas,
    }))
}

fn va_to_rva(va: u64, image_base: u64, size_of_image: u32) -> Result<u32, PeError> {
    va.checked_sub(image_base)
        .and_then(|rva| u32::try_from(rva).ok())
        .filter(|&rva| rva < size_of_image)
        .ok_or_else(|| PeError::Malformed(format!("VA 0x{va:x} is outside the image")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test binary itself is a real MSVC-linked PE32+ image.
    fn own_image() -> Vec<u8> {
        std::fs::read(std::env::current_exe().unwrap()).unwrap()
    }

    #[test]
    fn parses_own_test_binary() {
        let info = ImageInfo::parse(&own_image()).unwrap();
        assert_eq!(info.machine, MACHINE_AMD64);
        assert!(!info.is_dll());
        assert!(info.entry_rva != 0);
        assert!(info.has_exception_table, "x64 images always carry .pdata");
        assert!(info.sections.iter().any(|s| s.name.starts_with(".text")));
        let kernel32 = info
            .imports
            .iter()
            .find(|m| m.dll.eq_ignore_ascii_case("kernel32.dll"))
            .expect("Rust std imports kernel32");
        assert!(kernel32.entries.iter().all(|e| e.iat_slot_rva != 0));
        // Rust std uses static TLS, so a TLS directory must be present (its
        // callback array may legitimately be empty).
        assert!(info.tls.is_some());
    }

    #[test]
    fn rva_to_file_offset_maps_entry_point() {
        let data = own_image();
        let info = ImageInfo::parse(&data).unwrap();
        let off = info.rva_to_file_offset(info.entry_rva).unwrap() as usize;
        assert!(off < data.len());
    }

    #[test]
    fn rva_to_file_offset_rejects_offsets_past_4_gib() {
        let mut info = section_image(0x200, 0x200);
        assert_eq!(info.rva_to_file_offset(0x1020), Some(0x220));
        // Used to overflow (a panic in debug builds, a wrong offset in
        // release ones).
        info.sections[0].file_offset = u32::MAX - 0x10;
        assert_eq!(info.rva_to_file_offset(0x1000), Some(u32::MAX - 0x10));
        assert_eq!(info.rva_to_file_offset(0x1010), Some(u32::MAX));
        assert_eq!(info.rva_to_file_offset(0x1011), None);
        assert_eq!(info.rva_to_file_offset(0x1020), None);
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        assert!(ImageInfo::parse(b"").is_err());
        assert!(ImageInfo::parse(b"MZ not really").is_err());
        let mut truncated = own_image();
        truncated.truncate(0x200);
        assert!(ImageInfo::parse(&truncated).is_err());
    }

    #[test]
    fn rejects_iat_outside_image() {
        let mut data = own_image();
        let dd = layout::data_directory_offset(&data, dir::IMPORT).unwrap();
        let import_rva = layout::read_u32(&data, dd).unwrap();
        let desc = ImageInfo::parse(&data)
            .unwrap()
            .rva_to_file_offset(import_rva)
            .unwrap() as usize;
        // First descriptor's FirstThunk, at the 4 GiB limit: used to overflow.
        data[desc + 16..desc + 20].copy_from_slice(&0xFFFF_FFF8u32.to_le_bytes());
        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(_))
        ));
    }

    #[test]
    fn rejects_null_iat_for_regular_imports() {
        let mut data = own_image();
        let dd = layout::data_directory_offset(&data, dir::IMPORT).unwrap();
        let import_rva = layout::read_u32(&data, dd).unwrap();
        let info = ImageInfo::parse(&data).unwrap();
        let desc = info.rva_to_file_offset(import_rva).unwrap() as usize;
        assert_ne!(layout::read_u32(&data, desc).unwrap(), 0); // OriginalFirstThunk
        assert!(!info.imports[0].entries.is_empty());

        data[desc + 16..desc + 20].fill(0); // FirstThunk
        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(_))
        ));
    }

    #[test]
    fn rejects_header_iat_for_regular_imports() {
        let mut data = own_image();
        let dd = layout::data_directory_offset(&data, dir::IMPORT).unwrap();
        let import_rva = layout::read_u32(&data, dd).unwrap();
        let info = ImageInfo::parse(&data).unwrap();
        let desc = info.rva_to_file_offset(import_rva).unwrap() as usize;
        assert_ne!(layout::read_u32(&data, desc).unwrap(), 0); // OriginalFirstThunk
        assert!(!info.imports[0].entries.is_empty());

        data[desc + 16..desc + 20].copy_from_slice(&0x40u32.to_le_bytes()); // FirstThunk in headers
        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(_))
        ));
    }

    /// A minimal PE32+ image: one section at RVA 0x1000 holding `body`, with
    /// the export directory (RVA 0x1000, 0x80 bytes) pointing at it.
    fn synthetic_image(body: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; 0x200];
        d[0..2].copy_from_slice(b"MZ");
        d[0x3C..0x40].copy_from_slice(&0x40u32.to_le_bytes());
        d[0x40..0x44].copy_from_slice(b"PE\0\0");
        d[0x44..0x46].copy_from_slice(&MACHINE_AMD64.to_le_bytes());
        d[0x46..0x48].copy_from_slice(&1u16.to_le_bytes()); // sections
        d[0x54..0x56].copy_from_slice(&240u16.to_le_bytes()); // optional header size
        d[0x56..0x58].copy_from_slice(&0x2022u16.to_le_bytes()); // DLL, large address aware
        let opt = 0x58;
        d[opt..opt + 2].copy_from_slice(&0x20Bu16.to_le_bytes());
        d[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes());
        d[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        d[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes());
        d[opt + 56..opt + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        d[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes());
        d[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes());
        d[opt + 112..opt + 116].copy_from_slice(&0x1000u32.to_le_bytes());
        d[opt + 116..opt + 120].copy_from_slice(&0x80u32.to_le_bytes());
        let sec = opt + 240;
        d[sec..sec + 6].copy_from_slice(b".edata");
        d[sec + 8..sec + 12].copy_from_slice(&0x200u32.to_le_bytes()); // virtual size
        d[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        d[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // raw size
        d[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // raw offset
        d.extend_from_slice(body);
        d.resize(0x400, 0);
        d
    }

    /// An export-free section at RVA 0x1000, backed at file offset 0x200.
    fn section_image(virtual_size: u32, raw_size: u32) -> ImageInfo {
        let mut data = synthetic_image(&[0; 0x200]);
        let opt = layout::optional_header_offset(&data).unwrap();
        let export = layout::data_directory_offset(&data, 0).unwrap();
        data[export..export + 8].fill(0);
        let sec = opt + 240;
        data[sec + 8..sec + 12].copy_from_slice(&virtual_size.to_le_bytes());
        data[sec + 16..sec + 20].copy_from_slice(&raw_size.to_le_bytes());
        ImageInfo::parse(&data).unwrap()
    }

    #[test]
    fn section_rvas_stop_at_the_shorter_of_virtual_and_raw_data() {
        for (virtual_size, raw_size) in [(0x80, 0x200), (0x200, 0x80)] {
            let info = section_image(virtual_size, raw_size);
            // Raw padding beyond VirtualSize and a zero-filled virtual
            // tail beyond SizeOfRawData are both outside the file mapping.
            for (rva, expected) in [
                (0x0FFF, None),
                (0x1000, Some(0x200)),
                (0x107F, Some(0x27F)),
                (0x1080, None),
                (0x11FF, None),
                (0x1200, None),
            ] {
                assert_eq!(
                    info.rva_to_file_offset(rva),
                    expected,
                    "RVA {rva:#x}, virtual {virtual_size:#x}, raw {raw_size:#x}"
                );
            }
        }
    }

    #[test]
    fn section_rvas_use_raw_size_when_virtual_size_is_zero() {
        let info = section_image(0, 0x200);
        assert_eq!(info.rva_to_file_offset(0x1000), Some(0x200));
        assert_eq!(info.rva_to_file_offset(0x11FF), Some(0x3FF));
        assert_eq!(info.rva_to_file_offset(0x1200), None);
    }

    #[test]
    fn section_rvas_skip_empty_sections_and_gaps() {
        let mut info = section_image(0x180, 0x80);
        let mut empty = info.sections[0].clone();
        empty.rva = 0x2000;
        empty.file_size = 0;
        let mut later = info.sections[0].clone();
        later.rva = 0x3000;
        later.file_offset = 0x280;
        info.sections.extend([empty, later]);
        for _ in 0..2 {
            for (rva, expected) in [
                (0x1000, Some(0x200)),
                (0x107F, Some(0x27F)),
                (0x1080, None),
                (0x2000, None),
                (0x217F, None),
                (0x2FFF, None),
                (0x3000, Some(0x280)),
                (0x307F, Some(0x2FF)),
                (0x3080, None),
            ] {
                assert_eq!(info.rva_to_file_offset(rva), expected, "RVA {rva:#x}");
            }
            // Lookup must not depend on section-table order.
            info.sections.reverse();
        }
    }

    #[test]
    fn section_rvas_near_u32_max_do_not_wrap() {
        let mut info = section_image(0x200, 0x200);
        info.sections[0].rva = u32::MAX - 0x1F;
        assert_eq!(info.rva_to_file_offset(u32::MAX - 0x1F), Some(0x200));
        assert_eq!(info.rva_to_file_offset(u32::MAX), Some(0x21F));
        assert_eq!(info.rva_to_file_offset(u32::MAX - 0x20), None);
        assert_eq!(info.rva_to_file_offset(0), None);
    }

    #[test]
    fn rejects_null_iat_for_delay_imports() {
        let mut body = [0u8; 0x80];
        let delay_rva = 0x1000u32;
        let iat_rva = 0x1060u32;
        let name_rva = 0x1070u64;
        body[0..4].copy_from_slice(&1u32.to_le_bytes()); // RVA-based descriptor
        body[4..8].copy_from_slice(&0x1040u32.to_le_bytes()); // DllNameRVA
        body[12..16].copy_from_slice(&iat_rva.to_le_bytes()); // ImportAddressTableRVA
        body[16..20].copy_from_slice(&0x1050u32.to_le_bytes()); // ImportNameTableRVA
        body[0x40..0x49].copy_from_slice(b"test.dll\0");
        body[0x50..0x58].copy_from_slice(&name_rva.to_le_bytes());
        body[0x60..0x68].copy_from_slice(&name_rva.to_le_bytes());
        body[0x72..0x77].copy_from_slice(b"test\0"); // name after the zero hint

        let mut data = synthetic_image(&body);
        let export_dd =
            layout::data_directory_offset(&data, pe::IMAGE_DIRECTORY_ENTRY_EXPORT).unwrap();
        data[export_dd..export_dd + 8].fill(0);
        let delay_dd =
            layout::data_directory_offset(&data, pe::IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT).unwrap();
        data[delay_dd..delay_dd + 4].copy_from_slice(&delay_rva.to_le_bytes());
        data[delay_dd + 4..delay_dd + 8].copy_from_slice(&0x40u32.to_le_bytes());

        let info = ImageInfo::parse(&data).unwrap();
        assert_eq!(info.delay_imports.len(), 1);
        let module = &info.delay_imports[0];
        assert_eq!(module.dll, "test.dll");
        assert_eq!(module.entries.len(), 1);
        assert_eq!(module.entries[0].iat_slot_rva, iat_rva);

        let desc = info.rva_to_file_offset(delay_rva).unwrap() as usize;
        data[desc + 12..desc + 16].fill(0); // ImportAddressTableRVA
        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(_))
        ));
    }

    #[test]
    fn rejects_section_extending_beyond_file() {
        // Create a synthetic PE image.
        let mut data = synthetic_image(&vec![0u8; 0x200]);

        // Remove the export directory to isolate section validation.
        let opt = 0x58;
        data[opt + 112..opt + 120].fill(0);

        // Verify the baseline PE is accepted.
        assert!(
            ImageInfo::parse(&data).is_ok(),
            "baseline synthetic image must be valid"
        );

        // The section header starts after the 240-byte optional header.
        let sec = opt + 240;

        // Declare 4096 bytes of raw section data.
        // The actual file contains only 1024 bytes.
        data[sec + 16..sec + 20].copy_from_slice(&0x1000u32.to_le_bytes());

        // Investigate whether the parser rejects this malformed image.
        assert!(
            ImageInfo::parse(&data).is_err(),
            "section extends beyond physical file bounds"
        );
    }

    #[test]
    fn accepts_section_ending_at_file_boundary() {
        let mut data = synthetic_image(&vec![0u8; 0x200]);

        // Disable the export directory to isolate section validation.
        let opt = 0x58;
        data[opt + 112..opt + 120].fill(0);

        // Raw section: offset 0x200 + size 0x200 = file length 0x400.
        let info = ImageInfo::parse(&data);

        assert!(
            info.is_ok(),
            "section ending exactly at EOF should be accepted"
        );
    }

    #[test]
    fn accepts_section_with_zero_raw_size() {
        let mut data = synthetic_image(&vec![0u8; 0x200]);

        // Disable the export directory.
        let opt = 0x58;
        data[opt + 112..opt + 120].fill(0);

        // Section header begins after the optional header.
        let sec = opt + 240;

        // A section may occupy virtual memory without having raw data.
        data[sec + 16..sec + 20].copy_from_slice(&0u32.to_le_bytes());

        let info = ImageInfo::parse(&data);

        assert!(
            info.is_ok(),
            "section with zero raw size should be accepted"
        );
    }

    /// `synthetic_image` with no exports, its section moved to end with the
    /// image (RVA 0x1E00..0x2000), and a TLS directory without callbacks
    /// declared at `directory_rva`.
    fn image_with_tls_directory(directory_rva: u32) -> Vec<u8> {
        let mut data = synthetic_image(&[0; 0x200]);
        let opt = 0x58;
        let export = opt + layout::OPT_DATA_DIRECTORY;
        data[export..export + layout::DATA_DIRECTORY_SIZE].fill(0);
        let tls = export + pe::IMAGE_DIRECTORY_ENTRY_TLS * layout::DATA_DIRECTORY_SIZE;
        data[tls..tls + 4].copy_from_slice(&directory_rva.to_le_bytes());
        data[tls + 4..tls + 8].copy_from_slice(&TLS_DIRECTORY_SIZE.to_le_bytes());
        let sec = opt + 240;
        data[sec + 12..sec + 16].copy_from_slice(&0x1E00u32.to_le_bytes());
        data
    }

    #[test]
    fn accepts_tls_directory_ending_at_the_image_end() {
        // 40 bytes from 0x1FD8 end at 0x2000, the size of the image.
        let data = image_with_tls_directory(0x1FD8);

        let tls = ImageInfo::parse(&data).unwrap().tls.unwrap();

        assert_eq!(tls.directory_rva, 0x1FD8);
        assert_eq!(tls.callbacks_va, 0);
    }

    #[test]
    fn rejects_tls_directory_running_past_the_image_end() {
        let data = image_with_tls_directory(0x1FD9);

        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(reason))
                if reason == "TLS directory at 0x1fd9 runs outside the image"
        ));
    }

    #[test]
    fn rejects_tls_directory_running_past_the_rva_range() {
        let mut data = image_with_tls_directory(0xFFFF_FFF0);
        // The largest image, and the section under the directory: its end is
        // past 4 GiB, which no image size admits.
        let opt = 0x58;
        data[opt + 56..opt + 60].copy_from_slice(&u32::MAX.to_le_bytes());
        let sec = opt + 240;
        data[sec + 12..sec + 16].copy_from_slice(&0xFFFF_FF00u32.to_le_bytes());

        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(reason))
                if reason == "TLS directory at 0xfffffff0 runs outside the image"
        ));
    }

    #[test]
    fn unused_export_slots_are_not_exports() {
        // Ordinal base 1; three address slots: "alpha", an unused hole (0),
        // and "gamma". The Windows loader treats address 0 as "not exported".
        let mut body = vec![0u8; 0x100];
        let put32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut body, 12, 0x1040); // name of the DLL
        put32(&mut body, 16, 1); // ordinal base
        put32(&mut body, 20, 3); // number of functions
        put32(&mut body, 24, 2); // number of names
        put32(&mut body, 28, 0x1028); // address table
        put32(&mut body, 32, 0x1034); // name pointer table
        put32(&mut body, 36, 0x103C); // name ordinal table
        put32(&mut body, 0x28, 0x1200);
        put32(&mut body, 0x2C, 0);
        put32(&mut body, 0x30, 0x1204);
        put32(&mut body, 0x34, 0x1044);
        put32(&mut body, 0x38, 0x104A);
        body[0x3C..0x3E].copy_from_slice(&0u16.to_le_bytes());
        body[0x3E..0x40].copy_from_slice(&2u16.to_le_bytes());
        body[0x40..0x44].copy_from_slice(b"t.dl");
        body[0x44..0x4A].copy_from_slice(b"alpha\0");
        body[0x4A..0x50].copy_from_slice(b"gamma\0");

        let info = ImageInfo::parse(&synthetic_image(&body)).unwrap();
        let exports: Vec<_> = info
            .exports
            .iter()
            .map(|e| (e.name.as_deref(), e.ordinal, e.rva))
            .collect();
        assert_eq!(
            exports,
            [
                (Some("alpha"), 1, Some(0x1200)),
                (Some("gamma"), 3, Some(0x1204))
            ]
        );
    }

    #[test]
    fn every_name_of_a_shared_ordinal_survives() {
        // Two names for address slot 0, one for slot 1. The PE-specific
        // export table keeps only one name per slot, so "alpha" would be
        // lost; the generic iterator keeps both.
        let mut body = vec![0u8; 0x100];
        let put32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut body, 12, 0x1040); // name of the DLL
        put32(&mut body, 16, 1); // ordinal base
        put32(&mut body, 20, 2); // number of functions
        put32(&mut body, 24, 3); // number of names
        put32(&mut body, 28, 0x1028); // address table
        put32(&mut body, 32, 0x1034); // name pointer table
        put32(&mut body, 36, 0x1044); // name ordinal table
        put32(&mut body, 0x28, 0x1200);
        put32(&mut body, 0x2C, 0x1204);
        put32(&mut body, 0x34, 0x1050);
        put32(&mut body, 0x38, 0x1056);
        put32(&mut body, 0x3C, 0x105C);
        body[0x40..0x46].copy_from_slice(b"t.dll\0");
        body[0x44..0x46].copy_from_slice(&0u16.to_le_bytes());
        body[0x46..0x48].copy_from_slice(&0u16.to_le_bytes());
        body[0x48..0x4A].copy_from_slice(&1u16.to_le_bytes());
        body[0x50..0x56].copy_from_slice(b"alpha\0");
        body[0x56..0x5B].copy_from_slice(b"beta\0");
        body[0x5C..0x62].copy_from_slice(b"gamma\0");

        let info = ImageInfo::parse(&synthetic_image(&body)).unwrap();
        let exports: Vec<_> = info
            .exports
            .iter()
            .map(|e| (e.name.as_deref(), e.ordinal, e.rva))
            .collect();
        assert_eq!(
            exports,
            [
                (Some("alpha"), 1, Some(0x1200)),
                (Some("beta"), 1, Some(0x1200)),
                (Some("gamma"), 2, Some(0x1204)),
            ]
        );
    }

    #[test]
    fn rejects_tls_callback_outside_size_of_image() {
        let image_base = 0x1_8000_0000u64;
        let size_of_image = 0x2000u64;
        // Section starts at RVA 0x1000: place TLS directory at +0x00 and callback array at +0x40.
        let mut body = vec![0u8; 0x200];
        let callbacks_va = image_base + 0x1040;
        body[layout::TLS64_ADDRESS_OF_CALLBACKS..layout::TLS64_ADDRESS_OF_CALLBACKS + 8]
            .copy_from_slice(&callbacks_va.to_le_bytes());

        // Callback VA inside the image is accepted.
        body[0x40..0x48].copy_from_slice(&(image_base + 0x1100).to_le_bytes());
        let mut data = synthetic_image(&body);
        let opt = 0x58;
        data[opt + 112..opt + 120].fill(0); // clear export dir
        let tls_dd = opt + 112 + pe::IMAGE_DIRECTORY_ENTRY_TLS * 8;
        data[tls_dd..tls_dd + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        data[tls_dd + 4..tls_dd + 8].copy_from_slice(&40u32.to_le_bytes());
        let ok = ImageInfo::parse(&data).unwrap();
        assert_eq!(ok.tls.unwrap().callback_rvas, vec![0x1100]);

        // Callback VA at `image_base + size_of_image` (fits in u32 RVA, but outside image) fails.
        body[0x40..0x48].copy_from_slice(&(image_base + size_of_image).to_le_bytes());
        let mut bad = synthetic_image(&body);
        bad[opt + 112..opt + 120].fill(0);
        bad[tls_dd..tls_dd + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        bad[tls_dd + 4..tls_dd + 8].copy_from_slice(&40u32.to_le_bytes());
        assert!(matches!(ImageInfo::parse(&bad), Err(PeError::Malformed(_))));
    }

    /// `IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`.
    const DELAY_IMPORT: usize = 13;

    /// A synthetic image whose delay-load directory lists `delay.dll` with two
    /// imports, "Func" (hint 5) and ordinal 7, and whose IAT starts at
    /// `iat_rva`. The section is RVA 0x1000..0x1200 and the image ends at
    /// 0x2000.
    fn delay_image(iat_rva: u32) -> Vec<u8> {
        let mut body = vec![0u8; 0x200];
        let put32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        // The descriptor, then the all-zero one that ends the table.
        put32(&mut body, 0x00, 1); // attributes: RVAs, not VAs
        put32(&mut body, 0x04, 0x1040); // DLL name
        put32(&mut body, 0x0C, iat_rva); // import address table
        put32(&mut body, 0x10, 0x1060); // import name table
        body[0x40..0x4A].copy_from_slice(b"delay.dll\0");
        // The name table: "Func" by hint/name, ordinal 7, the terminator.
        body[0x60..0x68].copy_from_slice(&0x1100u64.to_le_bytes());
        body[0x68..0x70].copy_from_slice(&(1u64 << 63 | 7).to_le_bytes());
        body[0x100..0x102].copy_from_slice(&5u16.to_le_bytes());
        body[0x102..0x107].copy_from_slice(b"Func\0");

        let mut data = synthetic_image(&body);
        let opt = 0x58;
        // No export directory: the body is not one.
        data[opt + 112..opt + 120].fill(0);
        let dd = opt + 112 + DELAY_IMPORT * 8;
        data[dd..dd + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        data[dd + 4..dd + 8].copy_from_slice(&0x40u32.to_le_bytes());
        data
    }

    #[test]
    fn delay_imports_list_names_and_ordinals_with_their_iat_slots() {
        let info = ImageInfo::parse(&delay_image(0x1080)).unwrap();
        assert!(info.imports.is_empty(), "delay imports are not imports");
        assert_eq!(info.delay_imports.len(), 1);
        let module = &info.delay_imports[0];
        assert_eq!(module.dll, "delay.dll");
        let entries: Vec<_> = module
            .entries
            .iter()
            .map(|e| (e.symbol.clone(), e.iat_slot_rva))
            .collect();
        assert_eq!(
            entries,
            [
                (
                    ImportSymbol::Name {
                        hint: 5,
                        name: "Func".into()
                    },
                    0x1080
                ),
                (ImportSymbol::Ordinal(7), 0x1088),
            ]
        );
    }

    #[test]
    fn an_image_without_a_delay_load_directory_has_no_delay_imports() {
        let mut data = delay_image(0x1080);
        let dd = 0x58 + 112 + DELAY_IMPORT * 8;
        data[dd..dd + 8].fill(0);
        let info = ImageInfo::parse(&data).unwrap();
        assert!(info.delay_imports.is_empty());
    }

    #[test]
    fn a_delay_iat_ending_at_the_file_backed_end_of_the_section_is_accepted() {
        // Two 8-byte slots: the second ends exactly at the file-backed end of
        // the section (0x1200; the section is RVA 0x1000..0x1200).
        let info = ImageInfo::parse(&delay_image(0x1200 - 16)).unwrap();
        assert_eq!(info.delay_imports[0].entries[1].iat_slot_rva, 0x1200 - 8);
    }

    #[test]
    fn a_delay_iat_running_past_the_file_backed_section_is_rejected() {
        // One byte past the file-backed end of the section: the second slot
        // would end at 0x1201.
        assert!(matches!(
            ImageInfo::parse(&delay_image(0x1200 - 15)),
            Err(PeError::Malformed(_))
        ));
    }

    #[test]
    fn a_delay_iat_running_past_the_image_is_rejected() {
        // One byte too far: the second slot would end at 0x2001.
        assert!(matches!(
            ImageInfo::parse(&delay_image(0x2000 - 15)),
            Err(PeError::Malformed(_))
        ));
        // At the 4 GiB limit, where the slot's end overflows a u32.
        assert!(matches!(
            ImageInfo::parse(&delay_image(0xFFFF_FFF8)),
            Err(PeError::Malformed(_))
        ));
    }
}
