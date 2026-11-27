//! `d3d12_xs.dll`: virtual memory setup, device creation, root signature
//! serialization. The device and everything it creates live in
//! `victory-gpu`.

use std::ffi::c_void;

use victory_gpu::device;
use victory_gpu::object::{HRESULT, Iid, Object};

use crate::{HleExport, HleModule};

/// Device creation; first hands the device the title's general allocator
/// (`XMemAlloc`, which honours the title's hooks) for resource memory.
extern "system" fn create_device(
    adapter: *const c_void,
    parameters: *const u32,
    iid: *const Iid,
    out: *mut *const Object,
) -> HRESULT {
    victory_gpu::resource::use_allocator(super::xmem::XMemAlloc, super::xmem::XMemFree);
    victory_gpu::resource::use_page_mapper(super::xmem::map_alias);
    #[cfg(windows)]
    victory_gpu::on_failure(host_gpu_failed);
    device::create(adapter, parameters, iid, out)
}

/// A host GPU failure ends the run like a crash: reported to the crash log,
/// then the process ends as the console ends a title.
#[cfg(windows)]
fn host_gpu_failed(what: &str) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    victory_loader::log_crash(&format!("VIctory: host GPU failure: {what}"));
    const DXGI_ERROR_DEVICE_REMOVED: u32 = 0x887A_0005;
    // SAFETY: ends our own process.
    unsafe { TerminateProcess(GetCurrentProcess(), DXGI_ERROR_DEVICE_REMOVED) };
    unreachable!("the process has ended")
}

/// Keeps the host's translations of the title's shaders in `dir` (a
/// folder of the cache) across runs, and its compiled pipelines beside it
/// (`pipelines`).
pub fn use_shader_cache(dir: std::path::PathBuf) {
    #[cfg(windows)]
    {
        victory_gpu::pipeline::use_pipeline_cache(dir.with_file_name("pipelines"));
        victory_gpu::shader_cache::use_shader_cache(dir);
    }
    #[cfg(not(windows))]
    let _ = dir;
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "d3d12x",
        dlls: &["d3d12_xs.dll"],
        exports: vec![
            HleExport {
                name: "D3D12ConfigureVirtualMemoryX",
                address: device::configure_virtual_memory as *const () as usize,
            },
            HleExport {
                name: "D3D12XboxCreateDevice",
                address: create_device as *const () as usize,
            },
            HleExport {
                name: "D3D12SerializeRootSignature",
                address: victory_gpu::root::serialize as *const () as usize,
            },
            HleExport {
                name: "D3D12SerializeVersionedRootSignature",
                address: victory_gpu::root::serialize_versioned as *const () as usize,
            },
            // The AMD upscaler's (Series S).
            HleExport {
                name: "D3D12CreateRootSignatureDeserializer",
                address: victory_gpu::root::create_deserializer as *const () as usize,
            },
            // A data import: the title reads the id at this address.
            HleExport {
                name: "IID_ID3D12Device",
                address: (&raw const device::IID) as usize,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HleResolver, HostPolicy};
    use victory_loader::ImportResolver;
    use victory_loader::Provenance;
    use victory_loader::Resolution;
    use victory_pe::ImportSymbol;

    fn symbol(name: &str) -> ImportSymbol {
        ImportSymbol::Name {
            hint: 0,
            name: name.into(),
        }
    }

    #[test]
    fn the_interface_id_import_is_bound_to_data() {
        let r = HleResolver::new(HostPolicy::none());
        let Resolution::Found { address, .. } =
            r.resolve("d3d12_xs.dll", &symbol("IID_ID3D12Device"))
        else {
            panic!("not bound");
        };
        // SAFETY: the address is our static id.
        assert_eq!(unsafe { *(address as *const [u8; 16]) }, device::IID);
    }

    #[test]
    fn module_serves_the_console_d3d12_dll() {
        let m = module();
        assert_eq!(m.name, "d3d12x");
        assert!(m.serves("d3d12_xs.dll"));
        assert!(m.serves("D3D12_XS.DLL"), "DLL names are case-insensitive");
        assert!(!m.serves("d3d12.dll"));
        assert!(!m.serves("dxgi.dll"));
    }

    #[test]
    fn every_export_resolves_as_hle_from_the_dll() {
        let resolver = HleResolver::new(HostPolicy::none());
        for export in module().exports {
            match resolver.resolve("d3d12_xs.dll", &symbol(export.name)) {
                Resolution::Found {
                    address,
                    provenance,
                } => {
                    assert_eq!(provenance, Provenance::Hle, "{}", export.name);
                    assert_eq!(address, export.address, "{}", export.name);
                }
                Resolution::Missing => panic!("{} is not bound", export.name),
            }
        }
    }

    #[test]
    fn the_interface_id_export_points_at_the_device_iid() {
        let export = module()
            .exports
            .into_iter()
            .find(|e| e.name == "IID_ID3D12Device")
            .expect("the interface id is exported as data");
        assert_eq!(export.address, (&raw const device::IID) as usize);
        // SAFETY: the export is the address of our static interface id.
        assert_eq!(unsafe { *(export.address as *const [u8; 16]) }, device::IID);
    }
}
