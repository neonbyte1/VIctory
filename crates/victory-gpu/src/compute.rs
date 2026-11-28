//! Shaders of our own: HLSL compiled at run time by the system's compiler
//! (`D3DCompile`, shader model 5.1), their root signature serialized beside
//! them from the parameters the host binds.
//!
//! The sources keep to what Wine's compiler takes as well as Microsoft's:
//! Wine's ignores a `RootSignature` attribute (the code then holds none to
//! make one from) and has no methods on writable byte address buffers
//! (`Load`, the interlocked ones), so a buffer a shader both reads and
//! writes is a structured buffer of `uint`, indexed in words. (After #1279,
//! Mumogg's argument clamp for Wine.)

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::PCSTR;

/// `count` root constants at `register` (`b`, space 0).
pub(crate) fn constants(register: u32, count: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: register,
                RegisterSpace: 0,
                Num32BitValues: count,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// A root view of `kind` (constant, shader resource or unordered view) at
/// `register` of `space`.
pub(crate) fn root_view(
    kind: D3D12_ROOT_PARAMETER_TYPE,
    register: u32,
    space: u32,
) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: register,
                RegisterSpace: space,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// A descriptor table of `ranges`, which must stay where they are until
/// the root signature is made.
#[cfg(test)]
pub(crate) fn table(ranges: &[D3D12_DESCRIPTOR_RANGE]) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: ranges.len() as u32,
                pDescriptorRanges: ranges.as_ptr(),
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// `count` views of `kind` from `register` of `space` (`u32::MAX`:
/// unbounded), after the table's ranges before.
#[cfg(test)]
pub(crate) fn range(
    kind: D3D12_DESCRIPTOR_RANGE_TYPE,
    register: u32,
    space: u32,
    count: u32,
) -> D3D12_DESCRIPTOR_RANGE {
    D3D12_DESCRIPTOR_RANGE {
        RangeType: kind,
        NumDescriptors: count,
        BaseShaderRegister: register,
        RegisterSpace: space,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    }
}

/// The bytes of a blob, alive as long as it is.
pub(crate) fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own buffer, borrowed for as long as the blob.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

/// `source` compiled at `entry` for `target`; the compiler's messages, if
/// it fails, are logged under `name`.
pub(crate) fn compile(
    source: &str,
    name: PCSTR,
    entry: PCSTR,
    target: PCSTR,
    flags: u32,
) -> windows::core::Result<ID3DBlob> {
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            entry,
            target,
            flags,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let (Err(_), Some(errors)) = (&compiled, errors) {
        tracing::warn!(
            "{}: {}",
            // SAFETY: a name the caller keeps, terminated.
            unsafe { name.display() },
            String::from_utf8_lossy(bytes(&errors)).trim_end()
        );
    }
    compiled?;
    code.ok_or_else(windows::core::Error::empty)
}

/// A root signature of `params`, serialized here (version 1.0: views and
/// their data may change until execution).
pub(crate) fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER],
    flags: D3D12_ROOT_SIGNATURE_FLAGS,
) -> windows::core::Result<ID3D12RootSignature> {
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: flags,
    };
    let (mut blob, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the description points at live arrays (the tables' ranges
    // are the caller's); out pointers ours; the blob is read whole before
    // it is released.
    unsafe {
        if let Err(e) = D3D12SerializeRootSignature(
            &desc,
            D3D_ROOT_SIGNATURE_VERSION_1_0,
            &mut blob,
            Some(&mut errors),
        ) {
            if let Some(errors) = errors {
                tracing::warn!(
                    "root signature: {}",
                    String::from_utf8_lossy(bytes(&errors)).trim_end()
                );
            }
            return Err(e);
        }
        device.CreateRootSignature(0, bytes(&blob.ok_or_else(windows::core::Error::empty)?))
    }
}

/// A compute shader (`main`) and its root signature of `params`.
pub(crate) fn compute(
    device: &ID3D12Device,
    name: PCSTR,
    source: &str,
    flags: u32,
    params: &[D3D12_ROOT_PARAMETER],
) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let code = compile(
        source,
        name,
        windows::core::s!("main"),
        windows::core::s!("cs_5_1"),
        flags,
    )?;
    let root = root_signature(device, params, D3D12_ROOT_SIGNATURE_FLAG_NONE)?;
    let code = bytes(&code);
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        },
        ..Default::default()
    };
    // SAFETY: the description points at live code and a live root signature.
    let pso = unsafe { device.CreateComputePipelineState(&desc) }?;
    Ok((root, pso))
}
