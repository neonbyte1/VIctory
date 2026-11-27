//! Image instructions DXIL has no form of (`Lower`, image lowering).

use super::*;

/// Image instructions DXIL has no form of are refused, not emitted as
/// operations the validator rejects: a gather from a 1D (array) image
/// ("gather requires resource declared as texture/2D/Cube/2DArray/
/// CubeArray"), a gather with offsets from a cube ("out of bound offset
/// must be undef"), a comparison sample of a 3D image ("samplec
/// requires resource declared as texture1D/2D/Cube/1DArray/2DArray/
/// CubeArray"), and a load at a mip level from a UAV ("uav load don't
/// support mipLevel/sampleIndex"). Each result is written to LDS, so the
/// refusal ends the translation. The 2D gather and comparison sample and
/// the plain UAV load lower.
#[test]
fn image_instructions_dxil_lacks_are_refused() {
    // The view table's first eight dwords (s[12:19]) and the sampler
    // table's first four (s[20:23]); then the instruction (v0 its
    // addresses, v4 its first result) and ds_write_b32 v1, v4.
    let program = |image: [u32; 2]| {
        [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0
            0xF408_0503,
            0xFA00_0000, // s_load_dwordx4 s[20:23], s[6:7], 0
            image[0],
            image[1],
            0xD834_0000,
            0x0000_0401, // ds_write_b32 v1, v4
            S_ENDPGM,
        ]
    };
    // MIMG words 1: v0 addresses, v4 data, s[12:19] image, s[20:23]
    // sampler, a16.
    const GATHER_1D_ARRAY: [u32; 2] = [0xF11C_0820, 0x40A3_0400]; // image_gather4_lz dmask:0x8 dim:1D_ARRAY
    const GATHER_2D: [u32; 2] = [0xF11C_0808, 0x40A3_0400]; // image_gather4_lz dmask:0x8 dim:2D
    const GATHER_O_CUBE: [u32; 2] = [0xF15C_0818, 0x40A3_0400]; // image_gather4_lz_o dmask:0x8 dim:CUBE
    const LOAD_MIP_2D: [u32; 2] = [0xF004_0808, 0x40A3_0400]; // image_load_mip dmask:0x8 dim:2D
    // image_sample_c_lz dmask:0x1, 32-bit addresses.
    const SAMPLE_C_LZ_2D: [u32; 2] = [0xF0BC_0108, 0x00A3_0400];
    const SAMPLE_C_LZ_3D: [u32; 2] = [0xF0BC_0110, 0x00A3_0400];
    const LOAD_2D: [u32; 2] = [0xF000_0808, 0x40A3_0400]; // image_load dmask:0x8 dim:2D
    let refused = |words: &[u32], layout, used, why: &str| {
        let e = lower_with(words, layout, used)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains(why), "{e}");
    };
    let (layout, used) = texture_tables();
    assert!(lower_with(&program(GATHER_2D), layout.clone(), used.clone()).is_ok());
    refused(
        &program(GATHER_1D_ARRAY),
        layout.clone(),
        used.clone(),
        "a gather from a 1D or 3D image",
    );
    refused(
        &program(GATHER_O_CUBE),
        layout.clone(),
        used.clone(),
        "a gather with offsets from a cube",
    );
    assert!(lower_with(&program(SAMPLE_C_LZ_2D), layout.clone(), used.clone()).is_ok());
    refused(
        &program(SAMPLE_C_LZ_3D),
        layout,
        used,
        "a comparison sample of a 3D image",
    );
    // The image as a UAV.
    let (layout, mut used) = texture_tables();
    used[0].kind = ResourceType::Uav;
    assert!(lower_with(&program(LOAD_2D), layout.clone(), used.clone()).is_ok());
    refused(
        &program(LOAD_MIP_2D),
        layout,
        used,
        "a load at a mip level from a UAV",
    );
}
