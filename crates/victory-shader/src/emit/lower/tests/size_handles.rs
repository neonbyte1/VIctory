//! Tests of the heap handles that a descriptor's size query and its accesses
//! are lowered to: a buffer's query has a handle of its own, an image's
//! shares the handle of its accesses. They count the handles written;
//! `stride_tests` runs a program of this shape on the host GPU and WARP.

use super::{S_ENDPGM, lower_with, root, texture_tables};
use crate::native::{ResourceType, RootParam, RootType, UsedResource};

/// A table of buffers of four-byte records at s0, and a root constant at s1.
fn buffer_table() -> (Vec<RootParam>, Vec<UsedResource>) {
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)],
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
    )
}

#[test]
fn a_buffers_size_query_does_not_share_its_handle_with_the_loads() {
    // The record count of the buffer at s0 (count - 1 as an index) and
    // the indexed read, through one descriptor load.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x810C_C10A, // s_add_i32 s12, s10, -1
        0x7E04_020C, // v_mov_b32 v2, s12
        0xE030_2000,
        0x8002_0102, // buffer_load_dword v1, v2, s[8:11], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let (layout, used) = buffer_table();
    let l = lower_with(&words, layout, used).unwrap();
    assert_eq!(l.heap_handles, 2);
}

#[test]
fn a_buffer_at_a_computed_table_offset_is_sized_through_its_own_handle() {
    // As above, with the table offset computed from a root constant
    // (s1 << 5): the descriptor's address is not fixed, so its handles
    // are made per block and load.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x8F14_8501, // s_lshl_b32 s20, s1, 5
        0xF408_0202,
        0x2800_0000, // s_load_dwordx4 s[8:11], s[4:5], s20
        0x810C_C10A, // s_add_i32 s12, s10, -1
        0x7E04_020C, // v_mov_b32 v2, s12
        0xE030_2000,
        0x8002_0102, // buffer_load_dword v1, v2, s[8:11], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let (layout, used) = buffer_table();
    let l = lower_with(&words, layout, used).unwrap();
    assert_eq!(l.heap_handles, 2);
}

#[test]
fn a_buffer_sized_at_each_of_its_reloads_has_one_size_handle() {
    // The buffer at s0 loaded twice, its record count read after each
    // load (their sum - 1 as an index), and the indexed read.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x810C_C10A, // s_add_i32 s12, s10, -1
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x810C_0C0A, // s_add_i32 s12, s10, s12
        0x7E04_020C, // v_mov_b32 v2, s12
        0xE030_2000,
        0x8002_0102, // buffer_load_dword v1, v2, s[8:11], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let (layout, used) = buffer_table();
    let l = lower_with(&words, layout, used).unwrap();
    // One for the two size queries, one for the load.
    assert_eq!(l.heap_handles, 2);
}

#[test]
fn a_buffers_unread_size_query_writes_no_handle() {
    // The indexed read of the buffer at s0; nothing reads its record
    // count (s10), so only the load's handle is written.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x7E04_0280, // v_mov_b32 v2, 0
        0xE030_2000,
        0x8002_0102, // buffer_load_dword v1, v2, s[8:11], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let (layout, used) = buffer_table();
    let l = lower_with(&words, layout, used).unwrap();
    assert_eq!(l.heap_handles, 1);
}

#[test]
fn an_images_size_read_as_data_shares_the_handle_of_its_sample() {
    // The texture at s0 sampled with the sampler at s1, and its width
    // bits read as data, through one descriptor load.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], null
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], null
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xD54E_0008,
        0x0278_1A0E, // v_alignbit_b32 v8, s14, s13, 30
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
        0xD834_0000,
        0x0000_0800, // ds_write_b32 v0, v8
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        S_ENDPGM,
    ];
    let (layout, used) = texture_tables();
    let l = lower_with(&words, layout, used).unwrap();
    // One for the texture, one for the sampler.
    assert_eq!(l.heap_handles, 2);
}
