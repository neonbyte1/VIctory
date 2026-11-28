# Heap sync: making slots cheaper (`VICTORY_HEAP_MAKE`, default off)

Branch `perf/heap-sync-make`, worktree `scratch/HeapSync/make`, base fork `main`
`4bf60796e`. Supersedes the page-skip experiment on `perf/heap-sync-skip`
(dropped: the scan is 0.13 ms of the sync's 19 ms, so skipping it saved nothing).

## Why: the 19 ms is the making

From scratch run 0991b/d and 0991a/c (see `local://HeapSync.md`): the sync costs
17.7-19.4 ms a frame over 258-282k slots for 6.4-6.7k changed records and 8.9-9.5k
going null - **15.5-16.0k `mirror` calls a frame**. The scan itself is 0.13 ms
(measured: 262k slots, 0.5 ns/slot). Per call, measured on this host with the
real `make_null` path, 0.47 µs; a `CopyDescriptorsSimple` of the same slot under
0.005 µs. So:

- 8.9-9.5k slots a frame only need a **null** descriptor: ~4.2-4.5 ms of
  `CreateShaderResourceView(None)` calls that one copy a run can replace.
- the other 6.1-6.6k are views: if a descriptor for the same record was already
  made this epoch (the title copies tables afresh with the same views in them),
  a copy replaces the make.

## What it does

Both behind `VICTORY_HEAP_MAKE=1` (off by default, one exe gives both arms).

1. **Batched nulls** (`HostHeap::null_slots`). `sync_slots` splits the stale slots
   by the record's tag: those whose record is not a view of ours (the 8.9-9.5k)
   are nulled together - the tables and each slot's held resource cleared a slot
   at a time (`keep` frees what it held), the descriptors copied from one
   pre-made null slot (`Make::null_at`, a one-slot CPU-side heap of the heap's
   kind) a **run of consecutive slots at a time**. `made` is held across the
   batch, as `mirror` holds it across one slot, so a title thread making a record
   there waits for the batch or is seen by it.
2. **A made-descriptor cache** (`Cache`). Key: the record (view index and
   generation, exactly the dwords the title's table holds) + `view_ident(view)`
   (the view's kind, the resource it names and its words: format, dimension,
   mips) + the epoch. On a hit the destination's descriptor is
   `CopyDescriptorsSimple(1)` from the source slot, and its stride-table dword,
   mark, `viewed` entry and held resource come from the entry/the source; on a
   miss the descriptor is made as before and remembered.
   - Bounded: 8,192 entries a heap, direct-mapped (`record` hashed), overwritten
     in place; `MakeEvicted` counts overwrites.
   - Not cached: buffer and constant views (their descriptor shows the copy of
     the innermost title buffer over them, which moves as buffers come and go -
     `owner_moved`), and textures the title placed itself (made lazily, on the
     draw that first reads them).
   - **Exactness**: an entry answers only for its own record and view, only while
     `HostHeap::made` still says its source slot holds that record, and only in
     the epoch it was made in. The epoch is bumped by everything that re-makes a
     slot without changing its record - a texture given up
     (`HostHeap::evict`), a placed texture made real
     (`descriptor::make_read`) - so a stale descriptor can never be copied.
     A record whose view died with its resource is nulled by `mirror` before the
     cache is asked (`view_at` finds no view), so a freed resource's descriptor
     is never made to read again, whatever its address is reused for.

## Tests

Device (`descriptor::host::tests`, own gate `/tmp/qheapsync.sh`, reading a
slot's descriptor bytes in place - vkd3d-proton faults on a
`CopyDescriptorsSimple` from a shader-visible heap into a CPU-side one):

- `a_record_made_before_is_copied_not_made_again` - the same record in three
  slots: one make, two hits, all three descriptors byte-equal.
- `a_record_of_a_freed_view_is_nulled_not_copied` - the resource released
  (`descriptor::forget`): the copy of its record in another slot is nulled, no
  hit, and its descriptor is a null view's, not the freed texture's.
- `a_second_view_of_one_texture_is_its_own_descriptor` - two views of one texture
  (two mip ranges): both made (no hit across views), the copy of the second hits
  with the second's descriptor, and the two descriptors differ.
- `null_batches_make_what_make_null_makes` - a batch of nulls equals
  `make_null`'s descriptor, in a view heap and in a sampler heap, and leaves no
  record noted.
- `descriptor_writes_cost_what_the_sync_spends` - the cost measurement below,
  printing a slot nulled one at a time against nulled in a batch.
- The module's existing `heap_syncs_make_the_slots_whose_records_changed` runs
  through `sync_slots` with the switch on, i.e. through the tag split and
  `null_slots` for the cleared record.

Note on what the cache cannot be asked to distinguish: a record is (view index,
generation), and a view entry's index is reused only under a new generation, so
"the same bytes, another resource" is the dead-entry case above, and a different
format or view kind is a different entry, hence a different record.

## Checks and artifacts

- `cargo fmt --all` clean; clippy `--workspace --all-targets -D warnings` **and**
  `--all-features`: clean (the tests' counter checks are behind the `diagnostics`
  feature, `copystat`'s own gate); `cargo xtask policy` reports only the two
  violations already in `4bf60796e` (`.git-blame-ignore-revs`'s file type and
  `lower.rs`'s size) - this branch adds no files.
- Commit `805309ac` on `perf/heap-sync-make` (worktree `scratch/HeapSync/make`).
- exe: `/tmp/vexe-heapmake-805309ac/victory.exe`,
  sha256 `a2ecd651dc4b33f366855bcc41fd73836cec677e7038407df2e05df5ddb86dff`.
- A/B: `VICTORY_LOG=info VICTORY_HEAP_MAKE=1 <exe>` against the default (same
  exe); the frame line's heap clause reads
  `... make: N nulled in batches, N made, N hits, N misses, N evicted`, with
  `ms taking / scanning / making` before it.

## Device results

`descriptor::` module on the test binary (`50a48ca2`), **both arms** (run by the
integrator, `/tmp/qhmtest.sh`, log `scratch/HeapSync/logs/qhmtest-by-main.log`):

| arm | result |
|---|---|
| `VICTORY_HEAP_MAKE=1` | **55 passed, 2 failed** |
| `VICTORY_HEAP_MAKE=0` | **55 passed, 2 failed** |

The two failures are the same on both arms and are the host's, not this change's:

- `capped_heap_keeps_full_dynamic_range` - the 5090's vkd3d-proton reports
  `MaxViewDescriptorHeapSize` 1,041,451 < `HEAP_VIEWS` (2,097,152), so
  `HostHeap::new` takes the capped branch and its title slots are 762,920,
  never the `mid` (918,080) the test expects. The same failure on
  `perf/heap-sync-skip` with the switch off.
- `unsupported_same_family_and_default_uavs_are_refused` - the test asserts the
  host does *not* support B8G8R8X8_UNORM typed UAV stores (`typed_store(88)`)
  and this host does; it never reaches the heap code this change touches.

The one failure this change's tests had (`null_batches_make_what_make_null_makes`
comparing vkd3d-proton's host bytes for a null created in the shader-visible heap
against one copied from a CPU-side heap) is fixed and passes on both arms. The same result came from my own gate runs of both arms in a gap in the queue
(`logs/gpu-make-on6.log`, `logs/gpu-make-off6.log`: 55 passed, 2 failed each,
the same two). An earlier arm of the module (`logs/gpu-make-module4.log`, exe
`38020746`) had the same two host failures plus the null-batch test before it was
fixed.

The game A/B (`1008a-d`) runs with `805309ac`.

## Measured, and expectations

The cost measurement above, on this host, from the module run:

```
a slot nulled: 0.46 us; in a batch: 0.24 us; a world frame's 15.5-16.0k writes are 7.3 ms nulled, 3.8 ms batched
```

- So a batched null is **half** an individual one: the `Create*View` goes, and
  what is left (0.24 µs) is the per-slot table work (`keep`, the stride dword,
  the mark, the view address, `viewed`). For a world frame's 8.9-9.5k nulls that
  is 4.1-4.4 ms -> 2.1-2.3 ms, i.e. **about -2 ms of the sync's 19.4 ms**
  [INFERENCE: the frame's own `making` split is what the A/B reports].
- Cache hits: a title that copies a table afresh re-uses the same views across
  tables; the frame line's `hits` is the measurement. If it is near zero the
  cache costs a lock and a lookup a made slot (~50 ns) and should be dropped,
  leaving the null batching.
