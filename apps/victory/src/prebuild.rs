//! The pipeline pre-build, the warm-up's second half: every pipeline of the
//! title's shader pack made ahead of a run into the host pipeline library
//! the run reads (`<cache>/pipelines`), so that a first run, and a first
//! visit to any place, compiles nothing the pack lists.
//!
//! Each pack record becomes the creation stream the engine would hand the
//! device for that pipeline (`victory_game::pipelines`), which goes through
//! the run's own code (`victory_gpu::pipeline::Prebuild`): a pipeline is
//! only ever used if its host description is the run's, byte for byte.
//! Records whose states cannot be named, or whose shaders are refused, are
//! counted by cause and skipped. The library is saved as it grows: an
//! interrupted pre-build keeps its work, and the next one finds those
//! pipelines there.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use victory_game::pipelines::{self, Named, Unnamed};
use victory_game::{sgd5, shaderpack};
use victory_gpu::pipeline::{self, Prebuild};
use victory_shader::{Container, FourCC, HwStage, native_programs};

use crate::shader::{Source, shader_files, sources};

/// Where a native program's container is: file, source, and its stage.
type Found = (usize, usize, HwStage);

/// One pipeline to make: a graphics or compute record of the pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Job {
    Compute(usize),
    Graphics(usize),
}

pub fn run(paths: &[PathBuf], threads: usize, keys_only: Option<PathBuf>) -> Result<()> {
    let started = Instant::now();
    let listing = keys_only.is_some();
    if !listing {
        pipeline::use_pipeline_cache(victory_core::Config::load()?.cache_subdir("pipelines")?);
    }
    // The files that hold shaders (or the pack itself).
    let files: Vec<(PathBuf, Vec<u8>)> = paths
        .iter()
        .flat_map(|root| shader_files(root, false))
        .filter_map(|e| Some((e.path().to_owned(), std::fs::read(e.path()).ok()?)))
        .filter(|(_, d)| d.starts_with(shaderpack::MAGIC) || !sources(d).is_empty())
        .collect();
    // It ships beside the shader databases.
    let found = (files.iter())
        .filter(|(_, d)| d.starts_with(shaderpack::MAGIC) || sgd5::is_sgd5(d))
        .find_map(|(p, d)| Some((p, shaderpack::find(d)?)));
    let Some((pack_path, pack)) = found else {
        println!("pipelines    no shader pack among the files: nothing to pre-build");
        return Ok(());
    };
    let pack = shaderpack::Pack::parse(&pack).map_err(anyhow::Error::msg)?;
    let graphics: Vec<shaderpack::Graphics> = pack.graphics().collect();
    let compute: Vec<shaderpack::Compute> = pack.compute().collect();

    // Every container by the hash instructions its native program starts
    // with, which the pack's programs start with too.
    let sources: Vec<Vec<Source>> = files.iter().map(|(_, d)| sources(d)).collect();
    let programs: Mutex<HashMap<[u8; 24], Found>> = Mutex::new(HashMap::new());
    for (f, list) in sources.iter().enumerate() {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads.min(list.len()) {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(source) = list.get(i) else { break };
                        let Some(bytes) = source.bytes() else {
                            continue;
                        };
                        for (id, stage) in program_ids(&bytes) {
                            lock(&programs).entry(id).or_insert((f, i, stage));
                        }
                    }
                });
            }
        });
    }
    let programs = programs.into_inner().unwrap_or_else(|e| e.into_inner());
    let databases: Vec<sgd5::Database> =
        files.iter().flat_map(|(_, d)| sgd5::databases(d)).collect();
    let named = Named::find(&pack, &databases);
    eprintln!(
        "{}: shader pack, {} graphics and {} compute pipelines; states named: {} depth-stencil, {} rasteriser, {} blend",
        pack_path.display(),
        graphics.len(),
        compute.len(),
        named.depth.len(),
        named.raster.len(),
        named.blend.len(),
    );

    let container = |code: &[u8]| -> Option<(Vec<u8>, HwStage)> {
        let &(f, i, stage) = programs.get(&shaderpack::program_id(code)?)?;
        let bytes = sources[f][i].bytes()?;
        // As the title passes it: the container's own size.
        let size = u32::from_le_bytes(bytes.get(0x18..0x1c)?.try_into().ok()?) as usize;
        Some((bytes.get(..size)?.to_vec(), stage))
    };
    // A job's pipeline, or why it has none.
    let prepare = |job: Job| -> Result<Prebuild, String> {
        match job {
            Job::Graphics(i) => prepare_graphics(&graphics[i], &named, &container),
            Job::Compute(i) => {
                let (cs, _) = container(compute[i].stage.program).ok_or("shader not found")?;
                Ok(Prebuild::compute(&cs))
            }
        }
    };

    // A job's shaders as a run names them (diagnostics).
    let shaders_of = |job: Job| match job {
        Job::Compute(i) => format_stages([("compute", compute[i].stage.program)], &container),
        Job::Graphics(i) => shaders_of_graphics(&graphics[i], &container),
    };
    let mut jobs: Vec<Job> = (0..compute.len()).map(Job::Compute).collect();
    jobs.extend((0..graphics.len()).map(Job::Graphics));
    // Diagnostics, `VICTORY_PREBUILD_KEYS=<file>`: make nothing, list each
    // record's key (what a run lists pipelines by) with its shaders and
    // state as a run's `VICTORY_PIPELINE_LOG` has them, or why it is
    // skipped; and the records each source named states for.
    if keys_only.is_some() {
        let mut by: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for g in &graphics {
            for (kind, h) in [
                ("depth-stencil", g.depth_stencil_hash),
                ("rasteriser", g.rasterizer_hash),
                ("blend", g.blend_hash),
            ] {
                *by.entry((kind, named.source.get(&h).copied().unwrap_or("not named")))
                    .or_default() += 1;
            }
        }
        for ((kind, source), n) in by {
            println!("  {kind:<14} {source:<12} {n:>6} records");
        }
    }
    let skipped: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    let skip = |why: String| *lock(&skipped).entry(why).or_default() += 1;
    // First every record's key, which needs no translation: what a
    // pre-build put in the library is skipped, what an earlier run drew
    // with is made first.
    // Write the jobs in submission order regardless of which worker finishes first.
    let lines: Mutex<BTreeMap<Job, String>> = Mutex::new(BTreeMap::new());
    let todo: Mutex<Vec<(bool, Job)>> = Mutex::new(Vec::new());
    let listed = AtomicUsize::new(0);
    each(&jobs, threads, |job| match prepare(job) {
        Ok(p) if keys_only.is_some() => {
            lock(&lines).insert(job, format!("{}\t{job:?}\t{}", p.key(), p.describe()));
        }
        Ok(p) if pipeline::prebuilt(&p.key()) => drop(listed.fetch_add(1, Ordering::Relaxed)),
        Ok(p) => lock(&todo).push((!pipeline::wanted_early(&p.key()), job)),
        Err(why) => {
            if keys_only.is_some() {
                lock(&lines).insert(job, format!("-\t{job:?}\t{}\t{why}", shaders_of(job)));
            }
            skip(why);
        }
    });
    let listed = listed.into_inner();
    if let Some(file) = keys_only {
        let lines = lines.into_inner().unwrap_or_else(|e| e.into_inner());
        let ordered: Vec<_> = lines.into_values().collect();
        std::fs::write(file, ordered.join("\n"))?;
    }
    let mut todo = todo.into_inner().unwrap_or_else(|e| e.into_inner());
    todo.sort_by_key(|&(later, _)| later);
    let todo: Vec<Job> = todo.into_iter().map(|(_, job)| job).collect();

    let next = AtomicUsize::new(0);
    let (made0, loaded0) = pipeline::pipeline_counts();
    let counts = || {
        let (made, loaded) = pipeline::pipeline_counts();
        (made - made0, loaded - loaded0)
    };
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.min(todo.len()))
            .map(|_| {
                s.spawn(|| {
                    while let Some(&job) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Err(why) = prepare(job).and_then(|p| p.make()) {
                            skip(why);
                        }
                    }
                })
            })
            .collect();
        let (n, what) = (todo.len(), "pipelines to make");
        crate::translate::report(pack_path, n, what, &next, started, &workers, || {
            let (made, loaded) = counts();
            format!("; {made} made, {loaded} found in the library")
        });
    });
    if !listing {
        pipeline::save_pipeline_cache();
    }
    let (made, loaded) = counts();
    println!(
        "pipelines    {} in the pack, {:.1} s: {made} made, {} already in the library",
        jobs.len(),
        started.elapsed().as_secs_f64(),
        listed + loaded as usize,
    );
    for (why, n) in lock(&skipped).iter() {
        println!("  {n:>7}  skipped: {why}");
    }
    Ok(())
}

/// Runs `f` on every job, on `threads` threads.
fn each(jobs: &[Job], threads: usize, f: impl Fn(Job) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(jobs.len()) {
            s.spawn(|| {
                while let Some(&job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    f(job);
                }
            });
        }
    });
}

/// The hash instructions each native program of a container starts with.
fn program_ids(bytes: &[u8]) -> Vec<([u8; 24], HwStage)> {
    let Ok(c) = Container::parse(bytes) else {
        return Vec::new();
    };
    let Some(part) = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) else {
        return Vec::new();
    };
    native_programs(part.data)
        .map(|programs| {
            (programs.iter())
                .filter_map(|p| Some((shaderpack::program_id(&p.code)?, p.stage)))
                .collect()
        })
        .unwrap_or_default()
}

/// Turns one graphics pack record into a `Prebuild`, rejecting records that
/// carry stages outside vertex, mesh, and pixel (including `VS + GS`).
fn prepare_graphics(
    g: &shaderpack::Graphics<'_>,
    named: &Named,
    container: &impl Fn(&[u8]) -> Option<(Vec<u8>, HwStage)>,
) -> Result<Prebuild, String> {
    let [ps, vs, gs, ..] = g.stages;
    if g.stages[3..].iter().any(Option::is_some) || (vs.is_some() && gs.is_some()) {
        return Err("stages other than vertex, mesh and pixel".into());
    }
    let (first, stage) = vs
        .xor(gs)
        .and_then(|s| container(s.program))
        .ok_or("shader not found")?;
    let pixel = match ps {
        Some(ps) => Some(container(ps.program).ok_or("shader not found")?.0),
        None => None,
    };
    let mesh = match stage {
        HwStage::Ms => true,
        HwStage::Vs => false,
        _ => return Err("first stage not a vertex or mesh shader".into()),
    };
    let stream = pipelines::stream(g, named, &first, mesh, pixel.as_deref()).map_err(|u| {
        match u {
            Unnamed::DepthStencil => "depth-stencil state not named",
            Unnamed::Rasterizer => "rasteriser state not named",
            Unnamed::Blend => "blend state not named",
            Unnamed::Metadata => "metadata not read",
        }
        .to_owned()
    })?;
    // SAFETY: the stream points at `first`, `pixel` and its own input
    // elements, all alive here.
    Ok(unsafe { Prebuild::graphics(stream.bytes()) })
}

fn stage_name(stage: HwStage) -> &'static str {
    match stage {
        HwStage::Ls => "ls",
        HwStage::Hs => "hs",
        HwStage::Es => "es",
        HwStage::Gs => "geometry",
        HwStage::Vs => "vertex",
        HwStage::Ps => "pixel",
        HwStage::Ms => "mesh",
        HwStage::Cs => "compute",
    }
}

fn format_stages<'a>(
    stages: impl IntoIterator<Item = (&'static str, &'a [u8])>,
    container: &impl Fn(&[u8]) -> Option<(Vec<u8>, HwStage)>,
) -> String {
    let names: Vec<String> = stages
        .into_iter()
        .map(|(default, code)| match container(code) {
            Some((c, stage)) => {
                format!("{} {}", stage_name(stage), &blake3::hash(&c).to_hex()[..16])
            }
            None => format!("{default} ?"),
        })
        .collect();
    format!("[{}]", names.join(", "))
}

/// Lists all non-empty stages of a graphics record without hiding `GS` behind `VS`.
fn shaders_of_graphics(
    g: &shaderpack::Graphics<'_>,
    container: &impl Fn(&[u8]) -> Option<(Vec<u8>, HwStage)>,
) -> String {
    let [ps, vs, gs, es, hs, ls, fetch] = g.stages;
    let gs_default = if vs.is_some() { "geometry" } else { "mesh" };
    let slots = [
        vs.map(|s| ("vertex", s.program)),
        gs.map(|s| (gs_default, s.program)),
        es.map(|s| ("es", s.program)),
        hs.map(|s| ("hs", s.program)),
        ls.map(|s| ("ls", s.program)),
        fetch.map(|s| ("fetch", s.program)),
        ps.map(|s| ("pixel", s.program)),
    ];
    format_stages(slots.into_iter().flatten(), container)
}

pub(crate) fn keys_output_path(value: Option<std::ffi::OsString>) -> Result<Option<PathBuf>> {
    let Some(file) = value else {
        return Ok(None);
    };
    let path = PathBuf::from(file);
    victory_core::ensure_outside_repo(&path)?;
    Ok(Some(path))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> [u8; 0x50] {
        let mut m = [0u8; 0x50];
        m[0x26] = 3; // triangle topology
        m[0x28..0x2c].copy_from_slice(&1u32.to_le_bytes()); // sample count = 1
        m[0x38..0x3c].copy_from_slice(&4u32.to_le_bytes()); // input layout bytes (count = 0)
        m
    }

    fn test_named() -> Named {
        let mut named = Named::default();
        named.depth.insert(0, [0u8; pipelines::DEPTH]);
        named.raster.insert(0, [0u8; pipelines::RASTER]);
        named.blend.insert(0, [0u8; pipelines::BLEND]);
        named
    }

    fn test_graphics<'a>(
        metadata: &'a [u8],
        stages: [Option<shaderpack::Stage<'a>>; 7],
    ) -> shaderpack::Graphics<'a> {
        shaderpack::Graphics {
            key: 1,
            shader_hash: 2,
            pipeline_hash: 3,
            depth_stencil_hash: 0,
            rasterizer_hash: 0,
            blend_hash: 0,
            packet: &[],
            metadata,
            stages,
        }
    }

    fn stage(program: &[u8]) -> Option<shaderpack::Stage<'_>> {
        Some(shaderpack::Stage {
            program,
            metadata: None,
        })
    }

    #[test]
    fn prepare_rejects_vs_plus_gs_while_allowing_vs_and_ms_pipelines() {
        let metadata = test_metadata();
        let named = test_named();
        let container = |code: &[u8]| match code {
            b"vs" => Some((b"vs-container".to_vec(), HwStage::Vs)),
            b"ms" => Some((b"ms-container".to_vec(), HwStage::Ms)),
            b"gs" => Some((b"gs-container".to_vec(), HwStage::Gs)),
            b"ps" => Some((b"ps-container".to_vec(), HwStage::Ps)),
            _ => None,
        };

        let vs_ps = test_graphics(
            &metadata,
            [stage(b"ps"), stage(b"vs"), None, None, None, None, None],
        );
        let built_vs = prepare_graphics(&vs_ps, &named, &container).expect("VS+PS succeeds");
        assert!(built_vs.describe().contains("vertex"));
        assert!(built_vs.describe().contains("pixel"));
        let vs_hash = &blake3::hash(b"vs-container").to_hex()[..16];
        let ps_hash = &blake3::hash(b"ps-container").to_hex()[..16];
        assert_eq!(
            shaders_of_graphics(&vs_ps, &container),
            format!("[vertex {vs_hash}, pixel {ps_hash}]")
        );

        let ms_ps = test_graphics(
            &metadata,
            [stage(b"ps"), None, stage(b"ms"), None, None, None, None],
        );
        let built_ms = prepare_graphics(&ms_ps, &named, &container).expect("MS+PS succeeds");
        assert!(built_ms.describe().contains("mesh"));
        assert!(built_ms.describe().contains("pixel"));
        let ms_hash = &blake3::hash(b"ms-container").to_hex()[..16];
        assert_eq!(
            shaders_of_graphics(&ms_ps, &container),
            format!("[mesh {ms_hash}, pixel {ps_hash}]")
        );

        let vs_gs_ps = test_graphics(
            &metadata,
            [
                stage(b"ps"),
                stage(b"vs"),
                stage(b"gs"),
                None,
                None,
                None,
                None,
            ],
        );
        assert_eq!(
            prepare_graphics(&vs_gs_ps, &named, &container)
                .err()
                .as_deref(),
            Some("stages other than vertex, mesh and pixel")
        );
        let gs_hash = &blake3::hash(b"gs-container").to_hex()[..16];
        assert_eq!(
            shaders_of_graphics(&vs_gs_ps, &container),
            format!("[vertex {vs_hash}, geometry {gs_hash}, pixel {ps_hash}]")
        );

        let vs_gs_unresolved = test_graphics(
            &metadata,
            [
                None,
                stage(b"vs"),
                stage(b"missing-gs"),
                None,
                None,
                None,
                None,
            ],
        );
        assert_eq!(
            prepare_graphics(&vs_gs_unresolved, &named, &container)
                .err()
                .as_deref(),
            Some("stages other than vertex, mesh and pixel")
        );
        assert_eq!(
            shaders_of_graphics(&vs_gs_unresolved, &container),
            format!("[vertex {vs_hash}, geometry ?]")
        );
    }

    #[test]
    fn prebuild_keys_inside_repo_is_refused() {
        assert_eq!(keys_output_path(None).unwrap(), None);

        let inside = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("keys.txt");
        let err = keys_output_path(Some(std::ffi::OsString::from(&inside))).unwrap_err();
        assert!(
            err.to_string().contains("inside the VIctory source tree"),
            "unexpected error: {err}"
        );

        let outside = std::env::temp_dir().join("victory-prebuild-keys-test.txt");
        assert_eq!(
            keys_output_path(Some(std::ffi::OsString::from(&outside))).unwrap(),
            Some(outside)
        );
    }
}
