//! GPU time per title program and per kind of pass (diagnostics,
//! `VICTORY_GPU_PASSES=1`; `=n` times the lists begun in one frame of n).
//!
//! A timed list takes a GPU timestamp before each command it notes
//! (`HostList::note_pass`: draws, dispatches, indirect executions, argument
//! clamps, copies, clears, resolves, title barriers) and one at its end;
//! the time from one stamp to the next is charged to the command, with
//! the barriers and state changes recorded after it. Its stamps are
//! resolved into its read-back buffer when it closes and read when its
//! submission has completed (`Host::finish`): each list in flight has its
//! own heap, so nothing waits for them. Lists run in the usual way (unlike
//! `VICTORY_GPU_TIMING`, which runs and waits for each list alone).
//!
//! Every five seconds the report gives, per timed frame: the time from
//! each list's first stamp to its last and from each submission's first
//! stamp to its last (the rest is lists not timed: state preludes, video
//! memory copies, uploads, and the GPU's switches between lists), totals
//! by kind of command, by how the programs were lowered
//! (`victory_shader::emit::Form`, worked out by lowering each program once
//! more on a thread of its own) and for the acceleration-structure builder,
//! then the 25 programs that took most, each with its commands per frame by
//! kind, its mean and longest single command and its form, and up to 10
//! more whose longest single command took over a millisecond (a dispatch
//! of seconds that runs once in a while).
//!
//! A timestamp waits for the commands before it to finish (the host GPU
//! writes it at the bottom of the pipe): commands of one list overlap less
//! than they would untimed, and each is charged the time from the end of
//! the one before to its own end.
//!
//! shortcut: a list times its first 4095 commands (`host::STAMPS`); the
//! rest are charged to the last timed one and counted as untimed.

use std::collections::{BTreeMap, HashMap};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::object::This;
use crate::pipeline::{PassProgram, Shader};
use victory_shader::emit::Form;

/// What a timed command is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Class {
    /// From a list's start to its first noted command: state moves, twins,
    /// hoisted clamps.
    ListStart,
    /// A list of ours (uploads) from its start.
    OurList,
    Draw,
    Indexed,
    Mesh,
    Dispatch,
    IndirectDraw,
    IndirectIndexed,
    IndirectMesh,
    IndirectDispatch,
    /// An indirect execution's argument clamp (`crate::indirect`).
    Clamp,
    /// A window of indirect executions clamped together.
    ClampWindow,
    Copy,
    Clear,
    Resolve,
    /// A title barrier and what it waits for.
    Barrier,
    /// Query results resolved.
    Queries,
    /// A draw or dispatch not run on the host.
    Dropped,
    Other,
}

impl Class {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Class::ListStart => "list start",
            Class::OurList => "list of ours",
            Class::Draw => "draw",
            Class::Indexed => "indexed draw",
            Class::Mesh => "mesh",
            Class::Dispatch => "dispatch",
            Class::IndirectDraw => "indirect draw",
            Class::IndirectIndexed => "indirect indexed draw",
            Class::IndirectMesh => "indirect mesh",
            Class::IndirectDispatch => "indirect dispatch",
            Class::Clamp => "argument clamp",
            Class::ClampWindow => "argument clamp window",
            Class::Copy => "copy",
            Class::Clear => "clear",
            Class::Resolve => "resolve",
            Class::Barrier => "title barrier",
            Class::Queries => "query resolve",
            Class::Dropped => "dropped",
            Class::Other => "other",
        }
    }

    fn compute(self) -> bool {
        matches!(self, Class::Dispatch | Class::IndirectDispatch)
    }
}

/// A timed command: its kind, the title pipeline's host pipeline it runs
/// and whether that is the acceleration-structure builder's.
pub(crate) struct Key {
    class: Class,
    builder: bool,
    program: Option<PassProgram>,
    /// A dispatch's or mesh dispatch's groups (else zeros).
    groups: [u32; 3],
}

impl Key {
    pub(crate) const fn class(class: Class) -> Key {
        Key {
            class,
            builder: false,
            program: None,
            groups: [0; 3],
        }
    }

    /// A command running the title pipeline `pipeline`.
    pub(crate) fn of(class: Class, pipeline: This) -> Key {
        match crate::pipeline::pass_program(pipeline) {
            Some((program, builder)) => Key {
                class,
                builder,
                program: Some(program),
                groups: [0; 3],
            },
            None => Key::class(class),
        }
    }

    /// The same, of a dispatch of `groups`.
    pub(crate) fn groups(self, groups: [u32; 3]) -> Key {
        Key { groups, ..self }
    }
}

/// `VICTORY_GPU_PASSES`: lists begun in one frame of this many are timed
/// (0: none). Unset or `0`: off; a number: that; anything else: every frame.
fn parse_every(value: Option<&str>) -> u64 {
    match value {
        None | Some("0") => 0,
        Some(v) => v.trim().parse().unwrap_or(1).max(1),
    }
}

static EVERY: LazyLock<u64> = LazyLock::new(|| {
    let every = parse_every(std::env::var("VICTORY_GPU_PASSES").ok().as_deref());
    if every != 0 {
        tracing::info!(
            every,
            "GPU passes: lists begun in one frame of {every} are timed per command (VICTORY_GPU_PASSES)"
        );
    }
    every
});

fn every() -> u64 {
    *EVERY
}

/// Whether pass timing is on.
pub(crate) fn on() -> bool {
    every() != 0
}

/// Tests: every list begun is timed.
#[cfg(test)]
static FORCE: AtomicBool = AtomicBool::new(false);

/// Whether a list begun now is timed.
pub(crate) fn sampled() -> bool {
    #[cfg(test)]
    if FORCE.load(Ordering::Relaxed) {
        return true;
    }
    let n = every();
    n != 0 && crate::frame::current().is_multiple_of(n)
}

/// One window's times.
#[derive(Default)]
struct Window {
    /// By kind, builder and host pipeline (its address, kept alive here).
    rows: HashMap<(Class, bool, usize), Row>,
    /// Timed lists, and their time from first stamp to last (ticks).
    lists: u64,
    list_ticks: u64,
    /// Submissions with timed lists, from their first timed stamp to their
    /// last (ticks).
    submissions: u64,
    submission_ticks: u64,
    /// Commands timed, and past the stamps.
    commands: u64,
    untimed: u64,
    /// Time reading and adding the stamps (on the completing thread).
    collecting: Duration,
}

struct Row {
    ticks: u64,
    count: u64,
    /// The longest one command took, and its groups.
    longest: (u64, [u32; 3]),
    program: Option<PassProgram>,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// One timed list run: its stamps (one per timed command, then its end),
/// the commands' keys, whether it is a list of ours, and its commands past
/// the stamps.
pub(crate) struct Timed<'a> {
    pub(crate) stamps: &'a [u64],
    pub(crate) keys: &'a [Key],
    pub(crate) ours: bool,
    pub(crate) untimed: usize,
}

/// Adds the timed lists of a completed submission (`each` hands each one
/// to the closure it is given), timing the adding itself.
pub(crate) fn submission(each: impl FnOnce(&mut dyn FnMut(Timed<'_>))) {
    let started = Instant::now();
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    let w = window.get_or_insert_with(Window::default);
    let mut span = (u64::MAX, 0);
    each(&mut |t: Timed<'_>| w.add(&t, &mut span));
    if span.0 <= span.1 {
        w.submissions += 1;
        w.submission_ticks += span.1 - span.0;
    }
    w.collecting += started.elapsed();
}

impl Window {
    /// Adds a timed list run, widening `span` (the submission's first and
    /// last stamp) to its stamps.
    fn add(&mut self, t: &Timed<'_>, span: &mut (u64, u64)) {
        let Some((&start, &end)) = t.stamps.first().zip(t.stamps.last()) else {
            return;
        };
        // Not resolved (a list that did not run), or stamps out of order.
        if start == 0 || end < start {
            return;
        }
        *span = (span.0.min(start), span.1.max(end));
        self.lists += 1;
        self.list_ticks += end - start;
        self.untimed += t.untimed as u64;
        for (key, pair) in t.keys.iter().zip(t.stamps.windows(2)) {
            let class = match key.class {
                Class::ListStart if t.ours => Class::OurList,
                class => class,
            };
            let at = key.program.as_ref().map_or(0, |p| Arc::as_ptr(p) as usize);
            let row = self
                .rows
                .entry((class, key.builder, at))
                .or_insert_with(|| Row {
                    ticks: 0,
                    count: 0,
                    longest: (0, [0; 3]),
                    program: key.program.clone(),
                });
            let ticks = pair[1].saturating_sub(pair[0]);
            row.ticks += ticks;
            row.count += 1;
            if ticks > row.longest.0 {
                row.longest = (ticks, key.groups);
            }
            self.commands += 1;
        }
    }
}

/// The forms of programs by hash: `None` while being worked out.
type Forms = HashMap<String, Option<Result<Form, String>>>;
static FORMS: LazyLock<Mutex<Forms>> = LazyLock::new(Mutex::default);

/// The forms thread: works out the form of each shader sent to it.
static WORK: LazyLock<Mutex<std::sync::mpsc::Sender<Arc<Shader>>>> = LazyLock::new(|| {
    let (send, receive) = std::sync::mpsc::channel::<Arc<Shader>>();
    let options = victory_shader::emit::Options {
        view_marks: true,
        target: crate::shader_cache::host_target(),
        pair_meetings: crate::shader_cache::pair_meetings(),
        dispatch_record: crate::shader_cache::dispatch_record(),
    };
    let spawned = std::thread::Builder::new()
        .name("victory gpu pass forms".into())
        .spawn(move || {
            for shader in receive {
                let form = std::panic::catch_unwind(|| {
                    victory_shader::emit::form_with(shader.container(), options)
                });
                let form = match form {
                    Ok(Ok(form)) => Ok(form),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(_) => Err("the lowering panicked".to_owned()),
                };
                FORMS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(shader.hash.clone(), Some(form));
            }
        });
    if let Err(e) = spawned {
        tracing::warn!("GPU passes: no thread to work out programs' forms ({e})");
    }
    Mutex::new(send)
});

/// Works out `shader`'s form on the forms thread, once.
fn ask_form(shader: &Arc<Shader>) {
    {
        let mut forms = FORMS.lock().unwrap_or_else(|e| e.into_inner());
        if forms.contains_key(&shader.hash) {
            return;
        }
        forms.insert(shader.hash.clone(), None);
    }
    let _ = WORK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(shader.clone());
}

/// A program's form: each stage's, by stage where they differ from the
/// plain per-lane form.
fn program_form(shaders: &[Arc<Shader>], forms: &Forms) -> String {
    let labels: Vec<(&str, String)> = shaders
        .iter()
        .map(|s| {
            let label = match forms.get(&s.hash) {
                Some(Some(Ok(form))) => form.to_string(),
                Some(Some(Err(e))) => format!("no form ({e})"),
                _ => "form pending".to_owned(),
            };
            (s.stage, label)
        })
        .collect();
    if let [(_, only)] = &labels[..] {
        return only.clone();
    }
    let rest: Vec<String> = labels
        .iter()
        .filter(|(_, l)| l != "per-lane")
        .map(|(stage, l)| format!("{stage} {l}"))
        .collect();
    if rest.is_empty() {
        "per-lane".to_owned()
    } else {
        rest.join(" + ")
    }
}

/// A program's times in a report.
#[derive(Default)]
struct Program {
    ticks: u64,
    count: u64,
    longest: (u64, [u32; 3]),
    builder: bool,
    by: BTreeMap<Class, u64>,
    shaders: Vec<Arc<Shader>>,
}

/// Logs the window's times per timed frame (`frames` frames shown since
/// the last report), and starts a new window.
pub(crate) fn log(frames: f64, frequency: u64) {
    let Some(w) = WINDOW.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    if w.lists == 0 {
        return;
    }
    let timed = (frames / every().max(1) as f64).max(1.0);
    // Of one command, then per timed frame.
    let once = |ticks: u64| ticks as f64 * 1e3 / frequency.max(1) as f64;
    let ms = |ticks: u64| once(ticks) / timed;
    let per = |n: u64| n as f64 / timed;
    tracing::info!(
        "GPU passes (one frame in {}, {timed:.1} timed): lists {:.2} ms/frame from first stamp to last \
         ({:.1} lists/frame), submissions {:.2} ms/frame from first timed stamp to last ({:.1}/frame; \
         {:.2} ms in lists not timed and between lists); {:.1} commands/frame timed, {:.1} past the stamps; \
         adding the stamps {:.3} ms/frame on the completing thread",
        every(),
        ms(w.list_ticks),
        per(w.lists),
        ms(w.submission_ticks),
        per(w.submissions),
        ms(w.submission_ticks.saturating_sub(w.list_ticks)),
        per(w.commands),
        per(w.untimed),
        w.collecting.as_secs_f64() * 1e3 / timed,
    );
    let mut classes: BTreeMap<Class, (u64, u64, u64)> = BTreeMap::new();
    let mut builder = (0, 0);
    let mut programs: HashMap<String, Program> = HashMap::new();
    for ((class, is_builder, _), row) in &w.rows {
        let c = classes.entry(*class).or_default();
        *c = (c.0 + row.ticks, c.1 + row.count, c.2.max(row.longest.0));
        if *is_builder {
            (builder.0, builder.1) = (builder.0 + row.ticks, builder.1 + row.count);
        }
        let Some(shaders) = row.program.as_ref().map(crate::pipeline::pass_shaders) else {
            continue;
        };
        let name = shaders
            .iter()
            .map(|s| format!("{} {}", s.stage, s.hash))
            .collect::<Vec<_>>()
            .join(", ");
        let p = programs.entry(name).or_default();
        p.ticks += row.ticks;
        p.count += row.count;
        if row.longest.0 > p.longest.0 {
            p.longest = row.longest;
        }
        p.builder |= *is_builder;
        *p.by.entry(*class).or_default() += row.count;
        if p.shaders.is_empty() {
            p.shaders = shaders.to_vec();
            for s in shaders {
                ask_form(s);
            }
        }
    }
    let mut classes: Vec<_> = classes.into_iter().collect();
    classes.sort_by_key(|(_, (ticks, ..))| std::cmp::Reverse(*ticks));
    for (class, (ticks, n, longest)) in classes {
        tracing::info!(
            "  GPU passes kind {}: {:.2} ms/frame, {:.1}/frame, {:.1} us each, longest {:.3} ms",
            class.name(),
            ms(ticks),
            per(n),
            ms(ticks) * 1e3 / per(n).max(1e-9),
            once(longest),
        );
    }
    if builder.1 > 0 {
        tracing::info!(
            "  GPU passes BVH builder: {:.2} ms/frame, {:.1} commands/frame",
            ms(builder.0),
            per(builder.1)
        );
    }
    let forms = FORMS.lock().unwrap_or_else(|e| e.into_inner());
    let mut by_form: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut programs: Vec<(String, Program, String)> = programs
        .into_iter()
        .map(|(name, p)| {
            let form = program_form(&p.shaders, &forms);
            let kind = if p.by.keys().any(|c| c.compute()) {
                "compute"
            } else {
                "graphics"
            };
            let f = by_form.entry(format!("{kind} {form}")).or_default();
            (f.0, f.1) = (f.0 + p.ticks, f.1 + p.count);
            (name, p, form)
        })
        .collect();
    drop(forms);
    let mut by_form: Vec<_> = by_form.into_iter().collect();
    by_form.sort_by_key(|(_, (ticks, _))| std::cmp::Reverse(*ticks));
    for (form, (ticks, n)) in by_form {
        tracing::info!(
            "  GPU passes form {form}: {:.2} ms/frame, {:.1} commands/frame",
            ms(ticks),
            per(n)
        );
    }
    let line = |what: &str, i: usize, (name, p, form): &(String, Program, String)| {
        let by: Vec<String> =
            p.by.iter()
                .map(|(class, n)| format!("{} {:.1}", class.name(), per(*n)))
                .collect();
        tracing::info!(
            "  GPU passes {what} {}: {:.2} ms/frame, {:.1}/frame ({}), {:.1} us each, longest {:.3} ms{}, {form}{}: {name}",
            i + 1,
            ms(p.ticks),
            per(p.count),
            by.join(", "),
            ms(p.ticks) * 1e3 / per(p.count).max(1e-9),
            once(p.longest.0),
            match p.longest.1 {
                [0, 0, 0] => String::new(),
                [x, y, z] => format!(" ({x}x{y}x{z} groups)"),
            },
            if p.builder { ", BVH builder" } else { "" },
        );
    };
    programs.sort_by_key(|(_, p, _)| std::cmp::Reverse(p.ticks));
    for (i, program) in programs.iter().take(25).enumerate() {
        line("program", i, program);
    }
    // Single long commands (a rare dispatch of seconds) by the longest.
    let mut rest: Vec<_> = programs.iter().skip(25).collect();
    rest.sort_by_key(|(_, p, _)| std::cmp::Reverse(p.longest.0));
    for (i, program) in rest
        .into_iter()
        .take_while(|(_, p, _)| once(p.longest.0) >= 1.0)
        .take(10)
        .enumerate()
    {
        line("longest command, below the 25", i, program);
    }
}

#[cfg(test)]
mod tests;
