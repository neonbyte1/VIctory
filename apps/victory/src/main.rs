//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
mod prebuild;
mod run;
mod shader;
mod translate;

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout. Shader packs
    /// among the paths are listed last, their stages named from the
    /// containers found.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found (and per pack pipeline).
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
        /// Instead list the effects of shader databases: technique, pass,
        /// shaders (name, container hash, engine hash) and pass state.
        #[arg(long)]
        effects: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader. With
        /// `--cache`: then make every pipeline of the title's shader pack
        /// into the host pipeline library `run` reads (on this GPU and
        /// driver; a second time only what is not there yet).
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Positive worker thread count (default: every logical core).
        #[arg(long)]
        jobs: Option<NonZeroUsize>,
        /// Only translate into the shader cache `run` reads (the warm-up):
        /// every container found, once, nothing else checked or listed
        /// (with `--pipelines`, then the shader pack's pipelines).
        #[arg(long, conflicts_with_all = ["validate", "list"])]
        cache: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model and resolution profile the title sees. Defaults
        /// to VICTORY_CONSOLE, then series-x.
        #[arg(long, value_enum)]
        console: Option<Console>,
        /// Cap the title's actual render targets to a supported profile.
        /// Supported: 1920x1080 (Series X profile only).
        #[arg(long, value_name = "WIDTHxHEIGHT")]
        resolution: Option<String>,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

/// The explicit command-line choice wins; otherwise use `VICTORY_CONSOLE`,
/// with Series X as the unchanged default.
fn resolve_console(cli: Option<Console>, env: Option<&str>) -> Result<Console> {
    if let Some(console) = cli {
        return Ok(console);
    }
    let Some(value) = env else {
        return Ok(Console::SeriesX);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "series-x" => Ok(Console::SeriesX),
        "series-s" => Ok(Console::SeriesS),
        _ => anyhow::bail!("invalid VICTORY_CONSOLE {value:?}; expected series-x or series-s"),
    }
}

fn resolve_resolution(
    cli: Option<&str>,
    env: Option<&str>,
) -> Result<Option<victory_core::Resolution>> {
    let Some(value) = cli.or(env) else {
        return Ok(None);
    };
    victory_core::Resolution::parse(value)
        .map(Some)
        .map_err(|message| {
            anyhow::anyhow!(
                "invalid {} resolution {value:?}: {message}",
                if cli.is_some() {
                    "--resolution"
                } else {
                    "VICTORY_RESOLUTION"
                }
            )
        })
}

#[cfg(test)]
mod console_tests {
    use super::{Console, resolve_console, resolve_resolution};

    #[test]
    fn cli_console_overrides_the_environment() {
        assert!(matches!(
            resolve_console(Some(Console::SeriesX), Some("series-s")),
            Ok(Console::SeriesX)
        ));
    }

    #[test]
    fn environment_selects_a_lower_resolution_console_profile() {
        assert!(matches!(
            resolve_console(None, Some("series-s")),
            Ok(Console::SeriesS)
        ));
    }

    #[test]
    fn unset_console_defaults_to_series_x() {
        assert!(matches!(resolve_console(None, None), Ok(Console::SeriesX)));
    }

    #[test]
    fn invalid_environment_value_is_rejected() {
        assert!(resolve_console(None, Some("1080p")).is_err());
    }

    #[test]
    fn resolution_cli_overrides_environment_and_unknown_sizes_fail_closed() {
        assert_eq!(
            resolve_resolution(Some("1920x1080"), Some("3840x2160")).unwrap(),
            Some(victory_core::Resolution::HD_1080)
        );
        assert!(resolve_resolution(None, Some("3840x2160")).is_err());
    }
}

/// Diagnostics: the heap bytes Rust code holds, for the statistics.
#[cfg(all(windows, feature = "diagnostics"))]
#[global_allocator]
static ALLOC: victory_gpu::CountingAlloc = victory_gpu::CountingAlloc;

fn main() -> Result<()> {
    victory_core::logging::init("info");
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
            effects,
        } => shader::run(paths, verbose || disasm, disasm, effects),
        Command::Translate {
            paths,
            jobs,
            pipelines,
            cache: true,
            ..
        } => translate::warm(paths, jobs.map(NonZeroUsize::get), pipelines),
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            cache: false,
        } => translate::run(
            paths,
            validate,
            pipelines,
            list,
            jobs.map(NonZeroUsize::get),
        ),
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            console,
            resolution,
            title_args: _,
        } => {
            let env_console = if console.is_none() {
                std::env::var_os("VICTORY_CONSOLE")
                    .map(|value| {
                        value.into_string().map_err(|_| {
                            anyhow::anyhow!("VICTORY_CONSOLE must contain valid Unicode")
                        })
                    })
                    .transpose()?
            } else {
                None
            };
            let console = resolve_console(console, env_console.as_deref())?;
            let env_resolution = if resolution.is_none() {
                std::env::var_os("VICTORY_RESOLUTION")
                    .map(|value| {
                        value.into_string().map_err(|_| {
                            anyhow::anyhow!("VICTORY_RESOLUTION must contain valid Unicode")
                        })
                    })
                    .transpose()?
            } else {
                None
            };
            let resolution = resolve_resolution(resolution.as_deref(), env_resolution.as_deref())?;
            if resolution.is_some() && matches!(console, Console::SeriesS) {
                anyhow::bail!("--resolution requires the Series X console profile");
            }
            let code = run::run(&exe, unimplemented, strict, execute, console, resolution)?;
            if code != 0 {
                // Preserve all 32 bits of the Windows process status.
                std::process::exit(code as i32);
            }
            Ok(())
        }
        Command::Config => show_config(),
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("fidelityfx  : {}", status(&cfg.fidelityfx_dir));
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    check_policy(&cfg)
}

/// Reports each configured path the asset policy refuses, and fails if
/// there is one, so that `victory config` exits nonzero.
fn check_policy(cfg: &victory_core::Config) -> Result<()> {
    let mut violations = 0;
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
            violations += 1;
        }
    }
    anyhow::ensure!(
        violations == 0,
        "{violations} configured path(s) violate the asset policy"
    );
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::check_policy;

    fn outside() -> PathBuf {
        std::env::temp_dir().join("victory-config-test")
    }

    #[test]
    fn config_check_passes_paths_outside_the_repository() {
        let cfg = victory_core::Config {
            cache_dir: outside().join("cache"),
            data_dir: outside().join("data"),
            ..Default::default()
        };
        assert!(check_policy(&cfg).is_ok());
    }

    #[test]
    fn config_check_fails_a_path_inside_the_repository() {
        let cfg = victory_core::Config {
            game_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("game")),
            cache_dir: outside().join("cache"),
            data_dir: outside().join("data"),
            ..Default::default()
        };
        let err = check_policy(&cfg).unwrap_err();
        assert!(err.to_string().contains("1 configured path"), "{err}");
    }
}
