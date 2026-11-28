//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`; empty or
//! whitespace-only values are ignored), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// The most verbose level this build can log.
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

pub fn init(default_level: &str) {
    let filter = build_filter(default_level, |k| std::env::var(k).ok());
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .try_init();
}

fn build_filter(default_level: &str, env: impl Fn(&str) -> Option<String>) -> EnvFilter {
    let get = |k: &str| env(k).filter(|v| !v.trim().is_empty());
    get("VICTORY_LOG")
        .or_else(|| get("RUST_LOG"))
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_victory_log_falls_back_to_rust_log_and_default() {
        let f = build_filter("info", |k| match k {
            "VICTORY_LOG" => Some(String::new()),
            "RUST_LOG" => Some("warn".into()),
            _ => None,
        });
        assert_eq!(f.to_string(), "warn");

        let f = build_filter("info", |k| match k {
            "VICTORY_LOG" => Some("   ".into()),
            "RUST_LOG" => Some("warn".into()),
            _ => None,
        });
        assert_eq!(f.to_string(), "warn");

        let f = build_filter("info", |k| match k {
            "VICTORY_LOG" => Some(String::new()),
            "RUST_LOG" => Some(" \t ".into()),
            _ => None,
        });
        assert_eq!(f.to_string(), "info");
    }
}
