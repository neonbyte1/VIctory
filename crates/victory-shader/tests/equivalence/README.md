# Wave32 execution equivalence

Run from the repository root with its pinned Rust toolchain.

```sh
# Default suite, including the fixed generated corpus
cargo test -p victory-shader --locked --test equivalence

# Opt-in generated campaign
cargo test -p victory-shader --locked --test equivalence --release search::campaign -- --exact --ignored --nocapture

# Replay one seed (replace 42 with the reported seed)
WAVE_SEED=42 WAVE_COUNT=1 cargo test -p victory-shader --locked --test equivalence --release search::campaign -- --exact --ignored --nocapture
```

See [PR #2090](https://github.com/Marvisak/VIctory/pull/2090) for the tested
subset, comparison design, and reducer.
