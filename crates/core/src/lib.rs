//! Sinus Sentinel core: audio pipeline, gate, DSP, inference, sessionizer,
//! event store, and sync engine. No UI dependencies — this crate compiles
//! unchanged into the desktop app, mobile shells, and the CLI harness.
//! See docs/SPEC.md at the repo root.

pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn scaffold_builds() {
        assert!(!super::CORE_VERSION.is_empty());
    }
}
