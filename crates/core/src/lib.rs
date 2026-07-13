//! Sinus Sentinel core: audio pipeline, gate, DSP, inference, sessionizer,
//! event store, and sync engine. No UI dependencies — this crate compiles
//! unchanged into the desktop app, mobile shells, and the CLI harness.
//! See docs/SPEC.md at the repo root.

pub mod audio;
pub mod classify;
pub mod error;
pub mod gate;
pub mod mel;
pub mod pipeline;
pub mod session;
pub mod store;
pub mod sync;
pub mod synth;
pub mod token;
pub mod types;

pub use error::{Error, Result};
pub use types::{Event, EventType, Source, SAMPLE_RATE};

pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");
