//! UI-independent application engine.
//!
//! Platform shells own microphone/session lifecycle and presentation. This crate
//! owns monitoring-session state, the streaming detector, persistence, settings,
//! and history projections shared by every UI.

pub mod flag;
pub mod instance;
pub mod monitor;
pub mod settings;
pub mod state;
pub mod sync;
pub mod teach;
