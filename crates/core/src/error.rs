//! Error type shared across the core pipeline.

use thiserror::Error;

/// Errors surfaced by the core pipeline. Constructors never panic; callers
/// (CLI, desktop app) render these gracefully — see SPEC §4/§8.
#[derive(Debug, Error)]
pub enum Error {
    #[error("audio i/o: {0}")]
    Audio(String),

    #[error("wav decode: {0}")]
    Wav(#[from] hound::Error),

    #[error("database: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("sync http: {0}")]
    Http(String),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("token store: {0}")]
    Token(String),

    /// The ONNX model file was not found / could not be loaded. The app surfaces
    /// this as a "model missing" state rather than crashing (SPEC §4 stage ③).
    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("invalid configuration: {0}")]
    Config(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
