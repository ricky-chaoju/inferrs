//! Multimodal (audio + vision) encoder plugins for `inferrs`.
//!
//! Compiled as a `cdylib` for runtime dlopen loading.  The `ffi` module
//! exports the C ABI entry points (`inferrs_mm_*`) that the host binary calls
//! after dlopening this library.
//!
//! Modules:
//! - `audio`         — log-mel spectrogram computation (pure CPU)
//! - `audio_encoder` — Gemma4 conformer audio encoder
//! - `vision_encoder`— SigLIP2 vision encoder
//! - `config`        — `AudioConfig`, `Gemma4VisionConfig`, etc.
//! - `ffi`           — C ABI exports (`inferrs_mm_*`)

pub mod audio;
pub mod audio_encoder;
pub mod config;
pub mod ffi;
pub mod vision_encoder;
