//! Multimodal configuration types for audio and vision encoders.
//!
//! These types are extracted from `inferrs/src/config.rs` so that the
//! `inferrs-multimodal` crate can be built independently of the main binary.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Audio encoder configuration
// ---------------------------------------------------------------------------

/// Audio encoder configuration from `audio_config` in `config.json`.
#[derive(Debug, Deserialize, Clone)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub output_proj_dims: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    #[allow(dead_code)]
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_logit_cap")]
    pub attention_logit_cap: f64,
    #[serde(default = "default_invalid_logit")]
    pub attention_invalid_logits_value: f64,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f64,
    #[allow(dead_code)]
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_conv_kernel_size() -> usize {
    5
}
fn default_logit_cap() -> f64 {
    50.0
}
fn default_invalid_logit() -> f64 {
    -1e9
}
fn default_residual_weight() -> f64 {
    0.5
}
fn default_gradient_clipping() -> f64 {
    1e10
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

// ---------------------------------------------------------------------------
// Vision encoder configuration
// ---------------------------------------------------------------------------

/// SigLIP2 ViT vision encoder configuration (Gemma4).
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub position_embedding_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f64,
    pub use_clipped_linears: bool,
    pub standardize: bool,
    #[serde(default = "default_vision_hidden_activation")]
    pub hidden_activation: String,
}

/// Qwen vision encoder configuration (shared by Qwen3.5 and Qwen3-VL).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
#[allow(dead_code)]
pub struct QwenVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub initializer_range: f64,
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_vision_head_dim() -> usize {
    64
}
fn default_vision_rope_theta() -> f64 {
    100.0
}
fn default_vision_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

/// Vision encoder configuration from `vision_config` in `config.json`.
#[derive(Debug, Clone)]
pub enum VisionConfig {
    Gemma4(Gemma4VisionConfig),
    Qwen(QwenVisionConfig),
}

/// Parameters the server needs for image preprocessing, regardless of encoder
/// architecture.
pub struct VisionPreprocessParams {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
}

impl VisionConfig {
    pub fn preprocess_params(&self) -> VisionPreprocessParams {
        match self {
            VisionConfig::Gemma4(c) => VisionPreprocessParams {
                patch_size: c.patch_size,
                pooling_kernel_size: c.pooling_kernel_size,
                default_output_length: c.default_output_length,
            },
            VisionConfig::Qwen(c) => VisionPreprocessParams {
                patch_size: c.patch_size,
                pooling_kernel_size: if c.spatial_merge_size > 0 {
                    c.spatial_merge_size
                } else {
                    3
                },
                default_output_length: 280,
            },
        }
    }
}

impl<'de> serde::Deserialize<'de> for VisionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let raw = serde_json::Value::deserialize(de)?;
        let model_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        match model_type {
            "qwen3_5" | "qwen3_vl" => {
                let cfg: QwenVisionConfig =
                    serde_json::from_value(raw).map_err(serde::de::Error::custom)?;
                Ok(VisionConfig::Qwen(cfg))
            }
            _ => {
                let cfg: Gemma4VisionConfig =
                    serde_json::from_value(raw).map_err(serde::de::Error::custom)?;
                Ok(VisionConfig::Gemma4(cfg))
            }
        }
    }
}

/// Lenient deserializer for `RawConfig.vision_config`: returns `None` for
/// unknown or unsupported vision encoder types instead of failing the whole
/// config parse.
pub fn deserialize_vision_config_opt<'de, D>(
    de: D,
) -> std::result::Result<Option<VisionConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(de)?;
    let Some(val) = raw else { return Ok(None) };

    let model_type = val.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    match model_type {
        "qwen3_5" | "qwen3_vl" => {
            let cfg: QwenVisionConfig =
                serde_json::from_value(val).map_err(serde::de::Error::custom)?;
            Ok(Some(VisionConfig::Qwen(cfg)))
        }
        // Gemma3's vision config uses "siglip_vision_model" which has a different
        // schema from Gemma4's SigLIP2 config — skip it silently.
        "siglip_vision_model" => Ok(None),
        _ => match serde_json::from_value::<Gemma4VisionConfig>(val) {
            Ok(cfg) => Ok(Some(VisionConfig::Gemma4(cfg))),
            Err(_) => Ok(None),
        },
    }
}
