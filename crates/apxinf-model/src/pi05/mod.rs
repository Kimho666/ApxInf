//! Physical Intelligence π0.5 vision-language-action model.
//!
//! OpenPI defines the model math, LeRobot defines the distributed checkpoint
//! contract. Loading selects model Blocks; Model owns model dataflow;
//! ModelRunner and prepare own execution policy and stable graph resources.
//! CUDA crates expose kernels and device primitives.

#[cfg(feature = "cuda-new")]
mod backend;
mod config;
#[cfg(feature = "cuda-new")]
mod load;
mod math;
#[cfg(feature = "cuda-new")]
mod model;
#[cfg(feature = "cuda-new")]
mod model_runner;
mod weights;

pub use config::{GemmaVariantConfig, ModelVariantChoice, Pi05Config, Pi05PerformanceProfile};
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};
#[cfg(feature = "cuda-new")]
pub use model::Pi05CalibrationObserver;
#[cfg(feature = "cuda-new")]
pub use model::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
#[cfg(feature = "cuda-new")]
pub use model::{
    action_layer_fp8_static, language_layer_fp8_static, vision_layer_fp8_static,
    vision_patch_embed_fp8_static, vision_patch_embed_fp8_static_native,
    Fp8StaticActionLayerOutput, Fp8StaticLanguageLayerOutput,
};
#[cfg(feature = "cuda-new")]
pub use model::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
#[cfg(feature = "cuda-new")]
pub use model_runner::{Pi05ModelRunner, Pi05PreparedInference};
pub use weights::*;

#[cfg(feature = "cuda-new")]
pub(crate) use load::load_with_cuda_new;

#[cfg(feature = "cuda-new")]
pub use backend::ImageLayout as Pi05ImageLayout;
#[cfg(feature = "cuda-new")]
pub use model::{
    build_bf16_model, build_fp8_static_model, build_int8_dynamic_model,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic,
};
#[cfg(feature = "cuda-new")]
pub use model::{Bf16Model, Fp8StaticModel, Int8DynamicModel};
#[cfg(feature = "cuda-new")]
pub use model::{Bf16PrefixKvCache, Fp8StaticPrefixKvCache, Int8DynamicPrefixKvCache};
#[cfg(feature = "cuda-new")]
pub use model_runner::{capture_patches, capture_rgb, CapturedGraph};
