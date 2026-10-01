use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::{Backend, Error, NormalGenerator, Result, SamplingBackend, Tensor};
use half::{bf16, f16};

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VlaContract, VlaRequest, VlaRuntime,
};

use super::{SmolVlaConfig, SmolVlaModel};

pub struct SmolVlaModelRunner {
    model: Arc<SmolVlaModel>,
    config: Arc<SmolVlaConfig>,
}

impl SmolVlaModelRunner {
    pub fn new(model: Arc<SmolVlaModel>, config: Arc<SmolVlaConfig>) -> Self {
        Self { model, config }
    }

    fn noise(&self, latent: InitialLatent<'_>) -> Result<Tensor> {
        match latent {
            InitialLatent::Provided(tensor) => {
                let shape = tensor.shape().dims();
                if tensor.device() != apxinf_core::Device::Cpu
                    || shape.len() != 2
                    || shape[0] != self.config.action_horizon
                    || shape[1] > self.config.max_action_dim
                {
                    return Err(Error::Other(format!(
                        "SmolVLA provided latent must be CPU [{}, 0..={}]",
                        self.config.action_horizon, self.config.max_action_dim
                    )));
                }
                let source_width = shape[1];
                let values = tensor.to_f32_vec()?;
                let mut padded = vec![0.0; self.config.action_horizon * self.config.max_action_dim];
                for (row, source) in values.chunks_exact(source_width).enumerate() {
                    let target = &mut padded
                        [row * self.config.max_action_dim..(row + 1) * self.config.max_action_dim];
                    target[..source_width].copy_from_slice(source);
                }
                let host = if self.model.uses_fp16_gemm() {
                    Tensor::from_f16(
                        vec![
                            self.config.action_horizon,
                            self.config.max_action_dim,
                        ],
                        &padded.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
                    )?
                } else {
                    Tensor::from_bf16(
                        vec![
                            self.config.action_horizon,
                            self.config.max_action_dim,
                        ],
                        &padded.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
                    )?
                };
                self.model.backend().to_device(&host)
            }
            InitialLatent::Generate { rng } => {
                let host = Tensor::zeros(
                    vec![self.config.action_horizon, self.config.max_action_dim],
                    self.model.dtype(),
                );
                let output = self.model.backend().to_device(&host)?;
                let mut generator = self.model.backend().create_normal_generator(output)?;
                generator.generate(rng)?;
                Ok(generator.output().clone())
            }
        }
    }

    fn infer_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Action, BTreeMap<String, f64>)> {
        let observation = request.observation;
        observation.validate()?;
        let token_bytes = observation
            .token_ids
            .iter()
            .flat_map(|token| u32::to_ne_bytes(*token))
            .collect::<Vec<_>>();
        let token_ids = crate::accelerator::cuda::DeviceBuffer::alloc(
            token_bytes.len(),
            self.model.backend().context().device_id(),
        )
        .map_err(Error::Cuda)?;
        token_ids
            .copy_from_host(&token_bytes)
        .map_err(Error::Cuda)?;
        let noise = self.noise(request.initial_latent)?;
        let (tensor, timing) =
            self.model
                .infer_with_timing(observation, &noise, &token_ids)?;
        Ok((Action::new(tensor), timing.as_map()))
    }

}

impl VlaRuntime for SmolVlaModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(if self.model.uses_fp16_gemm() {
            "fp16"
        } else {
            "bf16"
        })
    }

    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [self.config.action_horizon, self.config.action_dim],
            patch_shape: [
                self.config.num_views * self.config.patches_per_view(),
                3 * self.config.patch_size * self.config.patch_size,
            ],
            max_token_len: self.config.max_token_len,
            num_views: self.config.num_views,
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
            accepts_rgb_u8: true,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.infer_profiled(request)
            .map(|(action, _profile)| action)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::Eager)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        if spec.token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count {} exceeds maximum {}",
                spec.token_count, self.config.max_token_len
            )));
        }
        match spec.image_layout {
            None | Some(ImageLayout::Nhwc) | Some(ImageLayout::Nchw) => {}
        }
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::Other(
                "SmolVLA CUDA graph preparation is not supported yet".into(),
            ));
        }
        Ok(Box::new(SmolVlaPreparedInference {
            spec: *spec,
            model: Arc::clone(&self.model),
            config: Arc::clone(&self.config),
        }))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        sample.observation.validate()?;
        self.prepare_with_policy(&sample.observation.inference_spec(), policy)
    }

    fn clear_prepared(&self) -> Result<()> {
        self.model.backend().synchronize()
    }

    fn execution_mode(&self) -> &'static str {
        "eager"
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.model.backend().to_cpu(action.tensor())?.to_f32_vec()
    }

    fn infer_host_f32_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, BTreeMap<String, f64>)> {
        let (action, profile) = self.infer_profiled(request)?;
        let values = self
            .model
            .backend()
            .to_cpu(action.tensor())?
            .to_f32_vec()?;
        Ok((values, profile))
    }
}

struct SmolVlaPreparedInference {
    spec: InferenceSpec,
    model: Arc<SmolVlaModel>,
    config: Arc<SmolVlaConfig>,
}

impl PreparedInference for SmolVlaPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: None,
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "SmolVLA request does not match prepared inference spec".into(),
            ));
        }
        let runner = SmolVlaModelRunner::new(Arc::clone(&self.model), Arc::clone(&self.config));
        runner.infer(request)
    }
}
