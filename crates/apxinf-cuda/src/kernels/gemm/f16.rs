use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use half::f16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::kernels::elementwise::bias_f16;
use crate::kernels::fused::bias_residual_f16;
use crate::tuning::{
    AutoTuneConfig, AutoTuneEngine, CandidateMeasurement, DeviceFingerprint, Epilogue, GemmLayout,
    GemmOp, GemmTuningKey, ScaleMode, TacticBackend, TacticId, TuningDType, TuningOutcome,
};
use crate::workspace::output_buffer;

struct CudaEventPair {
    start: ffi::cudaEvent_t,
    stop: ffi::cudaEvent_t,
}

impl CudaEventPair {
    fn new() -> Result<Self> {
        let mut events = Self {
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
        };
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreate(&mut events.start)).map_err(Error::Cuda)?;
            if let Err(error) = ffi::check_cuda(ffi::cudaEventCreate(&mut events.stop)) {
                let _ = ffi::cudaEventDestroy(events.start);
                return Err(Error::Cuda(error));
            }
        }
        Ok(events)
    }

    fn measure(
        &self,
        ctx: &CudaContext,
        evictor: &mut ColdL2Evictor,
        launch: impl FnOnce() -> Result<()>,
    ) -> Result<f64> {
        evictor.evict(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.start, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
        }
        launch()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.stop, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventSynchronize(self.stop)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventElapsedTime(
                &mut milliseconds,
                self.start,
                self.stop,
            ))
            .map_err(Error::Cuda)?;
        }
        Ok(f64::from(milliseconds))
    }
}

impl Drop for CudaEventPair {
    fn drop(&mut self) {
        unsafe {
            if !self.start.is_null() {
                let _ = ffi::cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                let _ = ffi::cudaEventDestroy(self.stop);
            }
        }
    }
}

struct ColdL2Evictor {
    buffer: CudaBuffer,
    bytes: usize,
    seed: u32,
}

impl ColdL2Evictor {
    fn new(ctx: &CudaContext) -> Result<Self> {
        const CUDA_DEV_ATTR_L2_CACHE_SIZE: i32 = 38;
        let mut l2_cache_bytes = 0i32;
        unsafe {
            ffi::check_cuda(ffi::cudaDeviceGetAttribute(
                &mut l2_cache_bytes,
                CUDA_DEV_ATTR_L2_CACHE_SIZE,
                ctx.device_id() as i32,
            ))
            .map_err(Error::Cuda)?;
        }
        let l2_cache_bytes = usize::try_from(l2_cache_bytes)
            .ok()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::Other("CUDA reported an empty L2 cache".into()))?;
        let bytes = l2_cache_bytes
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(255))
            .map(|bytes| bytes & !255usize)
            .ok_or_else(|| Error::Other("cold-L2 eviction buffer size overflow".into()))?;
        Ok(Self {
            buffer: CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?,
            bytes,
            seed: 0,
        })
    }

    fn evict(&mut self, ctx: &CudaContext) -> Result<()> {
        self.seed = self.seed.wrapping_add(1);
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_evict_l2(
                self.buffer.ptr(),
                self.bytes,
                self.seed,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        }
    }
}

fn tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Fp16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::F16,
        weight_dtype: TuningDType::F16,
        output_dtype: TuningDType::F16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::None,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    }
}

fn copy_f16_output(output: &CudaBuffer, elements: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; elements * DType::F16.size_in_bytes()];
    output.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_tactic_f16(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    tactic: TacticId,
) -> Result<()> {
    match tactic.backend {
        TacticBackend::Vendor => ctx
            .cublas()
            .gemm(
                DType::F16,
                key.m,
                key.n,
                key.k,
                1.0,
                activation,
                weight,
                0.0,
                output,
            )
            .map_err(Error::Cuda),
        TacticBackend::CublasLt => unsafe {
            ffi::check_cublas(ffi::apxinf_static_f16_gemm(
                activation.ptr(),
                weight.ptr(),
                output.ptr(),
                key.m as i32,
                key.n as i32,
                key.k as i32,
                1.0,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
        _ => Err(Error::Other(format!(
            "FP16 online autotune cannot execute {tactic:?}"
        ))),
    }
}

fn prepare_tactic_f16(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    super::providers::prepare(key, tactic)
}

fn resolve_f16_plan(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
) -> Result<super::PreparedGemmPlan> {
    ctx.gemm_plans()
        .resolve_or_tune(ctx, key, super::plan::default_f16_tactic(), |preferred| {
            autotune_request_f16(ctx, key, activation, weight, preferred)
        })
}

fn autotune_request_f16(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let elements = key
        .m
        .checked_mul(key.n)
        .ok_or_else(|| Error::Other("FP16 autotune output size overflow".into()))?;
    let bytes = elements
        .checked_mul(DType::F16.size_in_bytes())
        .ok_or_else(|| Error::Other("FP16 autotune output size overflow".into()))?;
    let reference_output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let reference_tactic = TacticId {
        backend: TacticBackend::Vendor,
        value: 0,
    };
    prepare_tactic_f16(key, reference_tactic)?;
    launch_tactic_f16(ctx, key, activation, weight, &reference_output, reference_tactic)?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_f16_output(&reference_output, elements)?;
    drop(reference_output);

    let output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    let candidates = super::providers::candidates(key, 64).into_iter();
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        prepare_tactic_f16(key, candidate.tactic)?;
        launch_tactic_f16(ctx, key, activation, weight, &output, candidate.tactic)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_f16_output(&output, elements)?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.01, 0.9999);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            evictor.evict(ctx)?;
            launch_tactic_f16(ctx, key, activation, weight, &output, candidate.tactic)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || {
                launch_tactic_f16(ctx, key, activation, weight, &output, candidate.tactic)
            })?;
        }
        Ok(CandidateMeasurement {
            tactic: candidate.tactic,
            milliseconds: Some(milliseconds / config.benchmark_iterations as f64),
            correct: true,
        })
    })
}

pub(crate) fn set_cublaslt_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    heuristic_rank: i32,
) -> Result<()> {
    if !(0..64).contains(&heuristic_rank) {
        return Err(Error::Other(format!(
            "invalid FP16 cuBLASLt heuristic rank {heuristic_rank}"
        )));
    }
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_f16_gemm_heuristic(
            m as i32,
            n as i32,
            k as i32,
            heuristic_rank,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub(crate) fn prepare_cublaslt_gemm(m: usize, n: usize, k: usize) -> Result<()> {
    let status = unsafe { ffi::apxinf_static_prepare_f16_gemm(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

/// Physical FP16 GEMM contract: `[M,K] @ [K,N] -> [M,N]`.
pub fn gemm_f16(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() != DType::F16 || weight.dtype() != DType::F16 {
        return Err(Error::Other(format!(
            "gemm_f16 expects FP16 operands, got {} and {}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let activation_shape = activation.shape().dims();
    let weight_shape = weight.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
    {
        return Err(Error::Other(format!(
            "gemm_f16 shape mismatch: {activation_shape:?} @ {weight_shape:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.device()
            },
        });
    }

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = resolve_f16_plan(ctx, &key, &activation, &weight)?;
    if plan.tactic.backend == TacticBackend::CublasLt {
        let tuned_result = (|| -> Result<()> {
            unsafe {
                let status = ffi::apxinf_static_f16_gemm(
                    activation.ptr(),
                    weight.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    ctx.stream().handle(),
                );
                ffi::check_cublas(status).map_err(Error::Cuda)?;
            }
            Ok(())
        })();
        if tuned_result.is_ok() {
            return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
        }
        let error = tuned_result.unwrap_err();
        eprintln!(
            "[apxinf] FP16 tactic {:?} failed for {key:?}: {error}; using vendor fallback",
            plan.tactic
        );
        ctx.gemm_plans().fallback(ctx, &key)?;
    }
    ctx.cublas()
        .gemm(
            DType::F16,
            m,
            n,
            k,
            1.0,
            &activation,
            &weight,
            0.0,
            &output,
        )
        .map_err(Error::Cuda)?;
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16))
}

/// Physical FP16 residual GEMM: `[M,K] @ [K,N] + [M,N] -> [M,N]`.
pub fn gemm_f16_residual(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || residual.dtype() != DType::F16
    {
        return Err(Error::Other(format!(
            "gemm_f16_residual expects FP16 operands, got {}, {}, and {}",
            activation.dtype(),
            weight.dtype(),
            residual.dtype()
        )));
    }
    let activation_shape = activation.shape().dims();
    let weight_shape = weight.shape().dims();
    let residual_shape = residual.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
        || residual_shape != [activation_shape[0], weight_shape[1]]
    {
        return Err(Error::Other(format!(
            "gemm_f16_residual shape mismatch: {activation_shape:?} @ {weight_shape:?} + {residual_shape:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device
        || weight.device() != expected_device
        || residual.device() != expected_device
    {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else if weight.device() != expected_device {
                weight.device()
            } else {
                residual.device()
            },
        });
    }

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = resolve_f16_plan(ctx, &key, &activation_buffer, &weight_buffer)?;
    if plan.tactic.backend == TacticBackend::CublasLt {
        let fused_result = unsafe {
            ffi::apxinf_static_f16_gemm_residual(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                residual_buffer.ptr(),
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                1.0,
                ctx.stream().handle(),
            )
        };
        if ffi::check_cublas(fused_result).is_ok() {
            return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
        }
    }
    drop(output);
    drop(activation_buffer);
    drop(weight_buffer);
    drop(residual_buffer);
    let projected = gemm_f16(ctx, activation, weight)?;
    bias_residual_f16(ctx, &projected, None, residual)
}

/// Physical FP16 biased GEMM: `[M,K] @ [K,N] + [N] -> [M,N]`.
pub fn gemm_f16_bias(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || bias.dtype() != DType::F16
    {
        return Err(Error::Other(format!(
            "gemm_f16_bias expects FP16 operands, got {}, {}, and {}",
            activation.dtype(),
            weight.dtype(),
            bias.dtype()
        )));
    }
    let activation_shape = activation.shape().dims();
    let weight_shape = weight.shape().dims();
    let bias_shape = bias.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
        || bias_shape != [weight_shape[1]]
    {
        return Err(Error::Other(format!(
            "gemm_f16_bias shape mismatch: {activation_shape:?} @ {weight_shape:?} + {bias_shape:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device
        || weight.device() != expected_device
        || bias.device() != expected_device
    {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else if weight.device() != expected_device {
                weight.device()
            } else {
                bias.device()
            },
        });
    }

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = resolve_f16_plan(ctx, &key, &activation_buffer, &weight_buffer)?;
    if plan.tactic.backend == TacticBackend::CublasLt {
        let prepare_status = unsafe {
            ffi::apxinf_static_prepare_f16_gemm_bias(
                m as i32,
                n as i32,
                k as i32,
                bias_buffer.ptr(),
            )
        };
        if ffi::check_cublas(prepare_status).is_ok() {
            let launch_status = unsafe {
                ffi::apxinf_static_f16_gemm_bias(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    bias_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    ctx.stream().handle(),
                )
            };
            if ffi::check_cublas(launch_status).is_ok() {
                return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
            }
        }
    }
    drop(output);
    drop(activation_buffer);
    drop(weight_buffer);
    drop(bias_buffer);
    let projected = gemm_f16(ctx, activation, weight)?;
    bias_f16(ctx, &projected, Some(bias))
}

/// Physical FP16 biased residual GEMM: `[M,K] @ [K,N] + [N] + [M,N]`.
pub fn gemm_f16_bias_residual(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || bias.dtype() != DType::F16
        || residual.dtype() != DType::F16
    {
        return Err(Error::Other(format!(
            "gemm_f16_bias_residual expects FP16 operands, got {}, {}, {}, and {}",
            activation.dtype(),
            weight.dtype(),
            bias.dtype(),
            residual.dtype()
        )));
    }
    let activation_shape = activation.shape().dims();
    let weight_shape = weight.shape().dims();
    let bias_shape = bias.shape().dims();
    let residual_shape = residual.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
        || bias_shape != [weight_shape[1]]
        || residual_shape != [activation_shape[0], weight_shape[1]]
    {
        return Err(Error::Other(format!(
            "gemm_f16_bias_residual shape mismatch: {activation_shape:?} @ {weight_shape:?} + {bias_shape:?} + {residual_shape:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device
        || weight.device() != expected_device
        || bias.device() != expected_device
        || residual.device() != expected_device
    {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else if weight.device() != expected_device {
                weight.device()
            } else if bias.device() != expected_device {
                bias.device()
            } else {
                residual.device()
            },
        });
    }

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = resolve_f16_plan(ctx, &key, &activation_buffer, &weight_buffer)?;
    if plan.tactic.backend == TacticBackend::CublasLt {
        let prepare_status = unsafe {
            ffi::apxinf_static_prepare_f16_gemm_bias_residual(
                m as i32,
                n as i32,
                k as i32,
                bias_buffer.ptr(),
            )
        };
        if ffi::check_cublas(prepare_status).is_ok() {
            let launch_status = unsafe {
                ffi::apxinf_static_f16_gemm_bias_residual(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    bias_buffer.ptr(),
                    residual_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    ctx.stream().handle(),
                )
            };
            if ffi::check_cublas(launch_status).is_ok() {
                return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
            }
        }
    }
    drop(output);
    drop(activation_buffer);
    drop(weight_buffer);
    drop(bias_buffer);
    drop(residual_buffer);
    let projected = gemm_f16(ctx, activation, weight)?;
    bias_residual_f16(ctx, &projected, Some(bias), residual)
}
