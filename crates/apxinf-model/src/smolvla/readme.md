# SmolVLA

This module contains the native ApxInf implementation of SmolVLA for LIBERO.
It intentionally does not depend on PyTorch, Transformers, Hugging Face model
code, or another Python inference framework. The Python policy uses the
repository's native tokenizer implementation; all model execution is handled by
ApxInf Rust code and CUDA kernels.

## Module layout

| File | Responsibility |
| --- | --- |
| `config.rs` | Model dimensions, image/patch settings, action shape, flow schedule, and checkpoint config parsing. |
| `weights.rs` | Safetensors loading, tensor layout conversion, weight packing, upload, and BF16/FP16 conversion. |
| `model.rs` | Vision encoder, VLM prefix encoder, action expert, cross-attention, flow matching, and phase timing. |
| `runtime.rs` | `VlaRuntime` integration, request preparation, initial-noise handling, and inference contract. |
| `load.rs` | Registry loading and checkpoint path resolution. |

The Python-facing integration is in `python/apxinf/apxinf/policies/impls/smolvla.py`.
It owns observation normalization, action denormalization, tokenizer setup, and
the `smolvla` / `smolvla_libero` policy registration. The maintained benchmark
and LIBERO entry points are `scripts/bench_smolvla.py` and
`scripts/eval_smolvla_libero.py`.

## Inference path

1. Accept two RGB views plus robot state and the tokenized task prompt.
2. Convert HWC/NHWC `uint8` images directly into patch-major normalized tensors
   on the GPU.
3. Encode the vision tower, project robot state, and build the language/vision
   prefix.
4. Run the VLM transformer and retain per-layer prefix keys and values.
5. Run the 10-step flow-matching action expert. The schedule advances from
   `1.0` to `0.1`; each step contains action/time projections, expert
   self-attention, cross-attention to the VLM prefix, and SwiGLU MLPs.
6. Slice the final latent to the configured action dimensions. For the LIBERO
   checkpoint, the output is `[50, 7]`.

The default model variant is BF16. The optional FP16 variant converts the
uploaded model tensors to FP16 and uses the FP16 GEMM path, which is
tensor-core-capable on Xavier's `sm_72`. BF16 and FP16 dispatch to matching
CUDA kernels for normalization, activation, attention, preprocessing, and
elementwise operations.

## Optimizations

- **GPU image preprocessing:** the `uint8`-to-patch kernel performs
  normalization, channel reordering, and patch-major layout conversion in one
  GPU operation instead of constructing patches on the CPU.
- **Packed projections:** QKV projections and gate/up MLP projections are
  packed into single GEMMs. Related biases are batched where possible.
- **FP16 execution:** all model-side GEMM operands use FP16 in the FP16
  variant, avoiding mixed FP32/BF16 GEMM fallbacks.
- **Fused CUDA kernels:** FP16 paths include RMSNorm, SwiGLU, QKV split with
  RoPE, bias+SiLU, concat, Euler flow update, and output slicing kernels.
- **Specialized attention kernels:** prefix GQA, suffix causal GQA, and full
  cross-attention paths avoid unnecessary decode-time attention work and keep
  attention entirely on the GPU.
- **GEMM tactics:** exact-shape cuBLAS tactic selection can be supplied through
  `--tactics`; the reported measurements use the tuned Xavier tactic table.
- **Stream-ordered allocation reuse:** operator outputs use an exact-size,
  stream-keyed CUDA allocation cache by default. This removes the thousands of
  blocking `cudaMalloc`/`cudaFree` pairs formerly issued by one inference. Set
  `APXINF_CUDA_ALLOC_CACHE=0` to disable the cache.
- **Uninitialized GEMM output:** GEMM writes every output element with
  `beta = 0`, so its output buffer is allocated without an avoidable
  `cudaMemset`.
- **Phase profiling:** CUDA events separately measure preprocessing, prefix
  embedding, VLM transformer, action expert, and output slicing.

SmolVLA cross-attention uses half-split RoPE for the query:
`x[..., :d/2]` rotates with `x[..., d/2:]`. This is intentionally different
from the interleaved-pair RoPE used elsewhere in the CUDA stack. The dedicated
FP16 half-split kernel fixed the original numerical mismatch with the reference
implementation.

## Numerical validation

The FP16 implementation was compared against a fixed-input LeRobot reference
rollout after the RoPE correction:

| Metric | Result |
| --- | --- |
| End-to-end relative Frobenius error | `0.006532` |
| First-step velocity relative error | `0.007712` |
| Cross-attention layer relative errors | `0.0065`–`0.0179` |
| Output shape | `[50, 7]` |
| Output values | all finite |

The CUDA test
`rope_half_split_f16_matches_fp32_reference` in
`crates/apxinf-cuda/src/tests/operators.rs` covers the corrected RoPE layout.

## Performance

Measurements were taken on the local Xavier `sm_72` GPU with the FP16 variant,
two `512x512` cameras, 20 benchmark iterations after 3 warmups, and the tuned
GEMM tactic table:

| Stage or metric | p50 latency |
| --- | ---: |
| End-to-end | `395.7 ms` |
| Model | `390.4 ms` |
| Preprocess | `4.4 ms` |
| Prefix embedding | `193.0 ms` |
| VLM transformer | `29.1 ms` |
| VLM prefix total | `222.4 ms` |
| Action expert | `160.8 ms` |
| Output slicing | `0.09 ms` |

With stream-ordered output reuse, the remaining dominant cost is prefix
embedding, followed by the action expert. The VLM transformer itself is
comparatively small after the prefix is built.

## LIBERO spatial result

The corrected FP16 implementation was evaluated with
`scripts/eval_smolvla_libero.py` on `libero_spatial`, one rollout per task:

| Setting | Value |
| --- | --- |
| Tasks | 10 |
| Trials per task | 1 |
| Completed runs | 10 |
| Successes | 8 |
| Success rate | **80%** |
| Failed tasks | task `4`, task `7` |
| Max steps | 520 |
| Replan interval | 5 actions |
| Seed | 7 |
| Mean model time | `425.5 ms/call` |
| Mean inference time | `438.8 ms/call` |

The successful tasks completed in 68–116 environment action steps. The two
failures reached the 520-step timeout. This result is from the post-RoPE-fix,
allocation-reuse FP16 wheel and is not comparable to the earlier all-failure
debug run.

## Validation commands

```bash
CUDA_PATH=/path/to/cuda cargo check -p apxinf-model --features cuda
CUDA_PATH=/path/to/cuda cargo test -p apxinf-cuda rope_half_split_f16_matches_fp32_reference
python -m pytest python/apxinf/tests/test_smolvla_policy.py
```

Benchmark and LIBERO evaluation use the maintained scripts:

```bash
python scripts/bench_smolvla.py --model-dir /path/to/checkpoint \
  --model-variant fp16 --num-views 2 --image-size 512 --profile
python scripts/eval_smolvla_libero.py --model-dir /path/to/checkpoint \
  --model-variant fp16 --suite libero_spatial --tasks all \
  --trials-per-task 1
```
