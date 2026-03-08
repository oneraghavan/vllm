# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FLA Triton kernel warmup fix (issue #34954).

Validates that:
1. The _warmup_fla_kernels class method has correct flag management,
   tensor shapes, and one-shot semantics.
2. The _forward_core method calls warmup when attn_metadata is None
   (profile_run path).
3. On GPU, FLA kernels can be pre-warmed so that subsequent calls
   under memory pressure succeed (the core bug scenario).
"""

import threading
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform

_cuda_mod = torch.cuda
_empty_cache_fn = getattr(torch.accelerator, "empty_cache", _cuda_mod.empty_cache)


# Importing Qwen3NextGatedDeltaNet triggers the full model import chain
# including _custom_ops, which can fail in non-editable-install dev setups
# due to duplicate op registration.  Guard so the tests that need it are
# skipped gracefully, while the direct-kernel GPU tests always run.
try:
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    _CAN_IMPORT_GDN = True
except (RuntimeError, ImportError):
    _CAN_IMPORT_GDN = False

_requires_gdn = pytest.mark.skipif(
    not _CAN_IMPORT_GDN,
    reason="Cannot import Qwen3NextGatedDeltaNet "
    "(likely _custom_ops registration conflict in dev setup)",
)


def _make_mock_layer(
    *,
    num_k_heads=4,
    num_v_heads=4,
    tp_size=1,
    head_k_dim=64,
    head_v_dim=64,
    dtype=torch.bfloat16,
    device="cpu",
):
    """Minimal mock matching the attributes _warmup_fla_kernels reads."""

    class _MockLayer:
        num_k_heads: int
        num_v_heads: int
        tp_size: int
        head_k_dim: int
        head_v_dim: int

        def __init__(
            self, *, nkh: int, nvh: int, tp: int, kd: int, vd: int, param: torch.Tensor
        ):
            self.num_k_heads = nkh
            self.num_v_heads = nvh
            self.tp_size = tp
            self.head_k_dim = kd
            self.head_v_dim = vd
            self._param = param

        def parameters(self):
            yield self._param

    return _MockLayer(
        nkh=num_k_heads,
        nvh=num_v_heads,
        tp=tp_size,
        kd=head_k_dim,
        vd=head_v_dim,
        param=torch.zeros(1, dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Unit tests (CPU, no GPU required) — mock out fla_chunk_gated_delta_rule
# ---------------------------------------------------------------------------


@_requires_gdn
@pytest.mark.skip_global_cleanup
class TestWarmupFlagManagement:
    """Verify the one-shot flag logic without executing real kernels."""

    def setup_method(self):
        Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False

    def teardown_method(self):
        Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False

    def test_flag_starts_false(self):
        assert Qwen3NextGatedDeltaNet._fla_kernels_warmed_up is False

    def test_warmup_sets_flag(self):
        """After one call the flag must be True."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ):
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(_make_mock_layer())
            assert Qwen3NextGatedDeltaNet._fla_kernels_warmed_up is True

    def test_warmup_runs_only_once(self):
        """fla_chunk_gated_delta_rule must be called exactly once across
        multiple _warmup_fla_kernels invocations."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            layer = _make_mock_layer()
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)
            assert mock_fla.call_count == 1

    def test_warmup_thread_safe(self):
        """Concurrent calls from multiple threads must still invoke
        fla_chunk_gated_delta_rule exactly once (double-checked locking)."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            barrier = threading.Barrier(8)

            def _call():
                barrier.wait()
                Qwen3NextGatedDeltaNet._warmup_fla_kernels(_make_mock_layer())

            threads = [threading.Thread(target=_call) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert mock_fla.call_count == 1

    def test_warmup_passes_correct_shapes(self):
        """Verify dummy tensors have shapes derived from layer attributes."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            layer = _make_mock_layer(
                num_k_heads=16,
                num_v_heads=32,
                tp_size=1,
                head_k_dim=128,
                head_v_dim=128,
            )
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)

            kw = mock_fla.call_args.kwargs
            assert kw["q"].shape == (1, 128, 16, 128)
            assert kw["v"].shape == (1, 128, 32, 128)
            assert kw["g"].shape == (1, 128, 32)
            assert kw["beta"].shape == (1, 128, 32)
            assert kw["initial_state"].shape == (1, 32, 128, 128)
            assert kw["cu_seqlens"].tolist() == [0, 128]
            assert kw["output_final_state"] is True
            assert kw["use_qk_l2norm_in_kernel"] is True

    def test_warmup_respects_tp_size(self):
        """With tp_size > 1 the head counts must be divided."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            layer = _make_mock_layer(
                num_k_heads=16,
                num_v_heads=32,
                tp_size=2,
                head_k_dim=128,
                head_v_dim=128,
            )
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)

            kw = mock_fla.call_args.kwargs
            assert kw["q"].shape[2] == 8  # 16 / 2
            assert kw["v"].shape[2] == 16  # 32 / 2

    def test_warmup_casts_fp32_to_bf16(self):
        """fp32 layers should have warmup tensors in bf16."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(
                _make_mock_layer(dtype=torch.float32)
            )
            assert mock_fla.call_args.kwargs["q"].dtype == torch.bfloat16

    def test_warmup_preserves_bf16(self):
        """bf16 layers should keep bf16 dtype."""
        with mock.patch(
            "vllm.model_executor.models.qwen3_next.fla_chunk_gated_delta_rule"
        ) as mock_fla:
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(
                _make_mock_layer(dtype=torch.bfloat16)
            )
            assert mock_fla.call_args.kwargs["q"].dtype == torch.bfloat16


@_requires_gdn
@pytest.mark.skip_global_cleanup
class TestForwardCoreProfilePath:
    """Verify _forward_core calls warmup when attn_metadata is None."""

    def setup_method(self):
        Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False

    def teardown_method(self):
        Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False

    def test_forward_core_calls_warmup_when_attn_metadata_none(self):
        """During profile_run, attn_metadata is None and _warmup should
        be invoked."""
        mock_forward_ctx = mock.MagicMock()
        mock_forward_ctx.attn_metadata = None

        with (
            mock.patch(
                "vllm.model_executor.models.qwen3_next.get_forward_context",
                return_value=mock_forward_ctx,
            ),
            mock.patch.object(
                Qwen3NextGatedDeltaNet,
                "_warmup_fla_kernels",
            ) as mock_warmup,
        ):
            instance = mock.MagicMock(spec=Qwen3NextGatedDeltaNet)
            Qwen3NextGatedDeltaNet._forward_core(
                instance,
                mixed_qkv=torch.empty(0),
                b=torch.empty(0),
                a=torch.empty(0),
                core_attn_out=torch.empty(0),
            )
            mock_warmup.assert_called_once_with(instance)

    def test_forward_core_returns_early_during_profile(self):
        """_forward_core should return immediately during profile_run
        without processing attn_metadata further."""
        mock_forward_ctx = mock.MagicMock()
        mock_forward_ctx.attn_metadata = None

        with (
            mock.patch(
                "vllm.model_executor.models.qwen3_next.get_forward_context",
                return_value=mock_forward_ctx,
            ),
            mock.patch.object(
                Qwen3NextGatedDeltaNet,
                "_warmup_fla_kernels",
            ),
        ):
            instance = mock.MagicMock(spec=Qwen3NextGatedDeltaNet)
            result = Qwen3NextGatedDeltaNet._forward_core(
                instance,
                mixed_qkv=torch.empty(0),
                b=torch.empty(0),
                a=torch.empty(0),
                core_attn_out=torch.empty(0),
            )
            assert result is None


# ---------------------------------------------------------------------------
# GPU integration tests — use fla_chunk_gated_delta_rule directly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA GPU")
@pytest.mark.skip_global_cleanup
class TestFLAWarmupGPU:
    """Run actual FLA Triton kernels on GPU to verify the warmup pattern
    prevents OOM under memory pressure."""

    @pytest.fixture(autouse=True)
    def _setup_triton_allocator(self):
        """Triton 3.4+ requires an explicit allocator."""
        try:
            import triton

            device = torch.device("cuda:0")
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device=device, dtype=torch.int8
                )
            )
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def _run_fla_kernel(H_K, H_V, K, V, T=128):
        """Run chunk_gated_delta_rule with the given dimensions."""
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule,
        )

        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        q = torch.randn(1, T, H_K, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H_K, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H_V, V, dtype=dtype, device=device)
        g = torch.randn(1, T, H_V, dtype=dtype, device=device)
        beta = torch.rand(1, T, H_V, dtype=dtype, device=device).sigmoid()
        initial_state = torch.zeros(1, H_V, V, K, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

        o, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        torch.accelerator.synchronize()
        return o

    def test_fla_kernel_basic(self):
        """FLA kernel executes successfully with small dimensions."""
        o = self._run_fla_kernel(H_K=4, H_V=4, K=64, V=64)
        assert o.shape == (1, 128, 4, 64)

    def test_fla_kernel_qwen3next_dims(self):
        """FLA kernel executes with actual Qwen3-Next dimensions."""
        o = self._run_fla_kernel(H_K=16, H_V=32, K=128, V=128)
        assert o.shape == (1, 128, 32, 128)

    def test_warmup_then_memory_pressure(self):
        """Core scenario: after warmup, FLA kernels succeed even under
        memory pressure because Triton autotune results are cached."""
        device = torch.device("cuda:0")
        H_K, H_V, K, V = 4, 4, 64, 64

        # Step 1: "warmup" — run kernel while memory is free
        self._run_fla_kernel(H_K, H_V, K, V)

        # Step 2: consume most GPU memory
        _empty_cache_fn()
        free = torch.cuda.mem_get_info()[0]
        target_free = 512 * 1024 * 1024  # leave ~512 MB
        alloc_bytes = max(0, free - target_free)
        hog = torch.empty(alloc_bytes, dtype=torch.uint8, device=device)

        try:
            # Step 3: re-run kernel — reuses cached autotune config
            o = self._run_fla_kernel(H_K, H_V, K, V)
            assert o.shape == (1, 128, H_V, V)
        finally:
            del hog
            _empty_cache_fn()

    @_requires_gdn
    def test_warmup_via_class_method_on_gpu(self):
        """_warmup_fla_kernels executes real FLA kernels on GPU."""
        Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False
        try:
            layer = _make_mock_layer(
                num_k_heads=4,
                num_v_heads=4,
                tp_size=1,
                head_k_dim=64,
                head_v_dim=64,
                dtype=torch.bfloat16,
                device="cuda",
            )
            Qwen3NextGatedDeltaNet._warmup_fla_kernels(layer)
            assert Qwen3NextGatedDeltaNet._fla_kernels_warmed_up is True
        finally:
            Qwen3NextGatedDeltaNet._fla_kernels_warmed_up = False
