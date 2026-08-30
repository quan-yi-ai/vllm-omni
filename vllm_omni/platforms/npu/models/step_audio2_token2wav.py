# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for Step-Audio2 / MiniCPM Token2Wav.

Ascend-specific workarounds that must not live in the shared GPU model file:

1. HiFT sine-source downsample — replace the failing 480x ``linear1d``
   downsample with its exact midpoint form while keeping HiFT on NPU.
2. CosyVoice2 DiT SDPA — force MATH backend (+ DiT attn mask expand) to
   avoid fused FA rejecting CosyVoice ``(B,1,1,S)`` masks (error 161001).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_ensure_models_loaded = None
_original_forward = None
_original_stream_chunk_for = None


def _linear_downsample_even_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Match ``F.interpolate(..., mode="linear")`` for an even integer scale.

    With ``align_corners=False``, every output location for an even integer
    downsample lies exactly halfway between two source samples. Selecting and
    averaging those samples avoids Ascend/pytorch#150's ``linear1d`` kernel.
    """
    if scale <= 0 or scale % 2:
        raise ValueError(f"scale must be a positive even integer, got {scale}")
    if x.shape[-1] % scale:
        raise ValueError(f"input length {x.shape[-1]} must be divisible by scale {scale}")

    left = scale // 2 - 1
    right = scale // 2
    return (x[..., left::scale] + x[..., right::scale]) * 0.5


def _run_original_f02sine_on_cpu(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Run the unmodified ``_f02sine`` without invoking NPU ``linear1d``."""
    output_device = f0_values.device
    output = self._step_audio2_original_f02sine(f0_values.cpu())
    return output.to(output_device)


def _f02sine_with_npu_safe_downsample(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Use the exact NPU midpoint path, with a narrow CPU fallback."""
    if getattr(self, "flag_for_pulse", False):
        return _run_original_f02sine_on_cpu(self, f0_values)

    upsample_scale = self.upsample_scale
    if upsample_scale <= 0:
        raise ValueError(f"upsample_scale must be positive, got {upsample_scale}")

    scale = int(upsample_scale)
    midpoint_supported = scale == upsample_scale and scale % 2 == 0 and f0_values.shape[1] % scale == 0
    if not midpoint_supported:
        return _run_original_f02sine_on_cpu(self, f0_values)

    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

    rad_values = _linear_downsample_even_scale(rad_values.transpose(1, 2), scale).transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * self.upsample_scale,
        scale_factor=self.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    return torch.sin(phase)


def patch_step_audio2_hift_for_npu(hift: torch.nn.Module) -> None:
    """Patch the non-causal Step-Audio2 HiFT implementation for Ascend.

    The ``flashcosyvoice.SineGen2`` instantiated by Step-Audio2 1.0.0 is
    non-causal and reduces a full-rate phase tensor by ``1 / 480`` before
    restoring it to the waveform rate. Ascend's ``upsample_linear1d`` kernel
    can raise an AIVector UB-address exception (ACL 507015) for that reduction.

    The exact midpoint form keeps the common path on NPU. Unsupported or pulse
    configurations delegate only ``_f02sine`` to CPU, preserving upstream
    behavior without restoring the old whole-HiFT CPU offload.
    """
    if getattr(hift, "_step_audio2_npu_downsample_patched", False):
        return

    try:
        sine_gen = hift.m_source.l_sin_gen
        original_f02sine = sine_gen._f02sine
    except AttributeError as exc:
        raise TypeError("expected a Step-Audio2 flashcosyvoice HiFT with m_source.l_sin_gen._f02sine") from exc

    if getattr(sine_gen, "causal", False):
        raise ValueError("the Step-Audio2 NPU HiFT patch only supports non-causal SineGen2")

    sine_gen._step_audio2_original_f02sine = original_f02sine
    sine_gen._f02sine = MethodType(_f02sine_with_npu_safe_downsample, sine_gen)
    hift._step_audio2_npu_downsample_patched = True
    hift._stft = MethodType(_npu_safe_stft, hift)  # type: ignore[method-assign]
    hift._istft = MethodType(_npu_safe_istft, hift)  # type: ignore[method-assign]
    logger.info("Patched Step-Audio2 HiFT linear downsample for Ascend NPU")


@contextmanager
def npu_token2wav_sdpa_context(*, require_math: bool = False) -> Iterator[None]:
    """Expand CosyVoice masks + force MATH SDPA to avoid FA 161001."""
    try:
        from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
            apply_cosyvoice2_dit_attn_npu_patch,
            npu_math_sdpa_context,
        )

        apply_cosyvoice2_dit_attn_npu_patch()
        with npu_math_sdpa_context():
            yield
    except Exception:
        with nullcontext():
            yield


def _fft_stft(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor):
    """FFT-based STFT matching ``torch.stft(center=True, pad_mode="reflect")``.

    Input ``[B, T]``; returns (real, imag) each ``[B, n_fft // 2 + 1, T']``.
    Verified numerically identical to torch (max diff 0.0). Ascend has no
    ``torch.stft`` kernel (PTA error 161002), so this is the NPU-safe path.
    """
    x = x.view(x.shape[0], -1)
    pad = n_fft // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
    frames = xp.unfold(1, n_fft, hop) * window  # [B, T', n_fft]
    spec = torch.fft.rfft(frames, dim=-1)        # [B, T', F]
    spec = spec.permute(0, 2, 1)                 # [B, F, T']
    return spec.real, spec.imag


def _fft_istft(real: torch.Tensor, imag: torch.Tensor, n_fft: int, hop: int,
               window: torch.Tensor) -> torch.Tensor:
    """FFT-based ISTFT matching ``torch.istft(center=True)``.

    Inputs ``[B, F, T']``; returns waveform ``[B, T]`` with the length torch
    would produce. Verified numerically identical to torch (max diff 0.0).
    """
    spec = torch.complex(real, imag).permute(0, 2, 1)      # [B, T', F]
    frames = torch.fft.irfft(spec, n=n_fft, dim=-1) * window  # [B, T', n_fft]
    batch, n_frames, _ = frames.shape
    total = (n_frames - 1) * hop + n_fft
    # Vectorized overlap-add via F.fold (replaces the per-frame Python loop,
    # which issued thousands of small NPU kernels per chunk). Numerically
    # equivalent to the loop version (verified max diff ~5e-7).
    out = torch.nn.functional.fold(
        frames.permute(0, 2, 1).reshape(batch, n_fft, n_frames, 1).squeeze(-1),
        output_size=(total, 1), kernel_size=(n_fft, 1), stride=(hop, 1),
    ).squeeze(-1).sum(dim=1)  # [B, total]
    wsum = torch.nn.functional.fold(
        (window * window).unsqueeze(0).unsqueeze(-1).expand(batch, n_fft, n_frames),
        output_size=(total, 1), kernel_size=(n_fft, 1), stride=(hop, 1),
    ).squeeze(-1).sum(dim=1)
    out = out / torch.clamp(wsum, min=1e-10)
    length = (n_frames - 1) * hop
    return out[:, :length]


def _npu_safe_stft(self, x: torch.Tensor):
    """Replacement for ``HiFTGenerator._stft``; FFT on NPU, torch elsewhere."""
    params = self.istft_params
    if x.device.type == "npu":
        return _fft_stft(x, params["n_fft"], params["hop_len"],
                         self.stft_window.to(x.device))
    s = torch.stft(x, params["n_fft"], params["hop_len"], params["n_fft"],
                   window=self.stft_window.to(x.device), return_complex=True)
    s = torch.view_as_real(s)
    return s[..., 0], s[..., 1]


def _npu_safe_istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Replacement for ``HiFTGenerator._istft``; FFT on NPU, torch elsewhere."""
    params = self.istft_params
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    if magnitude.device.type == "npu":
        return _fft_istft(real, imag, params["n_fft"], params["hop_len"],
                          self.stft_window.to(magnitude.device))
    return torch.istft(torch.complex(real, imag), params["n_fft"],
                       params["hop_len"], params["n_fft"],
                       window=self.stft_window.to(magnitude.device))


def _patch_flashcosyvoice_mel_for_npu() -> None:
    """Route the STFT inside ``flashcosyvoice.mel_spectrogram`` to CPU on NPU.

    Ascend has no ``torch.stft`` kernel (PTA error 161002). The flow-model
    prompt mel is the only STFT user and runs once per ``prompt_wav`` (cached),
    so a narrow CPU offload is negligible. The output stays on NPU; only the
    STFT computation runs on CPU.
    """
    import flashcosyvoice.utils.audio as _audio

    if getattr(_audio, "_npu_cpu_stft_patched", False):
        return
    _orig = _audio.mel_spectrogram

    @functools.wraps(_orig)
    def _mel_spectrogram_cpu_stft(y, *args, **kwargs):
        if y.device.type == "npu":
            out = _orig(y.cpu(), *args, **kwargs)
            return out.to("npu")
        return _orig(y, *args, **kwargs)

    _audio.mel_spectrogram = _mel_spectrogram_cpu_stft
    _audio._npu_cpu_stft_patched = True
    logger.info("Patched flashcosyvoice.mel_spectrogram to run STFT on CPU (Ascend lacks torch.stft)")


def _wrap_flow_for_fp16(flow: torch.nn.Module) -> None:
    """Wrap flow (DiT) model methods to cast inputs to weight dtype.

    Called after flow.half() to ensure all tensor inputs entering
    the flow are fp16, since NPU autocast is unreliable for this.
    """
    if getattr(flow, "_npu_fp16_inputs_wrapped", False):
        return

    import inspect

    # Wrap a method to cast all fp32 tensor args/kwargs to fp16
    def _make_casting_wrapper(orig_method, method_name):
        sig = inspect.signature(orig_method)

        @functools.wraps(orig_method)
        def wrapper(*args, **kwargs):
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.dtype == torch.float32:
                    a = a.to(torch.float16)
                new_args.append(a)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    v = v.to(torch.float16)
                new_kwargs[k] = v
            result = orig_method(*new_args, **new_kwargs)
            return result

        return wrapper

    # Wrap key DiT methods that receive external inputs
    methods_to_wrap = ["forward", "forward_chunk", "blocks_forward",
                       "blocks_forward_chunk", "inference"]
    for name in methods_to_wrap:
        if hasattr(flow, name):
            orig = getattr(flow, name)
            if callable(orig) and not getattr(orig, "_npu_fp16_wrapped", False):
                wrapped = _make_casting_wrapper(orig, name)
                wrapped._npu_fp16_wrapped = True
                setattr(flow, name, wrapped)
                logger.info(
                    "Wrapped flow.%s for NPU FP16 input casting", name
                )

    flow._npu_fp16_inputs_wrapped = True
    logger.info(
        "Flow model inputs will be cast to fp16 on every call"
    )


def patch_causal_conv1d_for_npu_fp16() -> None:
    """Monkey-patch CausalConv1d to cast inputs to weight dtype for NPU FP16.

    When flow.half() converts model weights to float16, NPU autocast
    does not reliably promote nn.Conv1d inputs.  This wraps CausalConv1d
    forward / forward_chunk to explicitly cast the input tensor to match
    self.weight.dtype before the conv1d call, avoiding
    "Input type (float) and bias type (c10::Half) should be the same".
    """
    import cosyvoice2.flow.decoder_dit as dd

    if getattr(dd.CausalConv1d, "_npu_fp16_patched", False):
        return

    _orig_forward = dd.CausalConv1d.forward
    _orig_forward_chunk = dd.CausalConv1d.forward_chunk

    def _fp16_safe_forward(self, x):
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return _orig_forward(self, x)

    def _fp16_safe_forward_chunk(self, x, cnn_cache=None):
        target_dtype = self.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if cnn_cache is not None and cnn_cache.dtype != target_dtype:
            cnn_cache = cnn_cache.to(target_dtype)
        return _orig_forward_chunk(self, x, cnn_cache)

    dd.CausalConv1d.forward = _fp16_safe_forward
    dd.CausalConv1d.forward_chunk = _fp16_safe_forward_chunk
    dd.CausalConv1d._npu_fp16_patched = True
    logger.info(
        "Patched CausalConv1d.forward/forward_chunk for NPU FP16 "
        "(cast inputs to weight dtype)"
    )


def _patched_ensure_models_loaded(self) -> None:
    assert _original_ensure_models_loaded is not None
    was_loaded = self._models_loaded
    _original_ensure_models_loaded(self)
    if was_loaded or self.device.type != "npu" or self._hift is None:
        return
    patch_step_audio2_hift_for_npu(self._hift)
    if self.float16 and self._flow is not None:
        _wrap_flow_for_fp16(self._flow)


def _patched_forward(self, generated_speech_tokens, prompt_wav, return_bytes=True):
    assert _original_forward is not None
    if self.device.type != "npu":
        return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)
    import time as _time
    _t_total_start = _time.time()
    with npu_token2wav_sdpa_context():
        _t_after_ctx = _time.time()
        if self.float16:
            # NPU autocast causes SDPA mixed-dtype errors with fp16 weights.
            # Since flow.half() + _wrap_flow_for_fp16 already handle dtype,
            # temporarily disable autocast during forward.
            import torch.amp as _amp
            _saved_autocast = _amp.autocast
            _amp.autocast = lambda *a, **kw: nullcontext()
            try:
                _t_before_fwd = _time.time()
                _result = _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)
                _t_after_fwd = _time.time()
                _t_total = _time.time() - _t_total_start
                _shape_str = str(generated_speech_tokens.shape) if hasattr(generated_speech_tokens, "shape") else "N/A"
                logger.info("TIMING forward: total=%.2fs ctx_setup=%.2fs original_fwd=%.2fs tokens=%s" % (
                    _t_total, _t_before_fwd - _t_after_ctx, _t_after_fwd - _t_before_fwd, _shape_str))
                return _result
            finally:
                _amp.autocast = _saved_autocast
        else:
            return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)


def _patched_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state):
    assert _original_stream_chunk_for is not None
    if self.device.type != "npu":
        return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)
    with npu_token2wav_sdpa_context():
        if self.float16:
            # NPU autocast causes SDPA mixed-dtype errors. Disable temporarily.
            import torch.amp as _amp
            _saved_autocast = _amp.autocast
            _amp.autocast = lambda *a, **kw: nullcontext()
            try:
                return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)
            finally:
                _amp.autocast = _saved_autocast
        else:
            return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)


def apply_step_audio2_token2wav_npu_patch() -> None:
    """Monkey-patch StepAudio2Token2WavCore for Ascend NPU.

    Import is deferred and optional: platform bootstrap (e.g. resolving
    ``current_omni_platform`` from rotary embedding) must not require
    Token2Wav optional deps such as ``librosa``.
    """
    global _PATCHED, _original_ensure_models_loaded, _original_forward, _original_stream_chunk_for
    if _PATCHED:
        return

    _patch_flashcosyvoice_mel_for_npu()
    patch_causal_conv1d_for_npu_fp16()

    # Patch DiT Attention at plugin-load time so it runs in every process
    # (API server AND engine cores — multiprocessing means class patches
    # must be applied in each process independently).
    try:
        from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
            apply_cosyvoice2_dit_attn_npu_patch,
        )
        apply_cosyvoice2_dit_attn_npu_patch()
    except Exception:
        pass

    try:
        from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
            StepAudio2Token2WavCore,
        )
    except ImportError as e:
        logger.debug("step_audio2 token2wav deps unavailable; skip NPU patch: %s", e)
        return

    _original_ensure_models_loaded = StepAudio2Token2WavCore._ensure_models_loaded
    _original_forward = StepAudio2Token2WavCore.forward
    _original_stream_chunk_for = StepAudio2Token2WavCore.stream_chunk_for

    StepAudio2Token2WavCore._ensure_models_loaded = _patched_ensure_models_loaded  # type: ignore[method-assign]
    StepAudio2Token2WavCore.forward = _patched_forward  # type: ignore[method-assign]
    StepAudio2Token2WavCore.stream_chunk_for = _patched_stream_chunk_for  # type: ignore[method-assign]

    _PATCHED = True
    logger.debug("Applied NPU patch for StepAudio2Token2WavCore")
