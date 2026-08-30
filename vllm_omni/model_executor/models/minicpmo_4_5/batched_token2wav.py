# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch, time as _time
from vllm.logger import init_logger
_tlog = init_logger("di_timing")
import torch.nn as nn
import torch.nn.functional as F

_SILENCE_TOKEN = 4218


def _apply_hift_graph_buffer_fixes(hift: torch.nn.Module) -> None:
    """Move HiFT's per-call H2D constants onto the device (capture-safe).

    flashcosyvoice SineGen2 creates ``torch.FloatTensor([[range(...)]])`` on
    CPU every forward call and ``_stft``/``_istft`` do ``stft_window.to(x)``
    per call. ACLGraph capture forbids host-device memcpy (error 107030), so
    these must be device-resident before capture (#5869 does the same for
    CUDA via register_buffer). The FFT-manual-STFT patch already avoids
    torch.stft; this completes the graph-capture story.
    """
    window = hift.stft_window
    if window.device.type != "npu":
        hift.stft_window = window.to("npu")
    sine_gen = hift.m_source.l_sin_gen
    if not hasattr(sine_gen, "harmonic_ids"):
        device = next(hift.parameters()).device
        ids = torch.arange(
            1,
            sine_gen.harmonic_num + 2,
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        sine_gen.register_buffer("harmonic_ids", ids, persistent=False)

    def _forward_no_h2d(self, f0):
        fn = torch.multiply(f0, self.harmonic_ids)
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    from types import MethodType

    if not getattr(sine_gen, "_hift_graph_forward_patched", False):
        sine_gen.forward = MethodType(_forward_no_h2d, sine_gen)
        sine_gen._hift_graph_forward_patched = True


class HiFTNPUGraphWrapper:
    """Exact-signature NPUGraph capture/replay for ``hift.forward(mel, src)``.

    Production MiniCPM-o shapes: uncached [1,80,50]/[1,1,0] first chunk,
    then steady [1,80,58]/[1,1,3840] (mel_cache_len=8, chunk 25 tokens x2
    up-rate + 8 cache). Falls back to eager on unknown signatures once the
    lazy-graph budget is exhausted.
    """

    def __init__(self, hift: torch.nn.Module, *, max_graphs: int = 8):
        self.hift = hift
        self.max_graphs = max(0, int(max_graphs))
        self._graphs: dict[tuple[int, int, int], object] = {}

    @staticmethod
    def _key(mel: torch.Tensor, src: torch.Tensor) -> tuple[int, int, int]:
        return (int(mel.shape[0]), int(mel.shape[2]), int(src.shape[2]))

    def _capture(self, key: tuple[int, int, int], mel: torch.Tensor, src: torch.Tensor) -> None:
        from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

        runner = NPUExactGraphRunner(
            max_graphs=1,
            component_name="MiniCPM-o HiFT",
            disable_config_hint=(
                "set platforms.npu.stages[stage_id=2].additional_config.enable_hift_npu_graph=false"
            ),
        )
        # Warmup kernels before capture (first call can trigger lazy init).
        with torch.inference_mode():
            self.hift(mel, src)
        torch.npu.synchronize()
        self._graphs[key] = runner.capture(
            (mel, src),
            lambda a, b: self.hift(a, b),
        )

    def inference(self, mel: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(mel, src)
        graph = self._graphs.get(key)
        if graph is None:
            if len(self._graphs) >= self.max_graphs:
                return self.hift(mel, src)
            self._capture(key, mel, src)
            return self.hift(mel, src)
        return graph.replay((mel, src))





def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators that have not
    registered autocast support. Degrade to a no-op there: an enclosing region
    can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(self, token2wav: Any):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        # The upstream streaming path preallocates fixed-size CFM and DiT
        # caches. This adapter never calls that path and supplies dynamically
        # sized request-owned buffers to ``blocks_forward_chunk`` instead.
        decoder = self.flow.decoder
        for module in (decoder, decoder.estimator):
            for buffer_name in ("att_cache_buffer", "cnn_cache_buffer"):
                if buffer_name in module._buffers:
                    setattr(module, buffer_name, None)
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        import os as _tjs_os
        _tjs_env = _tjs_os.environ.get("OMNI_TJS_STOP", "1")
        self._tjs_stop = int(_tjs_env) if _tjs_env.isdigit() else 0
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        # T23-2 N1: cache of setup_batch (Conformer+CFM prompt decode) results,
        # keyed identically to _prompt_features. setup_batch is a pure function
        # of (features, batch_size); all batch rows are identical, so caching
        # the batch_size=1 split state covers any N. Evicted with evict_prompt.
        self._setup_batch_cache: dict[tuple[str, str], BatchedToken2WavState] = {}
        self._hift_graph_wrapper: HiFTNPUGraphWrapper | None = None

    def set_hift_graph_wrapper(self, wrapper: HiFTNPUGraphWrapper | None) -> None:
        self._hift_graph_wrapper = wrapper

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)
        self._setup_batch_cache.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time_embedding: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_embedding = time_embedding.unsqueeze(1)
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch_size)
        step_times: list[torch.Tensor] = []
        step_dts: list[torch.Tensor] = []
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            step_times.append(torch.cat((time, time), dim=0))
            step_dts.append(dt)
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
        # Build timestep embeddings before estimator execution because the
        # upstream embedder creates its frequency tensor on the host.
        time_embeddings = tuple(estimator.t_embedder(step_time) for step_time in step_times)
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        for step, dt in enumerate(step_dts):
            old_cnn = cnn_cache[step] if cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=torch.cat((x, x), dim=0),
                mu=mu_cfg,
                time_embedding=time_embeddings[step],
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + dt * velocity
            next_cnn.append(step_cnn)
            next_att.append(step_att)
            # TJS: 到 _tjs_stop 步后 break, 用解析解补齐剩余
            if getattr(self, "_tjs_stop", 0) > 0 and step + 1 >= self._tjs_stop and step + 1 < self.n_timesteps:
                # 当前 t = timeline[step+1], 剩余 1-t 用当前 velocity 补齐
                _t_cur = timeline[step + 1]
                _remain = 1.0 - _t_cur
                x = x + _remain * velocity
                # 补齐剩余 cache(用最后一层的 step_cnn/step_att 复制)
                for _s in range(step + 1, self.n_timesteps):
                    next_cnn.append(step_cnn)
                    next_att.append(step_att)
                break
        return x, torch.stack(next_cnn), torch.stack(next_att)

    @staticmethod
    def _split_flow_cache(cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    @staticmethod
    def _stack_flow_cache(states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
        prompt_cache_id: str | None = None,
        prompt_wav: str | None = None,
    ) -> list[BatchedToken2WavState]:
        cache_key = (prompt_cache_id, prompt_wav) if prompt_cache_id is not None else None
        if cache_key is not None and cache_key in self._setup_batch_cache:
            cached = self._setup_batch_cache[cache_key]
            import logging
            logging.getLogger(__name__).info(
                '[T23-N1] setup_batch CACHE HIT key=%s bs=%d', cache_key, batch_size)
            return [cached] * batch_size
        import logging
        logging.getLogger(__name__).info(
            '[T23-N1] setup_batch MISS key=%s bs=%d', cache_key, batch_size)
        prompt_tokens, speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        states = [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]
        if cache_key is not None and cache_key not in self._setup_batch_cache:
            # Store a defensive clone of the row-0 state: the estimator caches
            # are only .detach()ed (shared storage); clone makes the cached
            # copy fully independent so no decode in-place write can pollute it.
            first = states[0]
            cloned_flow = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in first.flow_cache.items()
            }
            self._setup_batch_cache[cache_key] = BatchedToken2WavState(
                flow_cache=cloned_flow,
                hift_cache=first.hift_cache,
            )
        return states

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        _t0 = _time.time()
        _tlog.debug("decode_batch START tokens=%s", str(list(tokens.shape)))
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            _t_cfm_start = _time.time()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )
            _t_cfm_end = _time.time()
            _tlog.debug("CFM done: %.2fs", _t_cfm_end - _t_cfm_start)

        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        if self._hift_graph_wrapper is not None:
            speech, source = self._hift_graph_wrapper.inference(mel, old_source)
        else:
            speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]

        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
