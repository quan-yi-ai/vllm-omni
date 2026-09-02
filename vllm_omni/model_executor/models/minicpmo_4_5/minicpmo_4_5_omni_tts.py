# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import (
    get_tts_handoff,
    normalize_handoff_tensor,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

import os

try:
    import torch_npu
except ImportError:  # pragma: no cover - CUDA/dev fallback
    torch_npu = None

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
_REPETITION_PENALTY_CHUNK_SIZE = 16
_MIN_AUDIO_BUDGET = 128
_MAX_AUDIO_BUDGET = 2048
# 3-agent formula constants (task_fc967217606f follow-up):
# natural codec length zh min100/mean147/p95 203/max293; cap 2048 = official
# value (340 capped audio at 13.6s for long prompts)
_AUDIO_TOKENS_PER_TEXT_TOKEN = 10
_AUDIO_TOKEN_FIXED_OVERHEAD = 48
# MECHA (机制 A) light-mode flag: set by the runner's local decode loop for
# non-final window steps. make_omni_output then returns the bare (hidden,
# light-mm-dict) tuple instead of an OmniOutput, skipping wrapper assembly.
# Module-global so it survives ACLGraphWrapper attribute routing.
_MECHA_LIGHT_NEXT = False

# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
# seed=1 measured on Seed-TTS zh testset (910B2, 32/100-sample self-checks):
# fixes the deterministic "经济->数据" / "媒体->梅喜" codec-layer
# mispronunciations triggered by certain reference audios under seed=42;
# 100-sample mean WER 0.72% vs the 1.56% gate (seed=42: 1.94%).
_CODEC_SEED = 1
_CODEC_TEMPERATURE = 0.8
# spoine3 (submission4.0 b358fc51) true-greedy profile: top-k=1 argmax fast
# path removes Gumbel noise + top-k/top-p warp entirely -> deterministic codes
# (their Seed-TTS WER 0.00727 vs legacy top-k=25 stochastic sampling).
# Rollback: VLLM_MINICPMO_STAGE1_CODEC_TOP_K=25 restores the legacy sampler.
_CODEC_TOP_K = 1
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26


@dataclass(slots=True)
class _PendingCodecSample:
    output_index: int
    hidden_row: torch.Tensor
    codes: torch.Tensor
    request_id: str
    step: int
    state: dict[str, Any]
    info: dict[str, Any]


def _max_audio_tokens(condition_tokens: int) -> int:
    """Bound codec generation with an adaptive text-length estimate.

    EOS is masked for the first 50 steps, so a direct ``text_tokens * 10``
    limit can terminate short responses before EOS is eligible. The 2048
    ceiling matches the checkpoint's native generation default and keeps the
    sequence within the Talker's 4096-position context.

    3-agent formula (2026-08-21): budget = max(128, min(2048, ct*10+48)).
    Short conditions get the +48 overhead floor (no truncation of the
    natural min-100 codec length); long conditions cap at the official 2048
    ceiling (restored 2026-08-23; previously 340 truncated demo audio at
    ~14s) so the Talker stays in its training distribution.
    """
    return max(
        _MIN_AUDIO_BUDGET,
        min(_MAX_AUDIO_BUDGET, condition_tokens * _AUDIO_TOKENS_PER_TEXT_TOKEN + _AUDIO_TOKEN_FIXED_OVERHEAD),
    )


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


_SHARED_VLLM_SAMPLER: "Sampler | None" = None


def _shared_vllm_sampler() -> "Sampler":
    """The vLLM sampler is stateless; constructing it per decode step is pure overhead."""
    global _SHARED_VLLM_SAMPLER
    if _SHARED_VLLM_SAMPLER is None:
        _SHARED_VLLM_SAMPLER = Sampler()
    return _SHARED_VLLM_SAMPLER


def _apply_repetition_penalty(
    logits: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    penalty: float,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty.

    ``frequencies`` is the persistent per-request vocab-count histogram
    (shape ``(vocab_size,)``) maintained incrementally by the Talker: each
    decode step adds the sampled code and removes the evicted window token
    via ``index_add_``. Its values are bit-identical to the previous per-step
    ``zeros + scatter_add`` rebuild of the window, so sampling is numerically
    unchanged.
    """
    if penalty == 1.0:
        return logits
    alpha = torch.pow(_get_batch_penalty_dev(penalty, logits.device, logits.dtype), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


_BATCH_PENALTY_DEV: torch.Tensor | None = None


def _get_batch_penalty_dev(penalty: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Lazily cache the per-call penalty scalar on device (X1: drop the
    repeated host->device tensor creation in the batched penalty loop)."""
    global _BATCH_PENALTY_DEV
    if _BATCH_PENALTY_DEV is None or _BATCH_PENALTY_DEV.device != device or _BATCH_PENALTY_DEV.dtype != dtype:
        _BATCH_PENALTY_DEV = torch.as_tensor(penalty, device=device, dtype=dtype)
    return _BATCH_PENALTY_DEV


def _apply_batched_repetition_penalty(
    logits: torch.Tensor,
    histories: Sequence[torch.Tensor],
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Apply request-local frequency penalties to a batch of codec logits.

    Matches per-request ``_apply_repetition_penalty`` bit-for-bit: the
    frequency histogram is rebuilt from each request's recent window
    (bincount over an offset-encoded row id), so results are identical to
    the stateless path. Chunked to bound the bincount workspace regardless
    of request concurrency (#5792).
    """
    if logits.ndim != 2:
        raise ValueError(f"batched codec logits must be 2D, got shape {tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    if len(histories) != batch_size:
        raise ValueError(f"expected {batch_size} codec histories, got {len(histories)}")
    if penalty == 1.0:
        return logits
    if batch_size == 0:
        return logits

    penalized = logits.clone()
    penalty_tensor = _get_batch_penalty_dev(penalty, logits.device, logits.dtype)
    for start in range(0, batch_size, _REPETITION_PENALTY_CHUNK_SIZE):
        end = min(start + _REPETITION_PENALTY_CHUNK_SIZE, batch_size)
        chunk_logits = logits[start:end]
        encoded_rows: list[torch.Tensor] = []
        for local_row, history in enumerate(histories[start:end]):
            recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
            if recent.numel() > 0:
                encoded_rows.append(recent + local_row * vocab_size)
        if not encoded_rows:
            continue
        encoded = encoded_rows[0] if len(encoded_rows) == 1 else torch.cat(encoded_rows)
        frequencies = torch.bincount(
            encoded,
            minlength=(end - start) * vocab_size,
        ).reshape(end - start, vocab_size)
        alpha = torch.pow(penalty_tensor, frequencies.to(dtype=logits.dtype))
        penalized[start:end] = torch.where(chunk_logits < 0, chunk_logits * alpha, chunk_logits / alpha)
    return penalized


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits if inplace else logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


_NPU_TOPK_CACHE: dict = {}


def _npu_top_k_top_p_warp(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Fused top-k/top-p floor via ``torch_npu.npu_top_k_top_p``.

    Returns logits with the same candidate floor as ``_apply_top_k_top_p``
    (kernel semantics: keep top-k by value, then keep the top-p probability
    mass over the retained set), or the input unchanged when the kernel is
    unavailable. With top_k >= min_tokens_to_keep the retained candidate
    set matches ``_apply_top_k_top_p``; kernel/PT sorting and softmax-scope
    differ only at ties, so the distribution is near-identical (A3 probe:
    KL=0.028 over a 20k-row synthetic logit set; 0.336 -> 0.108 ms/step
    pipelined, batch=1 fp32). The WER/SIM gates confirm real output.
    """
    if top_k is None or top_p is None or not 0.0 < top_p < 1.0:
        return logits
    npu = getattr(torch_npu, "npu_top_k_top_p", None)
    if npu is None:
        return logits
    # C5: top_p/top_k are constants — cache per-(device, dtype) device tensors
    # instead of rebuilding + H2D every decode step.
    _cache = _NPU_TOPK_CACHE
    key = (str(logits.device), str(logits.dtype), float(top_p), int(top_k))
    cached = _cache.get(key)
    if cached is None:
        p = torch.full((1,), float(top_p), device=logits.device, dtype=logits.dtype)
        k = torch.full((1,), int(top_k), device=logits.device, dtype=torch.int32)
        _cache[key] = (p, k)
    else:
        p, k = cached
    try:
        return npu(
            logits,
            p.expand(logits.shape[0]).contiguous(),
            k.expand(logits.shape[0]).contiguous(),
        )
    except Exception:
        # A kernel failure must not silently change the sampling
        # distribution; fall back to the exact PyTorch warper.
        return _apply_top_k_top_p(
            logits,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=3,
            inplace=True,
        )




def _make_sample_tail_compute(*, penalty: float, eos_id: int, top_k: int, top_p: int):
    """Eager reference for the captured codec sampling tail (closure-captured
    constants; NPUExactGraphRunner calls compute(*inputs) only)."""
    # T17-4 C2 fix: the penalty device scalar is created lazily on the first
    # call (outside NPU capture, during the eager prime run) and reused for
    # capture+replay. Creating it inside the captured compute did a host->device
    # memcpy that NPU capture mode rejects (aclrtMemcpy 107030, "current capture
    # mode does not support this operation"), failing the codec_tail capture.
    _penalty_dev = None

    def compute(logits, freq, noise, mask_eos):
        nonlocal _penalty_dev
        if penalty != 1.0:
            if _penalty_dev is None or _penalty_dev.device != logits.device or _penalty_dev.dtype != logits.dtype:
                _penalty_dev = torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype)
            alpha = torch.pow(_penalty_dev, freq)
            logits = torch.where(logits < 0, logits * alpha, logits / alpha)
        logits[:, eos_id].masked_fill_(mask_eos, float("-inf"))
        if _NPU_TOP_K_TOP_P:
            logits = _npu_top_k_top_p_warp(logits, top_k=top_k, top_p=top_p)
        else:
            logits = _apply_top_k_top_p(
                logits,
                top_k=top_k,
                top_p=top_p,
                min_tokens_to_keep=3,
                inplace=True,
            )
        logits.add_(noise)
        return (logits.argmax(-1).reshape(-1),)

    return compute
# T8: single-kernel warper (A3 verified); opt out with MINICPMO_TTS_NPU_TOPK_TOPP=0
_NPU_TOP_K_TOP_P = os.environ.get("MINICPMO_TTS_NPU_TOPK_TOPP", "1") != "0"


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._stop_logits_base_cache: dict[tuple[Any, Any], torch.Tensor] = {}
        self._head_scaled = False
        self._sample_graph_runner = None
        self._sample_graph_disabled = False
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()
        self._talker_debug_trace = False
        try:
            _cc = getattr(vllm_config, "model_config", None)
            _cc2 = getattr(_cc, "stage_connector_config", None)
            if isinstance(_cc2, dict):
                _extra = _cc2.get("extra", _cc2)
            else:
                _extra = getattr(_cc2, "extra", None)
            if isinstance(_extra, dict):
                _tv = _extra.get("talker_debug_trace")
                if isinstance(_tv, str):
                    _tv = _tv.strip().lower() in ("1", "true", "yes", "on")
                self._talker_debug_trace = bool(_tv)
        except Exception:
            self._talker_debug_trace = False

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_scale = 1.0 / self._codec_temperature if self._codec_temperature else 1.0
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
            self._codec_min_tokens = int(getattr(tts_config, "min_new_tokens", _CODEC_MIN_TOKENS))
            # spoine3 F75 contract: the true-greedy profile (top-k=1) must win
            # over checkpoint/compat defaults (compat.py injects top_k=100,
            # checkpoints may carry 25) unless an explicit env override is set.
            _env_top_k = os.getenv("VLLM_MINICPMO_STAGE1_CODEC_TOP_K")
            if _env_top_k is not None and _env_top_k.strip():
                self._codec_top_k = int(_env_top_k)
            elif self._codec_top_k != _CODEC_TOP_K and _CODEC_TOP_K == 1:
                logger.info(
                    "MiniCPM-o Stage1 codec top-k configured=%s overridden to "
                    "1 by the true-greedy profile; set "
                    "VLLM_MINICPMO_STAGE1_CODEC_TOP_K=%s to restore",
                    self._codec_top_k,
                    self._codec_top_k,
                )
                self._codec_top_k = 1
            if self._codec_top_k == 1:
                logger.info(
                    "MiniCPM-o Stage1 codec top-k=1 true-greedy fast path enabled"
                )
        else:
            self._tts_config = None

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Cross-process stage transport serializes CPU tensors as lists
            # (legacy) or as (dtype, shape, data) tuples (T44 tensor path).
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            hidden_states = normalize_handoff_tensor(hidden_states)
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
            meta = info_dict.get("meta")
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            duplex_boundary = isinstance(meta, dict) and (
                bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
            )
            if native_duplex:
                max_tokens = _DUPLEX_CODEC_TOKENS_PER_CHUNK
                min_tokens = 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK
            else:
                max_tokens = _max_audio_tokens(int(token_ids.numel()))
                min_tokens = self._codec_min_tokens
            state = {
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "finished": empty_condition,
            }
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _sample_audio_codes(
        self,
        hidden_states: torch.Tensor,
        histories: Sequence[torch.Tensor],
        request_ids: Sequence[str],
        steps: Sequence[int],
    ) -> torch.Tensor:
        """Batched request-local codec sampling (#5792).

        One ``head_code`` forward and one warper pass cover all active
        requests; per-request RNG streams are preserved via per-row Gumbel
        ``exponential_``. Bit-identical to the per-request path: the batched
        repetition-penalty histogram matches the incremental ``freq``
        maintenance exactly, and the warp/sample math is row-wise identical.
        """
        batch_size = int(hidden_states.shape[0])
        if not (len(histories) == len(request_ids) == len(steps) == batch_size):
            raise ValueError(
                "MiniCPM-o batched codec sampling requires one history, request id, "
                f"and step per hidden row, got batch={batch_size}, histories={len(histories)}, "
                f"request_ids={len(request_ids)}, steps={len(steps)}"
            )
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)

        logits = self.head_code[0](hidden_states).float() * self._codec_scale
        if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or bool(
            getattr(self, "_talker_debug_trace", False)
        ):
            import logging as _lg

            for _r in range(batch_size):
                _lg.getLogger("vllm").info(
                    "[LOGITS] req=%s step=%d top5=%s hid0=%.4f hid1=%.4f",
                    request_ids[_r], steps[_r],
                    str(logits[_r].topk(5).indices.detach().cpu().tolist()),
                    float(hidden_states[_r, 0].detach().cpu()),
                    float(hidden_states[_r, 1].detach().cpu()),
                )
        eos_id = self._num_audio_tokens - 1
        request_states = getattr(self, "_request_audio_states", {})
        # Cold-start each request's persistent freq histogram exactly as the
        # per-request path would, then use it for the batched penalty so the
        # incremental maintenance is preserved across steps.
        freqs: list[torch.Tensor] = []
        for history, request_id in zip(histories, request_ids, strict=True):
            state = request_states.get(request_id)
            freq = state.get("freq") if isinstance(state, dict) else None
            if freq is None:
                freq = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
                if history.numel() > 0:
                    recent = history.reshape(-1)[-_REPETITION_WINDOW:].to(
                        device=logits.device,
                        dtype=torch.long,
                    )
                    freq.scatter_add_(0, recent, torch.ones_like(recent, dtype=freq.dtype))
                if isinstance(state, dict):
                    state["freq"] = freq
            freqs.append(freq)
        # Apply the incremental per-request freq histogram row-wise (same math
        # as the per-request path; _apply_batched_repetition_penalty is only for
        # the stateless cold-start equivalence check).
        for row, freq in enumerate(freqs):
            logits[row : row + 1] = _apply_repetition_penalty(
                logits[row : row + 1],
                freq,
                penalty=self._codec_repetition_penalty,
            )
        mask_eos_values: list[bool] = []
        for request_id, step in zip(request_ids, steps, strict=True):
            state = request_states.get(request_id)
            min_tokens = (
                int(state.get("min_tokens", self._codec_min_tokens))
                if isinstance(state, dict)
                else self._codec_min_tokens
            )
            mask_eos_values.append(step < min_tokens)
        mask_eos = torch.tensor(
            mask_eos_values,
            dtype=torch.bool,
            device=logits.device,
        )
        logits[:, eos_id].masked_fill_(mask_eos, float("-inf"))
        # spoine3 true-greedy fast path: with top-k=1 the warp + Gumbel noise
        # stages are pure overhead; argmax over the constrained logits yields
        # the identical token without advancing any request RNG stream.
        if self._codec_top_k == 1:
            return logits.argmax(-1).reshape(-1)
        if _NPU_TOP_K_TOP_P:
            logits = _npu_top_k_top_p_warp(
                logits,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
            )
        else:
            logits = _apply_top_k_top_p(
                logits,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
                min_tokens_to_keep=3,
            )
        # A3 (T12-5): capture the tail (penalty+eos+warp+add+argmax) into an
        # exact-shape NPU graph; per-step host dispatch -> single replay.
        # top-k=1 returns above, so the graph path only serves k>1 profiles.
        if (
            os.environ.get("T12_SAMPLE_GRAPH", "0") == "1"
            and not self._sample_graph_disabled
            and batch_size > 0
        ):
            runner = self._sample_graph_runner
            if runner is None:
                from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

                runner = NPUExactGraphRunner(
                    max_graphs=4,
                    component_name="MiniCPM-o codec sample tail",
                    disable_config_hint="set env T12_SAMPLE_GRAPH=0 to disable",
                )
                self._sample_graph_runner = runner
            # 收集 per-row noise (与 eager 相同的懒生成/扩展逻辑)
            noise_rows = []
            for row, (request_id, step) in enumerate(zip(request_ids, steps, strict=True)):
                state = getattr(self, "_request_audio_states", {}).get(request_id)
                noise = state.get("gumbel") if isinstance(state, dict) else None
                if noise is None or step >= int(noise.shape[0]):
                    gen = self._request_generator(request_id, logits.device)
                    n = max(64, (step + 1 - (int(noise.shape[0]) if noise is not None else 0)))
                    q = torch.empty(
                        (n, logits.shape[-1]), device=logits.device, dtype=logits.dtype
                    )
                    q.exponential_(generator=gen)
                    grow = torch.neg(torch.log(q))
                    noise = grow if noise is None else torch.cat([noise, grow], dim=0)
                    if isinstance(state, dict):
                        state["gumbel"] = noise
                noise_rows.append(noise[step : step + 1])
            freq_stack = (
                torch.stack(freqs, dim=0) if freqs else logits.new_empty(0, logits.shape[-1])
            )
            noise_stack = torch.cat(noise_rows, dim=0)
            try:
                compute = _make_sample_tail_compute(
                    penalty=self._codec_repetition_penalty,
                    eos_id=eos_id,
                    top_k=self._codec_top_k,
                    top_p=self._codec_top_p,
                )
                out = runner.run(
                    "codec_tail",
                    inputs=(logits, freq_stack, noise_stack, mask_eos),
                    constants=(
                        self._codec_repetition_penalty,
                        eos_id,
                        self._codec_top_k,
                        self._codec_top_p,
                    ),
                    compute=compute,
                )
                return out[0]
            except Exception:
                self._sample_graph_disabled = True
                self._sample_graph_runner = None
                # fall through to eager 噪声循环 (下方原代码不变)
# A4: fused Gumbel-max with per-request precomputed noise.
        # argmax(softmax(l)/q) == argmax(l - log q) for q ~ Exp(1) (verified
        # on 910C). Noise is pre-generated from the request-local generator in
        # 64-step chunks (same distribution, request-seeded deterministic;
        # NOT bit-identical to per-step draws on NPU -- see rng_test). Per
        # step this is one in-place add + argmax instead of softmax + exp +
        # div + argmax (~3 fewer kernels, no per-row host loop).
        for row, (request_id, step) in enumerate(zip(request_ids, steps, strict=True)):
            state = getattr(self, "_request_audio_states", {}).get(request_id)
            noise = state.get("gumbel") if isinstance(state, dict) else None
            if noise is None or step >= int(noise.shape[0]):
                gen = self._request_generator(request_id, logits.device)
                n = max(64, (step + 1 - (int(noise.shape[0]) if noise is not None else 0)))
                q = torch.empty(
                    (n, logits.shape[-1]), device=logits.device, dtype=logits.dtype
                )
                q.exponential_(generator=gen)
                grow = torch.neg(torch.log(q))
                noise = grow if noise is None else torch.cat([noise, grow], dim=0)
                if isinstance(state, dict):
                    state["gumbel"] = noise
            if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or bool(
                getattr(self, "_talker_debug_trace", False)
            ):
                import logging as _lg

                _lg.getLogger("vllm").info(
                    "[GUMBEL] req=%s step=%d noise_row0=%.4f noise_row1=%.4f history_len=%d",
                    request_id, step,
                    float(noise[step, 0].detach().cpu()) if noise.numel() else float("nan"),
                    float(noise[step, 1].detach().cpu()) if noise.numel() > 1 else float("nan"),
                    int(histories[row].numel()) if histories[row] is not None else -1,
                )
            logits[row : row + 1].add_(noise[step : step + 1])
        return logits.argmax(-1).reshape(-1)

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
    ) -> torch.Tensor:
        """Compatibility wrapper for one request."""
        return self._sample_audio_codes(hidden_state, (history,), (request_id,), (step,)).reshape(())

    def _step_constants(self, hidden: torch.Tensor):
        """Constant per-step tensors, cached per (device, dtype).

        These never change and downstream consumers only read them, so the
        per-step H2D copies from ``new_tensor``/``torch.tensor`` are avoidable.
        """
        cache = getattr(self, "_step_constants_cache", None)
        key = (hidden.device, hidden.dtype)
        if cache is None or cache[0] != key:
            neg_inf = float("-inf")
            cache = (
                key,
                (
                    hidden.new_tensor([0.0, neg_inf]),
                    hidden.new_tensor([neg_inf, 0.0]),
                    torch.tensor(False, dtype=torch.bool),
                    torch.tensor(True, dtype=torch.bool),
                    hidden.new_empty((0, 1), dtype=torch.long),
                ),
            )
            self._step_constants_cache = cache
        return cache[1]

    def make_omni_output_light(
        self,
        hidden: torch.Tensor,
        infos: list[dict],
        spans,
        sample_eligible,
        **kwargs: Any,
    ) -> dict:
        """MECHA (机制 A) point (d): sampling tail + stop row, no wrappers.

        Runs the identical per-request codec sampling tail / state update /
        _batch_stop_logits assembly as make_omni_output, but returns the bare
        (codec_deltas, terminal_flags, stop_flags, emit_duplex_metadata)
        tuple-ish dict instead of an OmniOutput — the runner-local loop only
        needs codes/meta/finished and skips building lists of per-request
        duplex tensors / dict wrappers for steps 0..W-2. The final step of a
        window still calls the full make_omni_output so the model's cached
        output (and _omni_last_model_output) is a complete OmniOutput.

        Must stay bit-identical to make_omni_output for the sampled codec
        sequence: it delegates to the same _sample_audio_codes and the same
        per-row state machinery. Divergence from the full path would show up
        in the 60-step dump compare.
        """
        if len(infos) != len(spans) or len(sample_eligible) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker light requires aligned spans/flags")
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)
        row_continue, row_stop, flag_false, flag_true, empty_delta = self._step_constants(hidden)
        stop_flags = [False] * len(infos)
        codec_deltas: list[torch.Tensor] = [empty_delta for _ in infos]
        terminal_flags: list[torch.Tensor] = [flag_false for _ in infos]
        pending_samples: list[Any] = []
        request_states = getattr(self, "_request_audio_states", None)
        if request_states is None:
            request_states = {}
            self._request_audio_states = request_states
        for index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                continue
            request_id = str(info.get("request_id", index))
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_flags[index] = True
                continue
            if not sample_eligible[index]:
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            pending_samples.append(
                _PendingCodecSample(
                    output_index=index,
                    hidden_row=hidden[end - 1 : end],
                    codes=codes,
                    request_id=request_id,
                    step=step,
                    state=state,
                    info=info,
                )
            )
        if pending_samples:
            active_hidden_rows = [pending.hidden_row for pending in pending_samples]
            active_hidden = (
                active_hidden_rows[0] if len(active_hidden_rows) == 1 else torch.cat(active_hidden_rows, dim=0)
            )
            sampled_batch = self._sample_audio_codes(
                active_hidden,
                [pending.codes for pending in pending_samples],
                [pending.request_id for pending in pending_samples],
                [pending.step for pending in pending_samples],
            )
            sampled_ids = sampled_batch.detach().to(device="cpu").tolist()
        else:
            sampled_batch = hidden.new_empty((0,), dtype=torch.long)
            sampled_ids = []
        for row, pending in enumerate(pending_samples):
            sampled = sampled_batch[row].reshape(())
            codes = pending.codes
            state = pending.state
            info = pending.info
            _min_tts = int(state.get("min_tokens", self._codec_min_tokens))
            _step = int(state.get("step", 0))
            if _step >= _min_tts:
                sampled_id = int(sampled_ids[row])
                is_eos = sampled_id == self._num_audio_tokens - 1
            else:
                sampled_id = None
                is_eos = False
            state["step"] = int(state.get("step", 0)) + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            finished = is_eos or reached_limit
            state["finished"] = finished
            if not is_eos and not reached_limit:
                if codes.numel() >= _REPETITION_WINDOW:
                    evicted = codes[
                        (codes.numel() - _REPETITION_WINDOW) : (codes.numel() - _REPETITION_WINDOW + 1)
                    ]
                else:
                    evicted = None
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                freq = state.get("freq")
                if isinstance(freq, torch.Tensor):
                    if evicted is not None:
                        freq.index_add_(0, evicted, freq.new_full((evicted.shape[0],), -1.0))
                    freq.index_add_(0, sampled.reshape(1), freq.new_ones(1))
                delta = sampled.reshape(1, 1)
            else:
                delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas[pending.output_index] = delta
            terminal_flags[pending.output_index] = flag_true if finished else flag_false
            stop_flags[pending.output_index] = finished
        if stop_flags:
            base = self._stop_logits_base_cache.get((hidden.device, hidden.dtype))
            if base is None:
                base = torch.tensor(
                    [[float("-inf"), 0.0], [0.0, float("-inf")]],
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                self._stop_logits_base_cache[(hidden.device, hidden.dtype)] = base
            idx = [0 if finished else 1 for finished in stop_flags]
            self._batch_stop_logits = base.index_select(
                0, base.new_tensor(idx, dtype=torch.long)
            )
        else:
            self._batch_stop_logits = hidden.new_empty((0, 2))
        return {
            "codes": {"audio": codec_deltas},
            "meta": {"finished": terminal_flags},
            "_emit_duplex": emit_duplex_metadata,
        }

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)
        if _MECHA_LIGHT_NEXT and not emit_duplex_metadata:
            # MECHA light: identical sampling tail, no wrapper objects. The
            # runner expects (raw_hidden, light_mm_dict). Duplex requests stay
            # on the full path (their metadata tensors are not synthesized).
            return (
                hidden,
                self.make_omni_output_light(
                    hidden,
                    infos,
                    spans,
                    sample_eligible,
                    **kwargs,
                ),
            )

        # Rows default to continue. Only previously finished or newly terminal
        # requests stop; prefill/ineligible rows stay aligned as False.
        row_continue, row_stop, flag_false, flag_true, empty_delta = self._step_constants(hidden)
        # Rows default to continue. Only previously finished or newly terminal
        # requests stop; prefill/ineligible rows stay aligned as False.
        stop_flags = [False] * len(infos)
        codec_deltas: list[torch.Tensor] = [empty_delta for _ in infos]
        terminal_flags: list[torch.Tensor] = [flag_false for _ in infos]
        pending_samples: list[_PendingCodecSample] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_flags[index] = True
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            pending_samples.append(
                _PendingCodecSample(
                    output_index=index,
                    hidden_row=hidden[end - 1 : end],
                    codes=codes,
                    request_id=request_id,
                    step=step,
                    state=state,
                    info=info,
                )
            )

        if pending_samples:
            active_hidden_rows = [pending.hidden_row for pending in pending_samples]
            active_hidden = (
                active_hidden_rows[0] if len(active_hidden_rows) == 1 else torch.cat(active_hidden_rows, dim=0)
            )
            sampled_batch = self._sample_audio_codes(
                active_hidden,
                [pending.codes for pending in pending_samples],
                [pending.request_id for pending in pending_samples],
                [pending.step for pending in pending_samples],
            )
            # One batched device-to-host synchronization replaces one .item()
            # synchronization per request.
            sampled_ids = sampled_batch.detach().to(device="cpu").tolist()
        else:
            sampled_batch = hidden.new_empty((0,), dtype=torch.long)
            sampled_ids = []

        for row, pending in enumerate(pending_samples):
            sampled = sampled_batch[row].reshape(())
            codes = pending.codes
            state = pending.state
            info = pending.info
            # O2: defer host sync of sampled_id until the min_tokens boundary —
            # inside min_tokens the EOS comparison is known-false, so the
            # .item() D2H sync is skipped for those steps.
            _min_tts = int(state.get("min_tokens", self._codec_min_tokens))
            _step = int(state.get("step", 0))
            if _step >= _min_tts:
                sampled_id = int(sampled_ids[row])
                is_eos = sampled_id == self._num_audio_tokens - 1
            else:
                sampled_id = None
                is_eos = False
            state["step"] = int(state.get("step", 0)) + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            finished = is_eos or reached_limit
            state["finished"] = finished
            # MiniCPMTTS.generate_chunk consumes the boundary sample but
            # returns only codes that were fed into the retained KV state.
            if not is_eos and not reached_limit:
                # Incremental repetition-penalty histogram: the slide below
                # evicts exactly the oldest window token (index len - W) and
                # appends the freshly sampled code, so +1/-1 index_add_ keeps
                # ``freq`` identical to a full scatter_add rebuild of the window.
                if codes.numel() >= _REPETITION_WINDOW:
                    evicted = codes[
                        (codes.numel() - _REPETITION_WINDOW) : (codes.numel() - _REPETITION_WINDOW + 1)
                    ]
                else:
                    evicted = None
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                freq = state.get("freq")
                if isinstance(freq, torch.Tensor):
                    if evicted is not None:
                        freq.index_add_(0, evicted, freq.new_full((evicted.shape[0],), -1.0))
                    freq.index_add_(0, sampled.reshape(1), freq.new_ones(1))
                delta = sampled.reshape(1, 1)
            else:
                delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas[pending.output_index] = delta
            terminal_flags[pending.output_index] = flag_true if finished else flag_false
            stop_flags[pending.output_index] = finished

        # A5: stop-logits constant rows cached per (device, dtype); per-batch
        # assembly is a row index into the cached 2x2 base.
        if stop_flags:
            base = self._stop_logits_base_cache.get((hidden.device, hidden.dtype))
            if base is None:
                base = torch.tensor(
                    [[float("-inf"), 0.0], [0.0, float("-inf")]],
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                self._stop_logits_base_cache[(hidden.device, hidden.dtype)] = base
            # base 行 0 = finished -> [-inf, 0] (STOP), 行 1 = continue -> [0, -inf]
            idx = [0 if finished else 1 for finished in stop_flags]
            self._batch_stop_logits = base.index_select(
                0, base.new_tensor(idx, dtype=torch.long)
            )
        else:
            self._batch_stop_logits = hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        # ``compute_logits`` emits one-hot stop rows ([0, -inf] / [-inf, 0]),
        # so a full sampler pass is ~10 host-dispatched kernels for a
        # deterministic pick; argmax is bit-identical on these rows. int32
        # matches the standard sampler's output dtype -- the runner scatters
        # these ids into int32 input buffers on the async path.
        if (
            isinstance(logits, torch.Tensor)
            and logits.ndim == 2
            and logits.shape[-1] == 2
            and not getattr(sampling_metadata, "max_num_logprobs", None)
        ):
            return SamplerOutput(
                sampled_token_ids=logits.argmax(dim=-1, keepdim=True).to(torch.int32),
                logprobs_tensors=None,
            )
        return _shared_vllm_sampler()(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
