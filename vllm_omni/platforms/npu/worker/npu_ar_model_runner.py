# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
import torch_npu  # noqa: F401 — NPU stream/event/device ops
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, PerLayerAttnMetadata
from vllm.v1.worker.mamba_utils import preprocess_mamba
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, global_stream
from vllm_ascend.worker.model_runner_v1 import graph_capture

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.distributed.omni_connectors.utils.config import get_stage_connector_role, stage_sends_async_output
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRunnerMixin
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, to_payload_element
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids


# ---------------------------------------------------------------------------
# NPU async omni output helpers — mirror the GPU async output pattern in
# gpu_ar_model_runner.py. The KEY NPU-safe design is: all D2H copies happen
# on the main thread's dedicated copy_stream (async), and the background
# builder thread only waits on the recorded event and assembles the output
# from already-CPU snapshots (pure CPU work). torch.cuda.set_device is NOT
# redirected by vllm-ascend's torch.cuda wrapper, so the builder thread must
# use torch.npu.set_device explicitly.
# ---------------------------------------------------------------------------

class _AsyncNPUCPUPayloadSnapshot:
    """CPU payload snapshot with an NPU event for synchronisation."""

    def __init__(
        self,
        payload: Any,
        ready_event: torch.npu.Event | None,
        npu_sources: list[torch.Tensor],
    ) -> None:
        self.payload = payload
        self._ready_event = ready_event
        self._npu_sources = npu_sources
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        if self._ready_event is not None:
            self._ready_event.synchronize()
        self._npu_sources.clear()
        self._waited = True


def _clone_npu_tensor_payload(value: Any, sources: list[torch.Tensor]) -> Any:
    """Deep-clone NPU tensors on the current stream.

    The clone protects async output snapshots from graph output buffers
    that may be reused by subsequent decode steps.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type == "npu":
            cloned = value.detach().clone()
            sources.append(cloned)
            return cloned
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_npu_tensor_payload(v, sources) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_npu_tensor_payload(v, sources) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_npu_tensor_payload(v, sources) for v in value)
    return value


def _copy_npu_tensor_payload_to_cpu(value: Any, pin_memory: bool) -> Any:
    """Recursively copy NPU tensors to CPU (non-blocking)."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "npu":
            return value
        cpu = torch.empty_like(value, device="cpu", pin_memory=pin_memory)
        cpu.copy_(value, non_blocking=True)
        return cpu
    if isinstance(value, dict):
        return {k: _copy_npu_tensor_payload_to_cpu(v, pin_memory) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_npu_tensor_payload_to_cpu(v, pin_memory) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_npu_tensor_payload_to_cpu(v, pin_memory) for v in value)
    return value


def _snapshot_npu_tensor_payload_to_cpu_async(
    value: Any,
    *,
    copy_stream: torch.npu.Stream,
    pin_memory: bool,
) -> _AsyncNPUCPUPayloadSnapshot:
    """Clone NPU tensors and initiate async D2H copy on *copy_stream*."""
    npu_sources: list[torch.Tensor] = []
    cloned = _clone_npu_tensor_payload(value, npu_sources)
    if not npu_sources:
        return _AsyncNPUCPUPayloadSnapshot(cloned, None, npu_sources)

    source_stream = torch.npu.current_stream()
    ready_event = torch.npu.Event()
    with torch.npu.stream(copy_stream):
        copy_stream.wait_stream(source_stream)
        cpu_payload = _copy_npu_tensor_payload_to_cpu(cloned, pin_memory)
        ready_event.record(copy_stream)
    return _AsyncNPUCPUPayloadSnapshot(cpu_payload, ready_event, npu_sources)


# ---------------------------------------------------------------------------
# NPU async omni model runner output — mirrors OmniAsyncGPUModelRunnerOutput
# but NPU-safe: builder thread sets torch.npu device explicitly and only does
# CPU assembly (the D2H snapshot was taken on the main thread's copy_stream).
# ---------------------------------------------------------------------------

class OmniAsyncNPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    """Async NPU output that builds the Omni payload on a background thread."""

    def __init__(
        self,
        *,
        model_runner_output_builder: Callable[[], OmniModelRunnerOutput],
        npu_device: torch.device | int | str | None = None,
        **kwargs: Any,
    ) -> None:
        sampled_token_ids = kwargs.pop("sampled_token_ids")
        logprobs_tensors = kwargs.pop("logprobs_tensors")
        invalid_req_indices = kwargs.pop("invalid_req_indices")
        async_output_copy_stream = kwargs.pop("async_output_copy_stream")
        vocab_size = kwargs.pop("vocab_size")
        routed_experts = kwargs.pop("routed_experts", None)
        kwargs.pop("check_ep_fault", False)
        if kwargs:
            raise TypeError(
                f"Unexpected OmniAsyncNPUModelRunnerOutput kwargs: {sorted(kwargs)}"
            )

        self._model_runner_output = None
        self._invalid_req_indices = invalid_req_indices

        self.async_copy_ready_event = torch.npu.Event()
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None

        default_stream = torch.npu.current_stream()
        with torch.npu.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors is not None
                else None
            )
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking()
                if self._routed_experts is not None
                else None
            )
            self.async_copy_ready_event.record()

        self._model_runner_output_builder = model_runner_output_builder
        self._background_exception: BaseException | None = None
        self._background_thread: threading.Thread | None = None
        self._npu_device = npu_device
        self._background_thread = threading.Thread(
            target=self._build_output_in_background,
            daemon=True,
            name="omni-async-npu-output-builder",
        )
        self._background_thread.start()

    def _build_model_runner_output_once(self) -> None:
        if self._model_runner_output is not None:
            return
        with record_function_or_nullcontext("omni_async_npu_output:build"):
            self._model_runner_output = self._model_runner_output_builder()
        self._model_runner_output_builder = None

    def _build_output_in_background(self) -> None:
        try:
            # torch.cuda.set_device is NOT redirected on NPU; use npu explicitly.
            if self._npu_device is not None:
                torch.npu.set_device(self._npu_device)
            self._build_model_runner_output_once()
        except BaseException as exc:  # noqa: BLE001 — re-raised by get_output().
            self._background_exception = exc

    def get_output(self) -> OmniModelRunnerOutput:
        background_thread = getattr(self, "_background_thread", None)
        if background_thread is not None:
            background_thread.join()
            self._background_thread = None
            background_exception = getattr(self, "_background_exception", None)
            if background_exception is not None:
                raise background_exception
        self._build_model_runner_output_once()
        if self._model_runner_output is None:
            raise RuntimeError("OmniAsyncNPUModelRunnerOutput: output was never built.")
        if not hasattr(self, "_has_fault"):
            self._has_fault = None
        # super().get_output() synchronizes the sampled_token_ids copy event
        # and finalizes token parsing on the model_runner_output.
        with record_function_or_nullcontext(
            "omni_async_npu_output:finalize_async_sampled_tokens"
        ):
            return super().get_output()


def _ensure_tensor_values(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int, float, bool) are wrapped with ``torch.tensor()``.
    Values that cannot be safely converted are dropped with a warning.
    This enforces the tensor-only invariant required by the
    ``OmniEngineCoreOutput.multimodal_output`` wire field and msgspec
    serialization. Mirrors ``gpu_ar_model_runner._ensure_tensor_values``.
    """
    result: dict[str, torch.Tensor] = {}
    for key, val in payload.items():
        if isinstance(val, torch.Tensor):
            result[key] = val
        elif isinstance(val, (int, float, bool)):
            result[key] = torch.tensor(val)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val)
            except (ValueError, TypeError, RuntimeError):
                logger.warning(
                    "Dropping non-tensorizable multimodal output key '%s' (type=%s) from wire payload.",
                    key,
                    type(val).__name__,
                )
        else:
            logger.warning(
                "Dropping non-tensor multimodal output key '%s' (type=%s) from wire payload.",
                key,
                type(val).__name__,
            )
    return result


def _clone_npu_tensor_payload(value: Any, sources: list[torch.Tensor]) -> Any:
    """Clone NPU tensors on the current stream before async CPU copies."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "npu":
            cloned = value.detach().clone()
            sources.append(cloned)
            return cloned
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_npu_tensor_payload(v, sources) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_npu_tensor_payload(v, sources) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_npu_tensor_payload(v, sources) for v in value)
    return value


def _copy_tensor_payload_to_cpu(value: Any, pin_memory: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "npu":
            return value
        cpu = torch.empty_like(value, device="cpu", pin_memory=pin_memory)
        cpu.copy_(value, non_blocking=True)
        return cpu
    if isinstance(value, dict):
        return {k: _copy_tensor_payload_to_cpu(v, pin_memory) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_tensor_payload_to_cpu(v, pin_memory) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_tensor_payload_to_cpu(v, pin_memory) for v in value)
    return value


class _AsyncCPUPayloadSnapshot:
    def __init__(
        self,
        payload: Any,
        ready_event: "torch.npu.Event | None",
        npu_sources: list[torch.Tensor],
    ) -> None:
        self.payload = payload
        self._ready_event = ready_event
        self._npu_sources = npu_sources
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        if self._ready_event is not None:
            self._ready_event.synchronize()
        self._npu_sources.clear()
        self._waited = True


def _snapshot_tensor_payload_to_cpu_async(
    value: Any,
    *,
    copy_stream: "torch.npu.Stream",
    pin_memory: bool,
) -> _AsyncCPUPayloadSnapshot:
    npu_sources: list[torch.Tensor] = []
    cloned = _clone_npu_tensor_payload(value, npu_sources)
    if not npu_sources:
        return _AsyncCPUPayloadSnapshot(cloned, None, npu_sources)

    source_stream = torch.npu.current_stream()
    ready_event = torch.npu.Event()
    with torch.npu.stream(copy_stream):
        copy_stream.wait_stream(source_stream)
        cpu_payload = _copy_tensor_payload_to_cpu(cloned, pin_memory)
        ready_event.record(copy_stream)
    return _AsyncCPUPayloadSnapshot(cpu_payload, ready_event, npu_sources)


class OmniAsyncNPUGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    """NPU port of OmniAsyncGPUModelRunnerOutput: build Omni output on a
    background thread while the async D2H snapshot completes."""

    def __init__(
        self,
        *,
        model_runner_output_builder: Callable[[], "OmniModelRunnerOutput"],
        npu_device: torch.device | int | str | None = None,
        **kwargs: Any,
    ) -> None:
        sampled_token_ids = kwargs.pop("sampled_token_ids")
        logprobs_tensors = kwargs.pop("logprobs_tensors")
        invalid_req_indices = kwargs.pop("invalid_req_indices")
        async_output_copy_stream = kwargs.pop("async_output_copy_stream")
        vocab_size = kwargs.pop("vocab_size")
        routed_experts = kwargs.pop("routed_experts", None)
        kwargs.pop("check_ep_fault", False)
        if kwargs:
            raise TypeError(f"Unexpected OmniAsyncNPUGPUModelRunnerOutput kwargs: {sorted(kwargs)}")

        self._model_runner_output = None
        self._invalid_req_indices = invalid_req_indices

        self.async_copy_ready_event = torch.npu.Event()
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None

        default_stream = torch.npu.current_stream()
        with torch.npu.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking() if self._logprobs_tensors is not None else None
            )
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking() if self._routed_experts is not None else None
            )
            self.async_copy_ready_event.record()

        self._model_runner_output_builder = model_runner_output_builder
        self._background_exception: BaseException | None = None
        self._background_thread: threading.Thread | None = None
        self._npu_device = npu_device
        self._background_thread = threading.Thread(
            target=self._build_output_in_background,
            daemon=True,
            name="omni-async-npu-output-builder",
        )
        self._background_thread.start()

    def _build_model_runner_output_once(self) -> None:
        if self._model_runner_output is not None:
            return
        with record_function_or_nullcontext("omni_async_npu_output:get_output/build_model_runner_output"):
            self._model_runner_output = self._model_runner_output_builder()
        self._model_runner_output_builder = None

    def _build_output_in_background(self) -> None:
        try:
            if self._npu_device is not None:
                torch.npu.set_device(self._npu_device)
            self._build_model_runner_output_once()
        except BaseException as exc:  # noqa: BLE001 - re-raised by get_output().
            self._background_exception = exc

    def get_output(self) -> "OmniModelRunnerOutput":
        background_thread = getattr(self, "_background_thread", None)
        if background_thread is not None:
            background_thread.join()
            self._background_thread = None
            background_exception = getattr(self, "_background_exception", None)
            if background_exception is not None:
                raise background_exception
        self._build_model_runner_output_once()
        if not hasattr(self, "_has_fault"):
            self._has_fault = None
        with record_function_or_nullcontext("omni_async_npu_output:get_output/finalize_async_sampled_tokens"):
            return super().get_output()


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor
    multimodal_outputs: Any # Omni-Specific

class NPUARModelRunner(OmniNPUModelRunner, OmniConnectorModelRunnerMixin, DuplexSamplingRunnerMixin):
    """Autoregressive NPU model runner that returns hidden states per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        self._omni_async_step_budget = int(os.environ.get("VLLM_OMNI_NPU_ASYNC_OUTPUT_STEPS", "48"))
        self._omni_async_steps: dict[str, int] = {}

        _OMNI_CONNECTOR_INIT_ARCHS = {
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
            "CovoAudioForConditionalGeneration",
            "MiMoAudioModel",
            "Qwen3TTSTalkerForConditionalGeneration",
            "Qwen3TTSCode2Wav",
            "CosyVoice3Model",
            "DyninOmniForConditionalGeneration",
            "IndexTTS2TalkerForConditionalGeneration",
        }
        # Mirrors gpu_ar_model_runner: an arch missing from the hardcoded allowlist
        # still needs connectors when the deploy config hands the stage a
        # sender/receiver role (e.g. MiniCPM-o 4.5, whose archs are not listed but
        # whose YAML wires stage 1 -> stage 2). Without the role check the
        # full-payload (``--no-async-chunk``) handoff never initializes: nothing
        # accumulates, nothing flushes, and the downstream stage starves silently.
        if (
            getattr(self.model_config, "model_arch", None) in _OMNI_CONNECTOR_INIT_ARCHS
            or get_stage_connector_role(self.model_config) is not None
        ):
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        self._downstream_payload_cache: dict[str, bool] = {}
        self._init_duplex_sampling_state()
        self._init_talker_local_decode_config()

    def _init_talker_local_decode_config(self) -> None:
        """Read runner-local multi-step Talker decode knobs from the deploy
        connector ``extra`` block (same channel as the chunk config) with env
        overrides. All knobs are runner-side only; K=1 (default) disables the
        feature entirely and the code paths below are inert.
        """
        self._talker_local_steps = 8
        self._talker_local_stage_id: int | None = 1
        self._talker_cpu_slot_mapping = True
        self._talker_binary_argmax = True
        self._talker_debug_trace = False
        self._talker_dump_dir: str | None = None
        self._talker_e3 = False
        try:
            model_cfg = getattr(self.vllm_config, "model_config", None)
            connector_cfg = getattr(model_cfg, "stage_connector_config", None)
            if isinstance(connector_cfg, Mapping):
                extra_cfg = connector_cfg.get("extra", connector_cfg)
            else:
                extra_cfg = getattr(connector_cfg, "extra", None)
            if not isinstance(extra_cfg, Mapping):
                extra_cfg = {}
            steps = extra_cfg.get("talker_local_decode_steps")
            if steps is not None:
                self._talker_local_steps = max(1, int(steps))
            stage_id = extra_cfg.get("talker_local_decode_stage_id")
            if stage_id is not None:
                self._talker_local_stage_id = int(stage_id)
            slot_cfg = extra_cfg.get("talker_local_cpu_slot_mapping")
            if isinstance(slot_cfg, str):
                slot_cfg = slot_cfg.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(slot_cfg, bool):
                self._talker_cpu_slot_mapping = slot_cfg
            elif isinstance(slot_cfg, int):
                self._talker_cpu_slot_mapping = bool(slot_cfg)
            bin_cfg = extra_cfg.get("talker_binary_argmax")
            if isinstance(bin_cfg, str):
                bin_cfg = bin_cfg.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(bin_cfg, bool):
                self._talker_binary_argmax = bin_cfg
            elif isinstance(bin_cfg, int):
                self._talker_binary_argmax = bool(bin_cfg)
            trace_cfg = extra_cfg.get("talker_debug_trace")
            if isinstance(trace_cfg, str):
                trace_cfg = trace_cfg.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(trace_cfg, bool):
                self._talker_debug_trace = trace_cfg
            elif isinstance(trace_cfg, int):
                self._talker_debug_trace = bool(trace_cfg)
            dump_cfg = extra_cfg.get("talker_dump_dir")
            if isinstance(dump_cfg, str) and dump_cfg.strip():
                self._talker_dump_dir = dump_cfg.strip()
            e3_cfg = extra_cfg.get("talker_e3")
            if isinstance(e3_cfg, str):
                e3_cfg = e3_cfg.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(e3_cfg, bool):
                self._talker_e3 = e3_cfg
            elif isinstance(e3_cfg, int):
                self._talker_e3 = bool(e3_cfg)
        except Exception:  # pragma: no cover - config best-effort
            logger.warning("Failed to parse talker local decode config; keeping defaults", exc_info=True)
        # env kill switch: OMNI_TALKER_LOCAL_DECODE=0 disables; OMNI_TALKER_LOCAL_STEPS overrides K
        if os.environ.get("OMNI_TALKER_LOCAL_DECODE", "1").strip().lower() in ("0", "false", "no", "off"):
            self._talker_local_steps = 1
        env_steps = os.environ.get("OMNI_TALKER_LOCAL_STEPS")
        if env_steps is not None and env_steps.strip().isdigit():
            self._talker_local_steps = max(1, int(env_steps.strip()))
        if os.environ.get("OMNI_TALKER_CPU_SLOT_MAPPING", "1").strip().lower() in ("0", "false", "no", "off"):
            self._talker_cpu_slot_mapping = False
        if self._talker_local_steps > 1:
            logger.info(
                "Talker local decode ENABLED: K=%d stage_id=%s cpu_slot_mapping=%s trace=%s",
                self._talker_local_steps,
                self._talker_local_stage_id,
                self._talker_cpu_slot_mapping,
                self._talker_debug_trace,
            )
        # MECHA (机制 A): K-window sub-step prebuild. Env-gated; master kill
        # switch OMNI_TALKER_MECHA=0 disables everything; each point is
        # independently killable (META/COS/TOK/COLLECT) for bisection.
        _mecha_master = os.environ.get("OMNI_TALKER_MECHA", "1").strip().lower() not in (
            "0", "false", "no", "off"
        )
        self._mecha_enabled = _mecha_master and self._talker_local_steps > 1
        self._mecha_meta = _mecha_master and os.environ.get(
            "OMNI_TALKER_MECHA_META", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self._mecha_cos = _mecha_master and os.environ.get(
            "OMNI_TALKER_MECHA_COS", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self._mecha_tok = _mecha_master and os.environ.get(
            "OMNI_TALKER_MECHA_TOK", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        # NOTE: flag kept off the method name — ``_mecha_collect`` is a method
        # below; assigning a bool to the same name shadowed it and crashed the
        # call sites with TypeError: 'bool' object is not callable (old-machine
        # port fix, verified AST + runtime).
        self._mecha_collect_enabled = _mecha_master and os.environ.get(
            "OMNI_TALKER_MECHA_COLLECT", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self._mecha_light = _mecha_master and os.environ.get(
            "OMNI_TALKER_MECHA_LIGHT", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self._talker_light_next = False
        if _mecha_master:
            logger.info(
                "Talker MECHA (机制 A) ENABLED: meta=%s cos=%s tok=%s collect=%s light=%s",
                self._mecha_meta,
                self._mecha_cos,
                self._mecha_tok,
                self._mecha_collect_enabled,
                self._mecha_light,
            )

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        self._resolve_duplex_sampling_hook(force=True)

    def _update_states(self, scheduler_output: SchedulerOutput):
        deferred_state_corrections_fn = super()._update_states(scheduler_output)
        self._update_duplex_sampling_states(scheduler_output)
        return deferred_state_corrections_fn

    #  -------------------------------------- Omni-new -------------------------------------------------
    # R5: runner-local multi-step Talker decode. Runs K sequential 1-token-per-req
    # autoregressive decode steps inside ONE engine-core round trip, amortizing
    # the scheduler/IPC/output round-trip gap (0.54 ms/step) across K codec tokens.
    # -------------------------------------------------------------------------
    def _talker_local_decode_eligible(
        self,
        scheduler_output: SchedulerOutput,
        *,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        cudagraph_mode: Any,
        use_spec_decode: bool,
        has_encoder_input: bool,
        num_encoder_reqs: int,
    ) -> bool:
        """Gate for the local window. All conditions must hold:
        - K > 1 and this stage is the configured talker stage
        - sync scheduling (async placeholder accounting is not K-token aware)
        - FULL_DECODE_ONLY graph mode (bucket-1 replay path)
        - pure decode batch: uniform 1..K scheduled tokens per req, no prefill
          chunk, no spec decode, no encoder, no grammar
        - batch fits the captured graph buckets
        """
        _tr = getattr(self, '_talker_debug_trace', False) or os.environ.get('OMNI_TALKER_DEBUG_TRACE', '0') == '1'
        def _rej(why):
            if _tr:
                logger.info('[TALKER-LOCAL] eligible REJECT: %s (K=%d stage=%s/%s async=%s mode=%s reqs=%d sched=%s spec=%s enc=%s gram=%s)',
                            why, self._talker_local_steps, getattr(self, "_stage_id", None), self._talker_local_stage_id,
                            self.use_async_scheduling, cudagraph_mode, num_reqs,
                            list(num_scheduled_tokens_np[:num_reqs]) if num_scheduled_tokens_np is not None else None,
                            use_spec_decode, has_encoder_input or num_encoder_reqs > 0,
                            getattr(scheduler_output, 'has_structured_output_requests', False))
            return False
        if self._talker_local_steps <= 1:
            return _rej('K<=1')
        if self._talker_local_stage_id is not None and getattr(self, "_stage_id", None) != self._talker_local_stage_id:
            return _rej('stage-mismatch')
        if self.use_async_scheduling:
            return _rej('async')
        # With the scheduler-K knob the engine schedules K tokens/step but the
        # cudagraph dispatcher cannot know that (uniform_decode_query_len=1),
        # so a K-token batch dispatches to NONE. That is expected: the local
        # window re-runs the batch as K sequential 1-token FULL_DECODE_ONLY
        # forwards with its own descriptor. Only reject when the mode is
        # neither FULL_DECODE_ONLY nor the K-window NONE case.
        S_check = int(num_scheduled_tokens_np[0]) if num_reqs > 0 else 0
        if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY and not (
            cudagraph_mode == CUDAGraphMode.NONE and S_check == self._talker_local_steps
        ):
            return _rej(f'mode={cudagraph_mode}')
        if num_reqs <= 0:
            return _rej('no-reqs')
        if use_spec_decode or has_encoder_input or num_encoder_reqs > 0:
            return _rej('spec/enc')
        if getattr(scheduler_output, "has_structured_output_requests", False):
            return _rej('grammar')
        # uniform scheduled count S in [1, K]; all reqs decode (no prefill chunks)
        sched = num_scheduled_tokens_np[:num_reqs]
        S = int(sched[0])
        if S < 1 or S > self._talker_local_steps:
            return _rej(f'S={S} out of [1,K]')
        if not bool(np.all(sched == S)):
            return _rej('non-uniform')
        num_prompt = getattr(self.input_batch, "num_prompt_tokens", None)
        if num_prompt is not None:
            computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
            if np.any(computed < num_prompt[:num_reqs]):
                return _rej('prefill')
        # batch size must be capturable
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if capture_sizes and num_reqs > max(int(x) for x in capture_sizes):
            return _rej('batch-too-big')
        if _tr:
            logger.info('[TALKER-LOCAL] eligible PASS: S=%d reqs=%d', S, num_reqs)
        return True

    def _talker_local_window(self, req_idx: int, S: int) -> int:
        """Window size for one request.

        - S == 1 (bootstrap window right after prefill): run min(K, slots
          remaining in the current KV block). The engine allocated lazily for
          1 token, so local extra steps must never cross into an unallocated
          block. The runner returns the full window; the engine then
          self-tunes its schedule to the returned count (num_new =
          num_tokens - num_computed), so the steady state schedules exactly K.
        - S > 1 (steady state): the engine scheduled exactly S == K tokens
          this step and their slots were computed by the scheduled batch —
          run exactly S.
        """
        if S > 1:
            return S
        K = self._talker_local_steps
        pos = int(self.input_batch.num_computed_tokens_cpu[req_idx])  # first window position (0-indexed)
        bs = self.cache_config.block_size
        return min(K, bs - (pos % bs))

    def _talker_local_engine_token(self, hidden_rows: torch.Tensor) -> list[int]:
        """STOP/CONTINUE token (argmax of the 2-col stop logits) per request row
        WITHOUT consuming the model's cached batch stop logits (the final step's
        logits must still flow to _bookkeeping_sync)."""
        try:
            row_logits = self._batch_stop_logits_snapshot()
            if row_logits is not None:
                tok = row_logits.argmax(dim=-1, keepdim=True).to(torch.int32).reshape(-1).tolist()
                return [int(t) for t in tok]
        except Exception:
            pass
        # Fallback: recompute from compute_logits then re-arm the model's cache.
        try:
            logits = self.model.compute_logits(
                hidden_rows,
                sampling_metadata=self.input_batch.sampling_metadata,
            )
            if logits is None:
                return [0] * hidden_rows.shape[0]
            tok = logits.argmax(dim=-1, keepdim=True).to(torch.int32).reshape(-1).tolist()
            return [int(t) for t in tok]
        except Exception:
            return [0] * hidden_rows.shape[0]

    def _batch_stop_logits_snapshot(self) -> torch.Tensor | None:
        """Read the model's cached per-batch stop logits without consuming them
        (compute_logits nulls the cache; the final local step's logits must
        still be returned through the normal path)."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        # ACLGraphWrapper forwards attribute access to its runnable via
        # __getattr__, but the cached stop logits are set at runtime on the
        # wrapped model; read through unwrap() to be safe.
        unwrap = getattr(model, "unwrap", None)
        if callable(unwrap):
            try:
                model = unwrap()
            except Exception:
                pass
        # The top-level MiniCPM-o model forwards make_omni_output to its
        # `talker` sub-model, which owns _batch_stop_logits.
        cached = getattr(model, "_batch_stop_logits", None)
        if cached is None:
            talker = getattr(model, "talker", None)
            if talker is not None:
                cached = getattr(talker, "_batch_stop_logits", None)
        if getattr(self, "_talker_debug_trace", False) or os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1":
            logger.info(
                "[TALKER-BSL] model=%s cached=%s numel=%s",
                type(model).__name__,
                "None" if cached is None else "set",
                "n/a" if cached is None else str(cached.numel()),
            )
        if cached is None or cached.numel() == 0:
            return None
        return cached.detach()

    def _talker_local_step_feedback(
        self,
        req_idx: int,
        local_step: int,
    ) -> None:
        """Refresh the runner input buffers for one local decode step of one
        request: advance the computed-token counter, positions, seq_lens,
        optimistic seq_lens, the KV slot (CPU block-table formula, O40) and the
        codec embedding (model.preprocess reads the just-sampled codec token).
        Attention metadata is rebuilt by the caller (O31: reuse fails).

        The engine's counter is advanced by the SCHEDULED count (1 for the
        first local window, K in steady state) via ``_update_after_schedule``
        before the runner reads it, so the runner-side counter here always
        represents "tokens whose KV is already written". Advancing it by one
        per local step keeps positions/slots contiguous; the engine's counter
        catches up on the next step because ``num_tokens`` grew by K while the
        counter grew by 1, so the next schedule hands us exactly K tokens.
        """
        num_computed = self.input_batch.num_computed_tokens_cpu[req_idx]
        new_pos = int(num_computed) + 1
        bs = self.cache_config.block_size
        # CPU slot mapping: block_id * block_size + offset (O40). Only valid
        # when the block row exists (block id 0 = NULL sentinel). The kernel
        # slot for the scheduled token is already correct; this covers the
        # local steps beyond the scheduled batch.
        try:
            bt = self.input_batch.block_table[0]
            block_np = bt.block_table.np
            block_id = int(block_np[req_idx, new_pos // bs])
            if block_id != 0:
                slot = block_id * bs + (new_pos % bs)
                # write BOTH the GPU buffer (kernel consumers) and the CPU
                # buffer (AscendCommonAttentionMetadata.slot_mapping_cpu)
                bt.slot_mapping.gpu[req_idx].fill_(slot)
                bt.slot_mapping.cpu[req_idx] = slot
        except Exception as exc:  # never let feedback break the request
            logger.warning("talker local slot mapping failed: %s", exc)
        # advance counters (runner-local view; engine counter syncs via the
        # next step's schedule, see docstring)
        self.input_batch.num_computed_tokens_cpu[req_idx] = new_pos
        # sync the GPU counter tensor (the kernel path reads it; leaving it
        # stale by one makes local-step attention see the pre-window value)
        if hasattr(self, "num_computed_tokens"):
            self.num_computed_tokens[req_idx].fill_(new_pos)
        # positions / seq_lens / optimistic seq lens for this request's row
        self.positions[req_idx].fill_(new_pos)
        self.seq_lens[req_idx].fill_(new_pos + 1)
        if hasattr(self, "optimistic_seq_lens_cpu"):
            self.optimistic_seq_lens_cpu[req_idx] = new_pos + 1
        # embed the freshly sampled codec for the next forward
        req_id = self.input_batch.req_ids[req_idx]
        req_infos = self.model_intermediate_buffer.get(req_id, {})
        req_infos["request_id"] = req_id
        try:
            _ids, req_embeds, _upd = self.model.preprocess(
                self.input_ids.gpu[req_idx : req_idx + 1],
                None,
                **req_infos,
            )
            if req_embeds is not None and req_embeds.numel() > 0:
                self.inputs_embeds.gpu[req_idx : req_idx + 1].copy_(req_embeds[:1])
        except Exception as exc:  # never let feedback break the request
            logger.warning("talker local feedback preprocess failed: %s", exc)

    #  -------------------------------------- MECHA (机制 A) -----------------------------------------
    # K-window sub-step prebuild. Per-step attention metadata objects built
    # once at window start (each an INDEPENDENT copy carrying its own
    # step-varying fields — the O31 reuse trap was sharing one object whose
    # step state went stale; here each object owns its seq_lens), cos/sin table
    # precomputed for the whole window position range, engine STOP/CONTINUE
    # tokens read from the shared per-request state (make_omni_output derives
    # _batch_stop_logits FROM state['finished'], so the argmax + D2H tolist is
    # redundant), and a light per-step collect. All gated by OMNI_TALKER_MECHA
    # + per-point env; every path falls back to the original implementation on
    # any surprise (correctness first; bit-identity verified by dump compare).
    # -------------------------------------------------------------------------------------------------

    def _mecha_prebuild_metadata(
        self, num_reqs: int, req_ids: list[str], window: int
    ) -> Any | None:
        """Build `window` independent per-step attention-metadata dicts.

        Base = the real builder at the CURRENT buffer state (exactly what the
        loop's step-0 build would produce). Steps 1..W-1 are shallow copies of
        each per-layer AscendMetadata with only the step-varying fields
        (seq_lens_list / seq_lens / seq_lens_cpu) refreshed to base+j. For a
        pure decode window everything else is step-invariant: slot_mapping and
        block_tables are LIVE views of the runner buffers (feedback keeps them
        current), query_start_loc / attn_mask / attn_state are static, and FIA
        replay consumes only seq_lens_list + actual_seq_lengths_q per layer.
        Returns None on any failure (caller falls back to per-step builds).
        """
        try:
            base_meta, _ = self._build_attention_metadata(
                num_tokens=num_reqs,
                num_tokens_padded=num_reqs,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                max_query_len=1,
                ubatch_slices=None,
                logits_indices=None,
                use_spec_decode=False,
                num_scheduled_tokens={rid: 1 for rid in req_ids},
                num_scheduled_tokens_np=np.ones(num_reqs, dtype=np.int32),
            )
            if not isinstance(base_meta, Mapping) or not base_meta:
                return None
            base_lens = [
                int(self.optimistic_seq_lens_cpu[i])
                if hasattr(self, "optimistic_seq_lens_cpu")
                else int(self.seq_lens[i])
                for i in range(num_reqs)
            ]
            steps: list[Any] = []
            for j in range(window):
                meta_j: dict[str, Any] = {}
                for key, md in base_meta.items():
                    cm = copy(md)
                    seq_lens_j = [base_lens[i] + j for i in range(num_reqs)]
                    cm.seq_lens_list = list(seq_lens_j)
                    # replicate the builder's FIA TND dummy-request padding
                    pad = len(md.seq_lens_list) - num_reqs
                    if pad > 0:
                        cm.seq_lens_list += [1] * pad
                    sl_t = md.seq_lens.new_tensor(seq_lens_j)
                    cm.seq_lens = sl_t
                    cm.seq_lens_cpu = sl_t
                    # the builder's build() prefers _seq_lens_cpu when set;
                    # refresh it on the copy so no consumer sees stale values
                    if hasattr(cm, "_seq_lens_cpu"):
                        cm._seq_lens_cpu = sl_t
                    if hasattr(cm, "max_seq_len"):
                        cm.max_seq_len = max(seq_lens_j)
                    meta_j[key] = cm
                steps.append(meta_j)
            return steps
        except Exception as exc:  # never let prebuild break the request
            logger.warning("talker mecha prebuild metadata failed, falling back: %s", exc)
            return None

    def _mecha_cos_prepare(self, window: int, pos_start: int) -> bool:
        """Precompute _cos/_sin rows 0..W-1 for positions pos_start..+W-1.

        The captured 1-token decode graph reads row 0 of the base _cos/_sin
        buffers at replay; per step j the loop would call
        update_cos_sin(positions=[pos_start+j]), which index_selects one row
        from the cache and expands it. Prebuilding fills all W rows once and
        each step becomes a single row copy. Returns False when the rotary
        globals are not ready (caller keeps calling update_cos_sin per step).
        """
        try:
            import vllm_ascend.ops.rotary_embedding as _rot
            if (
                getattr(_rot, "_cos", None) is None
                or getattr(_rot, "_sin", None) is None
                or getattr(_rot, "_cos_sin_cache", None) is None
            ):
                return False
            all_pos = torch.tensor(
                [pos_start + j for j in range(window)], dtype=torch.long
            )
            update_cos_sin(all_pos)  # fills _cos[:, 0:W] and _sin[:, 0:W]
            return getattr(_rot, "_cos", None) is not None
        except Exception:
            return False

    def _mecha_cos_step(self, j: int) -> None:
        """Per-step row copy: move precomputed row j into row 0 (the rows the
        captured 1-token graph reads at replay). Same device values as a fresh
        update_cos_sin(positions=[pos_start+j])."""
        import vllm_ascend.ops.rotary_embedding as _rot
        _rot._cos[:, :1].copy_(_rot._cos[:, j : j + 1])
        _rot._sin[:, :1].copy_(_rot._sin[:, j : j + 1])

    def _mecha_engine_tokens(
        self, num_reqs: int, req_ids: list[str], hidden_rows: torch.Tensor | None = None
    ) -> list[int]:
        """Engine STOP/CONTINUE token per request from the shared per-request
        state. make_omni_output assembles _batch_stop_logits FROM
        state['finished'] (row 0 = [-inf,0] -> STOP token 1; row 1 = [0,-inf] ->
        CONTINUE token 0), so the engine token is exactly int(finished),
        already known on the host — the argmax + D2H tolist in
        _talker_local_engine_token is pure overhead. Trace mode cross-checks
        against the device snapshot and warns on any mismatch; any exception
        falls back to the device path (never return a guessed token)."""
        toks: list[int] = []
        ok = True
        try:
            for i in range(num_reqs):
                rid = req_ids[i]
                infos = self.model_intermediate_buffer.get(rid, {})
                st = infos.get("audio_state")
                if not isinstance(st, dict):
                    ok = False
                    break
                fin = bool(st.get("finished"))
                toks.append(1 if fin else 0)
        except Exception:
            ok = False
        if not ok or len(toks) != num_reqs:
            return self._talker_local_engine_token(hidden_rows)
        if (
            getattr(self, "_talker_debug_trace", False)
            or os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1"
        ):
            try:
                if hidden_rows is not None:
                    ref = self._talker_local_engine_token(hidden_rows)
                    for i in range(min(len(ref), len(toks))):
                        if ref[i] != toks[i]:
                            logger.warning(
                                "[TALKER-MECHA] engine-token mismatch req=%s state=%s device=%s",
                                req_ids[i] if i < len(req_ids) else "?",
                                toks[i],
                                ref[i],
                            )
            except Exception:
                pass
        return toks

    def _mecha_collect(
        self,
        mm: Any,
        tokens: list[int],
        codec_deltas: list[list[torch.Tensor]],
        finished_flags: list[list[bool]],
        engine_tokens: list[list[int]],
        num_reqs: int,
    ) -> bool:
        """Light per-step collect for the known local-batch output shape
        (codes.audio = per-req delta list, meta.finished = per-req bool list).
        Returns False on shape surprises so the caller falls back to the
        generic walk."""
        try:
            codes = mm.get("codes", {}) if isinstance(mm, Mapping) else None
            meta = mm.get("meta", {}) if isinstance(mm, Mapping) else None
            audio = codes.get("audio") if isinstance(codes, Mapping) else None
            fin = meta.get("finished") if isinstance(meta, Mapping) else None
            if not (
                isinstance(audio, (list, tuple))
                and isinstance(fin, (list, tuple))
                and len(audio) >= num_reqs
                and len(fin) >= num_reqs
            ):
                return False
            for i in range(num_reqs):
                codec_deltas[i].append(audio[i])
                finished_flags[i].append(bool(fin[i]))
                engine_tokens[i].append(tokens[i])
            return True
        except Exception:
            return False

    def _talker_local_decode_loop(
        self,
        scheduler_output: SchedulerOutput,
        *,
        num_reqs: int,
        req_ids: list[str],
        num_scheduled_tokens_np: np.ndarray,
        cudagraph_mode: Any,
        batch_desc: Any,
        use_spec_decode: bool,
        logits_indices: torch.Tensor,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        model_kwargs: dict[str, Any],
        num_tokens_padded: int,
        num_tokens_across_dp: int,
        has_encoder_input: bool,
        attn_metadata: Any,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any,
        from_scratch: bool,
    ) -> tuple[torch.Tensor, Any, torch.Tensor, list[list[int]], bool]:
        """Run a full local decode window of W sequential 1-token-per-req
        forwards within this engine step.

        - W == S == 1 (no local extra): fall through, nothing to do.
        - S == 1, W > 1 (first window after prefill): the scheduled forward
          already produced step 0's codec; run W-1 extra local steps.
        - S > 1 (steady state): the scheduled batch has S tokens but its
          multi-token forward samples garbage (rows 1..S-1 carry stale
          embeddings), so the whole window is re-run from scratch with
          ``from_scratch=True`` — every step is a fresh 1-token-per-req
          forward fed by the just-sampled codec embedding.

        Returns (hidden_states, multimodal_outputs, positions, engine_tokens,
        window_ran); engine_tokens[i] holds one STOP/CONTINUE token per
        executed step per request (the final step's flag is also the
        bookkeeping token — sample_tokens appends it to the list)."""
        K = self._talker_local_steps
        S = int(num_scheduled_tokens_np[0])
        if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or getattr(
            self, "_talker_debug_trace", False
        ):
            logger.info(
                "[TALKER-LOCAL] loop entry: K=%d S=%d num_reqs=%d from_scratch=%s async=%s mode=%s",
                K, S, num_reqs, from_scratch, self.use_async_scheduling, cudagraph_mode,
            )
        if K <= 1 or S < 1:
            return hidden_states, multimodal_outputs, positions, [[] for _ in range(num_reqs)], False

        window = self._talker_local_window(0, S) if num_reqs == 1 else min(
            (self._talker_local_window(i, S) for i in range(num_reqs))
        )
        if window <= 1 and not from_scratch:
            return hidden_states, multimodal_outputs, positions, [[] for _ in range(num_reqs)], False

        codec_deltas: list[list[torch.Tensor]] = [[] for _ in range(num_reqs)]
        finished_flags: list[list[bool]] = [[] for _ in range(num_reqs)]
        engine_tokens: list[list[int]] = [[] for _ in range(num_reqs)]

        def collect_step_outputs(mm: Any, tokens_override: list[int] | None = None) -> None:
            """Append this forward's codec deltas + finished flags + engine
            tokens to the per-request accumulators."""
            if isinstance(mm, Mapping):
                codes = mm.get("codes", {})
                if isinstance(codes, Mapping):
                    audio = codes.get("audio")
                    if isinstance(audio, (list, tuple)):
                        for i, d in enumerate(audio):
                            if i < num_reqs and isinstance(d, torch.Tensor):
                                codec_deltas[i].append(d)
                meta = mm.get("meta", {})
                if isinstance(meta, Mapping):
                    fin = meta.get("finished")
                    if isinstance(fin, (list, tuple)):
                        for i, f in enumerate(fin):
                            if i < num_reqs:
                                finished_flags[i].append(bool(f))
            if tokens_override is not None:
                for i in range(num_reqs):
                    engine_tokens[i].append(tokens_override[i])

        # save and override scheduled-token count so model kwargs / spans see
        # the 1-token-per-req local batch
        saved_nstp = getattr(self, "_omni_num_scheduled_tokens_np", None)
        self._omni_num_scheduled_tokens_np = np.ones(num_reqs, dtype=np.int32)

        # E3 timing: measure per-step host cost breakdown (config-gated, off default)
        _e3 = os.environ.get("OMNI_TALKER_E3", "0") == "1" or getattr(self, "_talker_e3", False)
        _e3_t0 = time.perf_counter()
        _e3_acc = {"prebuild": 0.0, "feedback": 0.0, "metadata": 0.0, "cos": 0.0, "forward": 0.0, "tok": 0.0, "collect": 0.0}

        local_batch_desc: Any = None
        try:
            if from_scratch:
                # Re-run the entire window: every step is a local step. The
                # first step reuses the scheduled batch's row-0 embedding /
                # position (they are correct for position C+1); feedback
                # advances inputs for steps 2..W.
                (_, local_batch_desc, _, _local_ntadp, _) = self._determine_batch_execution_and_padding(
                    num_tokens=num_reqs,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=np.ones(num_reqs, dtype=np.int32),
                    max_num_scheduled_tokens=1,
                    use_cascade_attn=False,
                    force_eager=False,
                    num_encoder_reqs=0,
                )
                # fresh 1-token-per-req query_start_loc
                self.query_start_loc.np[0] = 0
                self.query_start_loc.np[1 : num_reqs + 1] = np.arange(1, num_reqs + 1)
                self.query_start_loc.np[num_reqs + 1 :].fill(num_reqs)
                self.query_start_loc.copy_to_gpu()

                # position rows for step 0: first window position per req
                for i in range(num_reqs):
                    p0 = int(self.input_batch.num_computed_tokens_cpu[i])
                    self.positions[i].fill_(p0)
                    self.seq_lens[i].fill_(p0 + 1)
                    if hasattr(self, "optimistic_seq_lens_cpu"):
                        self.optimistic_seq_lens_cpu[i] = p0 + 1

                # MECHA: prebuild per-step metadata + cos table once per window
                _mecha_steps = None
                _mecha_cos_ok = False
                _mecha_pos0 = int(self.positions[0]) if num_reqs == 1 else -1
                if getattr(self, "_mecha_enabled", False) and num_reqs == 1:
                    if self._mecha_meta:
                        if _e3:
                            _t = time.perf_counter()
                        _mecha_steps = self._mecha_prebuild_metadata(
                            num_reqs, req_ids[:num_reqs], window
                        )
                        if _e3:
                            _e3_acc["prebuild"] += time.perf_counter() - _t
                    if self._mecha_cos:
                        _mecha_cos_ok = self._mecha_cos_prepare(window, _mecha_pos0)

                for step_idx in range(window):
                    # refresh inputs: step 0 uses the scheduled row-0 embeds;
                    # steps 1..W-1 get the freshly sampled codec embed
                    if _e3:
                        _t = time.perf_counter()
                    if step_idx > 0:
                        for i in range(num_reqs):
                            self._talker_local_step_feedback(i, step_idx)
                    if _e3:
                        _e3_acc["feedback"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                    if _mecha_steps is not None:
                        local_metadata = _mecha_steps[step_idx]
                    else:
                        local_metadata, _ = self._build_attention_metadata(
                            num_tokens=num_reqs,
                            num_tokens_padded=num_reqs,
                            num_reqs=num_reqs,
                            num_reqs_padded=num_reqs,
                            max_query_len=1,
                            ubatch_slices=None,
                            logits_indices=None,
                            use_spec_decode=False,
                            num_scheduled_tokens={rid: 1 for rid in req_ids[:num_reqs]},
                            num_scheduled_tokens_np=np.ones(num_reqs, dtype=np.int32),
                        )
                    if _e3:
                        _e3_acc["metadata"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                    if _mecha_cos_ok:
                        self._mecha_cos_step(step_idx)
                    else:
                        update_cos_sin(self.positions[:num_reqs])
                    if _e3:
                        _e3_acc["cos"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                    if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or getattr(
                        self, "_talker_debug_trace", False
                    ):
                        try:
                            logger.info(
                                "[TALKER-META] step=%d/%d pos=%s seq_lens=%s opt=%s num_comp_cpu=%s num_comp_gpu=%s",
                                step_idx, window,
                                [int(self.positions[i]) for i in range(num_reqs)],
                                [int(self.seq_lens[i]) for i in range(num_reqs)],
                                [int(self.optimistic_seq_lens_cpu[i]) for i in range(num_reqs)],
                                [int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(num_reqs)],
                                [int(self.num_computed_tokens[i]) for i in range(num_reqs)],
                            )
                        except Exception:
                            pass
                    _mecha_light_this = (
                        getattr(self, "_mecha_light", False) and step_idx < window - 1
                    )
                    if _mecha_light_this:
                        self._talker_light_next = True
                    try:
                        with (
                            record_function_or_nullcontext("talker_local_forward"),
                            set_ascend_forward_context(
                                local_metadata,
                                self.vllm_config,
                                num_tokens=num_reqs,
                                num_tokens_across_dp=_local_ntadp,
                                aclgraph_runtime_mode=(
                                    CUDAGraphMode.FULL if from_scratch else cudagraph_mode
                                ),
                                batch_descriptor=local_batch_desc,
                                num_actual_tokens=num_reqs,
                                model_instance=self.model,
                                max_tokens_across_pcp=0,
                                skip_compiled=has_encoder_input,
                            ),
                        ):
                            _fwd_out = self._model_forward(
                                num_reqs,
                                input_ids=self.input_ids.gpu[:num_reqs],
                                positions=self.positions[:num_reqs],
                                intermediate_tensors=None,
                                inputs_embeds=self.inputs_embeds.gpu[:num_reqs],
                                **model_kwargs,
                            )
                    finally:
                        self._talker_light_next = False
                    if _e3:
                        _e3_acc["forward"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                    if _mecha_light_this and isinstance(_fwd_out, tuple) and len(_fwd_out) == 2:
                        # MECHA light: (raw hidden, light mm dict) — skip the
                        # OmniOutput wrapper + generic walk for non-final steps
                        hidden_states, multimodal_outputs = _fwd_out
                    else:
                        hidden_states, multimodal_outputs = self.extract_multimodal_outputs(_fwd_out)
                    # engine token: read cached stop logits WITHOUT consuming
                    if getattr(self, "_mecha_tok", False):
                        tok = self._mecha_engine_tokens(
                            num_reqs, req_ids[:num_reqs], hidden_states[:num_reqs]
                        )
                    else:
                        tok = self._talker_local_engine_token(hidden_states[:num_reqs])
                    if _e3:
                        _e3_acc["tok"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                    if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or getattr(
                        self, "_talker_debug_trace", False
                    ):
                        try:
                            _bsl = self._batch_stop_logits_snapshot()
                            logger.info(
                                "[TALKER-TOK] step=%d/%d tok=%s bsl=%s",
                                step_idx, window, tok,
                                str(_bsl.detach().cpu().tolist()) if _bsl is not None else None,
                            )
                        except Exception:
                            pass
                    if getattr(self, "_mecha_collect_enabled", False) and self._mecha_collect(
                        multimodal_outputs,
                        tok,
                        codec_deltas,
                        finished_flags,
                        engine_tokens,
                        num_reqs,
                    ):
                        pass
                    else:
                        collect_step_outputs(multimodal_outputs, tokens_override=tok)
                    if _e3:
                        _e3_acc["collect"] += time.perf_counter() - _t
                    if os.environ.get("OMNI_TALKER_DEBUG_TRACE", "0") == "1" or getattr(
                        self, "_talker_debug_trace", False
                    ):
                        try:
                            _dbg_codes = multimodal_outputs.get("codes", {}).get("audio") if isinstance(
                                multimodal_outputs, Mapping
                            ) else None
                            _dbg_vals = []
                            if isinstance(_dbg_codes, (list, tuple)):
                                for _d in _dbg_codes:
                                    if isinstance(_d, torch.Tensor):
                                        _dbg_vals.append(int(_d.reshape(-1)[0]) if _d.numel() else -1)
                            logger.info(
                                "[TALKER-LOCAL] from_scratch step=%d/%d pos=%s tok=%s codec=%s fin_flags=%s",
                                step_idx, window,
                                [int(self.positions[i]) for i in range(num_reqs)],
                                tok, _dbg_vals,
                                [f[-1] if f else None for f in finished_flags],
                            )
                        except Exception:
                            pass
                    if all(flags and flags[-1] for flags in finished_flags):
                        break
            else:
                # S == 1: step 0 already ran (the scheduled forward). Capture
                # its deltas/flags/token, then run W-1 local extra steps.
                if getattr(self, "_mecha_tok", False):
                    tok0 = self._mecha_engine_tokens(
                        num_reqs, req_ids[:num_reqs], hidden_states[:num_reqs]
                    )
                else:
                    tok0 = self._talker_local_engine_token(hidden_states[:num_reqs])
                if getattr(self, "_mecha_collect_enabled", False) and self._mecha_collect(
                    multimodal_outputs,
                    tok0,
                    codec_deltas,
                    finished_flags,
                    engine_tokens,
                    num_reqs,
                ):
                    pass
                else:
                    collect_step_outputs(multimodal_outputs, tokens_override=tok0)
                if all(flags and flags[-1] for flags in finished_flags):
                    return hidden_states, multimodal_outputs, positions, engine_tokens, True
                (_, local_batch_desc, _, _local_ntadp, _) = self._determine_batch_execution_and_padding(
                    num_tokens=num_reqs,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=np.ones(num_reqs, dtype=np.int32),
                    max_num_scheduled_tokens=1,
                    use_cascade_attn=False,
                    force_eager=False,
                    num_encoder_reqs=0,
                )
                self.query_start_loc.np[0] = 0
                self.query_start_loc.np[1 : num_reqs + 1] = np.arange(1, num_reqs + 1)
                self.query_start_loc.np[num_reqs + 1 :].fill(num_reqs)
                self.query_start_loc.copy_to_gpu()

                # MECHA: prebuild per-step metadata + cos table once per window
                _mecha_steps = None
                _mecha_cos_ok = False
                _mecha_pos0 = int(self.positions[0]) if num_reqs == 1 else -1
                if getattr(self, "_mecha_enabled", False) and num_reqs == 1:
                    if self._mecha_meta:
                        if _e3:
                            _t = time.perf_counter()
                        _mecha_steps = self._mecha_prebuild_metadata(
                            num_reqs, req_ids[:num_reqs], window
                        )
                        if _e3:
                            _e3_acc["prebuild"] += time.perf_counter() - _t
                    if self._mecha_cos:
                        _mecha_cos_ok = self._mecha_cos_prepare(window, _mecha_pos0)

                for local_step in range(1, window):
                    for i in range(num_reqs):
                        self._talker_local_step_feedback(i, local_step)
                    if _mecha_steps is not None:
                        local_metadata = _mecha_steps[local_step]
                    else:
                        local_metadata, _ = self._build_attention_metadata(
                            num_tokens=num_reqs,
                            num_tokens_padded=num_reqs,
                            num_reqs=num_reqs,
                            num_reqs_padded=num_reqs,
                            max_query_len=1,
                            ubatch_slices=None,
                            logits_indices=None,
                            use_spec_decode=False,
                            num_scheduled_tokens={rid: 1 for rid in req_ids[:num_reqs]},
                            num_scheduled_tokens_np=np.ones(num_reqs, dtype=np.int32),
                        )
                    if _mecha_cos_ok:
                        self._mecha_cos_step(local_step)
                    else:
                        update_cos_sin(self.positions[:num_reqs])
                    _mecha_light_this = (
                        getattr(self, "_mecha_light", False) and local_step < window - 1
                    )
                    if _mecha_light_this:
                        self._talker_light_next = True
                    try:
                        with (
                            record_function_or_nullcontext("talker_local_forward"),
                            set_ascend_forward_context(
                                local_metadata,
                                self.vllm_config,
                                num_tokens=num_reqs,
                                num_tokens_across_dp=_local_ntadp,
                                aclgraph_runtime_mode=(
                                    CUDAGraphMode.FULL if from_scratch else cudagraph_mode
                                ),
                                batch_descriptor=local_batch_desc,
                                num_actual_tokens=num_reqs,
                                model_instance=self.model,
                                max_tokens_across_pcp=0,
                                skip_compiled=has_encoder_input,
                            ),
                        ):
                            _fwd_out = self._model_forward(
                                num_reqs,
                                input_ids=self.input_ids.gpu[:num_reqs],
                                positions=self.positions[:num_reqs],
                                intermediate_tensors=None,
                                inputs_embeds=self.inputs_embeds.gpu[:num_reqs],
                                **model_kwargs,
                            )
                    finally:
                        self._talker_light_next = False
                    if _mecha_light_this and isinstance(_fwd_out, tuple) and len(_fwd_out) == 2:
                        hidden_states, multimodal_outputs = _fwd_out
                    else:
                        hidden_states, multimodal_outputs = self.extract_multimodal_outputs(_fwd_out)
                    if getattr(self, "_mecha_tok", False):
                        tok = self._mecha_engine_tokens(
                            num_reqs, req_ids[:num_reqs], hidden_states[:num_reqs]
                        )
                    else:
                        tok = self._talker_local_engine_token(hidden_states[:num_reqs])
                    if getattr(self, "_mecha_collect_enabled", False) and self._mecha_collect(
                        multimodal_outputs,
                        tok,
                        codec_deltas,
                        finished_flags,
                        engine_tokens,
                        num_reqs,
                    ):
                        pass
                    else:
                        collect_step_outputs(multimodal_outputs, tokens_override=tok)
                    if all(flags and flags[-1] for flags in finished_flags):
                        break
        finally:
            if saved_nstp is not None:
                self._omni_num_scheduled_tokens_np = saved_nstp

        if _e3:
            _total = time.perf_counter() - _e3_t0
            logger.info(
                "[E3] window=%d total=%.3fms step_avg=%.3fms breakdown=%s",
                window, _total * 1000, (_total / max(window, 1)) * 1000,
                {k: round(v * 1000, 3) for k, v in _e3_acc.items()},
            )

        # merge codec deltas into per-request (W,1) tensors and keep the last
        # step's finished flags
        merged_codes: list[torch.Tensor] = []
        merged_finished: list[Any] = []
        for i in range(num_reqs):
            if codec_deltas[i]:
                try:
                    merged_codes.append(torch.cat(codec_deltas[i], dim=0).contiguous())
                except Exception:
                    merged_codes.append(codec_deltas[i][-1])
            else:
                merged_codes.append(torch.empty(0, dtype=torch.long, device=hidden_states.device))
            merged_finished.append(finished_flags[i][-1] if finished_flags[i] else False)
        if isinstance(multimodal_outputs, Mapping):
            merged_mm = dict(multimodal_outputs)
            codes = dict(merged_mm.get("codes", {}) or {})
            codes["audio"] = merged_codes
            merged_mm["codes"] = codes
            meta = dict(merged_mm.get("meta", {}) or {})
            meta["finished"] = merged_finished
            merged_mm["meta"] = meta
            multimodal_outputs = merged_mm

        logger.debug("talker local decode: K=%d S=%d window=%d from_scratch=%s", K, S, window, from_scratch)
        return hidden_states, multimodal_outputs, self.positions, engine_tokens, True

    #  -------------------------------------- Omni-new -------------------------------------------------

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    #  -------------------------------------- Omni-new -------------------------------------------------
    def capture_model(self) -> int:
        npugraph_memory_bytes = super().capture_model()
        self._capture_talker_mtp_graphs()
        return npugraph_memory_bytes

    def _capture_talker_mtp_graphs(self) -> None:
        if not self.has_talker_mtp or not isinstance(self.talker_mtp, ACLGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes, reverse=True)
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_ascend_forward_context(
                            None,
                            self.vllm_config,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.npu.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """See gpu_ar_model_runner._model_needs_full_prefix_hidden_states."""
        model = getattr(self, "model", None)
        return bool(getattr(model, "requires_full_prefix_cached_hidden_states", True))

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, Mapping):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hs_for_cache,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=self.input_batch.block_table[0].slot_mapping.cpu,
                num_tokens_padded=num_tokens_padded,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            if self._model_needs_full_prefix_hidden_states():
                combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                    query_start_loc=self.query_start_loc.cpu,
                    input_batch=self.input_batch,
                    hidden_states=hidden_states,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ):
        if combined_hidden_states is not None:
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        return hidden_states_cpu[start:end]


    def _build_multimodal_outputs(
        self,
        per_req_payloads: list[dict[str, object] | None] | None,
    ) -> list[dict[str, torch.Tensor] | None] | None:
        if self.vllm_config.model_config.engine_output_type == "text":
            return None
        if per_req_payloads is None:
            return None
        wire_payloads: list[dict[str, torch.Tensor] | None] = []
        for payload in per_req_payloads:
            if not payload:
                wire_payloads.append(None)
            else:
                wire_payloads.append(_ensure_tensor_values(payload))
        if all(item is None for item in wire_payloads):
            return None
        return wire_payloads


    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        final_stage_id = self._request_final_stage_id(req_id)
        needs_payload = final_stage_id is None or final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids_output_copy: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids_output_copy if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        # Single-stage AR TTS models (e.g. VoxCPM2) finish on this stage but still
        # need multimodal payloads for final audio postprocess/output.
        if engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids_output_copy
        return engine_output_type, downstream_req_ids

    @staticmethod
    def _sparse_mm_req_ids(multimodal_outputs: Any) -> list[str] | None:
        if not isinstance(multimodal_outputs, dict):
            return None
        meta = multimodal_outputs.get("meta")
        req_ids = None
        sparse_audio = False
        if isinstance(meta, dict):
            req_ids = meta.get("req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(meta.get("sparse_audio"))
        if req_ids is None:
            req_ids = multimodal_outputs.get("meta.req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(multimodal_outputs.get("meta.sparse_audio"))
        if not sparse_audio:
            return None
        if not isinstance(req_ids, list):
            return None
        return [rid for rid in req_ids if isinstance(rid, str)]

    @staticmethod
    def _is_sparse_audio_marker(value: Any) -> bool:
        if isinstance(value, list):
            return any(str(item).lower() in ("1", "true", "yes", "on") for item in value)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    #  -------------------------------------- Omni-new -------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()
        if self.ascend_config.profiling_chunk_config.enabled:
            self._sync_device()
            self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if self._pending_full_payload_send:
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in self.requests})
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if ((
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios.
            # It does not affect other use cases. This is a temporary workaround and
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
        #  -------------------------------------- Omni-new -------------------------------------------------

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(scheduler_output)

                #  -------------------------------------- Omni-new -------------------------------------------------
                if scheduler_output.finished_req_ids and hasattr(self.model, "on_requests_finished"):
                    self.model.on_requests_finished(scheduler_output.finished_req_ids)
                #  -------------------------------------- Omni-new -------------------------------------------------

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)

                        kv_ids = self.kv_extracted_req_ids
                        self.kv_extracted_req_ids = None

                        output = make_empty_encoder_model_runner_output(scheduler_output)
                        if kv_ids:
                            output = copy(output)
                            output.kv_extracted_req_ids = kv_ids
                        return self.attach_omni_connector_output(output)

                # `<= 0`: upstream can schedule a negative span, which is truthy (#5196).
                if num_scheduled_tokens <= 0:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    if not has_kv_transfer_group():
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    else:
                        output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids

                    return self.attach_omni_connector_output(output)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                # We simply utilize the implementation in vLLM.
                if self.cache_config.mamba_cache_mode == "align":
                    # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                    # to decide copy operations, so we must apply deferred
                    # corrections before it runs.
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        self._get_mamba_copy_bufs(),
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size * self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        batch_desc.num_reqs,
                    )

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            #  -------------------------------------- Omni-new -------------------------------------------------
            if hasattr(self.model, "prepare_runner_inputs"):
                input_ids, positions = self.model.prepare_runner_inputs(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    req_ids=req_ids[:num_reqs],
                    num_computed_tokens=self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    num_scheduled_tokens=num_scheduled_tokens_np[:num_reqs],
                    input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
                )
            #  -------------------------------------- Omni-new -------------------------------------------------

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # R5: decide local-decode eligibility BEFORE the scheduled forward so
        # the S>1 steady-state window can skip the multi-token forward entirely
        # (its rows 1..S-1 carry stale embeddings and its make_omni_output
        # would advance the request codec state with a garbage sample).
        talker_local_tokens: list[list[int]] | None = None
        _talker_local_active = self._talker_local_decode_eligible(
            scheduler_output,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cudagraph_mode=cudagraph_mode,
            use_spec_decode=use_spec_decode,
            has_encoder_input=has_encoder_input,
            num_encoder_reqs=num_encoder_reqs,
        )
        _talker_local_from_scratch = bool(_talker_local_active and int(num_scheduled_tokens_np[0]) > 1)
        # For the K-window the dispatcher returns NONE (it cannot know about
        # the scheduler-K knob); the local loop replays its own 1-token batches
        # in FULL_DECODE_ONLY mode, so the scheduled forward context below is
        # skipped entirely.
        _talker_local_mode = CUDAGraphMode.FULL_DECODE_ONLY if _talker_local_from_scratch else cudagraph_mode

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            if not _talker_local_from_scratch:
                hidden_states = self._model_forward(
                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
                )
            else:
                # S>1 local window: the scheduled multi-token forward is
                # skipped; hidden_states is produced by the first local step.
                hidden_states = None
        with record_function_or_nullcontext("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            # [Omni] Map pending ropes metadata to req_ids.
            flush_pending_metadata = getattr(self.model, "flush_pending_metadata", None)
            if callable(flush_pending_metadata):
                flush_pending_metadata(req_ids[:num_reqs])

            if not _talker_local_from_scratch:
                hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            else:
                # S>1 local window: no scheduled forward ran; the local loop
                # below produces hidden_states + multimodal_outputs.
                multimodal_outputs = None

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, Mapping)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            # R5: runner-local multi-step Talker decode. When eligible, run the
            # local window (K sequential 1-token decodes) and merge the K-1 extra
            # codec deltas into the multimodal output; the engine-visible token
            # list is extended to K per request in sample_tokens.
            if _talker_local_active:
                (
                    hidden_states,
                    multimodal_outputs,
                    positions,
                    talker_local_tokens,
                    _window_ran,
                ) = self._talker_local_decode_loop(
                    scheduler_output,
                    num_reqs=num_reqs,
                    req_ids=req_ids[:num_reqs],
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode,
                    batch_desc=batch_desc,
                    use_spec_decode=use_spec_decode,
                    logits_indices=logits_indices,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    model_kwargs=model_kwargs,
                    num_tokens_padded=num_reqs,
                    num_tokens_across_dp=num_reqs,
                    has_encoder_input=has_encoder_input,
                    attn_metadata=attn_metadata,
                    hidden_states=hidden_states,
                    multimodal_outputs=multimodal_outputs,
                    from_scratch=_talker_local_from_scratch,
                )
            #  -------------------------------------- Omni-new -------------------------------------------------
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]

            #  -------------------------------------- Omni-new -------------------------------------------------
            self._maybe_update_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
            )
            #  -------------------------------------- Omni-new -------------------------------------------------

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                # After a local decode window the hidden batch is 1 row per
                # request; the scheduled-batch logits_indices no longer match.
                effective_logits_indices = logits_indices
                if talker_local_tokens is not None and not self.broadcast_pp_output:
                    effective_logits_indices = torch.arange(num_reqs, device=logits_indices.device)

                sample_hidden_states = hidden_states[effective_logits_indices]
                #  -------------------------------------- Omni-new -------------------------------------------------
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
                #  -------------------------------------- Omni-new -------------------------------------------------
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    #  -------------------------------------- Omni-new -------------------------------------------------
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)
                    #  -------------------------------------- Omni-new -------------------------------------------------

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            # R5: pending local-window engine tokens ride a runner attribute,
            # NOT the ExecuteModelState tuple (the shared tuple is unpacked
            # positionally by the stage2 generation runner — adding a field
            # breaks it). Cleared in sample_tokens after consumption.
            self._talker_local_tokens_pending = talker_local_tokens
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
                multimodal_outputs, # Omni-specific
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        if self.vllm_config.model_config.enable_return_routed_experts and hasattr(self, "_positions_cpu"):
            self._omni_routed_experts_d2h(scheduler_output)

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            self.input_batch.update_async_output_token_ids()
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                if getattr(self.model, "model_sampler_needs_output_token_ids", True) or not getattr(
                    sampling_metadata, "no_penalties", False
                ):
                    prepared_sampling_metadata = self._sampling_metadata_for_model_sampler(sampling_metadata)
                else:
                    # Rebuilding decoded-token history costs a D2H sync + full
                    # list copies per step; skip it for model samplers that
                    # declare they never read it (unless penalties need it).
                    prepared_sampling_metadata = sampling_metadata
                self._apply_duplex_sampling(logits, prepared_sampling_metadata)
                sampler_output = model_sample(logits, prepared_sampling_metadata)
                if sampler_output is not None:
                    return sampler_output
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        combined_hidden_states = None
        combined_multimodal_outputs = None
        mm_cpu = {}
        #  -------------------------------------- Omni-new -------------------------------------------------


        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return self.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return self.attach_omni_connector_output(output)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
            multimodal_outputs, # Omni-Specific
        ) = self.execute_model_state
        # R5: local-window engine tokens ride a runner attribute (the shared
        # tuple must stay 13 fields for the stage2 generation runner).
        talker_local_tokens = getattr(self, "_talker_local_tokens_pending", None)
        self._talker_local_tokens_pending = None
        # Clear ephemeral state.
        self.execute_model_state = None
        hidden_seq_len = int(hidden_states.shape[0])
        scheduled_seq_len = int(scheduler_output.total_num_scheduled_tokens)

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Correct padding values of prompt_token_ids to match the logits vocabulary size.
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        # Drop min-tokens stop ids the head cannot emit (e.g. the text
        # tokenizer EOS folded into all_stop_token_ids on a narrow codec
        # talker head); they would index_put_ out of bounds (#4962).
        if logits is not None:
            sanitize_min_tokens_stop_ids(
                self.input_batch.sampling_metadata.logitsprocs,
                logits.shape[-1],
            )
        #  -------------------------------------- Omni-new -------------------------------------------------


        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu: torch.Tensor | None = None # type: ignore[no-redef]

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        # R5: runner-local decode — the engine must see all K engine tokens of
        # the local window so its per-step accounting (num_computed_tokens,
        # block allocation) advances in lockstep with the K codec tokens we
        # already wrote to KV. The loop already collected one STOP/CONTINUE
        # token per executed step; they replace the bookkeeping-sampled token.
        # The stop token (id 1, stop_token_ids=[1]) trims the request at the
        # right point in the engine's _update_request_with_output.
        if talker_local_tokens is not None and any(toks for toks in talker_local_tokens):
            final_tokens: list[list[int]] = []
            for rid in req_ids_output_copy:
                idx = req_id_to_index_output_copy[rid]
                local_toks = talker_local_tokens[idx]
                if local_toks:
                    final_tokens.append(local_toks)
                elif idx < len(valid_sampled_token_ids) and valid_sampled_token_ids[idx]:
                    final_tokens.append(valid_sampled_token_ids[idx])
                else:
                    final_tokens.append([0])
            valid_sampled_token_ids = final_tokens
            logprobs_lists = None

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model())
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

        routed_experts_lists = None
        if self.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(self.input_batch, "num_tokens_no_spec"):
                routed_experts_lists = self._omni_extract_routed_experts(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Snapshot per-request scheduling metadata on the main thread; the
        # background builder thread must NOT read self.input_batch /
        # self.query_start_loc (mutated by the next execute_model step).
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )
        query_start_loc_cpu = self.query_start_loc.cpu
        if callable(query_start_loc_cpu):
            query_start_loc_cpu = query_start_loc_cpu()

        _use_async_omni = self._should_use_async_omni_output()
        if _use_async_omni and self._omni_async_step_budget > 0:
            # Wave4 front-loaded: per-request step budget. Each request's
            # first N steps run async (TTFT/TTFP window); once the budget is
            # exhausted the request falls back to sync (RTF regression gone).
            _budgeted = False
            for _rid in req_ids_output_copy:
                _n = self._omni_async_steps.get(_rid, 0)
                if _n < self._omni_async_step_budget:
                    _budgeted = True
                self._omni_async_steps[_rid] = _n + 1
            if not _budgeted:
                _use_async_omni = False
        # free step counters for finished requests (bounded memory)
        if len(self._omni_async_steps) > 256:
            _live = set(req_ids_output_copy)
            self._omni_async_steps = {k: v for k, v in self._omni_async_steps.items() if k in _live}

        if _use_async_omni:
            _copy_stream = self._get_or_create_omni_payload_copy_stream()
            _hs_snap = _snapshot_npu_tensor_payload_to_cpu_async(
                hidden_states, copy_stream=_copy_stream, pin_memory=True
            )
            _mm_snap = _snapshot_npu_tensor_payload_to_cpu_async(
                multimodal_outputs, copy_stream=_copy_stream, pin_memory=True
            )

            def _build_omni_output() -> OmniModelRunnerOutput:
                _hs_snap.wait()
                _mm_snap.wait()
                return self._build_omni_output_from_tensors(
                    hidden_states=_hs_snap.payload,
                    multimodal_outputs=_mm_snap.payload,
                    scheduler_output=scheduler_output,
                    req_ids_output_copy=req_ids_output_copy,
                    req_id_to_index_output_copy=req_id_to_index_output_copy,
                    valid_sampled_token_ids=valid_sampled_token_ids,
                    logprobs_lists=logprobs_lists,
                    prompt_logprobs_dict=prompt_logprobs_dict,
                    kv_connector_output=kv_connector_output,
                    ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                    cudagraph_stats=cudagraph_stats,
                    routed_experts_lists=routed_experts_lists,
                    kv_extracted_req_ids=kv_extracted_req_ids,
                    hidden_seq_len=hidden_seq_len,
                    scheduled_seq_len=scheduled_seq_len,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    query_start_loc_cpu=query_start_loc_cpu,
                )

            model_runner_output = None
        else:
            model_runner_output = self._build_omni_output_from_tensors(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                scheduler_output=scheduler_output,
                req_ids_output_copy=req_ids_output_copy,
                req_id_to_index_output_copy=req_id_to_index_output_copy,
                valid_sampled_token_ids=valid_sampled_token_ids,
                logprobs_lists=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                cudagraph_stats=cudagraph_stats,
                routed_experts_lists=routed_experts_lists,
                kv_extracted_req_ids=kv_extracted_req_ids,
                hidden_seq_len=hidden_seq_len,
                scheduled_seq_len=scheduled_seq_len,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                query_start_loc_cpu=query_start_loc_cpu,
            )
        #  -------------------------------------- Omni-new -------------------------------------------------

        if not _use_async_omni:
            if self.ascend_config.profiling_chunk_config.enabled and hasattr(self, "_execution_start_time"):
                self._sync_device()
                model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

            if self.dynamic_eplb:
                with record_function_or_nullcontext("EPLB update"):
                    self.eplb_updator.forward_end()

            if self.debugger is not None:
                self.debugger.stop()
                self.debugger.step()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            if _use_async_omni:
                return _build_omni_output()
            return model_runner_output

        if _use_async_omni:
            async_output = OmniAsyncNPUModelRunnerOutput(
                model_runner_output_builder=_build_omni_output,
                npu_device=self.device,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self._get_or_create_omni_payload_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
            )
        else:
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    #  -------------------------------------- Omni-new -------------------------------------------------
    # Async omni output helpers (NPU-safe): D2H snapshot on the main thread's
    # copy_stream, output assembly on a background thread.
    # -------------------------------------------------------------------------

    @staticmethod
    def _model_omni_flag(model: Any, name: str, default: bool = False) -> bool:
        return bool(getattr(model, name, default)) if model is not None else default

    def _should_use_async_omni_output(self) -> bool:
        """Gate async omni output: async scheduling + async_chunk, no prefix
        cache / speculative decode / returned routed experts.

        Default ON: overlaps the Omni output D2H + connector serialization
        with the next scheduler step, improving TTFT/TTFP at single
        concurrency. Opt out via VLLM_OMNI_NPU_ASYNC_OUTPUT=0.
        """
        if os.environ.get("VLLM_OMNI_NPU_ASYNC_OUTPUT", "").lower() in (
            "0",
            "false",
            "no",
        ):
            return False
        if not self.use_async_scheduling:
            return False
        if getattr(self, "omni_prefix_cache", None) is not None:
            return False
        if self.speculative_config is not None:
            return False
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            model_config = getattr(
                getattr(self, "vllm_config", None), "model_config", None
            )
        if not bool(getattr(model_config, "async_chunk", False)):
            return False
        if bool(getattr(model_config, "enable_return_routed_experts", False)):
            return False
        return True

    def _get_or_create_omni_payload_copy_stream(self) -> torch.npu.Stream:
        stream = getattr(self, "_omni_payload_copy_stream", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._omni_payload_copy_stream = stream
        return stream

    def _build_omni_async_snapshot_payload(
        self,
        *,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"multimodal_outputs": multimodal_outputs}
        if self._model_omni_flag(
            getattr(self, "model", None),
            "omni_pooler_payload_include_hidden",
            default=True,
        ):
            payload["hidden_states"] = hidden_states
        return payload

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _build_omni_output_from_tensors(
        self,
        *,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any,
        scheduler_output: SchedulerOutput,
        req_ids_output_copy: list[str],
        req_id_to_index_output_copy: dict[str, int],
        valid_sampled_token_ids: list[list[int]],
        logprobs_lists: Any,
        prompt_logprobs_dict: dict[str, Any],
        kv_connector_output: Any,
        ec_connector_output: Any,
        cudagraph_stats: Any,
        routed_experts_lists: Any,
        kv_extracted_req_ids: Any,
        hidden_seq_len: int,
        scheduled_seq_len: int,
        num_scheduled_tokens_np: np.ndarray,
        query_start_loc_cpu: Any,
    ) -> OmniModelRunnerOutput:
        """Assemble the Omni pooler payload + OmniModelRunnerOutput.

        ``hidden_states`` / ``multimodal_outputs`` may be NPU or CPU tensors:
        for the async path they are already-CPU snapshots (so the ``.to("cpu")``
        calls below are no-ops), for the sync path they are live NPU tensors.
        """
        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        sparse_mm_req_ids = self._sparse_mm_req_ids(multimodal_outputs)
        sparse_mm_index = {rid: i for i, rid in enumerate(sparse_mm_req_ids or [])}
        if engine_output_type == "audio" and sparse_mm_req_ids is not None:
            sparse_req_id_set = set(sparse_mm_req_ids)
            downstream_req_ids = [rid for rid in req_ids_output_copy if rid in sparse_req_id_set]
        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        hidden_states_cpu = None
        req_hidden_states_cpu: dict[str, torch.Tensor] | None = None
        audio_sparse_output = engine_output_type == "audio" and sparse_mm_req_ids is not None
        needs_scheduled_hidden_payload = needs_pooler_payload and (
            self.omni_prefix_cache is None or not self._model_needs_full_prefix_hidden_states()
        )
        if needs_scheduled_hidden_payload:
            num_valid_tokens = min(
                int(scheduler_output.total_num_scheduled_tokens),
                int(hidden_states.shape[0]),
            )
            if audio_sparse_output:
                pass
            elif len(downstream_req_ids) == len(req_ids_output_copy):
                hidden_states_cpu = hidden_states[:num_valid_tokens].detach().to("cpu").contiguous()
            else:
                req_hidden_states_cpu = {}

        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            combined_hidden_states = None
            combined_multimodal_outputs = None
            mm_cpu = None
            if self.omni_prefix_cache is not None:
                (
                    combined_hidden_states,
                    combined_multimodal_outputs,
                ) = self._maybe_get_combined_prefix_cache_tensors(
                    hidden_states,
                    multimodal_outputs,
                    scheduler_output.num_scheduled_tokens,
                )
            if self.omni_prefix_cache is None or combined_multimodal_outputs is None:
                mm_cpu = build_mm_cpu(
                    flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs
                )

            self._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                combined_hidden_states,
                combined_multimodal_outputs,
                req_ids_filter=downstream_req_id_set,
            )

            if req_hidden_states_cpu is not None and combined_hidden_states is None:
                for rid in downstream_req_ids:
                    idx = req_id_to_index_output_copy[rid]
                    start = int(query_start_loc_cpu[idx])
                    sched = int(num_scheduled_tokens_np[idx])
                    end = start + sched
                    req_hidden_states_cpu[rid] = hidden_states[start:end].detach().to("cpu").contiguous()

            pooler_output = []
            for rid in req_ids_output_copy:
                if rid not in downstream_req_id_set:
                    pooler_output.append({})
                    continue
                idx = req_id_to_index_output_copy[rid]
                start = int(query_start_loc_cpu[idx])
                sched = int(num_scheduled_tokens_np[idx])
                end = start + sched
                payload: dict[str, object] = {}
                if not audio_sparse_output:
                    if req_hidden_states_cpu is not None and combined_hidden_states is None:
                        req_hidden_states = req_hidden_states_cpu[rid]
                    else:
                        req_hidden_states = self._resolve_req_hidden_states(
                            hidden_states_cpu,
                            combined_hidden_states,
                            rid,
                            start,
                            end,
                        )
                    payload["hidden"] = req_hidden_states

                mm_payload: dict[str, object] = {}
                if combined_multimodal_outputs or mm_cpu:
                    if combined_multimodal_outputs:
                        def _unwrap_lists(v):
                            if isinstance(v, list):
                                return v[idx] if idx < len(v) else v[0]
                            if isinstance(v, dict):
                                return {k: _unwrap_lists(sv) for k, sv in v.items()}
                            return v

                        for mm_key in combined_multimodal_outputs.keys():
                            mm_payload[mm_key] = _unwrap_lists(combined_multimodal_outputs[mm_key][rid])
                    else:
                        for mm_key, mm_val in mm_cpu.items():
                            if mm_key in {"meta.req_id", "meta.sparse_audio"}:
                                continue
                            if audio_sparse_output and isinstance(mm_val, list):
                                sparse_idx = sparse_mm_index.get(rid)
                                if sparse_idx is None:
                                    continue
                                if sparse_idx >= len(mm_val):
                                    logger.warning(
                                        "Sparse multimodal payload mismatch for request %s: index %d >= %d.",
                                        rid,
                                        sparse_idx,
                                        len(mm_val),
                                    )
                                    continue
                                sparse_val = mm_val[sparse_idx]
                                mm_payload[mm_key] = (
                                    sparse_val.clone() if isinstance(sparse_val, torch.Tensor) else sparse_val
                                )
                                continue
                            mm_payload[mm_key] = to_payload_element(
                                element=mm_val,
                                idx=idx,
                                start=start,
                                end=end,
                                pass_lists_through=False,
                                seq_len=hidden_seq_len,
                                scheduled_seq_len=scheduled_seq_len,
                            )
                    payload.update(mm_payload)
                pooler_output.append(flatten_payload(payload))
                # R5 bit-identical dump: OMNI_TALKER_DUMP_DIR=<dir> writes, per
                # request, the codec token sequence (from the merged mm deltas)
                # + engine token sequence (from valid_sampled_token_ids) for
                # comparison between K=1 baseline and K=2 runs.
                _dump_dir = os.environ.get("OMNI_TALKER_DUMP_DIR") or getattr(self, "_talker_dump_dir", None)
                if _dump_dir:
                    try:
                        _codes = mm_cpu.get("codes.audio")
                        _deltas = []
                        if isinstance(_codes, (list, tuple)) and idx < len(_codes):
                            _c = _codes[idx]
                            if isinstance(_c, torch.Tensor):
                                _deltas = [int(v) for v in _c.reshape(-1).tolist()]
                        _etoks = []
                        if idx < len(valid_sampled_token_ids) and valid_sampled_token_ids[idx]:
                            _etoks = [int(v) for v in valid_sampled_token_ids[idx]]
                        _req = str(rid)
                        with open(os.path.join(_dump_dir, f"codec_{_req}.txt"), "a") as _f:
                            _f.write(",".join(str(v) for v in _deltas) + "\n")
                        with open(os.path.join(_dump_dir, f"engine_{_req}.txt"), "a") as _f:
                            _f.write(",".join(str(v) for v in _etoks) + "\n")
                    except Exception as _exc:  # dump must never break serving
                        logger.warning("talker dump failed: %s", _exc)

        pooler_output = pooler_output or []
        if self._async_chunk and stage_sends_async_output(self.model_config):
            pooler_inter, pooler_client = partition_payload_list(pooler_output)
        else:
            pooler_inter, pooler_client = pooler_output, pooler_output

        if pooler_inter and self._should_accumulate_full_payload_output():
            with record_function_or_nullcontext("omni_output_builder:accumulate_full_payload_output"):
                for i, rid in enumerate(req_ids_output_copy):
                    req_state = self.requests.get(rid)
                    if req_state is not None and pooler_inter[i]:
                        self.accumulate_full_payload_output(rid, pooler_inter[i], req_state)

        inter_stage_outputs = self._build_multimodal_outputs(pooler_inter)
        mmo = (
            inter_stage_outputs if pooler_client is pooler_inter else self._build_multimodal_outputs(pooler_client)
        )
        model_runner_output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=None,
            multimodal_outputs=mmo,
            inter_stage_outputs=inter_stage_outputs,
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
        model_runner_output.routed_experts = routed_experts_lists
        with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
            model_runner_output.omni_connector_output = self.get_omni_connector_output()
        return model_runner_output

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
    #  -------------------------------------- Omni-new -------------------------------------------------
