# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable exact-signature NPUGraph capture and replay helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


@dataclass
class CapturedDeviceGraph:
    graph: _ReplayGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            for static, current in zip(self.static_inputs, inputs, strict=True):
                static.copy_(current)
            self.graph.replay()
            # Graph outputs are persistent and overwritten by the next replay.
            # Clone before they become request-owned streaming cache entries.
            return tuple(output.detach().clone() for output in self.static_outputs)


def _tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), str(value.device)


class NPUExactGraphRunner:
    """Capture and replay tensor-only functions for exact NPU signatures."""

    def __init__(
        self,
        *,
        max_graphs: int = 32,
        component_name: str = "device graph",
        disable_config_hint: str = "disable graph capture",
        recoverable: bool = True,
    ) -> None:
        self.max_graphs = max(0, int(max_graphs))
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        # A failed capture used to permanently poison the stage process (the
        # torch-npu allocator/RNG state may be invalid). Mark-and-skip keeps
        # the stage serving eagerly: this key falls back permanently, while
        # other signatures can still capture and replay.
        self.recoverable = recoverable
        self._enabled = self.max_graphs > 0
        self._graphs: dict[tuple[object, ...], CapturedDeviceGraph] = {}
        self._failed_keys: set[tuple[object, ...]] = set()
        self._hits = 0
        self._graph_pool: object | None = None

    @staticmethod
    def is_supported() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and all(
            hasattr(npu, name)
            for name in (
                "NPUGraph",
                "graph",
                "is_current_stream_capturing",
                "synchronize",
            )
        )

    @staticmethod
    def _stream_is_capturing() -> bool:
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except (RuntimeError, TypeError):
            return False

    def _eligible(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        if not self._enabled:
            return False
        if self._failed_keys and not self.recoverable:
            # Hard-fail mode still treats any failure as a poisoned process.
            return False
        return (
            bool(inputs)
            and all(value.device.type == "npu" for value in inputs)
            and self.is_supported()
            and not self._stream_is_capturing()
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captures": len(self._graphs),
            "failed": len(self._failed_keys),
            "hits": self._hits,
        }

    def capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> CapturedDeviceGraph:
        npu = torch.npu
        static_inputs = tuple(value.detach().clone() for value in inputs)
        npu.synchronize()
        graph = npu.NPUGraph()
        if self._graph_pool is None:
            from vllm.platforms import current_platform

            self._graph_pool = current_platform.get_global_graph_pool()
        with torch.inference_mode(), npu.graph(graph, pool=self._graph_pool):
            static_outputs = compute(*static_inputs)
        npu.synchronize()
        return CapturedDeviceGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
        )

    def run(
        self,
        operation: str,
        inputs: tuple[torch.Tensor, ...],
        constants: tuple[object, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        if self._failed_keys and not self.recoverable:
            raise RuntimeError(
                f"{self.component_name} cannot continue after a failed NPUGraph capture; "
                f"restart the stage process and {self.disable_config_hint} before retrying."
            )
        if not self._eligible(inputs):
            return compute(*inputs)

        key = (
            operation,
            constants,
            tuple(_tensor_signature(value) for value in inputs),
        )
        if key in self._failed_keys:
            # Recoverable mode: this signature already failed capture once;
            # serve it eagerly forever instead of poisoning the stage.
            return compute(*inputs)
        graph = self._graphs.get(key)
        if graph is not None:
            self._hits += 1
            if self._hits == 1:
                logger.info("%s started NPUGraph replay", self.component_name)
            return graph.replay(inputs)

        # Prime lazy kernels and allocator state before capture. The next call
        # with the same exact tensor signature replays this graph.
        eager_outputs = compute(*inputs)
        if len(self._graphs) >= self.max_graphs:
            logger.warning_once(
                "%s reached the %d-entry NPUGraph limit; new tensor shapes will use eager execution.",
                self.component_name,
                self.max_graphs,
            )
            return eager_outputs
        try:
            self._graphs[key] = self.capture(inputs, compute)
        except Exception as exc:
            self._failed_keys.add(key)
            if self.recoverable:
                # Failed capture invalidates graph-mode assumptions for this
                # signature only. Already-captured graphs keep replaying (they
                # captured cleanly), and this signature serves eagerly forever
                # instead of poisoning the whole stage process.
                logger.exception(
                    "%s failed to capture NPUGraph for %s; falling back to "
                    "eager execution for this tensor signature.",
                    self.component_name,
                    operation,
                )
                return eager_outputs
            self._enabled = False
            logger.exception(
                "%s failed to capture NPUGraph for %s; the torch-npu "
                "allocator/RNG capture state may be invalid and this stage "
                "process must be restarted.",
                self.component_name,
                operation,
            )
            raise RuntimeError(
                f"{self.component_name} NPUGraph capture failed for {operation}; "
                f"restart the stage process. To run eagerly, {self.disable_config_hint}."
            ) from exc
        logger.info(
            "%s captured NPUGraph %d/%d for %s",
            self.component_name,
            len(self._graphs),
            self.max_graphs,
            operation,
        )
        return eager_outputs
